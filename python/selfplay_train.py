"""Self-play PPO trainer with an ELO-rated opponent pool.

The learner plays a 1v1 match in each of NUM_ENVS parallel arenas. Per-env
opponents are sampled from a frozen pool; periodically the current learner
is snapshotted into the pool. ELO ratings update at the end of every match
so you can watch which snapshots stay competitive.

Run after building the extension (see python/README.md):

    cd python
    python selfplay_train.py

Throughput knobs at the top of the file. The defaults target ~2-3 GiB on a
4090 with NUM_ENVS=512 and HIDDEN=128.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import brogameagent as bg


# ─── Hyperparameters ────────────────────────────────────────────────────────

NUM_ENVS         = 2048
ROLLOUT_LEN      = 64
PPO_EPOCHS       = 4
MINIBATCH        = 8192
LR               = 3e-4
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
CLIP             = 0.2
ENTROPY_COEF     = 0.01
VALUE_COEF       = 0.5
MAX_GRAD_NORM    = 0.5

HIDDEN           = 256
SIM_DT           = 0.016                # one sim tick = 16 ms
ACTION_REPEAT    = 4                    # policy commits an action for 4 sim ticks (64 ms)
MAX_STEPS_EP     = 600                  # in sim ticks (~9.6 s game time)

UPDATES          = 200
WALL_BUDGET_SEC  = None                 # if set, stops when wall clock exceeds this
SNAPSHOT_EVERY   = 10
MAX_POOL         = 8
ELO_K            = 32.0
ELO_INIT         = 1000.0

SEED             = 1234
DEVICE           = "cuda:0" if torch.cuda.is_available() else "cpu"
RECORD_DIR       = "replays_selfplay"   # None to disable
RECORD_EVERY     = 25                    # record every Nth update (env 0 only)
LOG_EVERY        = 1


# ─── Policy ─────────────────────────────────────────────────────────────────


OBS_DIM     = bg.observation.TOTAL
K_ENEMIES   = bg.action_mask.N_ENEMY_SLOTS
N_ABILITIES = bg.action_mask.N_ABILITY_SLOTS
ATTACK_DIM  = K_ENEMIES   + 1   # +1 = "no attack" slot, always legal
ABILITY_DIM = N_ABILITIES + 1   # +1 = "no cast"   slot, always legal


class Policy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.move_mean    = nn.Linear(hidden, 2)
        self.move_logstd  = nn.Parameter(torch.full((2,), -0.5))
        self.attack_head  = nn.Linear(hidden, ATTACK_DIM)
        self.ability_head = nn.Linear(hidden, ABILITY_DIM)
        self.value_head   = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return (
            torch.tanh(self.move_mean(h)),
            self.move_logstd.exp().expand(obs.shape[0], 2),
            self.attack_head(h),
            self.ability_head(h),
            self.value_head(h).squeeze(-1),
        )


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    """Build a Categorical with -inf logits on illegal slots."""
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def _legal_with_noop(mask_slots: torch.Tensor) -> torch.Tensor:
    """Append an always-legal "no-op" column to a (B, K) bool mask."""
    noop = torch.ones(mask_slots.shape[0], 1, device=mask_slots.device,
                      dtype=mask_slots.dtype)
    return torch.cat([mask_slots, noop], dim=-1).bool()


def sample_actions(policy: Policy,
                   obs: torch.Tensor,
                   mask_enemies: torch.Tensor,
                   mask_abilities: torch.Tensor):
    """Sample (move, attack, ability) and return all + summed logp + value."""
    move_mean, move_std, atk_logits, ab_logits, value = policy(obs)

    move_dist = torch.distributions.Normal(move_mean, move_std)
    move      = move_dist.sample()
    move_logp = move_dist.log_prob(move).sum(dim=-1)

    atk_dist = _masked_categorical(atk_logits, _legal_with_noop(mask_enemies))
    attack   = atk_dist.sample()
    atk_logp = atk_dist.log_prob(attack)

    ab_dist  = _masked_categorical(ab_logits, _legal_with_noop(mask_abilities))
    ability  = ab_dist.sample()
    ab_logp  = ab_dist.log_prob(ability)

    return move, attack, ability, move_logp + atk_logp + ab_logp, value


def evaluate_actions(policy: Policy,
                     obs: torch.Tensor,
                     mask_enemies: torch.Tensor,
                     mask_abilities: torch.Tensor,
                     move: torch.Tensor,
                     attack: torch.Tensor,
                     ability: torch.Tensor):
    """Recompute logp, entropy, value for stored (move, attack, ability) (PPO)."""
    move_mean, move_std, atk_logits, ab_logits, value = policy(obs)

    move_dist = torch.distributions.Normal(move_mean, move_std)
    move_logp = move_dist.log_prob(move).sum(dim=-1)
    move_ent  = move_dist.entropy().sum(dim=-1)

    atk_dist = _masked_categorical(atk_logits, _legal_with_noop(mask_enemies))
    atk_logp = atk_dist.log_prob(attack)
    atk_ent  = atk_dist.entropy()

    ab_dist  = _masked_categorical(ab_logits, _legal_with_noop(mask_abilities))
    ab_logp  = ab_dist.log_prob(ability)
    ab_ent   = ab_dist.entropy()

    return (move_logp + atk_logp + ab_logp,
            move_ent  + atk_ent  + ab_ent,
            value)


# ─── Opponent pool ──────────────────────────────────────────────────────────


class OpponentPool:
    """Frozen policy snapshots indexed by integer id, with ELO ratings.

    Keeping all networks on-device (they're small) lets us do a forward pass
    per opponent without any state-dict shuffling per tick.
    """

    def __init__(self, max_size: int, device: str):
        self.max_size = max_size
        self.device = device
        self.entries: list[dict] = []   # {'id', 'net', 'elo', 'games'}
        self._next_id = 0

    def __len__(self) -> int:
        return len(self.entries)

    def ids(self) -> list[int]:
        return [e["id"] for e in self.entries]

    def add(self, policy: Policy, elo: float) -> int:
        net = copy.deepcopy(policy).to(self.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        entry = {"id": self._next_id, "net": net, "elo": elo, "games": 0}
        self._next_id += 1
        self.entries.append(entry)

        # Eviction: keep the original snapshot (entries[0]) plus top-(max-1) by ELO.
        if len(self.entries) > self.max_size:
            keep = [self.entries[0]] + sorted(self.entries[1:],
                                              key=lambda e: -e["elo"])[: self.max_size - 1]
            self.entries = keep
        return entry["id"]

    def index_of(self, opp_id: int) -> int:
        for i, e in enumerate(self.entries):
            if e["id"] == opp_id:
                return i
        raise KeyError(opp_id)

    def network(self, idx: int) -> Policy:
        return self.entries[idx]["net"]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, len(self.entries), size=n)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(learner_elo: float, opp_elo: float, learner_score: float,
               k: float = ELO_K) -> tuple[float, float]:
    e_l = expected_score(learner_elo, opp_elo)
    delta = k * (learner_score - e_l)
    return learner_elo + delta, opp_elo - delta


# ─── Action conversion (numpy ↔ AgentAction list) ───────────────────────────


def slots_to_action_arrays(move_t: torch.Tensor,
                            attack_t: torch.Tensor,
                            ability_t: torch.Tensor,
                            enemy_ids_np: np.ndarray):
    """Tensors-on-device + per-env enemy id table → numpy arrays ready for
    VecSimulation.apply_actions_raw.

    Slot conventions:
      attack  == K_ENEMIES     → no auto-attack (target_id = -1)
      attack  <  K_ENEMIES     → enemy slot, mapped through enemy_ids
      ability == N_ABILITIES   → no cast   (use_ability_id = -1)
      ability <  N_ABILITIES   → ability slot (0..7)
    """
    N = move_t.shape[0]
    move_np    = move_t.detach().cpu().numpy().astype(np.float32, copy=False)
    attack_np  = attack_t.detach().cpu().numpy().astype(np.int32,  copy=False)
    ability_np = ability_t.detach().cpu().numpy().astype(np.int32, copy=False)

    safe_idx   = np.clip(attack_np, 0, K_ENEMIES - 1)
    rows       = np.arange(N)
    target_ids = np.where(
        attack_np == K_ENEMIES,
        np.int32(-1),
        enemy_ids_np[rows, safe_idx].astype(np.int32, copy=False),
    ).astype(np.int32, copy=False)

    ability_ids = np.where(
        ability_np == N_ABILITIES,
        np.int32(-1),
        ability_np,
    ).astype(np.int32, copy=False)
    return move_np, target_ids, ability_ids


# ─── Opponent forward pass (grouped by opponent network) ────────────────────


@torch.no_grad()
def opponent_action_arrays(pool: OpponentPool,
                            opp_idx_per_env: np.ndarray,
                            obs_t: torch.Tensor,
                            mask_enemies_t: torch.Tensor,
                            mask_abilities_t: torch.Tensor,
                            enemy_ids_np: np.ndarray):
    """One forward pass per opponent network, then scatter the sampled
    actions back and return the numpy tuple expected by apply_actions_raw."""
    n = obs_t.shape[0]
    move_buf    = torch.zeros((n, 2), device=obs_t.device)
    attack_buf  = torch.zeros((n,), dtype=torch.long, device=obs_t.device)
    ability_buf = torch.zeros((n,), dtype=torch.long, device=obs_t.device)
    for grp_idx in np.unique(opp_idx_per_env):
        mask = (opp_idx_per_env == grp_idx)
        idx = np.where(mask)[0]
        idx_t = torch.from_numpy(idx).to(obs_t.device)
        sub_obs   = obs_t.index_select(0, idx_t)
        sub_emask = mask_enemies_t.index_select(0, idx_t)
        sub_amask = mask_abilities_t.index_select(0, idx_t)
        net = pool.network(int(grp_idx))
        move, attack, ability, _, _ = sample_actions(net, sub_obs, sub_emask, sub_amask)
        move_buf.index_copy_(0, idx_t, move)
        attack_buf.index_copy_(0, idx_t, attack)
        ability_buf.index_copy_(0, idx_t, ability)
    return slots_to_action_arrays(move_buf, attack_buf, ability_buf, enemy_ids_np)


# ─── PPO update ─────────────────────────────────────────────────────────────


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """rewards/values/dones: (T, N). last_values: (N,). Returns adv (T,N), ret (T,N)."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        next_v = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret


def ppo_update(policy, optimizer, batch):
    (obs, emask, amask, move, attack, ability,
     old_logp, adv, ret, value) = batch
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    losses = []
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(N, device=obs.device)
        for start in range(0, N, MINIBATCH):
            mb = perm[start:start + MINIBATCH]
            new_logp, entropy, new_value = evaluate_actions(
                policy, obs[mb], emask[mb], amask[mb],
                move[mb], attack[mb], ability[mb]
            )
            ratio = (new_logp - old_logp[mb]).exp()
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.mse_loss(new_value, ret[mb])
            ent         = entropy.mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))


# ─── Trainer ────────────────────────────────────────────────────────────────


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print(f"device={DEVICE}  num_envs={NUM_ENVS}  rollout={ROLLOUT_LEN}  "
          f"hidden={HIDDEN}  obs_dim={OBS_DIM}")

    # Build vec env.
    cfg = bg.VecSimConfig()
    cfg.num_envs = NUM_ENVS
    cfg.dt = SIM_DT
    cfg.max_steps_per_episode = MAX_STEPS_EP
    vec = bg.VecSimulation(cfg)
    vec.seed_and_reset(SEED)

    # Policy + opponent pool.
    policy = Policy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    pool = OpponentPool(MAX_POOL, DEVICE)
    pool.add(policy, ELO_INIT)
    learner_elo = ELO_INIT

    # Per-env opponent assignment (resampled when an env's episode ends).
    opp_idx = pool.sample(NUM_ENVS, rng).astype(np.int64)

    # Persistent buffers reused across rollouts.
    obs_buf      = torch.zeros(ROLLOUT_LEN, NUM_ENVS, OBS_DIM,        device=DEVICE)
    emask_buf    = torch.zeros(ROLLOUT_LEN, NUM_ENVS, K_ENEMIES,      device=DEVICE)
    amask_buf    = torch.zeros(ROLLOUT_LEN, NUM_ENVS, N_ABILITIES,    device=DEVICE)
    move_buf     = torch.zeros(ROLLOUT_LEN, NUM_ENVS, 2,              device=DEVICE)
    attack_buf   = torch.zeros(ROLLOUT_LEN, NUM_ENVS, dtype=torch.long, device=DEVICE)
    ability_buf  = torch.zeros(ROLLOUT_LEN, NUM_ENVS, dtype=torch.long, device=DEVICE)
    logp_buf     = torch.zeros(ROLLOUT_LEN, NUM_ENVS, device=DEVICE)
    value_buf    = torch.zeros(ROLLOUT_LEN, NUM_ENVS, device=DEVICE)
    reward_buf   = torch.zeros(ROLLOUT_LEN, NUM_ENVS, device=DEVICE)
    done_buf     = torch.zeros(ROLLOUT_LEN, NUM_ENVS, device=DEVICE)

    # Episode counters for logging / ELO bookkeeping.
    episodes_done   = 0
    wins, losses_, draws = 0, 0, 0

    if RECORD_DIR:
        os.makedirs(RECORD_DIR, exist_ok=True)

    t0 = time.time()
    upd = 0
    while True:
        upd += 1
        if WALL_BUDGET_SEC is not None:
            if time.time() - t0 >= WALL_BUDGET_SEC:
                print(f"wall budget reached after {upd-1} updates")
                break
        elif upd > UPDATES:
            break
        recorder = None
        if RECORD_DIR and (upd == 1 or upd % RECORD_EVERY == 0):
            path = os.path.join(RECORD_DIR, f"upd{upd:04d}_env0.bgar")
            recorder = bg.Recorder()
            if recorder.open(path, upd, SEED + upd, SIM_DT):
                recorder.write_roster([vec.hero(0), vec.opponent(0)])

        # ── Rollout ─────────────────────────────────────────────────────────
        for t in range(ROLLOUT_LEN):
            hobs_np = vec.observe(bg.VecSimulation.HERO_ID)
            hmask_np, hids_np = vec.action_mask(bg.VecSimulation.HERO_ID)
            oobs_np = vec.observe(bg.VecSimulation.OPPONENT_ID)
            omask_np, oids_np = vec.action_mask(bg.VecSimulation.OPPONENT_ID)

            hobs   = torch.from_numpy(hobs_np).to(DEVICE)
            hemask = torch.from_numpy(hmask_np[:, :K_ENEMIES]).to(DEVICE)
            hamask = torch.from_numpy(hmask_np[:, K_ENEMIES:]).to(DEVICE)
            with torch.no_grad():
                move, attack, ability, logp, value = sample_actions(
                    policy, hobs, hemask, hamask
                )

            obs_buf[t]     = hobs
            emask_buf[t]   = hemask
            amask_buf[t]   = hamask
            move_buf[t]    = move
            attack_buf[t]  = attack
            ability_buf[t] = ability
            logp_buf[t]    = logp
            value_buf[t]   = value

            h_move, h_target, h_ability = slots_to_action_arrays(
                move, attack, ability, hids_np
            )

            oobs_t   = torch.from_numpy(oobs_np).to(DEVICE)
            oemask_t = torch.from_numpy(omask_np[:, :K_ENEMIES]).to(DEVICE)
            oamask_t = torch.from_numpy(omask_np[:, K_ENEMIES:]).to(DEVICE)
            o_move, o_target, o_ability = opponent_action_arrays(
                pool, opp_idx, oobs_t, oemask_t, oamask_t, oids_np
            )

            # Action persistence: repeat the same (hero, opp) action for K sim
            # ticks so each policy decision spans ACTION_REPEAT * SIM_DT seconds.
            # Reward accumulator drains the entire window in one rewards() call.
            for _k in range(ACTION_REPEAT):
                vec.apply_actions_raw(bg.VecSimulation.HERO_ID,
                                      h_move, h_target, h_ability)
                vec.apply_actions_raw(bg.VecSimulation.OPPONENT_ID,
                                      o_move, o_target, o_ability)
                vec.step()
                if recorder is not None:
                    sub_t = t * ACTION_REPEAT + _k
                    recorder.record_frame(sub_t, sub_t * SIM_DT, vec.world(0))

            rh, _ro = vec.rewards()
            done, winner = vec.dones()

            reward_buf[t] = torch.from_numpy(rh).to(DEVICE)
            done_buf[t]   = torch.from_numpy(done.astype(np.float32)).to(DEVICE)

            # Handle terminations: ELO + resample opponent + reset env.
            if done.any():
                done_idxs = np.where(done == 1)[0]
                for env_i in done_idxs:
                    w = winner[env_i]
                    opp_pool_idx = int(opp_idx[env_i])
                    e_l = pool.entries[opp_pool_idx]["elo"]
                    if w == bg.VecSimulation.HERO_ID:
                        wins += 1
                        new_l, new_o = update_elo(learner_elo, e_l, 1.0)
                    elif w == bg.VecSimulation.OPPONENT_ID:
                        losses_ += 1
                        new_l, new_o = update_elo(learner_elo, e_l, 0.0)
                    else:
                        # Draws: no rating change. Aggression is biased via
                        # the in-game timeout reward, not the ELO metric —
                        # that keeps ELO readable across training.
                        draws += 1
                        new_l, new_o = learner_elo, e_l
                    learner_elo = new_l
                    pool.entries[opp_pool_idx]["elo"]   = new_o
                    pool.entries[opp_pool_idx]["games"] += 1
                    episodes_done += 1
                # Resample opponents for finished envs.
                new_opps = pool.sample(len(done_idxs), rng).astype(np.int64)
                opp_idx[done_idxs] = new_opps
                vec.reset_done()

        if recorder is not None:
            recorder.close()

        # Bootstrap final value for GAE.
        with torch.no_grad():
            hobs_np = vec.observe(bg.VecSimulation.HERO_ID)
            hobs    = torch.from_numpy(hobs_np).to(DEVICE)
            _, _, _, _, last_v = policy(hobs)

        adv, ret = compute_gae(reward_buf, value_buf, done_buf, last_v, GAMMA, GAE_LAMBDA)

        batch = (
            obs_buf.reshape(-1, OBS_DIM),
            emask_buf.reshape(-1, K_ENEMIES),
            amask_buf.reshape(-1, N_ABILITIES),
            move_buf.reshape(-1, 2),
            attack_buf.reshape(-1),
            ability_buf.reshape(-1),
            logp_buf.reshape(-1),
            adv.reshape(-1),
            ret.reshape(-1),
            value_buf.reshape(-1),
        )
        loss = ppo_update(policy, optimizer, batch)

        if upd % SNAPSHOT_EVERY == 0:
            new_id = pool.add(policy, learner_elo)
            print(f"  >> snapshot id={new_id} (pool size = {len(pool)})")

        if upd % LOG_EVERY == 0:
            elapsed = time.time() - t0
            sps = upd * NUM_ENVS * ROLLOUT_LEN / elapsed
            wr      = wins / max(1, episodes_done)
            decided = wins + losses_
            dwr     = wins / max(1, decided)
            pool_summary = ", ".join(
                f"{e['id']}:{int(e['elo'])}" for e in pool.entries
            )
            print(f"upd {upd:3d}  loss={loss:7.3f}  "
                  f"elo={learner_elo:6.1f}  "
                  f"W/L/D={wins}/{losses_}/{draws}  "
                  f"wr={wr:.2f} dec_wr={dwr:.2f}  "
                  f"sps={sps:6.0f}  "
                  f"pool=[{pool_summary}]")

    print(f"done. learner_elo={learner_elo:.1f}  "
          f"W/L/D = {wins}/{losses_}/{draws}")


if __name__ == "__main__":
    main()
