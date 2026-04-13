"""Play N matches using a saved checkpoint and record them as .bgar files.

Loads the hero policy from a checkpoint, picks an opponent from the pool
(the highest-ELO snapshot by default — i.e. the strongest training-mate),
runs sequentially episode-by-episode with recording on, and writes
replay files to RECORD_DIR.

Usage:
    python record_from_checkpoint.py [--checkpoint PATH] [--out DIR]
                                     [--episodes N] [--opp POOL_ID|best|latest|random]

Examples:
    python record_from_checkpoint.py
    python record_from_checkpoint.py --episodes 20 --opp best
    python record_from_checkpoint.py --checkpoint checkpoints/latest.pt --opp 3

Deterministic given --seed. Recorded files are named matchNNNN.bgar.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import brogameagent as bg
import selfplay_train as st   # reuse Policy / sample_actions / helpers


def pick_opponent(blob: dict, which: str) -> dict:
    pool = blob["pool"]
    if not pool:
        raise RuntimeError("checkpoint has an empty opponent pool")
    if which == "best":
        return max(pool, key=lambda e: e["elo"])
    if which == "latest":
        return max(pool, key=lambda e: e["id"])
    if which == "random":
        return pool[np.random.randint(len(pool))]
    try:
        pool_id = int(which)
    except ValueError:
        raise RuntimeError(f"--opp must be an int id, 'best', 'latest', or 'random' (got {which!r})")
    for e in pool:
        if e["id"] == pool_id:
            return e
    raise RuntimeError(f"pool id {pool_id} not found (available: "
                       f"{[e['id'] for e in pool]})")


@torch.no_grad()
def sample_single_env(policy: st.Policy, obs_np: np.ndarray,
                      mask_np: np.ndarray, enemy_ids_np: np.ndarray,
                      device: str):
    """Sample (move, target_id, ability_id) for env 0 only. Returns numpy
    arrays shaped (1, 2), (1,), (1,) ready for apply_actions_raw."""
    obs_t  = torch.from_numpy(obs_np[:1]).to(device)
    emask  = torch.from_numpy(mask_np[:1, :st.K_ENEMIES]).to(device)
    amask  = torch.from_numpy(mask_np[:1, st.K_ENEMIES:]).to(device)
    move, attack, ability, _, _ = st.sample_actions(policy, obs_t, emask, amask)
    return st.slots_to_action_arrays(move, attack, ability, enemy_ids_np[:1])


def run_match(hero: st.Policy, opp: st.Policy, seed: int, out_path: str,
              device: str, dt: float, max_steps_ep: int,
              action_repeat: int) -> dict:
    cfg = bg.VecSimConfig()
    cfg.num_envs = 1
    cfg.dt = dt
    cfg.max_steps_per_episode = max_steps_ep
    vec = bg.VecSimulation(cfg)
    vec.seed_and_reset(seed)

    rec = bg.Recorder()
    if not rec.open(out_path, seed, seed, dt):
        raise RuntimeError(f"failed to open recorder at {out_path}")
    rec.write_roster([vec.hero(0), vec.opponent(0)])

    sub_step = 0
    info = {"winner": -1, "policy_steps": 0}
    for t in range(max_steps_ep // action_repeat + 1):
        hobs  = vec.observe(bg.VecSimulation.HERO_ID)
        hmask, hids = vec.action_mask(bg.VecSimulation.HERO_ID)
        oobs  = vec.observe(bg.VecSimulation.OPPONENT_ID)
        omask, oids = vec.action_mask(bg.VecSimulation.OPPONENT_ID)

        h_move, h_tid, h_ab = sample_single_env(hero, hobs, hmask, hids, device)
        o_move, o_tid, o_ab = sample_single_env(opp,  oobs, omask, oids, device)

        for _k in range(action_repeat):
            vec.apply_actions_raw(bg.VecSimulation.HERO_ID,     h_move, h_tid, h_ab)
            vec.apply_actions_raw(bg.VecSimulation.OPPONENT_ID, o_move, o_tid, o_ab)
            vec.step()
            rec.record_frame(sub_step, sub_step * dt, vec.world(0))
            sub_step += 1

        done, winner = vec.dones()
        info["policy_steps"] = t + 1
        if done[0]:
            info["winner"] = int(winner[0])
            break

    rec.close()
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    p.add_argument("--out",        default="replays_checkpoint")
    p.add_argument("--episodes",   type=int, default=5)
    p.add_argument("--opp",        default="best",
                   help="pool id | 'best' | 'latest' | 'random'")
    p.add_argument("--seed",       type=int, default=1234)
    p.add_argument("--device",     default=st.DEVICE)
    p.add_argument("--sim-dt",     type=float, default=st.SIM_DT)
    p.add_argument("--max-steps",  type=int,   default=st.MAX_STEPS_EP)
    p.add_argument("--repeat",     type=int,   default=st.ACTION_REPEAT)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"loading checkpoint: {args.checkpoint}")
    blob = st.load_checkpoint(args.checkpoint, args.device)
    tr = blob["training"]
    pool_summary = [f"{e['id']}:{int(e['elo'])}" for e in blob['pool']]
    print(f"  learner_elo={tr['learner_elo']:.1f}  "
          f"update={tr['update']}  "
          f"pool={pool_summary}")

    hero = st.restore_policy_from_blob(blob, args.device).eval()
    for param in hero.parameters():
        param.requires_grad_(False)

    opp_entry = pick_opponent(blob, args.opp)
    print(f"  opponent: pool id {opp_entry['id']} (elo={opp_entry['elo']:.0f})")
    opp = st.Policy().to(args.device).eval()
    opp.load_state_dict(opp_entry["state_dict"])
    for param in opp.parameters():
        param.requires_grad_(False)

    wins = losses = draws = 0
    for i in range(args.episodes):
        out_path = os.path.join(args.out, f"match{i:04d}.bgar")
        info = run_match(hero, opp, seed=args.seed + i, out_path=out_path,
                         device=args.device, dt=args.sim_dt,
                         max_steps_ep=args.max_steps, action_repeat=args.repeat)
        w = info["winner"]
        if w == bg.VecSimulation.HERO_ID:
            verdict = "WIN "; wins += 1
        elif w == bg.VecSimulation.OPPONENT_ID:
            verdict = "LOSS"; losses += 1
        else:
            verdict = "DRAW"; draws += 1
        print(f"  [{i:3d}] {verdict}  steps={info['policy_steps']}  -> {out_path}")

    print(f"done. W/L/D = {wins}/{losses}/{draws}  ({args.episodes} episodes)")


if __name__ == "__main__":
    main()
