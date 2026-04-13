"""Minimal REINFORCE training loop for a brogameagent hero.

Scenario: hero starts at origin. A stationary enemy spawns at a random
position within ~10 units. The hero must close the distance and kill it.
Episode ends when the enemy dies or MAX_STEPS elapse.

Policy: MLP producing (1) a 2D gaussian over (move_x, move_z) and (2) a
discrete head over K_ENEMIES+1 choices (attack nearest-K or no-op).
Trained with vanilla REINFORCE (no baseline — it's intentionally minimal).

Run:
    python example_train.py

Build the extension first (from repo root):
    cmake -S . -B build_py -DBROGAMEAGENT_PYTHON=ON -DBROGAMEAGENT_TESTS=OFF
    cmake --build build_py --config Release

The extension lands in this directory as brogameagent.cp<ver>-<plat>.pyd/.so.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import brogameagent as bg


# --- Hyperparameters ---

DT              = 1.0 / 30.0
MAX_STEPS       = 300
EPISODES        = 400
HIDDEN          = 64
LR              = 3e-4
GAMMA           = 0.99
SEED            = 1234
LOG_EVERY       = 10


# --- Environment ---


def build_world(rng: np.random.Generator, seed: int):
    world = bg.World()
    world.seed(seed)

    hero = bg.Agent()
    hero.unit.id = 1
    hero.unit.team_id = 0
    hero.unit.hp = 200.0
    hero.unit.max_hp = 200.0
    hero.unit.damage = 20.0
    hero.unit.attack_range = 2.0
    hero.unit.attacks_per_sec = 2.0
    hero.unit.move_speed = 7.0
    hero.unit.radius = 0.4
    hero.set_position(0.0, 0.0)
    hero.set_max_accel(40.0)
    hero.set_max_turn_rate(math.pi * 4.0)
    world.add_agent(hero)

    # Stationary dummy within a ring around origin.
    angle = float(rng.uniform(-math.pi, math.pi))
    dist = float(rng.uniform(4.0, 10.0))
    ex, ez = dist * math.cos(angle), dist * math.sin(angle)

    enemy = bg.Agent()
    enemy.unit.id = 2
    enemy.unit.team_id = 1
    enemy.unit.hp = 60.0
    enemy.unit.max_hp = 60.0
    enemy.unit.radius = 0.4
    enemy.set_position(ex, ez)
    world.add_agent(enemy)

    return world, hero, enemy


# --- Policy ---


class Policy(nn.Module):
    def __init__(self, obs_dim: int, k_enemies: int, hidden: int = HIDDEN):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.move_mean  = nn.Linear(hidden, 2)
        self.move_logstd = nn.Parameter(torch.full((2,), -0.5))
        # k_enemies attack slots + 1 "no-op" slot
        self.attack_head = nn.Linear(hidden, k_enemies + 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        move_mean = torch.tanh(self.move_mean(h))
        move_std = self.move_logstd.exp().expand_as(move_mean)
        attack_logits = self.attack_head(h)
        return move_mean, move_std, attack_logits


def sample_action(policy: Policy,
                  obs: np.ndarray,
                  mask_enemies: np.ndarray,
                  enemy_ids: np.ndarray):
    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    move_mean, move_std, attack_logits = policy(obs_t)

    move_dist = torch.distributions.Normal(move_mean, move_std)
    move = move_dist.sample()
    move_logp = move_dist.log_prob(move).sum(dim=-1)

    # Last slot (no-op) is always legal; enemy slots masked by mask_enemies.
    K = mask_enemies.shape[0]
    legal = np.concatenate([mask_enemies, np.ones(1, dtype=np.float32)])
    legal_t = torch.from_numpy(legal).bool().unsqueeze(0)
    logits = attack_logits.masked_fill(~legal_t, -1e9)
    attack_dist = torch.distributions.Categorical(logits=logits)
    attack_choice = attack_dist.sample()
    attack_logp = attack_dist.log_prob(attack_choice)

    choice = int(attack_choice.item())
    target_id = -1 if choice == K else int(enemy_ids[choice])

    move_np = move.squeeze(0).detach().cpu().numpy()
    action = bg.AgentAction(
        move_x=float(move_np[0]),
        move_z=float(move_np[1]),
        attack_target_id=target_id,
    )
    total_logp = move_logp + attack_logp
    return action, total_logp.squeeze(0)


# --- Rollout ---


def run_episode(policy: Policy, rng: np.random.Generator, seed: int):
    world, hero, enemy = build_world(rng, seed)
    rt = bg.RewardTracker()
    rt.reset(hero, world)

    log_probs = []
    rewards = []
    info = {"steps": 0, "killed": False, "final_dist": 0.0}

    for step in range(MAX_STEPS):
        obs = bg.observation.build(hero, world)
        mask, ids = bg.action_mask.build(hero, world)
        mask_enemies = mask[: bg.action_mask.N_ENEMY_SLOTS]

        action, logp = sample_action(policy, obs, mask_enemies, ids)
        world.apply_action(hero, action, DT)

        # Advance the rest of the world (scripted agents, cooldowns, projectiles).
        # We don't use Simulation here so we can inject rewards between steps.
        for a in world.agents:
            if a.unit.id == hero.unit.id:
                continue
            a.update(DT)
            a.unit.tick_cooldowns(DT)
        world.step_projectiles(DT)
        world.cull_projectiles()

        d = rt.consume(hero, world)
        reward = (
            d.damage_dealt
            - 0.5 * d.damage_taken
            + 100.0 * d.kills
            - 0.02                          # per-step penalty to discourage stalling
        )
        log_probs.append(logp)
        rewards.append(reward)

        if not enemy.unit.alive:
            info["killed"] = True
            info["steps"] = step + 1
            break
    else:
        info["steps"] = MAX_STEPS

    dx = hero.x - enemy.x
    dz = hero.z - enemy.z
    info["final_dist"] = math.sqrt(dx * dx + dz * dz)
    return log_probs, rewards, info


# --- REINFORCE update ---


def reinforce_update(optimizer: optim.Optimizer,
                     log_probs: list[torch.Tensor],
                     rewards: list[float]):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.append(G)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    # Normalize for stability.
    if returns_t.numel() > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    logp_t = torch.stack(log_probs)
    loss = -(logp_t * returns_t).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# --- Main ---


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    policy = Policy(obs_dim=bg.observation.TOTAL,
                    k_enemies=bg.action_mask.N_ENEMY_SLOTS)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    kills_window = []
    for ep in range(1, EPISODES + 1):
        log_probs, rewards, info = run_episode(policy, rng, seed=SEED + ep)
        loss = reinforce_update(optimizer, log_probs, rewards)
        kills_window.append(1 if info["killed"] else 0)
        if len(kills_window) > 50:
            kills_window.pop(0)

        if ep % LOG_EVERY == 0 or ep == 1:
            kill_rate = sum(kills_window) / len(kills_window)
            total_r = sum(rewards)
            print(
                f"ep {ep:4d}  return={total_r:8.2f}  "
                f"steps={info['steps']:3d}  "
                f"killed={info['killed']!s:5}  "
                f"final_dist={info['final_dist']:5.2f}  "
                f"kill_rate(50)={kill_rate:.2f}  "
                f"loss={loss:8.3f}"
            )

    print("done.")


if __name__ == "__main__":
    main()
