# brogameagent — Python bindings

Pybind11 bindings for the C++ library plus a minimal PyTorch training
example.

## Build

From the repo root:

```sh
pip install pybind11 numpy torch
cmake -S . -B build_py -DBROGAMEAGENT_PYTHON=ON -DBROGAMEAGENT_TESTS=OFF
cmake --build build_py --config Release
```

The extension is placed next to this README as
`brogameagent.cp<pyver>-<platform>.pyd` (Windows) or `brogameagent.*.so`
(Linux/Mac), importable from this directory.

## Quick check

```sh
cd python
python -c "import brogameagent as bg; print(bg.observation.TOTAL)"
```

## Run the training example

```sh
cd python
python example_train.py
```

Scenario: a hero trains with REINFORCE to chase and kill a stationary
dummy spawned at a random offset. Expect the `kill_rate(50)` column to
climb over a few hundred episodes. This is an intentionally minimal demo
— no baseline, no entropy bonus, no vectorized envs. Extend from here.

## API surface at a glance

```python
import brogameagent as bg
import numpy as np

world = bg.World()
world.seed(42)

hero = bg.Agent()
hero.unit.id = 1
hero.unit.team_id = 0
hero.unit.move_speed = 6.0
hero.set_position(0, 0)
world.add_agent(hero)

# Observation + action mask as numpy arrays.
obs = bg.observation.build(hero, world)          # (58,) float32
mask, enemy_ids = bg.action_mask.build(hero, world)

# Drive via AgentAction.
action = bg.AgentAction(move_x=0.0, move_z=-1.0, attack_target_id=-1)
world.apply_action(hero, action, 1/60)

# Simulation harness (one policy per agent id).
sim = bg.Simulation(world)
sim.add_policy(1, lambda self, w: bg.AgentAction(move_z=-1.0))
sim.run_steps(1/60, 30)

# Snapshot / restore for episode resets.
snap = world.snapshot()
world.restore(snap)
```
