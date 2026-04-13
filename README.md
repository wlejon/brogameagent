# brogameagent

A C++20 simulation and reinforcement-learning substrate for MOBA-style 1v1
(and later NvN) combat, with Python bindings for training and a binary
replay format designed to be visualized by a companion renderer (`bro`).

## Intent

The project is split along one axis: **training and rendering are different
problems and should evolve independently.**

- **`brogameagent`** (this repo) runs headless: it owns the physics,
  combat, perception, observation, reward, and training loop. It produces
  fast numerical rollouts and optionally records them to disk.
- **`bro`** (sibling repo) owns visualization. It consumes the `.bgar`
  replay files produced here. Rendering does not block training; training
  does not carry a renderer's dependencies.

The contract between the two is a single header, `replay_format.h`,
containing only packed POD structs and `<cstdint>`. Any schema change is a
build-time mismatch on the `bro` side, never silent corruption.

The library is also built to host **self-play training with ELO-rated
opponent pools**, since that's the target workflow. Scenarios vary per
env (spawn randomization today; obstacles / arena variants are
future-work hooks), episodes are clean 1v1 matches, ratings update on
decisive outcomes, and draws are biased against via in-game reward
shaping rather than by corrupting the rating metric.

## Repository layout

```
include/brogameagent/        public headers (pure C++20)
  types.h unit.h agent.h world.h
  nav_grid.h steering.h perception.h
  observation.h action_mask.h reward.h
  projectile.h snapshot.h simulation.h
  vec_simulation.h              # batched self-play envs
  recorder.h replay_reader.h    # writing + reading .bgar
  replay_format.h               # on-disk schema (zero-dep)
src/                           implementations (one .cpp per header where applicable)
tests/test_main.cpp           ~100 single-file tests, no external framework
tools/replay_query.cpp        CLI inspector for .bgar files
python/
  bindings.cpp                 pybind11 wrapper
  example_train.py             single-env REINFORCE demo (scripted opponent)
  selfplay_train.py            vectorized self-play PPO + ELO pool
  README.md                    Python-side usage
```

## Building

This is a standard CMake project, but there are two conventional build
directories:

- `build/` — static lib + tests + `replay_query.exe` (main dev build)
- `build_py/` — same lib plus the pybind11 Python extension

```sh
# Tests + CLI tool
cmake -S . -B build
cmake --build build --config Release

# Python extension (separate dir to keep the .pyd away from the .lib)
cmake -S . -B build_py -DBROGAMEAGENT_PYTHON=ON -DBROGAMEAGENT_TESTS=OFF
cmake --build build_py --config Release
```

The Python extension lands in `python/brogameagent.cp<ver>-<plat>.pyd`
(Windows) or `.so` (Unix), importable from `python/`.

### Running tests

```sh
./build/tests/Release/brogameagent_test.exe
# → 102/102 tests passed
```

### `replay_query` CLI

```sh
./build/Release/replay_query.exe info     <file.bgar>
./build/Release/replay_query.exe roster   <file.bgar>
./build/Release/replay_query.exe frame    <file.bgar> <frame_idx>
./build/Release/replay_query.exe step     <file.bgar> <step_idx>
./build/Release/replay_query.exe agent    <file.bgar> <agent_id>
./build/Release/replay_query.exe events   <file.bgar> [attacker_id]
./build/Release/replay_query.exe dps      <file.bgar>
./build/Release/replay_query.exe dump     <file.bgar>
```

All output is tab-separated — pipe into `awk`, `csvkit`, `pandas`,
whatever.

## Core concepts

### `World` and `Agent`

A `World` holds a set of `Agent*`s plus shared obstacles, an event log,
projectiles, registered abilities, and a deterministic `mt19937_64` RNG.
Each `Agent` owns a `Unit` (HP, mana, damage, cooldowns, ability slots,
etc.), a 2D position and velocity, a movement yaw, and an aim yaw/pitch
decoupled from movement.

Two drive modes:

- **Scripted**: `setTarget(x, z)` + `update(dt)` — A*-pathed seek-arrive.
- **Policy**: `applyAction(AgentAction, dt)` — continuous-control with
  `maxAccel` / `maxTurnRate` clamps. This is what RL uses.

`Agent::applyAction` also ticks the acting agent's cooldowns internally,
so a policy-driven agent doesn't need a separate `tickCooldowns` call.

### Observation / action mask / reward

Three independent builders, all stable in layout so a policy keeps the
same input/output shape:

- `observation::build(self, world, float* out)` — 58 floats: self block
  (HP, mana, cooldowns, aim-vs-move delta), K=5 enemies, K=4 allies,
  all sorted nearest-first in the agent's local frame.
- `action_mask::build(self, world, outMask, outEnemyIds)` — 9 floats
  aligned to the observation's enemy slot order (5 attack slots +
  4 ability slots), plus the enemy id each slot refers to.
- `RewardTracker::consume(self, world)` — returns `(damageDealt,
  damageTaken, kills, deaths, distanceTravelled)` since the last
  `consume` or `reset`, using the world's event log as the source of
  truth.

### Combat resolution

- `World::resolveAttack(attacker, targetId)` — range/cooldown-gated
  auto-attack, writes a `DamageEvent` to the log.
- `World::resolveAbility(caster, slot, targetId)` — runs a registered
  ability function, gated by cooldown, mana, and (optional) range.
- `World::spawnProjectile(Projectile)` — Single / Pierce / AoE modes,
  optional homing via `targetId`, ownership for event attribution.

### Snapshot / restore

`World::snapshot()` / `restore(WorldSnapshot)` captures all resettable
state (agent positions, stats, projectiles, event log, RNG). Tests use
this to verify deterministic replay; the recorder does not, but it's
available for explicit checkpointing during long runs.

### `VecSimulation` — batched self-play envs

`VecSimulation` holds N independent 1v1 `World`s and exposes batched
paths that write directly into numpy-ready buffers. The per-tick
contract is:

```python
hero_obs   = vec.observe(HERO_ID)            # (N, 58) float32
opp_obs    = vec.observe(OPPONENT_ID)
hmask, hids = vec.action_mask(HERO_ID)       # (N, 9), (N, 5)
omask, oids = vec.action_mask(OPPONENT_ID)

# policy forward(s) → hero_actions, opp_actions (lists of AgentAction)

vec.apply_actions(HERO_ID,     hero_actions)
vec.apply_actions(OPPONENT_ID, opp_actions)
vec.step()                                    # one sim tick
# (repeat apply+step for ACTION_REPEAT substeps)

rh, ro = vec.rewards()                        # drains accumulators
done, winner = vec.dones()
vec.reset_done()                              # re-scenarios finished envs
```

Per-env RNGs make rollouts deterministic given a base seed. Agents and
stats are symmetric today; varying stats / obstacles per env is a
future hook.

### Replay recorder

`Recorder` is a streaming writer. Attach one per episode (or per run of
a chosen env), call `recordFrame` after each tick, `close` on end. It
auto-slices the world's event log into per-frame deltas via an internal
cursor, so you don't manage that yourself.

```cpp
Recorder rec;
rec.open("ep0.bgar", /*episodeId*/ 42, /*seed*/ 7, /*dt*/ 0.016f);
rec.writeRoster(world.agents());
for (int step = 0; step < N; step++) {
    sim.step(dt);
    rec.recordFrame(step, step * dt, world);
}
rec.close();
```

On close, the writer appends a frame index and a footer. Readers can
random-access any frame in O(1) by seeking to `EOF - sizeof(Footer)`.

### On-disk format (.bgar)

```
FileHeader                (magic='BGAR', version, episodeId, seed, dt)
uint32 rosterCount
AgentStatic[rosterCount]  (id, team, maxHp, maxMana, radius, attackRange)
Frame* (stream)
  FrameHeader             (stepIdx, elapsed, liveCount, projCount, eventCount)
  AgentState[liveCount]   (pos, vel, yaw, hp, cooldown, flags)
  ProjectileState[]
  DamageEventRec[]
IndexEntry[indexCount]    (stepIdx, offset)
Footer                    (indexOffset, indexCount)   # last 16 bytes
```

All records are packed POD, little-endian, native alignment. `bro`
should include `include/brogameagent/replay_format.h` directly so any
schema evolution is a compile error on its side.

## Usage patterns

### Single-env scripted play (C++)

```cpp
World w;
Agent hero, enemy;
hero.unit().id = 1; hero.unit().teamId = 0; hero.setPosition(0, 0);
enemy.unit().id = 2; enemy.unit().teamId = 1; enemy.unit().hp = 50;
enemy.setPosition(5, 0);
w.addAgent(&hero);
w.addAgent(&enemy);

w.resolveAttack(hero, /*targetId*/ 2);
for (const auto& e : w.events()) {
    // inspect DamageEvent
}
```

### REINFORCE demo (Python, single env)

See `python/example_train.py` — a hero chases a stationary dummy using
a small MLP trained with REINFORCE. Useful to sanity-check the bindings
and reward signal. Optional recording writes `replays/epNNNN.bgar` for
a subset of episodes.

```sh
cd python
python example_train.py
```

### Self-play PPO + ELO (Python, vectorized)

`python/selfplay_train.py` is the actual training entry point. It runs
NUM_ENVS 1v1 envs on GPU 0 (CUDA if available), samples opponents from
an ELO-rated pool of frozen policy snapshots, runs PPO with GAE, and
snapshots the current policy into the pool periodically.

Key knobs at the top of the file:

```python
NUM_ENVS         = 512      # parallel envs
ROLLOUT_LEN      = 64       # policy steps per PPO update
ACTION_REPEAT    = 4        # sim ticks per policy decision
SIM_DT           = 0.016    # 16 ms per sim tick
MAX_STEPS_EP     = 600      # sim ticks (9.6 s game time)
UPDATES          = 200
WALL_BUDGET_SEC  = None     # if set, overrides UPDATES with a time cap
SNAPSHOT_EVERY   = 10       # push current policy into opponent pool every N updates
MAX_POOL         = 8
ELO_K            = 32.0
RECORD_DIR       = "replays_selfplay"   # None disables recording
RECORD_EVERY     = 25                   # record env 0 every N updates
```

Each policy decision spans `ACTION_REPEAT * SIM_DT` = 64 ms of game
time, so the network is learning to commit to sequences of timed
actions rather than thrashing at the physics rate.

#### Logging columns

```
upd 200  loss= 39.3  elo=2674.3  W/L/D=2420/84/41588  wr=0.05 dec_wr=0.97  sps=30525  pool=[…]
```

- `elo` — current learner rating.
- `W/L/D` — cumulative wins / losses / draws.
- `wr` — overall win rate (including draws).
- `dec_wr` — decisive win rate, `W / (W + L)`. This is the useful one
  during early training when most episodes still time out.
- `sps` — env-steps / sec (a "step" here is one sim tick).
- `pool` — `id:elo` per frozen snapshot.

#### On draws and ELO

Draws / timeouts **do not update ELO** in this trainer. The reason is
that standard ELO math assumes total score per match = 1 (win=1, draw=0.5
each, loss=0). Treating draws as "both lose" (score 0 for both) injects
-1 of score into the system every game, which drags all ratings down
together and destroys the signal — empirically, ELO went to -460k in 9
min of training when we tried it.

Instead, aggression is biased at the **reward** level, not the rating
level: the `rewardTimeout` on `VecSimConfig` is applied to both agents
when an episode times out, so stalling is strictly worse than fighting.
The ELO metric stays a readable measurement of relative skill across
snapshots.

## Verified end-to-end

On an RTX 4090 (single GPU, cuda:0), a 9-min training run with
`NUM_ENVS=512, ROLLOUT_LEN=64, ACTION_REPEAT=4, HIDDEN=128`:

- ~27–55k env-steps/sec sustained (~100k sim-ticks/sec)
- 440+ PPO updates
- Learner ELO 1000 → 4000+
- Decisive win rate ≥97% from ~upd 50 onward; overall win rate climbing
  steadily from 2% to ≥11% as agents learn to engage rather than time out
- Plateau expected at 1.5–3 hours

Throughput is currently CPU-bound (per-tick Python `AgentAction` list
construction). A batched `apply_actions` path accepting raw `(N, K)`
action arrays would 3–4× this; it's a known next step.

## Roadmap

- **Batched `apply_actions`** (raw `(N, K)` arrays — kill Python overhead).
- **Scenario variation beyond spawn**: per-env obstacle layouts, arena
  size, asymmetric stats for matchup training.
- **NvN teams** with proper "neutral draw" semantics once team play
  makes draws meaningful.
- **Ability integration in self-play** — current self-play trainer uses
  only auto-attack and movement. Abilities are supported at the sim
  level; wiring them into the action mask path is straightforward.
- **`bro` visualizer** — consumes `.bgar` files via `replay_format.h`.

## Test coverage

`tests/test_main.cpp` — 102 tests across:

- Nav grid: walkable cells, padded obstacles, A* pathing, grid LOS.
- Steering: seek / arrive / flee / pursue / evade / follow-path.
- Perception: LOS, FOV, `canSee`, aim / lead-aim.
- Agents: position / velocity integration, `maxAccel` / `maxTurnRate`
  clamps, scripted vs policy paths.
- Combat: damage reduction, attacks, abilities, events, reward tracker,
  projectile modes (Single / Pierce / AoE / homing).
- World: snapshot / restore round-trip, deterministic RNG, observation
  layout, action mask alignment.
- `Simulation` / `VecSimulation`: per-tick semantics, cooldown ticks,
  determinism under seed, termination + winner, reward drain.
- Recorder / reader: round-trip, event slicing, bad-magic rejection,
  random access by step.

## License / authorship

Solo authorship; no license file yet. Add before any external share.
