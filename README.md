# brogameagent

A C++20 combat simulation for MOBA-style 1v1 (and later NvN) fights. Built
for algorithmic, rollout-based AI — *not* neural-network training.

## Intent

The sim is a fast, deterministic substrate for **sampling-based planning**:
snapshot the current world, fork N hypothetical futures under different AI
responses, step forward, score, commit the winning action. `VecSimulation`
makes the fork-and-step loop cheap enough to do per-frame at runtime.

## Repository layout

```
include/brogameagent/        public headers (pure C++20)
  types.h unit.h agent.h world.h
  nav_grid.h steering.h perception.h
  observation.h action_mask.h reward.h
  projectile.h snapshot.h simulation.h
  vec_simulation.h              # batched envs (for parallel rollouts)
  recorder.h replay_reader.h    # writing + reading .bgar
  replay_format.h               # on-disk schema (zero-dep)
src/                           implementations
tests/test_main.cpp            single-file test suite, no external framework
tools/replay_query.cpp         CLI inspector for .bgar files
```

## Building

```sh
cmake -S . -B build
cmake --build build --config Release
```

Produces the static lib, tests, `replay_query.exe`, and `mcts_bench.exe`.

### Running tests

```sh
./build/tests/Release/brogameagent_test.exe
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

All output is tab-separated — pipe into `awk`, `csvkit`, `pandas`, whatever.

### `mcts_bench` CLI

Runs N episodes with MCTS planning per decision and reports win/loss/draw
counts, mean terminal HP delta, and mean per-decision search cost as a
single TSV row:

```sh
./build/Release/mcts_bench.exe duel [flags]
./build/Release/mcts_bench.exe team [flags]
```

Key flags: `--episodes N`, `--iterations M`, `--budget-ms T`,
`--rollout {random|aggressive}`, `--opponent {idle|aggressive}`,
`--puct C`, `--pw A`, `--heroes H`, `--enemies E`,
`--planner {team|layered}`, `--seed S`, `--max-ticks K`.

Sweep by re-running across a grid and concatenating the output rows.

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
  `maxAccel` / `maxTurnRate` clamps.

### Observation / action mask / reward

Three independent builders, stable in layout:

- `observation::build(self, world, float* out)` — self block (HP, mana,
  cooldowns, aim-vs-move delta), K enemies, K allies, sorted nearest-first
  in the agent's local frame.
- `action_mask::build(self, world, outMask, outEnemyIds)` — attack + ability
  slot legality, aligned to the observation's enemy slot order.
- `RewardTracker::consume(self, world)` — returns `(damageDealt,
  damageTaken, kills, deaths, distanceTravelled)` since the last `consume`
  or `reset`, using the world's event log as the source of truth.

### Combat resolution

- `World::resolveAttack(attacker, targetId)` — range/cooldown-gated
  auto-attack, writes a `DamageEvent` to the log.
- `World::resolveAbility(caster, slot, targetId)` — runs a registered
  ability function, gated by cooldown, mana, and (optional) range.
- `World::spawnProjectile(Projectile)` — Single / Pierce / AoE modes,
  optional homing via `targetId`, ownership for event attribution.

### Snapshot / restore

`World::snapshot()` / `restore(WorldSnapshot)` captures all resettable
state (agent positions, stats, projectiles, event log, RNG). This is the
primitive that makes parallel-rollout planning possible: fork current state
into N copies, explore, keep the best.

### `VecSimulation` — batched envs

`VecSimulation` holds N independent `World`s and steps them in parallel.
Intended as the substrate for per-frame Monte Carlo rollouts: seed all N
envs from the current game state, apply a different candidate AI action in
each, step forward K ticks, score, pick a winner.

### Replay recorder

`Recorder` is a streaming writer. Attach one per scenario, call
`recordFrame` after each tick, `close` on end. It auto-slices the world's
event log into per-frame deltas via an internal cursor.

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

All records are packed POD, little-endian, native alignment. Consumers
should include `include/brogameagent/replay_format.h` directly so any
schema evolution is a compile error at the boundary.

## Test coverage

`tests/test_main.cpp` covers:

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
