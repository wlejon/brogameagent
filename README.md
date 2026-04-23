# brogameagent

A C++20 combat simulation and planner library for MOBA-style 1v1 (and
NvN) fights. Built for algorithmic, rollout-based AI — MCTS variants over
a snapshot-restorable sim, plus hand-crafted NN circuits for ExIt-style
self-improvement. Zero external dependencies (no Python, no libtorch, no
ONNX — every circuit is authored in plain C++).

## Intent

The sim is a fast, deterministic substrate for **sampling-based planning**:
snapshot the current world, fork N hypothetical futures under different AI
responses, step forward, score, commit the winning action. `VecSimulation`
makes the fork-and-step loop cheap enough to do per-frame at runtime.

Layered on top of that substrate:

- **A planner zoo** — single-hero `Mcts`, simultaneous-move `DecoupledMcts`,
  cooperative `TeamMcts`, hierarchical `TacticMcts` / `LayeredPlanner`,
  options-based `OptionMcts` / `TeamOptionMcts`, role-based `Commander`,
  partial-observability `InfoSetMcts`, and `root_parallel_search`.
- **Hand-crafted NN circuits** — a small library of eager, autograd-free
  layers (`Linear`, `DeepSetsEncoder`, `ValueHead`, `FactoredPolicyHead`)
  composed into a `SingleHeroNet` that plugs into any MCTS variant via the
  existing `IPrior` / `IEvaluator` interfaces. An `ExItTrainer` consumes
  MCTS-derived policy/value targets and hot-swaps weights through a
  `WeightsHandle` so the game never pauses for training.

## Repository layout

```
include/brogameagent/        public headers (pure C++20)
  # core sim
  types.h unit.h agent.h world.h
  nav_grid.h steering.h perception.h
  observation.h action_mask.h reward.h
  projectile.h snapshot.h simulation.h
  vec_simulation.h              # batched envs (for parallel rollouts)
  recorder.h replay_reader.h    # writing + reading .bgar
  replay_format.h               # on-disk schema (zero-dep)
  # planners
  mcts.h                        # all MCTS variants + options + Commander
  info_set_mcts.h               # IS-MCTS over particle beliefs
  belief.h observability.h      # partial-observability infra
  capability.h policy.h         # scripted policy primitives
  # NN circuits (hand-crafted, no autograd)
  nn/tensor.h nn/ops.h          # Tensor + GEMV/softmax/xent primitives
  nn/circuits.h                 # Linear, ReLU, Tanh
  nn/encoder.h                  # DeepSetsEncoder (permutation-invariant)
  nn/heads.h                    # ValueHead, FactoredPolicyHead
  nn/net.h                      # SingleHeroNet + WeightsHandle
  # Learning primitives
  learn/replay_buffer.h         # Situation ring buffer
  learn/search_trace.h          # extract policy targets from MCTS roots
  learn/trainer.h               # ExItTrainer (SGD+momentum minibatch)
  learn/neural_adapters.h       # NeuralEvaluator / NeuralPrior
  learn/gumbel.h                # GumbelNoisePrior + improved policy target
src/                           implementations
tests/test_main.cpp            single-file test suite, 191 tests, no framework
tools/replay_query.cpp         CLI inspector for .bgar files
tools/mcts_bench.cpp           MCTS strength/speed sweeps → TSV
tools/nn_check.cpp             finite-diff gradient verification for every circuit
tools/nn_train_value.cpp       capture MCTS targets → train SingleHeroNet → save .bgnn
tools/nn_exit.cpp              full ExIt loop (generate → train → eval, iterated)
```

## Building

```sh
cmake -S . -B build
cmake --build build --config Release
```

Produces the static lib, tests, `replay_query.exe`, `mcts_bench.exe`,
the NN CLIs (`nn_check.exe`, `nn_train_value.exe`, `nn_exit.exe`), and
the examples under `examples/` (see `examples/README.md` for the guided
tour from "hello world" to layered multi-agent search).

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

### NN CLIs (`nn_check`, `nn_train_value`, `nn_exit`)

```sh
# Finite-diff gradient verification for every circuit.
./build/Release/nn_check.exe [--verbose]

# Generate episodes with MCTS, capture (obs, π̂, z) targets, train
# SingleHeroNet, save .bgnn.
./build/Release/nn_train_value.exe \
    [--episodes N] [--iterations M] [--steps S] [--out F.bgnn] [--seed X]

# Full ExIt loop: iterated generate → train → eval, hot-swapping weights
# via WeightsHandle. Emits per-iter TSV metrics and an .bgnn checkpoint.
./build/Release/nn_exit.exe \
    [--iters K] [--episodes N] [--iterations M] [--max-ticks T] \
    [--steps S] [--eval E] [--out-prefix P] [--seed X]
```

All three emit TSV on stdout for sweep/log composition.

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

## NN circuits and ExIt learning

The `nn/` and `learn/` modules implement a dependency-free, hand-crafted
NN stack sized for realtime MCTS inside this sim. Each circuit owns its
own forward + backward (no autograd, no graph), so the whole stack reads
like ordinary C++ and single-steps cleanly in a debugger.

### Design choices

- **Eager, hand-coded backward.** Every circuit has its own `forward` +
  `backward` method; a net's `backward()` just calls them in reverse
  order. No tape, no JIT, no graph. Keeps the library small and every
  gradient readable.
- **Factored policy, not flat.** The action space is
  `(MoveDir × AttackSlot × AbilitySlot)`; the policy head emits three
  independent softmax distributions. Aligns with `action_mask::build` and
  is vastly more sample-efficient than a flat joint softmax.
- **DeepSets encoder, not convs.** `observation::build` already yields a
  slot-sorted egocentric vector; a permutation-invariant set encoder
  (per-slot MLP + mean-pool + concat with self block) respects that
  structure directly.
- **Legal-action masking is first-class.** Masked softmax + cross-entropy
  zero out illegal slots in both forward and backward, so the trainer
  never needs to post-filter.

### Circuits and loss primitives (`include/brogameagent/nn/`)

- `Tensor` — rank-1/2 owned float buffer, row-major, no broadcasting.
- `linear_forward` / `linear_backward` — GEMV + bias with hand-written
  backward.
- `Linear`, `Relu`, `Tanh` — circuits with SGD+momentum velocity state
  and per-tensor serialization.
- `softmax_forward` / `softmax_xent` — numerically-stable softmax with
  optional mask, fused softmax-xent whose gradient collapses to
  `(p − target)`.
- `DeepSetsEncoder` — per-stream MLP over self / enemy-slots / ally-slots,
  mean-pool over valid slots, concat. Invalid slots contribute zero
  gradient.
- `ValueHead` — `embed → hidden → 1 → tanh`; output in [−1, 1].
- `FactoredPolicyHead` — three linears: 9 move logits, 6 attack logits
  (N_ENEMY_SLOTS + "no-op"), 9 ability logits (MAX_ABILITIES + "no-op").
- `SingleHeroNet` — `DeepSetsEncoder → Linear+ReLU trunk → {ValueHead,
  FactoredPolicyHead}`. Default shape ~14K params.
- `WeightsHandle` — atomic publish/subscribe over a `.bgnn` blob via
  `shared_ptr` + mutex. Publishers bump a version; readers snapshot
  per-decision and reload only when the version advances. This is the
  primitive that lets a live planner consume fresh weights from a
  background trainer without stopping the game.

### `.bgnn` weights format

Tiny zero-dependency binary format:

```
magic('BGNN')  uint32
version        uint32
for each circuit in SingleHeroNet in save order:
    for each weight tensor (W, b):
        int32 rows
        int32 cols
        float[rows*cols]
```

Load via `SingleHeroNet::load(blob)`; the adapter classes
(`NeuralEvaluator`, `NeuralPrior`) invoke this automatically when
`WeightsHandle::version()` advances.

### Learning primitives (`include/brogameagent/learn/`)

- `Situation` — one training tuple: `obs`, legal-action masks, three
  factored policy targets, and a value target in [−1, 1].
- `ReplayBuffer` — fixed-capacity FIFO with uniform sampling.
- `SearchTrace::make_situation(world, hero, root)` — extracts the policy
  targets from a completed MCTS root by converting child visit counts
  into the three factored distributions.
- `ExItTrainer` — SGD+momentum minibatch trainer. Computes value MSE and
  factored-policy cross-entropy, backpropagates once per sample, steps
  the optimizer, and optionally publishes to a `WeightsHandle`.
- `NeuralEvaluator` / `NeuralPrior` — adapters that implement
  `mcts::IEvaluator` / `mcts::IPrior`. They compose with every MCTS
  variant in the library without engine changes — the integration is
  entirely through the existing interfaces.
- `GumbelNoisePrior` — wraps any inner prior with IID Gumbel noise in
  log-space (Danihelka et al. 2022, simplified). Adds per-decision
  exploration diversity without sacrificing the policy-improvement
  property of MCTS.
- `gumbel_improved_policy(root, ...)` — computes the paper's π'
  distribution from a completed search tree as a distillation target
  that's strictly better than raw visit counts when the search budget
  is small.

### ExIt loop (at the level of `nn_exit`)

```
for iter in 0..K:
    # Generate: run MCTS with current net as prior+evaluator (iter 0
    # falls back to classical MCTS + HpDelta so the data is informative).
    for ep in 0..N:
        run episode, capture (obs, mask, π̂_from_root, discounted_return)

    # Train: SGD on the replay buffer, publish weights periodically.
    trainer.step_n(S)

    # Eval: N_eval episodes vs a frozen scripted opponent.
    report win_rate, mean_hp_delta, elapsed_ms
    save iter_k.bgnn
```

MCTS is the expert, the net is the apprentice. The tree's visit
distribution is the policy-improvement target; the eventual
discounted return is the value target. Repeating this yields a
progressively stronger prior that short-circuits MCTS at tight
iteration budgets via `use_leaf_value` + a fast `NeuralPrior` seed.

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

NN circuits are additionally verified end-to-end by `nn_check.exe`,
which runs finite-difference gradient checks against every circuit's
analytic backward (9 checks, pass on clean build). Training plumbing
is exercised by `nn_train_value.exe` (value-loss convergence) and
`nn_exit.exe` (full loop, save/load/publish round-trip).

## License / authorship

MIT [LICENSE](LICENSE)
