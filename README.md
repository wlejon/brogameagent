# brogameagent

A C++20 algorithms library for sampling-based game AI: MCTS variants,
ExIt-style self-improvement, and a hand-crafted, autograd-free NN
circuit stack — all designed to plug into any snapshot-restorable
substrate. No Python, no libtorch, no ONNX; every circuit is authored
in plain C++. Sibling repos `bromath` (header-only math) and
`brotensor` (tensor + ops, CPU always-on / CUDA / Metal) supply the
low-level primitives; both vendor in as `add_subdirectory`, no system
deps.

Two reference substrates ship in-tree:

- A deterministic MOBA-style combat sim (1v1 / NvN) with snapshot /
  restore, projectiles, abilities, and a replay format — the original
  driver for the library and still the richest test bed.
- A tile-grid harness (`include/brogameagent/grid/`) — a smaller,
  domain-neutral substrate used to exercise the generic planner /
  trainer paths without combat-sim coupling.

New substrates plug in by implementing the `mcts::IEvaluator` /
`IPrior` interfaces (or the domain-agnostic `GenericMcts<State,
Action>`) and feeding `GenericTrainer` + `WeightsHandle`. Nothing in
the planner or learning stack is combat-specific.

## Intent

Whatever the substrate, the loop is the same: snapshot current state,
fork N hypothetical futures under different actions, step forward,
score, commit the winner. `VecSimulation` (combat) and the grid
harness both make that fork-and-step loop cheap enough to do per-frame
at runtime; `ExItTrainer` closes the loop by distilling the search's
visit distribution back into a prior that short-circuits the next
round of search.

Layered on top of that substrate:

- **A planner zoo** — single-hero `Mcts`, simultaneous-move `DecoupledMcts`,
  cooperative `TeamMcts`, hierarchical `TacticMcts` / `LayeredPlanner`,
  options-based `OptionMcts` / `TeamOptionMcts`, role-based `Commander`,
  partial-observability `InfoSetMcts` (with `Belief` / `Observability`
  filtering), `root_parallel_search`, and a domain-agnostic
  `GenericMcts<State, Action>` for non-combat substrates.
- **A circuit library** — eager, autograd-free layers spanning the usual
  toolkit: `Linear` / `Relu` / `Tanh`, `LayerNorm`, `Embedding`, `GRU`,
  `MultiHeadAttention` / `TransformerBlock` / `TransformerEncoder`,
  `SetTransformer`, `DeepSetsEncoder`, `Autoencoder` (with `Decoder`),
  `ForwardModel`, `Ensemble`, factored / categorical / distributional
  heads, and the prebuilt `SingleHeroNet` and `PolicyValueNet` that plug
  into any MCTS variant via the existing `IPrior` / `IEvaluator`
  interfaces. Tensor storage and ops come from sibling
  [brotensor](../brotensor) — one `brotensor::Tensor` carries a runtime
  `Device` tag, and a single op surface dispatches CPU / CUDA / Metal.
- **A learning stack** — `ExItTrainer` consumes MCTS-derived
  policy/value targets and hot-swaps weights through a `WeightsHandle`
  so the game never pauses for training. A more general
  `GenericTrainer` + `GenericReplayBuffer` pair lets the same machinery
  drive non-combat tasks. A batched `InferenceServer` (with pluggable
  `InferenceBackend`) lets many search threads share one forward pass.
  Distillation extras: `gumbel_improved_policy`, `GumbelNoisePrior`,
  contrastive (`learn/contrastive.h`), and a `ForwardModel` skeleton
  for MuZero-shaped extensions.
- **A grid harness** — `include/brogameagent/grid/` is a tile-grid
  training substrate (best-crop curriculum, failure tape, BC ingest,
  observation window, frame stack, reward shaping, generic recorder)
  decoupled from the combat sim; used as the substrate for the
  `15_grid_corridor` example and lighter-weight RL experiments.

## Building

```sh
# CPU default — brotensor links in CPU-only.
cmake -S . -B build
cmake --build build --config Release

# Opt in to GPU dispatch via brotensor (mutually exclusive):
cmake -S . -B build -DBROGAMEAGENT_WITH_CUDA=ON
cmake -S . -B build -DBROGAMEAGENT_WITH_METAL=ON
```

Produces the static lib, tests, `replay_query.exe`, `mcts_bench.exe`,
the NN CLIs (`nn_check.exe`, `nn_train_value.exe`, `nn_exit.exe`,
`nn_pretrain_ae.exe`), and the examples under `examples/` (see
`examples/README.md` for the guided tour from "hello world" through
multi-agent search and into the grid harness). The GPU options
additionally enable the batched `InferenceServer` adapter and the
`nn_pretrain_ae_gpu` / `nn_exit_gpu` tools.

### Running tests

```sh
brogameagent_test.exe
```

### `replay_query` CLI

```sh
replay_query.exe info     <file.bgar>
replay_query.exe roster   <file.bgar>
replay_query.exe frame    <file.bgar> <frame_idx>
replay_query.exe step     <file.bgar> <step_idx>
replay_query.exe agent    <file.bgar> <agent_id>
replay_query.exe events   <file.bgar> [attacker_id]
replay_query.exe dps      <file.bgar>
replay_query.exe dump     <file.bgar>
```

All output is tab-separated — pipe into `awk`, `csvkit`, `pandas`, whatever.

### `mcts_bench` CLI

Runs N episodes with MCTS planning per decision and reports win/loss/draw
counts, mean terminal HP delta, and mean per-decision search cost as a
single TSV row:

```sh
mcts_bench.exe duel [flags]
mcts_bench.exe team [flags]
```

Key flags: `--episodes N`, `--iterations M`, `--budget-ms T`,
`--rollout {random|aggressive}`, `--opponent {idle|aggressive}`,
`--puct C`, `--pw A`, `--heroes H`, `--enemies E`,
`--planner {team|layered}`, `--seed S`, `--max-ticks K`.

Sweep by re-running across a grid and concatenating the output rows.

### NN CLIs (`nn_check`, `nn_train_value`, `nn_exit`)

```sh
# Finite-diff gradient verification for every circuit.
nn_check.exe [--verbose]

# Generate episodes with MCTS, capture (obs, π̂, z) targets, train
# SingleHeroNet, save .bgnn.
nn_train_value.exe \
    [--episodes N] [--iterations M] [--steps S] [--out F.bgnn] [--seed X]

# Full ExIt loop: iterated generate → train → eval, hot-swapping weights
# via WeightsHandle. Emits per-iter TSV metrics and an .bgnn checkpoint.
nn_exit.exe \
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

### Capabilities and policies

A higher-level behaviour layer sits above raw `applyAction`. A
`Capability` (`include/brogameagent/capability.h`) is one "tool" an
agent can invoke — `MoveTo`, `LaneWalk`, `BasicAttack`, `CastAbility`,
`Flee`, `Hold` ship as built-in factories (`makeMoveToCapability()` etc.,
`addAllBuiltinCapabilities(set)`). Each cap has `gate` (cheap legality
pre-flight), `start`, `advance`, and `cancel`, and mutates an in-flight
`Action` struct rather than holding per-action state. A `CapabilitySet`
is the per-agent bag of caps (stable integer ids, lane-waypoint and
fallback-point scratch) and exposes `buildBuiltinMask` for scripted
selection.

A `Policy` (`include/brogameagent/policy.h`) picks which capability to
start each think tick. `ScriptedMinionPolicy` is the built-in default
(attack-in-range → lane-walk → hold) for bulk minion behaviour; the
JS-primary path in the host engine writes the pending `Action` directly
instead. `CapabilityId` reserves ids ≥ `kJsCapFirst` (100) for
JS-registered caps. See `examples/13_capabilities_demo.cpp` for a tower /
minion / hero wired through capability sets.

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

The tensor type (`brotensor::Tensor`) and the underlying ops
(`linear_forward`, `softmax_forward`, `attention_forward`, the
activations, `xavier_init`, …) live in the sibling `brotensor`
library — see `<brotensor/tensor.h>` and `<brotensor/ops.h>`. A
`brotensor::Tensor` carries a runtime `Device` tag, and every op is
device-neutral: it dispatches to the CPU, CUDA, or Metal backend by its
operands' device. There is no separate host/device tensor type and no
`_cpu` / `_gpu` op split. Layers below own the higher-level circuit
structure (autograd-free, parameter/gradient/optimizer tensors,
serialization) and call those device-neutral ops; a layer's
`to(Device)` migrates every owned tensor at once, so the same
forward/backward code runs on CPU or — when `BROGAMEAGENT_WITH_CUDA` /
`BROGAMEAGENT_WITH_METAL` is on — on the GPU.

Core building blocks:

- `Linear`, `Relu`, `Tanh` — circuits with SGD+momentum velocity state
  and per-tensor serialization.
- `LayerNorm`, `Embedding`, `FeedForward` — standard transformer
  building blocks.
- `MultiHeadAttention`, `TransformerBlock`, `TransformerEncoder`,
  `SetTransformer` — attention stacks for sequence / set inputs.
- `GRU` — recurrent cell for sequential observation histories.
- `DeepSetsEncoder` — per-stream MLP over self / enemy-slots / ally-slots,
  mean-pool over valid slots, concat. Invalid slots contribute zero
  gradient.
- `Autoencoder` / `Decoder` — reconstruction pretraining (see
  `nn_pretrain_ae`).
- `ForwardModel` — latent-dynamics skeleton for MuZero-shaped extensions.
- `Ensemble` — homogeneous N-way wrapper with per-member parameters and
  averaged forward.

Heads and assembled nets:

- `ValueHead` — `embed → hidden → 1 → tanh`; output in [−1, 1].
- `FactoredPolicyHead` — three linears: 9 move logits, 6 attack logits
  (N_ENEMY_SLOTS + "no-op"), 9 ability logits (MAX_ABILITIES + "no-op").
- `CategoricalHead` / distributional heads (`heads_dist.h`) — for
  discrete-action and value-distribution outputs (C51-style).
- `SingleHeroNet` — `DeepSetsEncoder → Linear+ReLU trunk → {ValueHead,
  FactoredPolicyHead}`. Default shape ~14K params.
- `SingleHeroNetST` — set-transformer variant: a `SetTransformer` over the
  entity slots in place of the DeepSets mean-pool, same head pairing.
- `SingleHeroNetTX` — transformer variant: a per-stream
  `TransformerEncoder<n_blocks, num_heads, d_ff>` over the enemy / ally
  slots, masked mean-pool, concat with the self block, then the same trunk +
  heads. Implements `ICircuit` and `learn::BatchedNet`, so it drives the
  batched `InferenceServer` and the GPU MCTS-server path; it's the net the
  transformer / batched / GPU-dispatch tests exercise.
- `PolicyValueNet` — generic policy+value head pairing (also a
  `learn::BatchedNet`) used by the grid harness and other non-combat
  substrates.
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
- `InferenceServer` / `InferenceBackend` — many search threads enqueue
  observations; the server batches them through one net forward per
  tick. Backend is pluggable (CPU `BatchedNet`, GPU on opt-in builds).
- `GenericReplayBuffer<Situation>` / `GenericTrainer<Net, Loss>` —
  substrate-agnostic versions of `ReplayBuffer` / `ExItTrainer` used by
  the grid harness and any non-combat task that wants the same
  generate→train→publish loop.
- `Contrastive` (`learn/contrastive.h`) — auxiliary representation
  losses for the encoder stack.

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

## Grid harness (`include/brogameagent/grid/`)

A small tile-grid training substrate, completely independent of the
combat sim, exercising the same `GenericMcts` / `GenericTrainer` /
`WeightsHandle` machinery on a simpler state space. Useful as a fast
correctness substrate and as a template for porting the stack to new
domains.

- `Harness` — wires generate → train → eval against a user-supplied
  step function; owns the replay buffer, trainer thread, and weights
  handle.
- `BestCrop` — curriculum buffer that keeps the highest-return episode
  prefixes seen so far and seeds future searches from them.
- `FailureTape` — bounded ring of recent failure tails for targeted
  replay / inspection.
- `bc_ingest` — behavioral-cloning ingest path that turns recorded
  expert trajectories into `Situation`s for warm-start training.
- `ObsWindow`, `FrameStack` — observation construction utilities.
- `shaping` — pluggable reward-shaping functions.
- `GenericRecorder` — substrate-neutral episode recorder, mirroring
  `Recorder` but free of combat-sim types.

See `examples/15_grid_corridor.cpp` for the end-to-end shape.

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
- `Mcts`: legal-action enumeration, attack routing, side-effect-free
  search, prefers-attack-when-in-range, determinism under seed, wall-time
  budget, `advance_root` subtree reuse, terminal HP-delta evaluation.
- Recorder / reader: round-trip, event slicing, bad-magic rejection,
  random access by step.

NN circuits are additionally verified end-to-end by `nn_check.exe`,
which runs finite-difference gradient checks against every circuit's
analytic backward — Linear / ReLU / Tanh / Sigmoid, softmax-xent,
DeepSetsEncoder, decoder / autoencoder, LayerNorm, Embedding, attention,
GRU, SetTransformer, the distributional + opponent heads, `ForwardModel`,
InfoNCE, and the assembled `SingleHeroNet` / `SingleHeroNetST` /
`PolicyValueNet` (save/load + SGD-step). It prints a per-check TSV and a
`# summary pass=… fail=…` footer, returning non-zero on any failure.
Transformer circuits (MHA, FeedForward, TransformerBlock,
TransformerEncoder) and the `SingleHeroNetTX` net have their own CPU
correctness binary (`brogameagent_transformer_test` /
`brogameagent_single_hero_net_tx_test`), and Adam bias-correction has
`brogameagent_adam_test`. Training plumbing is exercised by
`nn_train_value.exe` (value-loss convergence) and `nn_exit.exe` (full
loop, save/load/publish round-trip).

When built with `BROGAMEAGENT_WITH_CUDA` / `BROGAMEAGENT_WITH_METAL`,
`tests/gpu/` additionally exercises GPU dispatch — per-layer
host↔device migration round-trips and the batched inference / MCTS
server paths. (CPU↔GPU parity for brotensor's op surface itself is
tested in brotensor.)

## License / authorship

MIT [LICENSE](LICENSE)
