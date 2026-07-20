# brogameagent

[![CI](https://github.com/wlejon/brogameagent/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brogameagent/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wlejon/brogameagent/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brogameagent/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A C++20 game-AI library: navigation and movement, MCTS planners, and a
hand-crafted autograd-free NN stack with ExIt-style self-improvement.
No Python, no libtorch, no ONNX. Sibling repos `bromath` (header-only
math) and `brotensor` (tensor + ops, CPU / CUDA / Metal) vendor in as
`add_subdirectory`; `recastnavigation` is the only external dependency
and it's optional.

Two halves, separable at configure time: a **core** (navigation,
steering, avoidance, perception, the combat sim, the planners) with no
tensor dependency at all, and a **neural layer** (`nn/`, `learn/`) built
on brotensor. `-DBROGAMEAGENT_WITH_NN=OFF` builds the core alone.

## What's in it

**Movement**: `NavGrid` (2D A\* + smoothing + grid LOS), `NavMesh`
(Recast-baked polygon navmesh: slopes, bridges, multi-level interiors,
off-mesh links, runtime dynamic obstacles), `steering` (seek / arrive /
flee / pursue / evade / followPath), `AvoidanceSim` (2D ORCA with
priorities, layer/mask and elevation filtering), `perception` (LOS, FOV,
aim / lead-aim).

**Planners**: single-hero `Mcts`, simultaneous-move `DecoupledMcts`,
cooperative `TeamMcts`, hierarchical `TacticMcts` / `LayeredPlanner`,
options-based `OptionMcts` / `TeamOptionMcts`, role-based `Commander`,
partial-observability `InfoSetMcts` (with `Belief` / `Observability`),
`root_parallel_search`, and a domain-agnostic `GenericMcts<State,
Action>`.

**Circuits**: eager, autograd-free layers: `Linear` / `Relu` / `Tanh`,
`LayerNorm`, `Embedding`, `GRU`, `MultiHeadAttention` /
`TransformerBlock` / `TransformerEncoder`, `SetTransformer`,
`DeepSetsEncoder`, `Autoencoder`, `ForwardModel`, `Ensemble`, factored /
categorical / distributional heads, and the prebuilt `SingleHeroNet`
(DeepSets, ~14K params), `SingleHeroNetST` (set-transformer),
`SingleHeroNetTX` (per-stream transformer, also a `learn::BatchedNet`),
and `PolicyValueNet`.

**Learning**: `ExItTrainer` consumes MCTS-derived policy/value targets
and hot-swaps weights through a `WeightsHandle` so the game never pauses
for training. `GenericTrainer` + `GenericReplayBuffer` do the same for
non-combat tasks. A batched `InferenceServer` lets many search threads
share one forward pass. Plus `gumbel_improved_policy`,
`GumbelNoisePrior`, contrastive losses, and a `ForwardModel` skeleton
for MuZero-shaped extensions.

**Substrates**: a deterministic MOBA-style combat sim (snapshot /
restore, projectiles, abilities, `.bgar` replays) and a domain-neutral
tile-grid harness (`grid/`: best-crop curriculum, failure tape, BC
ingest, observation window, frame stack, reward shaping). New substrates
plug in via `mcts::IEvaluator` / `IPrior` or `GenericMcts`; nothing in
the planner or learning stack is combat-specific.

## Building

```sh
cmake -S . -B build && cmake --build build --config Release
ctest --test-dir build -C Release
```

| Option | Default | Effect |
|---|---|---|
| `BROGAMEAGENT_WITH_NN` | `ON` | `nn/` + `learn/`, the only users of brotensor. `OFF` drops that dependency entirely. Forced `ON` by either GPU option. |
| `BROGAMEAGENT_WITH_CUDA` / `_METAL` | `OFF` | GPU dispatch through brotensor. Mutually exclusive. |
| `BROGAMEAGENT_WITH_NAVMESH` | `ON` standalone, `OFF` as a subdirectory | Polygon `NavMesh` via `recastnavigation`. |
| `BROGAMEAGENT_TOOLS` / `_EXAMPLES` / `_TESTS` | top-level | CLI tools, examples, test binaries. |

The navmesh is the only option needing an external package. Standalone
builds turn it on but soft-disable (with a STATUS message) when
`find_package(recastnavigation)` comes up empty, so consumers without
vcpkg still build:

```sh
vcpkg install recastnavigation
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
```

When enabled the library defines `BROGAMEAGENT_HAS_NAVMESH=1` on its
interface. `nav_mesh.h` exposes no Recast/Detour types, so it ships
either way; only the implementation gates.

The build produces the static lib, tests, `replay_query`, `mcts_bench`,
the NN CLIs (`nn_check`, `nn_train_value`, `nn_exit`, `nn_pretrain_ae`,
plus `nn_pretrain_ae_gpu` / `nn_exit_gpu` on GPU builds), and the
examples under `examples/`. See [examples/README.md](examples/README.md)
for the guided tour from "hello world" through multi-agent search and
into the grid harness.

## Navigation

`NavGrid(minX, minZ, maxX, maxZ, cellSize)` rasterizes padded AABB
obstacles into walkable cells, runs 8-directional A\*, and smooths the
result with LOS checks. `obstacles()` returns the raw boxes so the same
walls can be bridged into `World::addAvoidanceObstacle`, keeping ORCA
consistent with what A\* paths around.

`NavMesh` is the 3D counterpart for worlds a flat grid can't represent.
`bake()` takes a y-up triangle soup (CCW from above); Detour answers
`findPath` (funnel-straightened), `nearestPoint`, `raycast`, and seeded
`randomPoint`. Recast/Detour stay behind a pimpl. Two bake modes:

- **Static**: one Detour tile. Supports the detail mesh,
  `saveTo()` / `loadFrom()` (cache the bake; it's the expensive part),
  and bake-time off-mesh links (jumps, drops, ladders, teleporters:
  the Godot `NavigationLink` analog). Paths mark link takeoffs with
  `NavMeshPath::kLinkStart` so followers can play an animation.
- **Dynamic-obstacle**: tiled bake via `dtTileCache`. `addObstacle`
  (cylinder), `addBoxObstacle` (axis-aligned or Y-rotated) and
  `removeObstacle` queue changes; `update()` rebuilds one touched tile
  per call and `generation()` bumps when the surface settles, so
  followers know to re-plan. Trades away the detail mesh,
  serialization, and off-mesh links.

Both clamp an unreachable goal to the closest reachable point rather
than failing; the `findPathEx` variants report `partial` and accept
`requireFullPath=true` for hard-fail semantics.

## Local avoidance

`AvoidanceSim` implements ORCA (van den Berg, Guy, Lin & Manocha) in the
XZ plane: one half-plane per neighbor and obstacle segment, then a small
2D LP for the admissible velocity closest to the preferred one, falling
back to a 3D LP that minimizes the worst violation in dense crowds.
Obstacle constraints are never relaxed. Per-agent knobs beyond radius /
speed / horizons:

- **`priority`** (0..1): how avoidance effort splits across a pair. The
  lower-priority agent corrects more; shares sum to 1, preserving ORCA's
  reciprocal guarantee, and equal priorities reproduce the plain 50/50
  solver bit-for-bit.
- **`layers` / `mask`**: A avoids B only when `(A.mask & B.layers) != 0`,
  filtered during neighbor gathering, so avoidance is per-side.
- **`height` + `setElevation`**: non-overlapping vertical spans are
  ignored, so bridge traffic doesn't dodge tunnel traffic. The solve
  stays 2D.
- **`setResponsive(false)`**: still avoided by everyone else, never
  solved itself; for units keeping scripted movement.

Deterministic: identical inputs give bit-identical velocities.
`World::setAvoidanceEnabled(true)` inserts the filter between an agent's
desired velocity and integration.

## Combat sim

A `World` holds `Agent*`s, obstacles, projectiles, registered abilities,
an event log, and a deterministic `mt19937_64`. Each `Agent` owns a
`Unit` (HP, mana, cooldowns, ability slots), position/velocity, and an
aim yaw/pitch decoupled from movement yaw. Agents drive either scripted
(`setTarget` + `update`, A\*-pathed seek-arrive) or by policy
(`applyAction`, continuous control with accel/turn-rate clamps).

- **Observation / mask / reward**: `observation::build` (self block, K
  enemies, K allies, nearest-first, egocentric), `action_mask::build`
  (attack + ability legality, aligned to the observation's slot order),
  `RewardTracker::consume` (damage dealt/taken, kills, deaths, distance,
  from the event log).
- **Combat**: `resolveAttack`, `resolveAbility`, `spawnProjectile`
  (Single / Pierce / AoE, optional homing).
- **Capabilities**: a `Capability` is one tool an agent can invoke
  (`MoveTo`, `LaneWalk`, `BasicAttack`, `CastAbility`, `Flee`, `Hold`
  ship built in) with `gate` / `start` / `advance` / `cancel`; a
  `CapabilitySet` is the per-agent bag; a `Policy` picks which to start
  each think tick. See `examples/13_capabilities_demo.cpp`.
- **Snapshot / restore**: `World::snapshot()` / `restore()` captures all
  resettable state. This is the primitive that makes fork-N-futures
  planning possible; `VecSimulation` steps N independent worlds for
  batched rollouts.
- **Recorder**: streaming `.bgar` writer; `recordFrame` per tick,
  auto-slicing the event log into per-frame deltas. On close it appends
  a frame index and footer, so readers random-access any frame in O(1).

## ExIt loop

MCTS is the expert, the net is the apprentice. Each iteration generates
episodes with the current net as prior + evaluator, captures `(obs,
mask, π̂, discounted return)` from the search tree via
`learn::SearchTrace`, trains on the replay buffer, evaluates against a
frozen opponent, and publishes weights through `WeightsHandle`. The
tree's visit distribution is the policy target; the eventual return is
the value target. `tools/nn_exit.cpp` is the end-to-end implementation.

Design choices worth knowing: policies are **factored**
(`MoveDir × AttackSlot × AbilitySlot`), not a flat joint softmax;
legal-action masking is first-class in both forward and backward, so the
trainer never post-filters; and every circuit hand-codes its own
backward, so the whole stack single-steps in a debugger.

## CLI tools

```sh
replay_query info|roster|frame|step|agent|events|dps|dump <file.bgar>
mcts_bench   duel|team [--episodes N --iterations M --budget-ms T ...]
nn_check     [--verbose]                    # finite-diff gradient checks
nn_train_value [--episodes N --steps S --out F.bgnn ...]
nn_exit      [--iters K --episodes N --eval E --out-prefix P ...]
```

All emit TSV on stdout, so pipe into `awk`, `csvkit`, `pandas`, whatever.

## File formats

Both are packed-POD, little-endian, with magic + version headers;
include the headers directly so schema changes are a compile error at
the boundary.

- **`.bgar`** replays (`replay_format.h`): file header, roster, a
  stream of frames (agent states, projectiles, damage events), then a
  frame index and 16-byte footer.
- **`.bgnn`** weights: magic, version, then `rows`/`cols`/`float[]` per
  weight tensor in net save order. `WeightsHandle` publishes these
  atomically; `NeuralEvaluator` / `NeuralPrior` reload when the version
  advances.

## Tests

`tests/test_main.cpp` covers nav grid, steering, perception, agent
integration and clamps, combat and projectiles, snapshot/restore,
observation and mask layout, `Simulation` / `VecSimulation` determinism,
`Mcts` semantics (legal actions, side-effect freedom, time budget,
`advance_root` reuse), and the recorder/reader round-trip.

Core-only suites that build with the NN layer off:
`brogameagent_avoidance_test` (ORCA solver + `World` integration:
crossing crowds, priorities, layers, elevation, obstacles, determinism)
and `brogameagent_nav_mesh_test` (bake, slopes, stacked levels, off-mesh
links, queries, save/load, dynamic obstacles; gated on
`BROGAMEAGENT_WITH_NAVMESH`).

NN-side: `nn_check` runs finite-difference gradient checks against every
circuit's analytic backward and returns non-zero on any failure;
`brogameagent_transformer_test`, `brogameagent_adam_test`, and
`brogameagent_single_hero_net_tx_test` cover the transformer stack, Adam
bias-correction, and the TX net. On GPU builds, `tests/gpu/` exercises
per-layer host↔device migration and the batched inference / MCTS server
paths. (Op-level CPU↔GPU parity is tested in brotensor.)

## License

MIT, see [LICENSE](LICENSE)
