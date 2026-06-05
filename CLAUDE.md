# CLAUDE.md — brogameagent

Game-agent algorithms library: MCTS variants (single-hero, decoupled, team, layered, options, info-set, root-parallel), AlphaZero-style ExIt training, and hand-crafted autograd-free NN circuits, all over a snapshot-restorable combat sim. MuZero and other model-based extensions land here.

## Layout

```
include/brogameagent/
  grid/            tile-grid simulation harness
  learn/           ExIt trainer, replay, neural adapters, inference server
  nn/              circuit classes (Linear, Attention, Encoder, …) plus the
                   assembled hero nets: SingleHeroNet (DeepSets),
                   SingleHeroNetST (set-transformer), SingleHeroNetTX
                   (per-stream transformer; also a learn::BatchedNet),
                   PolicyValueNet — tensor + scalar ops live in brotensor
  *.h              World, Agent, Mcts variants (incl. info_set_mcts/belief/
                   observability, generic_mcts), perception/observation/reward,
                   capability + policy (Capability/CapabilitySet/Policy/
                   ScriptedMinionPolicy), recorder, simulation, vec_simulation

src/                 implementation
tests/               GoogleTest-style; tests/gpu/ is GPU-conditional
tools/               replay_query, mcts_bench, nn_* CLIs
examples/            tutorial-grade demos
```

The `nn/` module owns higher-level circuit structure (SGD/Adam updates,
serialization, `to(Device)` migration). The underlying tensor type and
per-op math live in [brotensor](../brotensor): one `brotensor::Tensor`
that carries a runtime `Device` tag, and a single device-neutral op
surface (`brotensor::linear_forward(...)`, etc.) that dispatches to the
CPU/CUDA/Metal backend by its operands' device. **Do not reintroduce
tensor storage or op math into brogameagent** — a missing CPU op gets
added to brotensor's CPU backend, not hand-rolled here.

## Build

```sh
# CPU default — brotensor's CPU backend is always built in.
cmake -S . -B build && cmake --build build --config Release

# Add a GPU backend (brotensor compiles it as an extra self-registering
# static lib; the dispatcher routes by tensor device tag at runtime):
cmake -S . -B build -DBROGAMEAGENT_WITH_CUDA=ON
cmake -S . -B build -DBROGAMEAGENT_WITH_METAL=ON
```

`brotensor` is an unconditional `add_subdirectory` dependency at `../brotensor`. `bromath` (header-only) is also a sibling at `../bromath`. Both have first-loader-wins guards so they're safe inside a parent project that pulls in multiple siblings.

Defines that propagate from brotensor: `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL` / `BROTENSOR_HAS_GPU`. Code rarely needs them — the unified `Tensor` and op surface compile the same regardless of backend; reach for them only to gate a test path that genuinely needs a GPU device present.

## Tests

```sh
ctest --test-dir build -C Release
# or
build/Release/brogameagent_test.exe
```

`tests/test_main.cpp` is the main CPU-side test entry (sim, combat, MCTS,
recorder, …). Alongside it: `test_transformer.cpp` (MHA / FeedForward /
TransformerBlock / TransformerEncoder), `test_adam.cpp` (Adam
bias-correction), and `test_single_hero_net_tx.cpp` (SingleHeroNetTX
forward / FD-gradient / save-load) are separate CPU test binaries.
`tools/nn_check` runs finite-diff gradient checks over every circuit.
`tests/gpu/` is built only when a GPU backend is on and exercises the
dispatch layer (host↔device parity, batched ops, layer dispatch
round-trips, the MCTS / inference-server paths).

## Conventions

- **Tensor + ops are external.** A layer holds `brotensor::Tensor` members and calls the device-neutral ops (`brotensor::linear_forward(...)`, `brotensor::layernorm_forward(...)`, …); the op dispatches on its operands' `Device` tag. There is no separate host/device tensor type and no `_cpu`/`_gpu` op suffix. Fully-qualified call sites are the convention.
- **Device migration pattern.** A parameter-bearing layer keeps one `brotensor::Tensor` per parameter/grad/optimizer-buffer and a `brotensor::Device device_` field that caches where they live. `to(Device d)` does `member_ = member_.to(d)` for every owned tensor (and recurses into sub-layers), then sets `device_`. `Tensor::to()` to an unregistered backend throws on its own — no separate guard needed.
- **`resize()` does not zero.** `brotensor::Tensor::resize()` leaves contents undefined; after resizing a gradient / velocity / Adam-moment buffer in `init()`/`load_from()`, call `.zero()` explicitly. `Tensor::mat`/`vec` factories are zero-filled.
- **Autograd-free.** Every circuit owns its own `forward` and `backward`. A net's `backward()` calls them in reverse order. No tape, no graph. Keep gradients hand-readable.
- **Factored heads.** Policies are factored (`MoveDir × AttackSlot × AbilitySlot`), not flat-joint. Aligns with `action_mask::build`. Don't collapse to a flat softmax.
- **Mask everywhere.** Masked softmax + cross-entropy zero out illegal slots in fwd and bwd; trainer never post-filters.
- **WeightsHandle.** Atomic publish/subscribe for hot-swapping nets during a live game. Publishers bump a version; readers snapshot per-decision and reload on version change. Don't hold raw weight pointers across MCTS decisions.

## When extending

- New MCTS variant: implement `mcts::IEvaluator` / `mcts::IPrior` and your search class; reuse `NeuralEvaluator` / `NeuralPrior` so a `SingleHeroNet` plugs in without engine changes.
- New layer: live entirely in `include/brogameagent/nn/` + `src/nn/`. If you need a CPU op that doesn't exist in brotensor yet, add it there (see brotensor's CLAUDE.md), not here.
- New model-based agent (MuZero, etc.): goes in `learn/` next to `generic_trainer`, `gumbel`, `forward_model`. Treat it as an additional ExIt-shaped consumer of MCTS-derived targets.
- Replay format (`.bgar`) and weights format (`.bgnn`) are packed-POD with magic numbers; bump the version field on any schema change.
