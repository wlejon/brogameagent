# CLAUDE.md — brogameagent

Game-agent algorithms library: MCTS variants (single-hero, decoupled, team, layered, options, info-set, root-parallel), AlphaZero-style ExIt training, and hand-crafted autograd-free NN circuits, all over a snapshot-restorable combat sim. MuZero and other model-based extensions land here.

## Layout

```
include/brogameagent/
  grid/            tile-grid simulation harness
  learn/           ExIt trainer, replay, neural adapters, inference server
  nn/              circuit classes (Linear, Attention, Encoder, …)
                   — tensor + scalar ops live in sibling brotensor
  *.h              World, Agent, Mcts variants, perception/observation/reward,
                   recorder, capability system, simulation, vec_simulation

src/                 implementation
tests/               GoogleTest-style; tests/gpu/ is GPU-conditional
tools/               replay_query, mcts_bench, nn_* CLIs
examples/            tutorial-grade demos
```

The `nn/` module owns higher-level circuit structure (parameter
mirrors, SGD+momentum, serialization, `Device::to()` migration). The
underlying tensor type and per-op math live in
[brotensor](../brotensor) — `brotensor::Tensor`, `brotensor::GpuTensor`,
and the `*_cpu` / `*_gpu` op surfaces. **Do not reintroduce CPU tensor
or op code into brogameagent.**

## Build

```sh
# CPU default — brotensor links in CPU-only.
cmake -S . -B build && cmake --build build --config Release

# GPU dispatch (mutually exclusive):
cmake -S . -B build -DBROGAMEAGENT_WITH_CUDA=ON
cmake -S . -B build -DBROGAMEAGENT_WITH_METAL=ON
```

`brotensor` is an unconditional `add_subdirectory` dependency at `../brotensor`. `bromath` (header-only) is also a sibling at `../bromath`. Both have first-loader-wins guards so they're safe inside a parent project that pulls in multiple siblings.

Defines that propagate from brotensor: `BROTENSOR_HAS_CUDA` / `BROTENSOR_HAS_METAL` / `BROTENSOR_HAS_GPU`. Use these to gate any GPU-only code.

## Tests

```sh
ctest --test-dir build -C Release
# or
build/Release/brogameagent_test.exe
```

`tests/test_main.cpp` is the monolithic CPU-side test entry. `tests/gpu/` is built only when a GPU backend is on and exercises the dispatch layer (host↔device parity, batched ops, layer dispatch round-trips).

## Conventions

- **Tensor + ops are external.** Any new layer should use `brotensor::Tensor` (host) and `brotensor::GpuTensor` (device). For CPU ops call `brotensor::*_cpu(...)`; for GPU call `brotensor::*_gpu(...)`. Don't add `using namespace brogameagent::nn { using brotensor::… }` shims — fully-qualified call sites are the convention since the rename.
- **Device migration pattern.** Parameter-bearing layers hold `Device device_` (from `<brotensor/device.h>`), a host `Tensor` for each parameter, and an optional `GpuTensor` mirror gated on `BROTENSOR_HAS_GPU`. `to(Device::GPU)` uploads, `to(Device::CPU)` downloads. CPU-only builds throw at `to(GPU)` via `brotensor::device_require_gpu(layer_name)`.
- **Autograd-free.** Every circuit owns its own `forward` and `backward`. A net's `backward()` calls them in reverse order. No tape, no graph. Keep gradients hand-readable.
- **Factored heads.** Policies are factored (`MoveDir × AttackSlot × AbilitySlot`), not flat-joint. Aligns with `action_mask::build`. Don't collapse to a flat softmax.
- **Mask everywhere.** Masked softmax + cross-entropy zero out illegal slots in fwd and bwd; trainer never post-filters.
- **WeightsHandle.** Atomic publish/subscribe for hot-swapping nets during a live game. Publishers bump a version; readers snapshot per-decision and reload on version change. Don't hold raw weight pointers across MCTS decisions.

## When extending

- New MCTS variant: implement `mcts::IEvaluator` / `mcts::IPrior` and your search class; reuse `NeuralEvaluator` / `NeuralPrior` so a `SingleHeroNet` plugs in without engine changes.
- New layer: live entirely in `include/brogameagent/nn/` + `src/nn/`. If you need a CPU op that doesn't exist in brotensor yet, add it there (see brotensor's CLAUDE.md), not here.
- New model-based agent (MuZero, etc.): goes in `learn/` next to `generic_trainer`, `gumbel`, `forward_model`. Treat it as an additional ExIt-shaped consumer of MCTS-derived targets.
- Replay format (`.bgar`) and weights format (`.bgnn`) are packed-POD with magic numbers; bump the version field on any schema change.
