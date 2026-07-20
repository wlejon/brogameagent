# brogameagent examples

Self-contained demonstrations of the library, ordered by increasing
complexity. Each file is a single `.cpp` with a `main()` that prints a
short summary of what it did. Build with the top-level CMake
(`BROGAMEAGENT_EXAMPLES=ON` by default); executables land in
`build/examples/Release/`.

| # | File | Demonstrates |
|---|---|---|
| 01 | `01_hello_world.cpp` | `World`, `Agent`, team/position/HP; nearest-enemy / ally queries. |
| 02 | `02_nav_and_combat.cpp` | `NavGrid` + obstacles, A\* pathing, scripted `setTarget`/`update`, `World::resolveAttack`, event log. |
| 03 | `03_projectiles.cpp` | `Projectile` with `Single` / homing (`targetId`), `World::spawnProjectile` + `stepProjectiles`. |
| 04 | `04_obs_mask_reward.cpp` | `observation::build`, `action_mask::build`, `RewardTracker` deltas. |
| 05 | `05_simulation.cpp` | `Simulation` harness with a per-agent `PolicyFn`; deterministic seed. |
| 06 | `06_snapshot_rollout.cpp` | `World::snapshot`/`restore` as the fork primitive; 1-ply manual "MCTS" over three candidate actions. |
| 07 | `07_mcts_duel.cpp` | `Mcts` + `MctsConfig` with `AggressiveRollout`, `AttackBiasPrior` (PUCT), progressive widening, `advance_root`. |
| 08 | `08_decoupled_1v1.cpp` | `DecoupledMcts`: simultaneous-move 1v1 with joint output (hero + opponent best response). |
| 09 | `09_team_mcts.cpp` | `TeamMcts`: cooperative multi-agent search with shared `TeamHpDeltaEvaluator`. |
| 10 | `10_layered_planner.cpp` | `LayeredPlanner`: `TacticMcts` (coarse) over `TeamMcts` (fine) with `TacticPrior` bias; shows replan schedule. |
| 11 | `11_root_parallel.cpp` | `root_parallel_search`: N `std::thread`s over independent `World` copies, merged root. |
| 12 | `12_record_and_read.cpp` | `Recorder` writes `.bgar`; `ReplayReader` reads header/roster/frames and queries a trajectory. |
| 13 | `13_capabilities_demo.cpp` | `Capability` / `CapabilitySet` / `Policy`: a tower, minions and a hero driven by `ScriptedMinionPolicy` and the built-in capability factories. |
| 14 | `14_fog_of_war.cpp` | `Observability` filtering, `Belief` tracking, and `InfoSetMcts` planning under partial observability. |
| 15 | `15_grid_corridor.cpp` | The `grid/` harness: `GenericMcts` over a `PolicyValueNet` (prior + value via `WeightsHandle`), trained by `grid::Harness` with `bc_ingest` warm-start, on a tile-grid substrate independent of the combat sim. |

Recommended reading order matches the numbering. The MCTS-focused files
(07 onward) assume familiarity with the primitives in 01-06.

The NN / ExIt layer is demonstrated by CLIs rather than examples. See
`nn_check.cpp`, `nn_train_value.cpp`, and `nn_exit.cpp` under `tools/`.
`nn_exit` in particular is the end-to-end tour:
it exercises the same MCTS engines used here (`Mcts` with `NeuralPrior` +
`NeuralEvaluator`), captures targets via `learn::SearchTrace`, trains
`SingleHeroNet` with `ExItTrainer`, and hot-swaps weights through a
`WeightsHandle`.
