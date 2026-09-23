# ORCA local-avoidance solver + World integration. Core-only — builds with
# the NN layer off.
add_executable(brogameagent_avoidance_test
    test_avoidance.cpp
)
target_link_libraries(brogameagent_avoidance_test PRIVATE brogameagent)
if(MSVC)
    target_compile_options(brogameagent_avoidance_test PRIVATE /W4)
else()
    target_compile_options(brogameagent_avoidance_test PRIVATE -Wall -Wextra)
endif()
add_test(NAME brogameagent_avoidance_test COMMAND brogameagent_avoidance_test)

# Weighted hex-grid navigator against a linear-scan reference (tie order,
# identical paths, fields, clearance, components). Core-only.
add_executable(brogameagent_hex_nav_test
    test_hex_nav.cpp
)
target_link_libraries(brogameagent_hex_nav_test PRIVATE brogameagent)
if(MSVC)
    target_compile_options(brogameagent_hex_nav_test PRIVATE /W4)
else()
    target_compile_options(brogameagent_hex_nav_test PRIVATE -Wall -Wextra)
endif()
add_test(NAME brogameagent_hex_nav_test COMMAND brogameagent_hex_nav_test)

# Replay readers (.bgar and generic grid replays) against corrupt counts and
# offsets. Core-only.
add_executable(brogameagent_replay_reader_test
    test_replay_reader.cpp
)
target_link_libraries(brogameagent_replay_reader_test PRIVATE brogameagent)
if(MSVC)
    target_compile_options(brogameagent_replay_reader_test PRIVATE /W4)
else()
    target_compile_options(brogameagent_replay_reader_test PRIVATE -Wall -Wextra)
endif()
add_test(NAME brogameagent_replay_reader_test COMMAND brogameagent_replay_reader_test)

# Simulation add/removePolicy from inside a policy during step(). Core-only.
add_executable(brogameagent_simulation_test
    test_simulation.cpp
)
target_link_libraries(brogameagent_simulation_test PRIVATE brogameagent)
if(MSVC)
    target_compile_options(brogameagent_simulation_test PRIVATE /W4)
else()
    target_compile_options(brogameagent_simulation_test PRIVATE -Wall -Wextra)
endif()
add_test(NAME brogameagent_simulation_test COMMAND brogameagent_simulation_test)

# Polygon navmesh (Recast/Detour). Core-only — builds with the NN layer off,
# but needs the navmesh feature compiled in.
if(BROGAMEAGENT_WITH_NAVMESH)
    add_executable(brogameagent_nav_mesh_test
        test_nav_mesh.cpp
    )
    target_link_libraries(brogameagent_nav_mesh_test PRIVATE brogameagent)
    if(MSVC)
        target_compile_options(brogameagent_nav_mesh_test PRIVATE /W4)
    else()
        target_compile_options(brogameagent_nav_mesh_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_nav_mesh_test COMMAND brogameagent_nav_mesh_test)
endif()

# Bronze JavaScript binding (brogameagent_api): mount points, TypeErrors on
# bad arguments, a NavGrid findPath round trip. Core-only.
# brogameagent_test_api_restored covers the surface the bronze port dropped
# and this tree restored: World.findById/registerAbility/seed,
# Agent.applyAction, HexNav.field, and the nn / learn / grid namespaces.
if(BROGAMEAGENT_ENABLE_API AND TARGET brogameagent_api)
    # With the NN layer on, brogameagent_api links brotensor_api, whose
    # bronze-compiled tensor.o references brass_gc_write_barrier the way a
    # loaded module would — and that symbol is defined BOTH by brass.lib and
    # by bronze_runtime_shared.dll. Naming brass first and /INCLUDE-ing a
    # symbol only brass's GC object defines forces that object in before any
    # static reference exists, so the DLL's member is never pulled (LNK2005
    # otherwise). Same recipe as bro's bronze_host and brotensor's own
    # test_api.
    set(_bga_brass "")
    if(TARGET brass)
        set(_bga_brass brass)
    endif()

    foreach(_bga_api_test IN ITEMS
            "brogameagent_test_api:test_game_ai_api.cpp"
            "brogameagent_test_api_restored:test_game_ai_api_restored.cpp"
            "brogameagent_test_api_cycles:test_game_ai_api_cycles.cpp")
        string(REPLACE ":" ";" _bga_pair "${_bga_api_test}")
        list(GET _bga_pair 0 _t)
        list(GET _bga_pair 1 _src)

        add_executable(${_t} ${_src})
        target_link_libraries(${_t} PRIVATE
            ${_bga_brass} brogameagent_api bronze_runtime_shared)
        if(MSVC AND _bga_brass)
            target_link_options(${_t} PRIVATE "/INCLUDE:brass_gc_card_table_base")
        endif()
        if(MSVC)
            target_compile_options(${_t} PRIVATE /W4)
        else()
            target_compile_options(${_t} PRIVATE -Wall -Wextra)
        endif()
        # bronze's shared runtime is built into BRONZE_SHARED_RUNTIME_DIR, not
        # beside this executable, and the PE loader only looks beside the .exe
        # (0xc0000135 + a modal "dll was not found" otherwise). Stage it next to
        # the test as a post-build step and put both dirs on ctest's PATH.
        if(WIN32)
            add_custom_command(TARGET ${_t} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:bronze_runtime_shared>
                    $<TARGET_FILE_DIR:${_t}>
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_LINKER_FILE:bronze_runtime_shared>
                    $<TARGET_FILE_DIR:${_t}>
            )
        endif()
        # It runs JavaScript, which brass compiles to machine code for the
        # target; on one brass has no code generator for (BRASS_HOST_BACKEND,
        # decided in brass's CMakeLists — Apple Silicon today) that code cannot
        # run. The executable still builds there, which is the check that the
        # embed links; the test is not registered rather than registered to
        # bus-error.
        if(NOT BRASS_HOST_BACKEND)
            message(STATUS "brogameagent: ${_t} not registered (brass has no code generator for this target)")
        else()
            add_test(NAME ${_t} COMMAND ${_t})
            # The same run with bronze collecting on every allocation: a raw
            # Value held across an allocating call in the binding crashes
            # here instead of once in a blue moon in an app.
            add_test(NAME ${_t}_gcstress COMMAND ${_t})
            if(WIN32)
                set_tests_properties(${_t} PROPERTIES
                    ENVIRONMENT "PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:${_t}>;$ENV{PATH}")
                set_tests_properties(${_t}_gcstress PROPERTIES
                    ENVIRONMENT "BRONZE_GC_STRESS=1;BRONZE_GC_POISON=1;PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:${_t}>;$ENV{PATH}")
            else()
                set_tests_properties(${_t}_gcstress PROPERTIES
                    ENVIRONMENT "BRONZE_GC_STRESS=1;BRONZE_GC_POISON=1")
            endif()
        endif()
    endforeach()
endif()

# Tree-of-Thought MCTS reasoning engine unit tests. Core-only.
add_executable(test_tot_reasoning
    test_tot_reasoning.cpp
)
target_link_libraries(test_tot_reasoning PRIVATE brogameagent)
if(MSVC)
    target_compile_options(test_tot_reasoning PRIVATE /W4)
else()
    target_compile_options(test_tot_reasoning PRIVATE -Wall -Wextra)
endif()
add_test(NAME test_tot_reasoning COMMAND test_tot_reasoning)

# The current test suite exercises the NN + learning layer (test_main.cpp
# itself includes nn/factored.h), so it only builds when that layer is on.
if(NOT BROGAMEAGENT_WITH_NN)
    message(STATUS "brogameagent: NN layer off (BROGAMEAGENT_WITH_NN=OFF) — skipping the NN test suite")
    return()
endif()

add_executable(brogameagent_test
    test_main.cpp
)

target_link_libraries(brogameagent_test PRIVATE brogameagent)

if(MSVC)
    target_compile_options(brogameagent_test PRIVATE /W4)
else()
    target_compile_options(brogameagent_test PRIVATE -Wall -Wextra)
endif()

add_test(NAME brogameagent_test COMMAND brogameagent_test)

# Transformer CPU correctness (MHA, FF, TransformerBlock, TransformerEncoder).
add_executable(brogameagent_transformer_test
    test_transformer.cpp
)
target_link_libraries(brogameagent_transformer_test PRIVATE brogameagent)
if(MSVC)
    target_compile_options(brogameagent_transformer_test PRIVATE /W4)
else()
    target_compile_options(brogameagent_transformer_test PRIVATE -Wall -Wextra)
endif()
add_test(NAME brogameagent_transformer_test COMMAND brogameagent_transformer_test)

# Adam CPU correctness (bias-correction reference + convergence).
add_executable(brogameagent_adam_test
    test_adam.cpp
)
target_link_libraries(brogameagent_adam_test PRIVATE brogameagent)
if(MSVC)
    target_compile_options(brogameagent_adam_test PRIVATE /W4)
else()
    target_compile_options(brogameagent_adam_test PRIVATE -Wall -Wextra)
endif()
add_test(NAME brogameagent_adam_test COMMAND brogameagent_adam_test)

# SingleHeroNetTX CPU correctness (forward smoke, FD gradient, save/load).
add_executable(brogameagent_single_hero_net_tx_test
    test_single_hero_net_tx.cpp
)
target_link_libraries(brogameagent_single_hero_net_tx_test PRIVATE brogameagent)
if(MSVC)
    target_compile_options(brogameagent_single_hero_net_tx_test PRIVATE /W4)
else()
    target_compile_options(brogameagent_single_hero_net_tx_test PRIVATE -Wall -Wextra)
endif()
add_test(NAME brogameagent_single_hero_net_tx_test
         COMMAND brogameagent_single_hero_net_tx_test)

if(BROGAMEAGENT_WITH_CUDA OR BROGAMEAGENT_WITH_METAL)
    # Per-op CPU↔GPU parity tests for the brotensor op surface now live in
    # brotensor/tests/ (test_*_parity.cpp). What remains here exercises
    # brogameagent's own layer / server / trainer GPU dispatch.

    # End-to-end layer GPU-dispatch parity tests (per-layer host<->device).
    set(_bga_gpu_dispatch_tests
        layer_layernorm
        layer_layernorm_adam
        layer_attention
        layer_mha
        layer_feedforward
        layer_transformer_block
        layer_transformer_encoder
        layer_autoencoder
        single_hero_net_tx
    )
    foreach(_name IN LISTS _bga_gpu_dispatch_tests)
        set(_target brogameagent_gpu_${_name}_dispatch_test)
        add_executable(${_target} gpu/test_${_name}_dispatch.cpp)
        target_link_libraries(${_target} PRIVATE brogameagent)
        if(BROGAMEAGENT_WITH_CUDA)
            target_include_directories(${_target} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        endif()
        if(MSVC)
            target_compile_options(${_target} PRIVATE /W4)
        else()
            target_compile_options(${_target} PRIVATE -Wall -Wextra)
        endif()
        add_test(NAME ${_target} COMMAND ${_target})
    endforeach()

    # CPU↔GPU parity test for the batched GenericExItTrainer GPU step.
    add_executable(brogameagent_gpu_generic_trainer_batched_parity_test
        gpu/test_generic_trainer_batched_parity.cpp
    )
    target_link_libraries(brogameagent_gpu_generic_trainer_batched_parity_test
        PRIVATE brogameagent)
    if(BROGAMEAGENT_WITH_CUDA)
        target_include_directories(brogameagent_gpu_generic_trainer_batched_parity_test
            PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(brogameagent_gpu_generic_trainer_batched_parity_test
            PRIVATE /W4)
    else()
        target_compile_options(brogameagent_gpu_generic_trainer_batched_parity_test
            PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_gpu_generic_trainer_batched_parity_test
             COMMAND brogameagent_gpu_generic_trainer_batched_parity_test)

    # End-to-end GPU smoke test for GenericExItTrainer.
    add_executable(brogameagent_gpu_exit_trainer_test
        gpu/test_exit_trainer_gpu.cpp
    )
    target_link_libraries(brogameagent_gpu_exit_trainer_test PRIVATE brogameagent)
    if(BROGAMEAGENT_WITH_CUDA)
        target_include_directories(brogameagent_gpu_exit_trainer_test PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(brogameagent_gpu_exit_trainer_test PRIVATE /W4)
    else()
        target_compile_options(brogameagent_gpu_exit_trainer_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_gpu_exit_trainer_test
             COMMAND brogameagent_gpu_exit_trainer_test)

    # PolicyValueNet::forward_batched parity test.
    add_executable(brogameagent_gpu_pvn_batched_test
        gpu/test_policy_value_net_batched.cpp
    )
    target_link_libraries(brogameagent_gpu_pvn_batched_test PRIVATE brogameagent)
    if(BROGAMEAGENT_WITH_CUDA)
        target_include_directories(brogameagent_gpu_pvn_batched_test PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(brogameagent_gpu_pvn_batched_test PRIVATE /W4)
    else()
        target_compile_options(brogameagent_gpu_pvn_batched_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_gpu_pvn_batched_test
             COMMAND brogameagent_gpu_pvn_batched_test)

    # BatchedInferenceServer functional + stress + bench.
    add_executable(brogameagent_gpu_inference_server_test
        gpu/test_inference_server.cpp
    )
    target_link_libraries(brogameagent_gpu_inference_server_test PRIVATE brogameagent)
    if(BROGAMEAGENT_WITH_CUDA)
        target_include_directories(brogameagent_gpu_inference_server_test PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(brogameagent_gpu_inference_server_test PRIVATE /W4)
    else()
        target_compile_options(brogameagent_gpu_inference_server_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_gpu_inference_server_test
             COMMAND brogameagent_gpu_inference_server_test)

    # MCTS + BatchedInferenceServer integration (equivalence + bench).
    add_executable(brogameagent_gpu_mcts_server_test
        gpu/test_mcts_server.cpp
    )
    target_link_libraries(brogameagent_gpu_mcts_server_test PRIVATE brogameagent)
    if(BROGAMEAGENT_WITH_CUDA)
        target_include_directories(brogameagent_gpu_mcts_server_test PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(brogameagent_gpu_mcts_server_test PRIVATE /W4)
    else()
        target_compile_options(brogameagent_gpu_mcts_server_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_gpu_mcts_server_test
             COMMAND brogameagent_gpu_mcts_server_test)

    # SingleHeroNetTX::forward_batched parity test.
    add_executable(brogameagent_gpu_tx_batched_test
        gpu/test_single_hero_net_tx_batched.cpp
    )
    target_link_libraries(brogameagent_gpu_tx_batched_test PRIVATE brogameagent)
    if(BROGAMEAGENT_WITH_CUDA)
        target_include_directories(brogameagent_gpu_tx_batched_test PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(brogameagent_gpu_tx_batched_test PRIVATE /W4)
    else()
        target_compile_options(brogameagent_gpu_tx_batched_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_gpu_tx_batched_test
             COMMAND brogameagent_gpu_tx_batched_test)

    # MCTS + BatchedInferenceServer integration with the TX net.
    add_executable(brogameagent_gpu_mcts_server_tx_test
        gpu/test_mcts_server_tx.cpp
    )
    target_link_libraries(brogameagent_gpu_mcts_server_tx_test PRIVATE brogameagent)
    if(BROGAMEAGENT_WITH_CUDA)
        target_include_directories(brogameagent_gpu_mcts_server_tx_test PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    if(MSVC)
        target_compile_options(brogameagent_gpu_mcts_server_tx_test PRIVATE /W4)
    else()
        target_compile_options(brogameagent_gpu_mcts_server_tx_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_gpu_mcts_server_tx_test
             COMMAND brogameagent_gpu_mcts_server_tx_test)
endif()
