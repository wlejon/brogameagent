cmake_minimum_required(VERSION 3.24)
project(brogameagent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/MP /permissive- /utf-8)
    # Static CRT only when building standalone (not consumed as a subdirectory)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- brogameagent library ---

add_library(brogameagent STATIC
    src/nav_grid.cpp
    src/steering.cpp
    src/agent.cpp
    src/perception.cpp
    src/world.cpp
    src/observation.cpp
    src/action_mask.cpp
    src/reward.cpp
    src/simulation.cpp
    src/recorder.cpp
    src/replay_reader.cpp
    src/vec_simulation.cpp
    src/mcts.cpp
    src/generic_mcts.cpp
    src/capability_set.cpp
    src/capabilities.cpp
    src/scripted_minion_policy.cpp
    src/observability.cpp
    src/belief.cpp
    src/info_set_mcts.cpp
    # --- NN circuits ---
    src/nn/ops.cpp
    src/nn/circuits.cpp
    src/nn/encoder.cpp
    src/nn/decoder.cpp
    src/nn/autoencoder.cpp
    src/nn/heads.cpp
    src/nn/net.cpp
    src/nn/policy_value_net.cpp
    src/nn/layernorm.cpp
    src/nn/embedding.cpp
    src/nn/attention.cpp
    src/nn/gru.cpp
    src/nn/set_transformer.cpp
    src/nn/net_st.cpp
    src/nn/heads_dist.cpp
    src/nn/ensemble.cpp
    src/nn/forward_model.cpp
    # --- Learning primitives + adapters ---
    src/learn/trainer.cpp
    src/learn/generic_trainer.cpp
    src/learn/search_trace.cpp
    src/learn/neural_adapters.cpp
    src/learn/gumbel.cpp
    src/learn/contrastive.cpp
    # --- Grid-world / platformer training kit ---
    src/grid/obs_window.cpp
    src/grid/generic_recorder.cpp
    src/grid/harness.cpp
)
add_library(brogameagent::brogameagent ALIAS brogameagent)

target_include_directories(brogameagent PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(brogameagent PRIVATE src)
target_compile_features(brogameagent PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(brogameagent PRIVATE /W4)
else()
    target_compile_options(brogameagent PRIVATE -Wall -Wextra)
endif()

# --- Install / export ---
option(BROGAMEAGENT_INSTALL "Generate install targets for brogameagent" ${PROJECT_IS_TOP_LEVEL})
if(BROGAMEAGENT_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS brogameagent
        EXPORT brogameagentTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT brogameagentTargets
        FILE brogameagentTargets.cmake
        NAMESPACE brogameagent::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brogameagent
    )

    set(_cfg_in "${CMAKE_CURRENT_BINARY_DIR}/brogameagentConfig.cmake.in")
    file(WRITE "${_cfg_in}"
        "@PACKAGE_INIT@\n"
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/brogameagentTargets.cmake\")\n"
    )
    configure_package_config_file("${_cfg_in}"
        "${CMAKE_CURRENT_BINARY_DIR}/brogameagentConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brogameagent
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/brogameagentConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brogameagent
    )
endif()

# --- CLI tools ---
option(BROGAMEAGENT_TOOLS "Build brogameagent CLI tools" ${PROJECT_IS_TOP_LEVEL})
if(BROGAMEAGENT_TOOLS)
    add_executable(replay_query tools/replay_query.cpp)
    target_link_libraries(replay_query PRIVATE brogameagent)

    add_executable(mcts_bench tools/mcts_bench.cpp)
    target_link_libraries(mcts_bench PRIVATE brogameagent)

    add_executable(nn_check tools/nn_check.cpp)
    target_link_libraries(nn_check PRIVATE brogameagent)

    add_executable(nn_train_value tools/nn_train_value.cpp)
    target_link_libraries(nn_train_value PRIVATE brogameagent)

    add_executable(nn_exit tools/nn_exit.cpp)
    target_link_libraries(nn_exit PRIVATE brogameagent)

    add_executable(nn_pretrain_ae tools/nn_pretrain_ae.cpp)
    target_link_libraries(nn_pretrain_ae PRIVATE brogameagent)
endif()

# Tests
option(BROGAMEAGENT_TESTS "Build brogameagent tests" ON)
if(BROGAMEAGENT_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
option(BROGAMEAGENT_EXAMPLES "Build brogameagent examples" ${PROJECT_IS_TOP_LEVEL})
if(BROGAMEAGENT_EXAMPLES AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(examples)
endif()
