cmake_minimum_required(VERSION 3.24)
project(brogameagent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    # Restrict MSVC-only flags to CXX. nvcc forwards host args verbatim and
    # things like /MP, /permissive-, /utf-8, /W4 either confuse its pipeline
    # (single-input rule for /MP) or get parsed as nvcc options themselves.
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    # Static CRT only when building standalone (not consumed as a subdirectory)
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- Optional GPU dispatch (CUDA or Metal via brotensor sibling) ---
#
# brogameagent dispatches GPU work to brotensor (sibling repo / submodule).
# Selecting one of these options pulls in brotensor with the matching backend
# and sets the BROTENSOR_HAS_CUDA / _METAL / _GPU defines on us transitively.
# The two are mutually exclusive at brotensor's level.
option(BROGAMEAGENT_WITH_CUDA  "Dispatch GPU work via brotensor (CUDA)"  OFF)
option(BROGAMEAGENT_WITH_METAL "Dispatch GPU work via brotensor (Metal)" OFF)
if(BROGAMEAGENT_WITH_CUDA AND BROGAMEAGENT_WITH_METAL)
    message(FATAL_ERROR
        "BROGAMEAGENT_WITH_CUDA and BROGAMEAGENT_WITH_METAL are mutually exclusive.")
endif()

# --- bromath (shared header-only math, sibling repo) ---
# Guard against double-add — when consumed as a sibling lib from another
# parent project (bro, etc.) the parent or another sibling may have already
# added bromath. First-loader wins.
if(NOT TARGET bromath)
    set(BROMATH_DIR "${CMAKE_SOURCE_DIR}/../bromath" CACHE PATH "bromath sibling")
    if(EXISTS "${BROMATH_DIR}/CMakeLists.txt")
        add_subdirectory("${BROMATH_DIR}" "${CMAKE_BINARY_DIR}/bromath" EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "bromath not found at ${BROMATH_DIR}")
    endif()
endif()

# --- brotensor (host + GPU tensor + ops, sibling repo) ---
# brotensor is now a hard dependency — it owns the host Tensor type and CPU
# ops as well as the optional CUDA/Metal kernels. Always added; the
# BROTENSOR_WITH_CUDA / BROTENSOR_WITH_METAL cache vars are forwarded based on
# the matching brogameagent option. When neither GPU backend is requested,
# both are forced OFF so brotensor builds CPU-only. Double-add guard mirrors
# bromath: first-loader wins when consumed as a sibling.
if(BROGAMEAGENT_WITH_CUDA)
    set(BROTENSOR_WITH_CUDA  ON  CACHE BOOL "" FORCE)
    set(BROTENSOR_WITH_METAL OFF CACHE BOOL "" FORCE)
elseif(BROGAMEAGENT_WITH_METAL)
    set(BROTENSOR_WITH_CUDA  OFF CACHE BOOL "" FORCE)
    set(BROTENSOR_WITH_METAL ON  CACHE BOOL "" FORCE)
else()
    set(BROTENSOR_WITH_CUDA  OFF CACHE BOOL "" FORCE)
    set(BROTENSOR_WITH_METAL OFF CACHE BOOL "" FORCE)
endif()
if(NOT TARGET brotensor)
    set(BROTENSOR_DIR "${CMAKE_SOURCE_DIR}/../brotensor" CACHE PATH "brotensor sibling")
    if(EXISTS "${BROTENSOR_DIR}/CMakeLists.txt")
        add_subdirectory("${BROTENSOR_DIR}" "${CMAKE_BINARY_DIR}/brotensor" EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "brotensor not found at ${BROTENSOR_DIR}")
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
    src/nn/factored.cpp
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
    src/nn/multi_head_attention.cpp
    src/nn/feedforward.cpp
    src/nn/transformer_block.cpp
    src/nn/transformer_encoder.cpp
    src/nn/net_tx.cpp
    # --- Learning primitives + adapters ---
    src/learn/trainer.cpp
    src/learn/generic_trainer.cpp
    src/learn/search_trace.cpp
    src/learn/neural_adapters.cpp
    src/learn/gumbel.cpp
    src/learn/contrastive.cpp
    src/learn/inference_backend.cpp
    src/learn/inference_server.cpp
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
# bromath is header-only INTERFACE; expose its include dirs without putting
# the bromath target itself into our install-export set.
target_link_libraries(brogameagent PUBLIC $<BUILD_INTERFACE:bromath::bromath>)
# brotensor is a hard dependency (host Tensor + CPU ops live there); GPU
# kernels are an additive feature of brotensor itself.
target_link_libraries(brogameagent PUBLIC $<BUILD_INTERFACE:brotensor::brotensor>)

if(MSVC)
    target_compile_options(brogameagent PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brogameagent PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
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

    if(BROGAMEAGENT_WITH_CUDA OR BROGAMEAGENT_WITH_METAL)
        add_executable(nn_pretrain_ae_gpu tools/nn_pretrain_ae_gpu.cpp)
        target_link_libraries(nn_pretrain_ae_gpu PRIVATE brogameagent)

        add_executable(nn_exit_gpu tools/nn_exit_gpu.cpp)
        target_link_libraries(nn_exit_gpu PRIVATE brogameagent)
    endif()
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
