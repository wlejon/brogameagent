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

# --- Optional GPU dispatch (CUDA or Metal via brotensor) ---
#
# brogameagent dispatches GPU work to brotensor (a pinned dependency).
# Selecting one of these options pulls in brotensor with the matching backend
# and sets the BROTENSOR_HAS_CUDA / _METAL / _VULKAN / _GPU defines on us
# transitively. CUDA and Metal are mutually exclusive at brotensor's level.
option(BROGAMEAGENT_WITH_CUDA  "Dispatch GPU work via brotensor (CUDA)"  OFF)
option(BROGAMEAGENT_WITH_METAL "Dispatch GPU work via brotensor (Metal)" OFF)
# Vulkan (the AMD GPU path) is additive: it coexists with either of the above.
option(BROGAMEAGENT_WITH_VULKAN "Dispatch GPU work via brotensor (Vulkan)" OFF)
if(BROGAMEAGENT_WITH_CUDA AND BROGAMEAGENT_WITH_METAL)
    message(FATAL_ERROR
        "BROGAMEAGENT_WITH_CUDA and BROGAMEAGENT_WITH_METAL are mutually exclusive.")
endif()

# --- NN + learning layer (optional) ---
#
# brotensor is used by exactly two source trees: src/nn/* and src/learn/*.
# The core — navmesh, pathfinding, steering, perception, MCTS, grid harness —
# has zero brotensor references and never includes an nn/ or learn/ header, so
# the whole neural layer gates cleanly as one unit. With BROGAMEAGENT_WITH_NN=OFF
# brogameagent builds with no brotensor dependency at all, which is what a
# minimal `bro` build wants. GPU dispatch is a feature of the NN layer, so
# requesting a GPU backend implies it. Default ON preserves the full surface.
option(BROGAMEAGENT_WITH_NN "Build the neural-net + learning layer (requires brotensor)" ON)
if(BROGAMEAGENT_WITH_CUDA OR BROGAMEAGENT_WITH_METAL OR BROGAMEAGENT_WITH_VULKAN)
    set(BROGAMEAGENT_WITH_NN ON CACHE BOOL "" FORCE)
endif()

# --- Polygon navmesh (optional, Recast/Detour from vcpkg) ---
#
# NavMesh (include/brogameagent/nav_mesh.h) bakes and queries polygon
# navmeshes via the recastnavigation package — the 3D counterpart to NavGrid
# (slopes, bridges, multi-level interiors). It is the library's only external
# (non-sibling) dependency, so it gates:
#   - Standalone builds default ON, but soft-disable (STATUS message) when
#     find_package can't see recastnavigation — consumers without vcpkg still
#     build. Point CMake at it via CMAKE_TOOLCHAIN_FILE (vcpkg toolchain) or
#     CMAKE_PREFIX_PATH=<vcpkg>/installed/<triplet>.
#   - Subdirectory consumers (bro etc.) default OFF and must opt in with
#     -DBROGAMEAGENT_WITH_NAVMESH=ON, so adding this feature never changes an
#     existing parent configure.
# When ON, brogameagent defines BROGAMEAGENT_HAS_NAVMESH=1 on its interface.
if(PROJECT_IS_TOP_LEVEL)
    set(_bga_navmesh_default ON)
else()
    set(_bga_navmesh_default OFF)
endif()
option(BROGAMEAGENT_WITH_NAVMESH
    "Polygon navmesh bake + queries (requires recastnavigation)" ${_bga_navmesh_default})
if(BROGAMEAGENT_WITH_NAVMESH)
    # A standalone MSVC build links the static CRT (/MT, above), so it needs
    # Recast from a static vcpkg triplet. Left to itself find_package also
    # searches the prefixes of PATH entries, and a vcpkg
    # installed/x64-windows/bin on PATH (for its DLLs) hands it the /MD copy,
    # which does not link. So: no PATH search, and a vcpkg tree seen on PATH or
    # under VCPKG_ROOT is tried in its -static triplet. A vcpkg toolchain
    # (VCPKG_TARGET_TRIPLET) or an explicit CMAKE_PREFIX_PATH /
    # recastnavigation_DIR still decides as before.
    # A /MD copy an earlier configure cached is forgotten too.
    set(_bga_recast_args)
    set(_bga_static_crt OFF)
    if(MSVC AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT VCPKG_TARGET_TRIPLET)
        set(_bga_static_crt ON)
        if(recastnavigation_DIR MATCHES "/installed/([^/]+)/share/"
           AND NOT CMAKE_MATCH_1 MATCHES "-static$")
            unset(recastnavigation_DIR CACHE)
        endif()
    endif()
    if(_bga_static_crt AND NOT recastnavigation_DIR)
        set(_bga_recast_hints)
        file(TO_CMAKE_PATH "$ENV{PATH}" _bga_path)
        foreach(_p IN LISTS _bga_path)
            if(_p MATCHES "^(.*/installed)/([^/]+)/(debug/)?bin/?$")
                set(_root "${CMAKE_MATCH_1}")
                string(REGEX REPLACE "-static(-md)?$" "" _t "${CMAKE_MATCH_2}")
                list(APPEND _bga_recast_hints "${_root}/${_t}-static")
            endif()
        endforeach()
        if(DEFINED ENV{VCPKG_ROOT})
            file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _bga_vcpkg)
            list(APPEND _bga_recast_hints "${_bga_vcpkg}/installed/x64-windows-static")
        endif()
        list(REMOVE_DUPLICATES _bga_recast_hints)
        set(_bga_recast_args HINTS ${_bga_recast_hints} NO_SYSTEM_ENVIRONMENT_PATH)
    endif()
    find_package(recastnavigation CONFIG QUIET ${_bga_recast_args})
    if(NOT recastnavigation_FOUND)
        message(STATUS
            "brogameagent: recastnavigation not found — building without NavMesh. "
            "Install it (vcpkg install recastnavigation) and pass the vcpkg "
            "toolchain or CMAKE_PREFIX_PATH to enable.")
        set(BROGAMEAGENT_WITH_NAVMESH OFF)
    endif()
endif()

# --- Dependencies: bromath (header-only), brotensor (NN layer), bronze ---
# Each resolves through cmake/bro_deps.cmake, like every repo in the bro
# ecosystem: an existing target (a parent project such as bro got there first,
# and its pins win); otherwise the ../<name> working tree beside the top-level
# project; otherwise the commit pinned here, fetched at configure. Leaves first.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bro_deps.cmake)

bro_dependency(bromath GITHUB wlejon/bromath REF 8b511fbd69c014bf0584d2b42e6c96a8cf720f6a TARGET bromath::bromath)

# --- brotensor (host + GPU tensor + ops) ---
# brotensor is now a hard dependency — it owns the host Tensor type and CPU
# ops as well as the optional CUDA/Metal kernels. Always added; the
# BROTENSOR_WITH_CUDA / BROTENSOR_WITH_METAL cache vars are forwarded based on
# the matching brogameagent option. When neither GPU backend is requested,
# both are forced OFF so brotensor builds CPU-only. Double-add guard mirrors
# bromath: first-loader wins when consumed as a sibling.
if(BROGAMEAGENT_WITH_NN)
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
    # Turned on, never forced off: a parent (bro) that enables brotensor's
    # Vulkan backend keeps it whatever this option says.
    if(BROGAMEAGENT_WITH_VULKAN)
        set(BROTENSOR_WITH_VULKAN ON CACHE BOOL "" FORCE)
    endif()
    bro_dependency(brotensor GITHUB wlejon/brotensor REF a332d9283c1e23b9f39e8ecaa4393f937dd26c0a)
endif()
# --- bronze, the JavaScript runtime the api binds (it brings brass, its code
# generator). The api is bound across bronze's C++ embed boundary, so it has to
# build against the bronze the surrounding tree builds against.
bro_dependency(bronze GITHUB wlejon/bronze REF 15ce4cf5fc25d8afe03882985ae95adcba46a6a9 TARGET bronze_runtime_shared
               OPTIONS BRONZE_BUILD_SHARED_RUNTIME=ON BRONZE_BUILD_TESTS=OFF)

# --- brogameagent library ---

add_library(brogameagent STATIC
    src/nav_grid.cpp
    src/nav_grid_field.cpp
    src/hex_nav.cpp
    src/steering.cpp
    src/avoidance.cpp
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
    src/tot_reasoning.cpp
    src/capability_set.cpp
    src/capabilities.cpp
    src/scripted_minion_policy.cpp
    src/observability.cpp
    src/belief.cpp
    src/info_set_mcts.cpp
    # --- Grid-world observation/recording (brotensor-free) ---
    src/grid/obs_window.cpp
    src/grid/generic_recorder.cpp
)
add_library(brogameagent::brogameagent ALIAS brogameagent)

# Polygon navmesh — the only source that depends on Recast/Detour. The
# header is Recast-free, so it ships unconditionally; only the impl gates.
if(BROGAMEAGENT_WITH_NAVMESH)
    target_sources(brogameagent PRIVATE src/nav_mesh.cpp)
    target_link_libraries(brogameagent PRIVATE
        RecastNavigation::Recast RecastNavigation::Detour
        RecastNavigation::DetourTileCache)
    target_compile_definitions(brogameagent PUBLIC BROGAMEAGENT_HAS_NAVMESH=1)
endif()

# The neural + learning layer — the only sources that depend on brotensor.
if(BROGAMEAGENT_WITH_NN)
    target_sources(brogameagent PRIVATE
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
        # --- Grid-world training harness (pulls nn/ + learn/) ---
        src/grid/harness.cpp
    )
endif()

target_include_directories(brogameagent PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(brogameagent PRIVATE src)
target_compile_features(brogameagent PUBLIC cxx_std_20)
# bromath is header-only INTERFACE; expose its include dirs without putting
# the bromath target itself into our install-export set.
target_link_libraries(brogameagent PUBLIC $<BUILD_INTERFACE:bromath::bromath>)
# brotensor backs the NN + learning layer only (host Tensor + CPU ops; GPU
# kernels are an additive feature of brotensor itself). Linked only when that
# layer is built.
if(BROGAMEAGENT_WITH_NN)
    target_link_libraries(brogameagent PUBLIC $<BUILD_INTERFACE:brotensor::brotensor>)
    # Lets consumers (notably src/api) compile the nn/learn/grid JS surface
    # only when the layer behind it exists, the way BROGAMEAGENT_HAS_NAVMESH
    # gates the navmesh surface.
    target_compile_definitions(brogameagent PUBLIC BROGAMEAGENT_HAS_NN=1)
endif()

if(MSVC)
    target_compile_options(brogameagent PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(brogameagent PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

# --- Standalone Bronze JavaScript API ---
option(BROGAMEAGENT_ENABLE_API "Build Bronze JavaScript API" ON)
if(BROGAMEAGENT_ENABLE_API)
    add_subdirectory(src/api)
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

    # NN training/inspection tools need the neural layer (and brotensor).
    if(BROGAMEAGENT_WITH_NN)
        add_executable(nn_check tools/nn_check.cpp)
        target_link_libraries(nn_check PRIVATE brogameagent)

        add_executable(nn_train_value tools/nn_train_value.cpp)
        target_link_libraries(nn_train_value PRIVATE brogameagent)

        add_executable(nn_exit tools/nn_exit.cpp)
        target_link_libraries(nn_exit PRIVATE brogameagent)

        add_executable(nn_pretrain_ae tools/nn_pretrain_ae.cpp)
        target_link_libraries(nn_pretrain_ae PRIVATE brogameagent)

        if(BROGAMEAGENT_WITH_CUDA OR BROGAMEAGENT_WITH_METAL OR BROGAMEAGENT_WITH_VULKAN)
            add_executable(nn_pretrain_ae_gpu tools/nn_pretrain_ae_gpu.cpp)
            target_link_libraries(nn_pretrain_ae_gpu PRIVATE brogameagent)

            add_executable(nn_exit_gpu tools/nn_exit_gpu.cpp)
            target_link_libraries(nn_exit_gpu PRIVATE brogameagent)
        endif()
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
