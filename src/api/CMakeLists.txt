add_library(brogameagent_api STATIC
    api.cpp
    host_class.cpp
    host_ai_game.cpp
    host_ai_navgrid.cpp
    host_ai_navmesh.cpp
    host_ai_agent.cpp
    host_ai_unit.cpp
    host_ai_extras.cpp
    host_ai_mcts.cpp
    host_ai_mcts_support.cpp
    host_ai_primitives.cpp
    host_ai_planner.cpp
    host_ai_sim.cpp
    host_ai_belief.cpp
    host_ai_world_extra.cpp
    # The nn / learn / grid surface. Each of these compiles to nothing (or to
    # an `{ available: false }` stub) unless BROGAMEAGENT_HAS_NN is defined,
    # which brogameagent sets PUBLIC when BROGAMEAGENT_WITH_NN is on.
    host_ai_nn_support.cpp
    host_ai_nn.cpp
    host_ai_nn_nets.cpp
    host_ai_nn_ops.cpp
    host_ai_learn.cpp
    host_ai_learn_generic.cpp
    host_ai_grid.cpp
    host_ai_grid_recording.cpp
)

target_include_directories(brogameagent_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(brogameagent_api PUBLIC
    brogameagent
    bromath::bromath
    bronze_runtime_shared
)

if(BROGAMEAGENT_WITH_NN AND TARGET brotensor_api)
    # The nn bindings hand tensors to JS as bro.tensor GpuTensor handles
    # (getTensorFromHandle / createGpuTensorValue) rather than reviving the
    # old AITensor class, so they need brotensor's own binding. brotensor's
    # api is INSTALLED by the host exactly once — brogameagent never calls
    # installTensor() itself.
    target_link_libraries(brogameagent_api PUBLIC brotensor_api)
endif()

target_compile_features(brogameagent_api PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(brogameagent_api PRIVATE /W4)
else()
    target_compile_options(brogameagent_api PRIVATE -Wall -Wextra)
endif()
