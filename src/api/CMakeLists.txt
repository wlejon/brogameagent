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

target_compile_features(brogameagent_api PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(brogameagent_api PRIVATE /W4)
else()
    target_compile_options(brogameagent_api PRIVATE -Wall -Wextra)
endif()
