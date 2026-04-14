set(_brogameagent_examples
    01_hello_world
    02_nav_and_combat
    03_projectiles
    04_obs_mask_reward
    05_simulation
    06_snapshot_rollout
    07_mcts_duel
    08_decoupled_1v1
    09_team_mcts
    10_layered_planner
    11_root_parallel
    12_record_and_read
)

foreach(ex IN LISTS _brogameagent_examples)
    add_executable(example_${ex} ${ex}.cpp)
    target_link_libraries(example_${ex} PRIVATE brogameagent)
    set_target_properties(example_${ex} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )
endforeach()
