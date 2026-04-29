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

if(BROGAMEAGENT_WITH_CUDA)
    add_executable(brogameagent_gpu_tensor_test
        gpu/test_gpu_tensor.cpp
    )
    target_link_libraries(brogameagent_gpu_tensor_test PRIVATE brogameagent)
    if(MSVC)
        target_compile_options(brogameagent_gpu_tensor_test PRIVATE /W4)
    else()
        target_compile_options(brogameagent_gpu_tensor_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME brogameagent_gpu_tensor_test COMMAND brogameagent_gpu_tensor_test)

    # CPU↔GPU numerical parity tests. One executable per op group so failures
    # are surfaced individually by ctest.
    set(_bga_gpu_parity_tests
        linear
        elementwise
        softmax
        layernorm
        attention
        optim
    )
    foreach(_name IN LISTS _bga_gpu_parity_tests)
        set(_target brogameagent_gpu_${_name}_parity_test)
        add_executable(${_target} gpu/test_gpu_${_name}.cpp)
        target_link_libraries(${_target} PRIVATE brogameagent)
        target_include_directories(${_target} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        if(MSVC)
            target_compile_options(${_target} PRIVATE /W4)
        else()
            target_compile_options(${_target} PRIVATE -Wall -Wextra)
        endif()
        add_test(NAME ${_target} COMMAND ${_target})
    endforeach()
endif()
