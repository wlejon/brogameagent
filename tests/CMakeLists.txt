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
        adam
        reduce
        loss
        embedding
        concat
        mha
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
        target_include_directories(${_target} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        if(MSVC)
            target_compile_options(${_target} PRIVATE /W4)
        else()
            target_compile_options(${_target} PRIVATE -Wall -Wextra)
        endif()
        add_test(NAME ${_target} COMMAND ${_target})
    endforeach()
endif()
