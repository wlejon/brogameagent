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
