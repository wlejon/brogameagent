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

# Position-independent code so a static libbrogameagent can be linked
# into the Python extension .so/.pyd on all platforms.
set_property(TARGET brogameagent PROPERTY POSITION_INDEPENDENT_CODE ON)

# --- Python bindings ---
option(BROGAMEAGENT_PYTHON "Build Python bindings (requires pybind11)" OFF)
if(BROGAMEAGENT_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(brogameagent_py python/bindings.cpp)
    target_link_libraries(brogameagent_py PRIVATE brogameagent)
    target_compile_features(brogameagent_py PRIVATE cxx_std_20)
    # Output to python/ with module name brogameagent so `import brogameagent`
    # works from that directory. Placed in its own subfolder so the .pyd's
    # import lib doesn't collide with the static library's .lib on MSVC.
    set_target_properties(brogameagent_py PROPERTIES
        OUTPUT_NAME brogameagent
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}/python"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}/python"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python_archive"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/python_archive"
    )
endif()

# Tests
option(BROGAMEAGENT_TESTS "Build brogameagent tests" ON)
if(BROGAMEAGENT_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()
