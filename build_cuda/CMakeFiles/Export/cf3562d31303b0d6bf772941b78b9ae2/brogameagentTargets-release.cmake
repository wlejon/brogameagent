#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "brogameagent::brogameagent" for configuration "Release"
set_property(TARGET brogameagent::brogameagent APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(brogameagent::brogameagent PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/brogameagent.lib"
  )

list(APPEND _cmake_import_check_targets brogameagent::brogameagent )
list(APPEND _cmake_import_check_files_for_brogameagent::brogameagent "${_IMPORT_PREFIX}/lib/brogameagent.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
