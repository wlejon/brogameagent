#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "brogameagent::brogameagent" for configuration "MinSizeRel"
set_property(TARGET brogameagent::brogameagent APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(brogameagent::brogameagent PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CUDA;CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/brogameagent.lib"
  )

list(APPEND _cmake_import_check_targets brogameagent::brogameagent )
list(APPEND _cmake_import_check_files_for_brogameagent::brogameagent "${_IMPORT_PREFIX}/lib/brogameagent.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
