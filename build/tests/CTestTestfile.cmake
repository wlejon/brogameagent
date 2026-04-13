# CMake generated Testfile for 
# Source directory: D:/projects/brogameagent/tests
# Build directory: D:/projects/brogameagent/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[brogameagent_test]=] "D:/projects/brogameagent/build/tests/Debug/brogameagent_test.exe")
  set_tests_properties([=[brogameagent_test]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/projects/brogameagent/tests/CMakeLists.txt;13;add_test;D:/projects/brogameagent/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[brogameagent_test]=] "D:/projects/brogameagent/build/tests/Release/brogameagent_test.exe")
  set_tests_properties([=[brogameagent_test]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/projects/brogameagent/tests/CMakeLists.txt;13;add_test;D:/projects/brogameagent/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[brogameagent_test]=] "D:/projects/brogameagent/build/tests/MinSizeRel/brogameagent_test.exe")
  set_tests_properties([=[brogameagent_test]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/projects/brogameagent/tests/CMakeLists.txt;13;add_test;D:/projects/brogameagent/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[brogameagent_test]=] "D:/projects/brogameagent/build/tests/RelWithDebInfo/brogameagent_test.exe")
  set_tests_properties([=[brogameagent_test]=] PROPERTIES  _BACKTRACE_TRIPLES "D:/projects/brogameagent/tests/CMakeLists.txt;13;add_test;D:/projects/brogameagent/tests/CMakeLists.txt;0;")
else()
  add_test([=[brogameagent_test]=] NOT_AVAILABLE)
endif()
