# CMake generated Testfile for 
# Source directory: /Users/eric/git/tinygrad.c
# Build directory: /Users/eric/git/tinygrad.c/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_tensor "/Users/eric/git/tinygrad.c/build/test_tensor")
set_tests_properties(test_tensor PROPERTIES  _BACKTRACE_TRIPLES "/Users/eric/git/tinygrad.c/CMakeLists.txt;153;add_test;/Users/eric/git/tinygrad.c/CMakeLists.txt;0;")
add_test(test_ops "/Users/eric/git/tinygrad.c/build/test_ops")
set_tests_properties(test_ops PROPERTIES  _BACKTRACE_TRIPLES "/Users/eric/git/tinygrad.c/CMakeLists.txt;154;add_test;/Users/eric/git/tinygrad.c/CMakeLists.txt;0;")
add_test(test_resnet18 "/Users/eric/git/tinygrad.c/build/test_resnet18")
set_tests_properties(test_resnet18 PROPERTIES  _BACKTRACE_TRIPLES "/Users/eric/git/tinygrad.c/CMakeLists.txt;155;add_test;/Users/eric/git/tinygrad.c/CMakeLists.txt;0;")
