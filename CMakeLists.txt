cmake_minimum_required(VERSION 3.16)
project(tinygradc C)

# ---- Language & global options ----
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
option(TG_ASAN "Enable AddressSanitizer" OFF)

if(ENABLE_WARNINGS)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
  endif()
endif()

# Make all libs PIC so we can make plugins later (backends)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optional: AddressSanitizer support (via -DTG_ASAN=ON or ASAN=1 env)
if (TG_ASAN OR DEFINED ENV{ASAN})
  message(STATUS "AddressSanitizer enabled")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  add_link_options(-fsanitize=address)
endif()

# ---- Find dependencies ----
find_package(unity CONFIG REQUIRED)
find_package(GSL CONFIG REQUIRED)
find_package(ZLIB QUIET)
find_package(CURL CONFIG QUIET)
find_package(capstone CONFIG QUIET)
# Conan-provided packages
find_package(glib CONFIG QUIET)
find_package(libuv CONFIG QUIET)
# sqlite3 is optional; prefer Conan's CONFIG package, fallback to CMake module
find_package(sqlite3 CONFIG QUIET)
if (NOT sqlite3_FOUND)
  find_package(SQLite3 QUIET)
endif()
# Ensure Conan generators dir is on CMAKE_PREFIX_PATH for Z3
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/build/conan/build/Release/generators")
find_package(Z3 CONFIG REQUIRED)

# ---- numpy compat library ----
add_library(numpy STATIC
  src/compat/numpy.c
)
target_include_directories(numpy PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/compat)
target_link_libraries(numpy PUBLIC GSL::gsl GSL::gslcblas)

# ---- tqdm library ----
add_library(tqdm STATIC
  src/compat/tqdm.c
)
target_include_directories(tqdm PUBLIC ${CMAKE_SOURCE_DIR}/src/compat)

# ---- fetch library (optional CURL/ZLIB at compile time) ----
add_library(fetch STATIC
  src/compat/fetch.c
)
target_include_directories(fetch PUBLIC ${CMAKE_SOURCE_DIR}/src/compat)
if (TARGET CURL::libcurl)
  target_link_libraries(fetch PUBLIC CURL::libcurl)
  target_compile_definitions(fetch PUBLIC TG_HAVE_CURL=1)
endif()
if (TARGET ZLIB::ZLIB)
  target_link_libraries(fetch PUBLIC ZLIB::ZLIB)
  target_compile_definitions(fetch PUBLIC TG_HAVE_ZLIB=1)
endif()

# ---- Core object libraries ----
add_library(tg_tensor OBJECT
  src/tensor/tensor.c
  src/tensor/tensor_autograd.c
)
target_include_directories(tg_tensor PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/compat)
target_link_libraries(tg_tensor PUBLIC numpy GSL::gsl GSL::gslcblas)

add_library(tg_engine OBJECT
  src/engine/graph.c
  src/engine/autograd.c
)
target_include_directories(tg_engine PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/compat)

add_library(tg_ops_cpu OBJECT
  src/runtime/ops_cpu/ops.c
  src/runtime/ops_cpu/kernels/reduce.c
  src/runtime/ops_cpu/kernels/logsoftmax_nll.c
  src/runtime/ops_cpu/kernels/batchnorm.c
  src/runtime/ops_cpu/kernels/relu.c
  src/runtime/ops_cpu/kernels/pool.c
  src/runtime/ops_cpu/kernels/conv_im2col.c
  src/runtime/ops_cpu/kernels/linear.c
  src/runtime/uop_interpreter.c
)
target_include_directories(tg_ops_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/compat)
target_link_libraries(tg_ops_cpu PUBLIC numpy GSL::gsl GSL::gslcblas)

add_library(tg_nn OBJECT
  src/nn/sgd.c
)
target_include_directories(tg_nn PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/compat)

# ---- Stubbed modules (auto-discovered) ----
file(GLOB STUB_DTYPE       CONFIGURE_DEPENDS src/dtype/*.c)
file(GLOB STUB_SHAPE       CONFIGURE_DEPENDS src/shape/*.c)
file(GLOB STUB_HELPERS     CONFIGURE_DEPENDS src/helpers/*.c)
file(GLOB STUB_DEVICE      CONFIGURE_DEPENDS src/device/*.c)
file(GLOB STUB_UOP         CONFIGURE_DEPENDS src/uop/*.c)
file(GLOB STUB_GRADIENT    CONFIGURE_DEPENDS src/gradient/*.c)
file(GLOB STUB_CODEGEN     CONFIGURE_DEPENDS src/codegen/*.c)
file(GLOB STUB_CODEGEN_OPT CONFIGURE_DEPENDS src/codegen/opt/*.c)
file(GLOB STUB_RENDERER    CONFIGURE_DEPENDS src/renderer/*.c)
file(GLOB STUB_FRONTEND    CONFIGURE_DEPENDS src/frontend/*.c)
file(GLOB STUB_SCHEDULE    CONFIGURE_DEPENDS src/schedule/*.c)
file(GLOB STUB_RTGRAPH     CONFIGURE_DEPENDS src/runtime/graph/*.c)
# extra NN stubs (exclude sgd.c which is real)
file(GLOB STUB_NN_EXTRA    CONFIGURE_DEPENDS src/nn/*.c)
list(FILTER STUB_NN_EXTRA EXCLUDE REGEX ".*/sgd\\.c$")

macro(add_stub_obj name files)
  if(${files})
    add_library(${name} OBJECT ${${files}})
    target_include_directories(${name} PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/compat)
  endif()
endmacro()

add_stub_obj(tg_dtype         STUB_DTYPE)
add_stub_obj(tg_shape         STUB_SHAPE)
add_stub_obj(tg_helpers       STUB_HELPERS)
if(TARGET tg_helpers)
  if (TARGET ZLIB::ZLIB)
    target_compile_definitions(tg_helpers PUBLIC TG_HAVE_ZLIB=1)
  endif()
  if (TARGET CURL::libcurl)
    target_compile_definitions(tg_helpers PUBLIC TG_HAVE_CURL=1)
  endif()
  if (TARGET capstone::capstone OR TARGET Capstone::capstone)
    target_compile_definitions(tg_helpers PUBLIC TG_HAVE_CAPSTONE=1)
    if (TARGET capstone::capstone)
      target_link_libraries(tg_helpers PUBLIC capstone::capstone)
    elseif (TARGET Capstone::capstone)
      target_link_libraries(tg_helpers PUBLIC Capstone::capstone)
    endif()
  endif()
  if (TARGET sqlite3::sqlite3 OR TARGET SQLite::SQLite3)
    target_compile_definitions(tg_helpers PUBLIC TG_HAVE_SQLITE3=1)
    if (TARGET sqlite3::sqlite3)
      target_link_libraries(tg_helpers PUBLIC sqlite3::sqlite3)
    elseif (TARGET SQLite::SQLite3)
      target_link_libraries(tg_helpers PUBLIC SQLite::SQLite3)
    endif()
  endif()
endif()
add_stub_obj(tg_device        STUB_DEVICE)
add_stub_obj(tg_uop           STUB_UOP)
if (TARGET tg_uop)
  # Ensure Z3 headers are available when compiling uop sources
  target_link_libraries(tg_uop PUBLIC z3::libz3)
  target_compile_definitions(tg_uop PUBLIC TG_HAVE_Z3=1 HAVE_Z3=1)
endif()
add_stub_obj(tg_gradient      STUB_GRADIENT)
if(TARGET tg_gradient)
  target_link_libraries(tg_gradient PUBLIC numpy GSL::gsl GSL::gslcblas)
endif()
add_stub_obj(tg_codegen       STUB_CODEGEN)
add_stub_obj(tg_codegen_opt   STUB_CODEGEN_OPT)
add_stub_obj(tg_renderer      STUB_RENDERER)
add_stub_obj(tg_frontend      STUB_FRONTEND)
add_stub_obj(tg_schedule      STUB_SCHEDULE)
add_stub_obj(tg_runtime_graph STUB_RTGRAPH)
add_stub_obj(tg_nn_extra      STUB_NN_EXTRA)

# ---- Final library ----
add_library(tinygradc STATIC
  $<TARGET_OBJECTS:tg_tensor>
  $<TARGET_OBJECTS:tg_engine>
  $<TARGET_OBJECTS:tg_ops_cpu>
  $<TARGET_OBJECTS:tg_nn>

  $<$<TARGET_EXISTS:tg_dtype>:$<TARGET_OBJECTS:tg_dtype>>
  $<$<TARGET_EXISTS:tg_shape>:$<TARGET_OBJECTS:tg_shape>>
  $<$<TARGET_EXISTS:tg_helpers>:$<TARGET_OBJECTS:tg_helpers>>
  $<$<TARGET_EXISTS:tg_device>:$<TARGET_OBJECTS:tg_device>>
  $<$<TARGET_EXISTS:tg_uop>:$<TARGET_OBJECTS:tg_uop>>
  $<$<TARGET_EXISTS:tg_gradient>:$<TARGET_OBJECTS:tg_gradient>>
  $<$<TARGET_EXISTS:tg_codegen>:$<TARGET_OBJECTS:tg_codegen>>
  $<$<TARGET_EXISTS:tg_codegen_opt>:$<TARGET_OBJECTS:tg_codegen_opt>>
  $<$<TARGET_EXISTS:tg_renderer>:$<TARGET_OBJECTS:tg_renderer>>
  $<$<TARGET_EXISTS:tg_frontend>:$<TARGET_OBJECTS:tg_frontend>>
  $<$<TARGET_EXISTS:tg_schedule>:$<TARGET_OBJECTS:tg_schedule>>
  $<$<TARGET_EXISTS:tg_runtime_graph>:$<TARGET_OBJECTS:tg_runtime_graph>>
  $<$<TARGET_EXISTS:tg_nn_extra>:$<TARGET_OBJECTS:tg_nn_extra>>
)
target_include_directories(tinygradc PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/compat)

# Link compat libs and GSL to tinygradc
target_link_libraries(tinygradc PUBLIC numpy tqdm fetch GSL::gsl GSL::gslcblas)
if (TARGET ZLIB::ZLIB)
  target_link_libraries(tinygradc PUBLIC ZLIB::ZLIB)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_ZLIB=1)
endif()
if (TARGET CURL::libcurl)
  target_link_libraries(tinygradc PUBLIC CURL::libcurl)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_CURL=1)
endif()
if (TARGET capstone::capstone)
  target_link_libraries(tinygradc PUBLIC capstone::capstone)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_CAPSTONE=1)
elseif (TARGET Capstone::capstone)
  target_link_libraries(tinygradc PUBLIC Capstone::capstone)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_CAPSTONE=1)
endif()
if (TARGET sqlite3::sqlite3)
  target_link_libraries(tinygradc PUBLIC sqlite3::sqlite3)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_SQLITE3=1)
elseif (TARGET SQLite::SQLite3)
  target_link_libraries(tinygradc PUBLIC SQLite::SQLite3)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_SQLITE3=1)
endif()
target_link_libraries(tinygradc PUBLIC z3::libz3 stdc++)
target_compile_definitions(tinygradc PUBLIC TG_HAVE_Z3=1 HAVE_Z3=1)

# Link glib components if available (gio, gio-unix, glib, gobject)
if (TARGET glib::gio)
  target_link_libraries(tinygradc PUBLIC glib::gio)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_GIO=1)
endif()
if (TARGET glib::gio-unix)
  target_link_libraries(tinygradc PUBLIC glib::gio-unix)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_GIO_UNIX=1)
endif()
if (TARGET glib::glib)
  target_link_libraries(tinygradc PUBLIC glib::glib)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_GLIB=1)
endif()
if (TARGET glib::gobject)
  target_link_libraries(tinygradc PUBLIC glib::gobject)
endif()

# Link libuv if available
if (TARGET uv)
  target_link_libraries(tinygradc PUBLIC uv)
  target_compile_definitions(tinygradc PUBLIC TG_HAVE_LIBUV=1)
endif()

# Link libm on Linux
if(UNIX AND NOT APPLE)
  target_link_libraries(tinygradc PUBLIC m)
endif()

# ---- Examples ----
if(BUILD_EXAMPLES)
  add_executable(resnet18_cpu examples/resnet18_cpu.c)
  target_link_libraries(resnet18_cpu PRIVATE tinygradc)
endif()

# ---- Tests (Unity via Conan) ----
if(BUILD_TESTS)
  # Optional ASAN for tests (off by default)
  option(TG_ENABLE_ASAN "Enable AddressSanitizer for tests" OFF)
  include(CTest)
  enable_testing()
  add_library(test_common INTERFACE)
  target_include_directories(test_common INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/compat
    ${CMAKE_SOURCE_DIR}/tests
  )
  # Ensure Unity supports float/double in tests
  target_compile_definitions(test_common INTERFACE UNITY_INCLUDE_DOUBLE UNITY_INCLUDE_FLOAT)
  target_link_libraries(test_common INTERFACE tinygradc unity::unity)

  # Enable AddressSanitizer for tests when requested
  if (TG_ENABLE_ASAN)
    if (CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(test_common INTERFACE -fsanitize=address -fno-omit-frame-pointer -g)
      target_link_options(test_common INTERFACE -fsanitize=address)
    endif()
  endif()
  # Propagate optional feature macros to tests
  if (TARGET ZLIB::ZLIB)
    target_compile_definitions(test_common INTERFACE TG_HAVE_ZLIB=1)
  endif()
  if (TARGET CURL::libcurl)
    target_compile_definitions(test_common INTERFACE TG_HAVE_CURL=1)
  endif()
  if (TARGET capstone::capstone OR TARGET Capstone::capstone)
    target_compile_definitions(test_common INTERFACE TG_HAVE_CAPSTONE=1)
  endif()
  if (TARGET sqlite3::sqlite3 OR TARGET SQLite::SQLite3)
    target_compile_definitions(test_common INTERFACE TG_HAVE_SQLITE3=1)
  endif()

  # Discover and register tests
  file(GLOB_RECURSE TEST_SOURCES tests/*.c)
  foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE test_common)
    add_test(NAME ${test_name} COMMAND ${test_name})
    # Derive a suite label from the filename: test_<label>[_...].c -> <label>
    string(REGEX REPLACE "^test_([^_]+).*$" "\\1" test_label ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS ${test_label})
  endforeach()

  # Tools for CI checks (not registered as tests)
  # Removed: tool_default_float_invalid helper (file not present). CI check handled elsewhere.
endif()
