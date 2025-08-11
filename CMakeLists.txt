cmake_minimum_required(VERSION 3.16)
project(tinygradc C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)

if(ENABLE_WARNINGS)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU|AppleClang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
  endif()
endif()

# Make all libs PIC so we can make plugins later (backends)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python for manifest generation
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ---------------------------
# Core object libraries
# ---------------------------
add_library(tg_tensor OBJECT
  src/tensor/tensor.c
)
target_include_directories(tg_tensor PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

add_library(tg_engine OBJECT
  src/engine/graph.c
  src/engine/autograd.c
)
target_include_directories(tg_engine PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

add_library(tg_ops_cpu OBJECT
  src/runtime/ops_cpu/ops.c
  src/runtime/ops_cpu/kernels/reduce.c
  src/runtime/ops_cpu/kernels/logsoftmax_nll.c
  src/runtime/ops_cpu/kernels/batchnorm.c
  src/runtime/ops_cpu/kernels/relu.c
  src/runtime/ops_cpu/kernels/pool.c
  src/runtime/ops_cpu/kernels/conv_im2col.c
  src/runtime/ops_cpu/kernels/linear.c
)
target_include_directories(tg_ops_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

add_library(tg_nn OBJECT
  src/nn/sgd.c
)
target_include_directories(tg_nn PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# ---------------------------
# Stubbed modules (auto-discovered)
# These compile harmlessly; you’ll replace them piecemeal while porting.
# ---------------------------
file(GLOB STUB_DTYPE    CONFIGURE_DEPENDS src/dtype/*.c)
file(GLOB STUB_SHAPE    CONFIGURE_DEPENDS src/shape/*.c)
file(GLOB STUB_HELPERS  CONFIGURE_DEPENDS src/helpers/*.c)
file(GLOB STUB_DEVICE   CONFIGURE_DEPENDS src/device/*.c)
file(GLOB STUB_UOP      CONFIGURE_DEPENDS src/uop/*.c)
file(GLOB STUB_CODEGEN  CONFIGURE_DEPENDS src/codegen/*.c)
file(GLOB STUB_CODEGEN_OPT CONFIGURE_DEPENDS src/codegen/opt/*.c)
file(GLOB STUB_RENDERER CONFIGURE_DEPENDS src/renderer/*.c)
file(GLOB STUB_FRONTEND CONFIGURE_DEPENDS src/frontend/*.c)
file(GLOB STUB_SCHEDULE CONFIGURE_DEPENDS src/schedule/*.c)
file(GLOB STUB_RTGRAPH  CONFIGURE_DEPENDS src/runtime/graph/*.c)
# extra NN stubs (exclude sgd.c which is real)
file(GLOB STUB_NN_EXTRA CONFIGURE_DEPENDS src/nn/*.c)
list(FILTER STUB_NN_EXTRA EXCLUDE REGEX ".*/sgd\\.c$")

macro(add_stub_obj name files)
  if(${files})
    add_library(${name} OBJECT ${${files}})
    target_include_directories(${name} PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  endif()
endmacro()

add_stub_obj(tg_dtype        STUB_DTYPE)
add_stub_obj(tg_shape        STUB_SHAPE)
add_stub_obj(tg_helpers      STUB_HELPERS)
add_stub_obj(tg_device       STUB_DEVICE)
add_stub_obj(tg_uop          STUB_UOP)
add_stub_obj(tg_codegen      STUB_CODEGEN)
add_stub_obj(tg_codegen_opt  STUB_CODEGEN_OPT)
add_stub_obj(tg_renderer     STUB_RENDERER)
add_stub_obj(tg_frontend     STUB_FRONTEND)
add_stub_obj(tg_schedule     STUB_SCHEDULE)
add_stub_obj(tg_runtime_graph STUB_RTGRAPH)
add_stub_obj(tg_nn_extra     STUB_NN_EXTRA)

# ---------------------------
# Final library
# ---------------------------
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
  $<$<TARGET_EXISTS:tg_codegen>:$<TARGET_OBJECTS:tg_codegen>>
  $<$<TARGET_EXISTS:tg_codegen_opt>:$<TARGET_OBJECTS:tg_codegen_opt>>
  $<$<TARGET_EXISTS:tg_renderer>:$<TARGET_OBJECTS:tg_renderer>>
  $<$<TARGET_EXISTS:tg_frontend>:$<TARGET_OBJECTS:tg_frontend>>
  $<$<TARGET_EXISTS:tg_schedule>:$<TARGET_OBJECTS:tg_schedule>>
  $<$<TARGET_EXISTS:tg_runtime_graph>:$<TARGET_OBJECTS:tg_runtime_graph>>
  $<$<TARGET_EXISTS:tg_nn_extra>:$<TARGET_OBJECTS:tg_nn_extra>>
)
target_include_directories(tinygradc PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Link libm on Linux
if(UNIX AND NOT APPLE)
  target_link_libraries(tinygradc PUBLIC m)
endif()

# ---------------------------
# Manifest generation (for client generators)
# ---------------------------
add_custom_target(tinygradc_manifest
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/generators/manifest/gen_manifest.py
          ${CMAKE_SOURCE_DIR}/include/tg.h
          ${CMAKE_SOURCE_DIR}/generators/manifest/tg_manifest.json
  DEPENDS ${CMAKE_SOURCE_DIR}/generators/manifest/gen_manifest.py ${CMAKE_SOURCE_DIR}/include/tg.h
  COMMENT "Generating tg_manifest.json"
)

# ---------------------------
# Examples
# ---------------------------
if(BUILD_EXAMPLES)
  add_executable(resnet18_cpu examples/resnet18_cpu.c)
  target_link_libraries(resnet18_cpu PRIVATE tinygradc)
endif()

# ---------------------------
# Tests
# ---------------------------
if(BUILD_TESTS)
  add_executable(test_tensor   tests/test_tensor.c)
  add_executable(test_ops      tests/test_ops.c)
  add_executable(test_resnet18 tests/test_resnet18.c)
  target_link_libraries(test_tensor   PRIVATE tinygradc)
  target_link_libraries(test_ops      PRIVATE tinygradc)
  target_link_libraries(test_resnet18 PRIVATE tinygradc)

  enable_testing()
  add_test(NAME test_tensor   COMMAND test_tensor)
  add_test(NAME test_ops      COMMAND test_ops)
  add_test(NAME test_resnet18 COMMAND test_resnet18)
endif()
