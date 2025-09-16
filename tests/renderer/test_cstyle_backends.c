#include "test_common.h"
#include "renderer/cstyle.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_renderer_opencl_signature_and_params) {
  Renderer* r = renderer_cstyle_opencl();
  TEST_ASSERT_NOT_NULL(r);
  UOp* buf = uop_define_global(dtypes.uint8, 0);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* bidx = uop_index(buf, idx);
  UOp* load = uop_load(bidx, dtypes.uint8);
  UOp* arr[] = {buf, idx, bidx, load};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "__kernel void kernel_main("));
  TEST_ASSERT_NOT_NULL(strstr(src, "__global uchar* data0"));
  free(src); uop_unref(load); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_renderer_opencl_bitcast_as_type) {
  Renderer* r = renderer_cstyle_opencl();
  UOp* x = uop_const(dtypes.uint8, 7);
  UOp* bc = uop_bitcast(x, dtypes.uint8);
  UOp* arr[] = {x, bc};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "as_uchar("));
  free(src); uop_unref(bc); uop_unref(x); free(r);
}

TEST(test_renderer_opencl_vector_literal_style) {
  Renderer* r = renderer_cstyle_opencl();
  UOp* one = uop_const(dtypes.float32, 1.0);
  UOp* vec = uop_broadcast(one, 4);
  UOp* arr[] = {one, vec};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "(float4)("));
  free(src); uop_unref(vec); uop_unref(one); free(r);
}

TEST(test_renderer_metal_signature_and_params) {
  Renderer* r = renderer_cstyle_metal();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 1);
  UOp* bidx = uop_index(buf, idx);
  UOp* load = uop_load(bidx, dtypes.float32);
  UOp* arr[] = {buf, idx, bidx, load};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(strstr(src, "kernel void kernel_main("));
  TEST_ASSERT_NOT_NULL(strstr(src, "device float* data0"));
  free(src); uop_unref(load); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_renderer_metal_bitcast_as_type_template) {
  Renderer* r = renderer_cstyle_metal();
  UOp* x = uop_const(dtypes.int32, 1);
  UOp* bc = uop_bitcast(x, dtypes.int32);
  UOp* arr[] = {x, bc};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "as_type<int>("));
  free(src); uop_unref(bc); uop_unref(x); free(r);
}

TEST(test_renderer_metal_vector_literal_style) {
  Renderer* r = renderer_cstyle_metal();
  UOp* one = uop_const(dtypes.float32, 1.0);
  UOp* vec = uop_broadcast(one, 4);
  UOp* arr[] = {one, vec};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "float4("));
  free(src); uop_unref(vec); uop_unref(one); free(r);
}

TEST(test_renderer_cuda_signature_contains_global) {
  Renderer* r = renderer_cstyle_cuda("sm_80");
  UOp* x = uop_const(dtypes.int32, 0);
  char* src = r->render(r, &x, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "__global__ void kernel_main("));
  free(src); uop_unref(x); free(r);
}

TEST(test_renderer_qcom_is_opencl_style) {
  Renderer* r = renderer_cstyle_qcom();
  UOp* x = uop_const(dtypes.int32, 0);
  char* src = r->render(r, &x, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "__kernel void kernel_main("));
  free(src); uop_unref(x); free(r);
}

TEST(test_renderer_opencl_image_load_store) {
  Renderer* r = renderer_cstyle_opencl();
  // image param as DEFINE_GLOBAL with dtype "imagef"
  DType imgdt = to_dtype("imagef");
  UOp* img = uop_define_global(imgdt, 0);
  // int2 index: represent as vectorize two ints
  UOp* i0 = uop_const(dtypes.int32, 3);
  UOp* i1 = uop_const(dtypes.int32, 4);
  UOpArg va={0}; UOp* vsrc[2]={i0,i1}; va.type=ARG_NONE; UOp* idx2 = uop_new(OPS_VECTORIZE, dtype_vec(&dtypes.int32, 2), vsrc, 2, &va, NULL);
  UOp* bidx = uop_index(img, idx2);
  UOp* val4 = uop_broadcast(uop_const(dtypes.float32, 1.0), 4);
  UOp* ld = uop_load(bidx, dtype_vec(&dtypes.float32,4));
  UOp* st = uop_store(bidx, val4);
  UOp* arr[] = {img, i0, i1, idx2, bidx, val4, ld, st};
  char* src = r->render(r, arr, 8);
  TEST_ASSERT_NOT_NULL(strstr(src, "__kernel void kernel_main"));
  TEST_ASSERT_NOT_NULL(strstr(src, "image2d_t data0"));
  TEST_ASSERT_NOT_NULL(strstr(src, "const sampler_t smp"));
  TEST_ASSERT_NOT_NULL(strstr(src, "read_imagef(data0"));
  TEST_ASSERT_NOT_NULL(strstr(src, "write_imagef(data0"));
  free(src); uop_unref(st); uop_unref(ld); uop_unref(val4); uop_unref(bidx); uop_unref(idx2); uop_unref(i1); uop_unref(i0); uop_unref(img); free(r);
}

TEST(test_renderer_amd_ocml_unary) {
  Renderer* r = renderer_cstyle_amd("gfx");
  UOp* x = uop_const(dtypes.float32, 4.0);
  UOp* s = uop_sqrt(x);
  UOp* sn = uop_sin(x);
  UOp* arr[] = {x,s,sn};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(strstr(src, "__ocml_sqrt_f32"));
  TEST_ASSERT_NOT_NULL(strstr(src, "__ocml_sin_f32"));
  free(src); uop_unref(sn); uop_unref(s); uop_unref(x); free(r);
}

TEST(test_renderer_local_and_barrier) {
  Renderer* r = renderer_cstyle_opencl();
  UOp* lbuf = uop_define_local(dtypes.float32, 8);
  UOp* bar = uop_barrier(lbuf);
  UOp* arr[] = {lbuf, bar};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "__local float"));
  TEST_ASSERT_NOT_NULL(strstr(src, "barrier("));
  free(src); uop_unref(bar); uop_unref(lbuf); free(r);
}

TEST_MAIN()
