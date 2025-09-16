#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_wgsl_barrier) {
  Renderer* r = renderer_wgsl();
  UOp* b = uop_barrier(NULL);
  UOp* arr[] = {b};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "workgroupBarrier"));
  free(src); uop_unref(b); free(r);
}

TEST(test_wgsl_emits_enable_f16) {
  Renderer* r = renderer_wgsl();
  UOp* c = uop_const(dtypes.half, 1.0);
  UOp* arr[] = {c};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "enable f16;"));
  free(src); uop_unref(c); free(r);
}

TEST(test_wgsl_load_gated_select) {
  Renderer* r = renderer_wgsl();
  UOp* buf = uop_define_global(dtypes.int32, 0);
  UOp* idx = uop_const(dtypes.int32, 3);
  UOp* gate = uop_const(dtypes.bool_, 0);
  UOp* idx_srcs[3] = { buf, idx, gate };
  UOp* bidx = uop_new(OPS_INDEX, buf->dtype, idx_srcs, 3, &(UOpArg){0}, NULL);
  UOp* ld = uop_load(bidx, dtypes.int32);
  UOp* arr[] = {buf, idx, gate, bidx, ld};
  char* src = r->render(r, arr, 5);
  TEST_ASSERT_NOT_NULL(strstr(src, "select("));
  free(src); uop_unref(ld); uop_unref(bidx); uop_unref(gate); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_wgsl_store_gated_if) {
  Renderer* r = renderer_wgsl();
  UOp* buf = uop_define_global(dtypes.int32, 0);
  UOp* idx = uop_const(dtypes.int32, 4);
  UOp* gate = uop_const(dtypes.bool_, 1);
  UOp* idx_srcs[3] = { buf, idx, gate };
  UOp* bidx = uop_new(OPS_INDEX, buf->dtype, idx_srcs, 3, &(UOpArg){0}, NULL);
  UOp* val = uop_const(dtypes.int32, 42);
  UOp* st = uop_store(bidx, val);
  UOp* arr[] = {buf, idx, gate, bidx, val, st};
  char* src = r->render(r, arr, 6);
  TEST_ASSERT_NOT_NULL(strstr(src, "if ("));
  free(src); uop_unref(st); uop_unref(val); uop_unref(bidx); uop_unref(gate); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_wgsl_define_local_decl) {
  Renderer* r = renderer_wgsl();
  UOp* loc = uop_define_local(dtypes.char_, 16); // packed type
  UOp* arr[] = {loc};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "var<workgroup>"));
  TEST_ASSERT_NOT_NULL(strstr(src, "atomic<u32>"));
  free(src); uop_unref(loc); free(r);
}

TEST(test_wgsl_define_reg_decl) {
  Renderer* r = renderer_wgsl();
  UOp* reg = uop_define_reg(dtypes.int32);
  UOp* arr[] = {reg};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "var "));
  free(src); uop_unref(reg); free(r);
}

TEST(test_wgsl_const_uint32_negative_bitcast) {
  Renderer* r = renderer_wgsl();
  UOp* c = uop_const(dtypes.uint32, -1.0);
  UOp* arr[] = {c};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "bitcast<u32>(-1)"));
  free(src); uop_unref(c); free(r);
}

TEST(test_wgsl_min_signature) {
  Renderer* r = renderer_wgsl();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* arr[] = {buf};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size"));
  TEST_ASSERT_NOT_NULL(strstr(src, "var<storage, read_write> data0"));
  TEST_ASSERT_NOT_NULL(strstr(src, "@builtin(workgroup_id)"));
  TEST_ASSERT_NOT_NULL(strstr(src, "@builtin(local_invocation_id)"));
  free(src); uop_unref(buf); free(r);
}

TEST(test_wgsl_nan_check_cmpne_same) {
  Renderer* r = renderer_wgsl();
  UOp* x = uop_const(dtypes.float32, 2.0);
  UOp* ne = uop_ne(x, x);
  UOp* arr[] = {x, ne};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "min("));
  TEST_ASSERT_NOT_NULL(strstr(src, "max("));
  free(src); uop_unref(ne); uop_unref(x); free(r);
}

TEST(test_wgsl_packed_store_i8_atomic) {
  Renderer* r = renderer_wgsl();
  UOp* buf = uop_define_global(dtypes.char_, 0); // i8 buffer id=0
  UOp* idx = uop_const(dtypes.int32, 5);
  UOp* bidx = uop_index(buf, idx);
  UOp* val = uop_const(dtypes.char_, 7);
  UOp* st = uop_store(bidx, val);
  UOp* arr[] = {buf, idx, bidx, val, st};
  char* src = r->render(r, arr, 5);
  TEST_ASSERT_NOT_NULL(strstr(src, "atomicAnd("));
  TEST_ASSERT_NOT_NULL(strstr(src, "atomicAdd("));
  free(src); uop_unref(st); uop_unref(val); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_wgsl_packed_load_i8_atomic) {
  Renderer* r = renderer_wgsl();
  UOp* buf = uop_define_global(dtypes.char_, 0);
  UOp* idx = uop_const(dtypes.int32, 2);
  UOp* bidx = uop_index(buf, idx);
  UOp* ld = uop_load(bidx, dtypes.char_);
  UOp* arr[] = {buf, idx, bidx, ld};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(strstr(src, "atomicLoad("));
  free(src); uop_unref(ld); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_wgsl_bitcast_masks_u8_u16) {
  Renderer* r = renderer_wgsl();
  UOp* u32c = uop_const(dtypes.uint32, 255);
  UOp* bc8 = uop_bitcast(u32c, dtypes.char_);
  UOp* bc16 = uop_bitcast(u32c, dtypes.short_);
  UOp* arr[] = {u32c, bc8, bc16};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(strstr(src, "& 0xFF"));
  TEST_ASSERT_NOT_NULL(strstr(src, "& 0xFFFF"));
  free(src); uop_unref(bc16); uop_unref(bc8); uop_unref(u32c); free(r);
}

TEST(test_wgsl_where_select) {
  Renderer* r = renderer_wgsl();
  UOp* c = uop_const(dtypes.bool_, 1);
  UOp* t = uop_const(dtypes.int32, 5);
  UOp* f = uop_const(dtypes.int32, 6);
  UOp* w = uop_where(c, t, f);
  UOp* arr[] = {c,t,f,w};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(strstr(src, "select("));
  free(src); uop_unref(w); uop_unref(f); uop_unref(t); uop_unref(c); free(r);
}

TEST(test_wgsl_workgroup_size_from_special) {
  Renderer* r = renderer_wgsl();
  // two local dims: x=4, y=8
  UOp* l0 = uop_special_ex("lidx0", 0, 4, dtypes.int32);
  UOp* l1 = uop_special_ex("lidx1", 1, 8, dtypes.int32);
  UOp* arr[] = {l0, l1};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "@workgroup_size(4,8,1)"));
  free(src); uop_unref(l1); uop_unref(l0); free(r);
}

TEST_MAIN()

