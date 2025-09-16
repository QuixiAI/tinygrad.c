#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

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

TEST_MAIN()

