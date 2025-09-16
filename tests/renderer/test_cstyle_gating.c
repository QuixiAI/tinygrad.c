#include "test_common.h"
#include "renderer/cstyle.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_renderer_cstyle_gated_load_ternary) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  // buffer and index with gate
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 5);
  UOp* gate = uop_const(dtypes.bool_, 1);
  UOp* bidx = uop_new(OPS_INDEX, dtypes.float32, (UOp*[]){buf, idx, gate}, 3, NULL, NULL);
  // alt value for gated load
  UOp* alt = uop_const(dtypes.float32, 7.0);
  UOp* load = uop_new(OPS_LOAD, dtypes.float32, (UOp*[]){bidx, alt}, 2, NULL, NULL);
  UOp* arr[] = {buf, idx, gate, bidx, alt, load};
  char* src = r->render(r, arr, (int)(sizeof(arr)/sizeof(arr[0])));
  TEST_ASSERT_NOT_NULL(src);
  // expect ternary form in output
  TEST_ASSERT_NOT_NULL(strstr(src, "?*"));
  TEST_ASSERT_NOT_NULL(strstr(src, ":"));
  free(src);
  // cleanup
  uop_unref(load); uop_unref(alt); uop_unref(bidx); uop_unref(gate); uop_unref(idx); uop_unref(buf);
  free(r);
}

TEST(test_renderer_cstyle_gated_store_if) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  // buffer and index with gate
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 2);
  UOp* gate = uop_const(dtypes.bool_, 1);
  UOp* bidx = uop_new(OPS_INDEX, dtypes.float32, (UOp*[]){buf, idx, gate}, 3, NULL, NULL);
  UOp* val = uop_const(dtypes.float32, 3.0);
  // store with explicit gate as third src
  UOp* store = uop_new(OPS_STORE, dtypes.void_, (UOp*[]){bidx, val, gate}, 3, NULL, NULL);
  UOp* arr[] = {buf, idx, gate, bidx, val, store};
  char* src = r->render(r, arr, (int)(sizeof(arr)/sizeof(arr[0])));
  TEST_ASSERT_NOT_NULL(src);
  // expect if-guarded store
  TEST_ASSERT_NOT_NULL(strstr(src, "if ("));
  TEST_ASSERT_NOT_NULL(strstr(src, "*"));
  free(src);
  // cleanup
  uop_unref(store); uop_unref(val); uop_unref(bidx); uop_unref(gate); uop_unref(idx); uop_unref(buf);
  free(r);
}

TEST(test_renderer_cstyle_param_dtype_from_define_global) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  // define a uint8 buffer at id 0 and load from it
  UOp* buf = uop_define_global(dtypes.uint8, 0);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* bidx = uop_index(buf, idx);
  UOp* load = uop_load(bidx, dtypes.uint8);
  UOp* arr[] = {buf, idx, bidx, load};
  char* src = r->render(r, arr, (int)(sizeof(arr)/sizeof(arr[0])));
  TEST_ASSERT_NOT_NULL(src);
  // header should have unsigned char* data0
  TEST_ASSERT_NOT_NULL(strstr(src, "unsigned char* data0"));
  free(src);
  uop_unref(load); uop_unref(bidx); uop_unref(idx); uop_unref(buf);
  free(r);
}

TEST_MAIN()

