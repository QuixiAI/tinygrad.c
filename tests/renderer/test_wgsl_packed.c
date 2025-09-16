#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

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

TEST_MAIN()

