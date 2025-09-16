#include "test_common.h"
#include "renderer/ptx.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_ptx_ld_st_index) {
  Renderer* r = renderer_ptx("sm_80", "NV");
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 4);
  UOp* bidx = uop_index(buf, idx);
  UOp* v = uop_load(bidx, dtypes.float32);
  UOp* st = uop_store(bidx, v);
  UOp* arr[] = {buf, idx, bidx, v, st};
  char* src = r->render(r, arr, 5);
  TEST_ASSERT_NOT_NULL(strstr(src, "ld.param.u64"));
  TEST_ASSERT_NOT_NULL(strstr(src, "mul.wide.s32"));
  TEST_ASSERT_NOT_NULL(strstr(src, "add.s64"));
  TEST_ASSERT_NOT_NULL(strstr(src, "ld.global.f32"));
  TEST_ASSERT_NOT_NULL(strstr(src, "st.global.f32"));
  free(src); uop_unref(st); uop_unref(v); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST_MAIN()

