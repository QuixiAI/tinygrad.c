#include "test_common.h"
#include "renderer/ptx.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_ptx_if_ctrlflow) {
  Renderer* r = renderer_ptx("sm_80", "NV");
  UOp* a = uop_const(dtypes.int32, 1);
  UOp* b = uop_const(dtypes.int32, 2);
  UOp* lt = uop_lt(a, b);
  UOp* If = uop_new(OPS_IF, dtypes.void_, &lt, 1, NULL, NULL);
  UOp* EndIf = uop_new(OPS_ENDIF, dtypes.void_, NULL, 0, NULL, NULL);
  UOp* arr[] = {a,b,lt,If,EndIf};
  char* src = r->render(r, arr, 5);
  TEST_ASSERT_NOT_NULL(strstr(src, "setp.lt.s32"));
  TEST_ASSERT_NOT_NULL(strstr(src, "@!%p"));
  TEST_ASSERT_NOT_NULL(strstr(src, "bra IF_END_"));
  free(src); uop_unref(EndIf); uop_unref(If); uop_unref(lt); uop_unref(b); uop_unref(a); free(r);
}

TEST(test_ptx_range_loop) {
  Renderer* r = renderer_ptx("sm_80", "NV");
  UOp* N = uop_const(dtypes.int32, 3);
  UOp* R = uop_range(N, 0);
  UOp* ER = uop_new(OPS_ENDRANGE, dtypes.void_, NULL, 0, NULL, NULL);
  UOp* arr[] = {N,R,ER};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(strstr(src, "setp.lt.s32"));
  TEST_ASSERT_NOT_NULL(strstr(src, "bra LOOP_"));
  TEST_ASSERT_NOT_NULL(strstr(src, "add.s32"));
  free(src); uop_unref(ER); uop_unref(R); uop_unref(N); free(r);
}

TEST(test_ptx_min_signature) {
  Renderer* r = renderer_ptx("sm_80", "NV");
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* arr[] = {buf};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, ".entry kernel_main"));
  TEST_ASSERT_NOT_NULL(strstr(src, ".param .u64 data0"));
  free(src); uop_unref(buf); free(r);
}

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

