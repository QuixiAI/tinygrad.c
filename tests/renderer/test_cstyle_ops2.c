#include "test_common.h"
#include "renderer/cstyle.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_renderer_cstyle_params_have_restrict) {
  Renderer* r = renderer_cstyle_clang();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* bidx = uop_index(buf, idx);
  UOp* v = uop_load(bidx, dtypes.float32);
  UOp* arr[] = {buf, idx, bidx, v};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(strstr(src, "restrict"));
  free(src); uop_unref(v); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_renderer_cstyle_bitops_and_mod_and_cmps) {
  Renderer* r = renderer_cstyle_clang();
  UOp* a = uop_const(dtypes.int32, 13);
  UOp* b = uop_const(dtypes.int32, 2);
  UOp* shl = uop_shl(a,b);
  UOp* shr = uop_shr(a,b);
  UOp* mod = uop_mod(a,b);
  UOp* eq  = uop_eq(a,b);
  UOp* ne  = uop_cmpne(a,b);
  UOp* arr[] = {a,b,shl,shr,mod,eq,ne};
  char* src = r->render(r, arr, 7);
  TEST_ASSERT_NOT_NULL(strstr(src, "<<"));
  TEST_ASSERT_NOT_NULL(strstr(src, ">>"));
  TEST_ASSERT_NOT_NULL(strstr(src, "%"));
  TEST_ASSERT_NOT_NULL(strstr(src, "=="));
  TEST_ASSERT_NOT_NULL(strstr(src, "!="));
  free(src); uop_unref(ne); uop_unref(eq); uop_unref(mod); uop_unref(shr); uop_unref(shl); uop_unref(b); uop_unref(a); free(r);
}

TEST(test_renderer_cstyle_nan_inf_constants) {
  Renderer* r = renderer_cstyle_clang();
  UOp* pinf = uop_const(dtypes.float32, INFINITY);
  UOp* ninf = uop_const(dtypes.float32, -INFINITY);
  UOp* nnan = uop_const(dtypes.float32, NAN);
  UOp* arr[] = {pinf, ninf, nnan};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(strstr(src, "__builtin_inff"));
  TEST_ASSERT_NOT_NULL(strstr(src, "-__builtin_inff"));
  TEST_ASSERT_NOT_NULL(strstr(src, "__builtin_nanf"));
  free(src); uop_unref(nnan); uop_unref(ninf); uop_unref(pinf); free(r);
}

TEST_MAIN()

