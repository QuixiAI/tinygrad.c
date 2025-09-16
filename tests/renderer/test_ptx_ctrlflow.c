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

TEST_MAIN()

