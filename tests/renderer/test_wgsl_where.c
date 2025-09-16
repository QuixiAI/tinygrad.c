#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

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

TEST_MAIN()

