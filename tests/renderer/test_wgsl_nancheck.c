#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

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

TEST_MAIN()

