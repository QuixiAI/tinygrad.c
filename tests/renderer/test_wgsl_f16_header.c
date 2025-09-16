#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_wgsl_emits_enable_f16) {
  Renderer* r = renderer_wgsl();
  UOp* c = uop_const(dtypes.half, 1.0);
  UOp* arr[] = {c};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "enable f16;"));
  free(src); uop_unref(c); free(r);
}

TEST_MAIN()

