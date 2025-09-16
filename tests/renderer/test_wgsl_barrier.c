#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_wgsl_barrier) {
  Renderer* r = renderer_wgsl();
  UOp* b = uop_barrier(NULL);
  UOp* arr[] = {b};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "workgroupBarrier"));
  free(src); uop_unref(b); free(r);
}

TEST_MAIN()

