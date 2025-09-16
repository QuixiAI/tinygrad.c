#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_wgsl_workgroup_size_from_special) {
  Renderer* r = renderer_wgsl();
  // two local dims: x=4, y=8
  UOp* l0 = uop_special_ex("lidx0", 0, 4, dtypes.int32);
  UOp* l1 = uop_special_ex("lidx1", 1, 8, dtypes.int32);
  UOp* arr[] = {l0, l1};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "@workgroup_size(4,8,1)"));
  free(src); uop_unref(l1); uop_unref(l0); free(r);
}

TEST_MAIN()

