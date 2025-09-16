#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_wgsl_min_signature) {
  Renderer* r = renderer_wgsl();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* arr[] = {buf};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size"));
  TEST_ASSERT_NOT_NULL(strstr(src, "var<storage, read_write> data0"));
  TEST_ASSERT_NOT_NULL(strstr(src, "@builtin(workgroup_id)"));
  TEST_ASSERT_NOT_NULL(strstr(src, "@builtin(local_invocation_id)"));
  free(src); uop_unref(buf); free(r);
}

TEST_MAIN()
