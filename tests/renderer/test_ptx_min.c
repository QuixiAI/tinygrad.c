#include "test_common.h"
#include "renderer/ptx.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_ptx_min_signature) {
  Renderer* r = renderer_ptx("sm_80", "NV");
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* arr[] = {buf};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, ".entry kernel_main"));
  TEST_ASSERT_NOT_NULL(strstr(src, ".param .u64 data0"));
  free(src); uop_unref(buf); free(r);
}

TEST_MAIN()
