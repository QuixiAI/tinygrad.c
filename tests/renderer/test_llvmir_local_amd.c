#include "test_common.h"
#include "renderer/llvmir.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_llvmir_amd_define_local_addrspace) {
  Renderer* r = renderer_llvm_amd("gfx942");
  UOp* loc = uop_define_local(dtypes.float32, 4);
  UOp* arr[] = {loc};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "addrspace(3) global"));
  TEST_ASSERT_NOT_NULL(strstr(src, "addrspacecast"));
  free(src); uop_unref(loc); free(r);
}

TEST_MAIN()

