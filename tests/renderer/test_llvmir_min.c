#include "test_common.h"
#include "renderer/llvmir.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_llvmir_min_signature) {
  Renderer* r = renderer_llvm_generic();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 4);
  UOp* bidx = uop_index(buf, idx);
  UOp* v = uop_load(bidx, dtypes.float32);
  UOp* arr[] = {buf, idx, bidx, v};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "define void @kernel_main("));
  // parameter list contains data0 pointer
  TEST_ASSERT_NOT_NULL(strstr(src, "data0"));
  TEST_ASSERT_NOT_NULL(strstr(src, "ret void"));
  free(src); uop_unref(v); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST_MAIN()
