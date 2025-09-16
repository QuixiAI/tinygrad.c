#include "test_common.h"
#include "renderer/llvmir.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_llvmir_wmma_mfma_suffixes) {
  Renderer* r = renderer_llvm_amd("gfx942");
  // Construct a minimal WMMA: a: <16 x half>, b: <16 x half>, acc: <8 x float>
  DType a_dt = dtype_vec(&dtypes.float16, 16);
  DType b_dt = dtype_vec(&dtypes.float16, 16);
  DType c_dt = dtype_vec(&dtypes.float32, 8);
  UOp* a = uop_vconst(a_dt, NULL, 0);
  UOp* b = uop_vconst(b_dt, NULL, 0);
  UOp* acc = uop_vconst(c_dt, NULL, 0);
  int first[1] = {0}; int second[1] = {0};
  UOp* w = uop_wmma(a, b, acc, first, second, 1);
  UOp* arr[] = {a,b,acc,w};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "llvm.amdgcn.mfma.f32.16x16x16.f16"));
  free(src); uop_unref(w); uop_unref(acc); uop_unref(b); uop_unref(a); free(r);
}

TEST_MAIN()

