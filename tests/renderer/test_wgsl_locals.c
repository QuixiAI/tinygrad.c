#include "test_common.h"
#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_wgsl_define_local_decl) {
  Renderer* r = renderer_wgsl();
  UOp* loc = uop_define_local(dtypes.char_, 16); // packed type
  UOp* arr[] = {loc};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "var<workgroup>"));
  TEST_ASSERT_NOT_NULL(strstr(src, "atomic<u32>"));
  free(src); uop_unref(loc); free(r);
}

TEST(test_wgsl_define_reg_decl) {
  Renderer* r = renderer_wgsl();
  UOp* reg = uop_define_reg(dtypes.int32);
  UOp* arr[] = {reg};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "var "));
  free(src); uop_unref(reg); free(r);
}

TEST(test_wgsl_const_uint32_negative_bitcast) {
  Renderer* r = renderer_wgsl();
  UOp* c = uop_const(dtypes.uint32, -1.0);
  UOp* arr[] = {c};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "bitcast<u32>(-1)"));
  free(src); uop_unref(c); free(r);
}

TEST_MAIN()

