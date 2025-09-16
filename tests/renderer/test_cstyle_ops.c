#include "test_common.h"
#include "renderer/cstyle.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_renderer_cstyle_clang_sqrt_builtin) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* c4 = uop_const(dtypes.float32, 4.0);
  UOp* s = uop_sqrt(c4);
  UOp* arr[2] = {c4, s};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "__builtin_sqrtf"));
  free(src);
  uop_unref(s);
  uop_unref(c4);
  free(r);
}

TEST(test_renderer_cstyle_clang_vectorize_literal) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* one = uop_const(dtypes.float32, 1.0);
  UOp* vec = uop_broadcast(one, 4);
  UOp* arr[2] = {one, vec};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(src);
  // Expect C-style vector literal cast: (float4){ ... }
  TEST_ASSERT_NOT_NULL(strstr(src, "(float4){"));
  free(src);
  uop_unref(vec);
  uop_unref(one);
  free(r);
}

TEST_MAIN()

