#include "test_common.h"
#include "renderer/cstyle.h"

void setUp(void) {}
void tearDown(void) {}

TEST(test_renderer_cstyle_clang_minimal) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  TEST_ASSERT_EQUAL_STRING("CPU", r->device);
  TEST_ASSERT_TRUE(r->supports_float4);
  TEST_ASSERT_FALSE(r->has_local);
  TEST_ASSERT_NOT_NULL(r->render);
  char* src = r->render(r, NULL, 0);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_EQUAL(-1, (int)strstr(src, "kernel_main") - (int)src);
  free(src);
  free(r);
}

TEST_MAIN()

