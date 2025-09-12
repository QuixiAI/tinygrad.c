#include "test_common.h"
#include "renderer/renderer.h"

void setUp(void) {}
void tearDown(void) {}

TEST(test_to_function_name_sanitizes) {
  char* s = renderer_to_function_name("my fun(%):name");
  TEST_ASSERT_NOT_NULL(s);
  // Letters and digits remain, others hex-encoded (2 hex chars)
  // Reference Python: to_function_name('my fun(%):name')
  // becomes 'my20fun2825293Aname' (space=0x20, '('=28, '%'=25, ')'=29, ':'=3A)
  TEST_ASSERT_EQUAL_STRING("my20fun2825293Aname", s);
  free(s);
}

TEST_MAIN()
