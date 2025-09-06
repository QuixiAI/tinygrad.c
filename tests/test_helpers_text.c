#include "test_common.h"
#include "helpers/helpers.h"
#include <string.h>
#include <stdlib.h>

TEST(test_colored_and_strip){
  char* s = tg_colored("hello", "green", 0);
  TEST_ASSERT_NOT_NULL(s);
  // should contain ESC and [ and end with m...\x1b[0m
  TEST_ASSERT_TRUE(strstr(s, "\x1b[") != NULL);
  char* t = tg_ansistrip(s);
  TEST_ASSERT_EQUAL_STRING("hello", t);
  TEST_ASSERT_EQUAL_INT(5, tg_ansilen(s));
  free(s); free(t);
}

TEST(test_time_to_str){
  char* s1 = tg_time_to_str(11.0, 8);
  TEST_ASSERT_NOT_NULL(s1);
  TEST_ASSERT_TRUE(strstr(s1, "s ") != NULL);
  free(s1);
  char* s2 = tg_time_to_str(0.02, 8);
  TEST_ASSERT_TRUE(strstr(s2, "ms") != NULL);
  free(s2);
  char* s3 = tg_time_to_str(0.0001, 8);
  TEST_ASSERT_TRUE(strstr(s3, "us") != NULL);
  free(s3);
}

TEST(test_strip_parens){
  char* a = tg_strip_parens("(abc)");
  TEST_ASSERT_EQUAL_STRING("abc", a);
  free(a);
  char* b = tg_strip_parens("(a(b)c)");
  TEST_ASSERT_EQUAL_STRING("a(b)c", b);
  free(b);
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()
