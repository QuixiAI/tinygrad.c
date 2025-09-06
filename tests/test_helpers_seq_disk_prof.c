#include "test_common.h"
#include "helpers/helpers.h"
#include <string.h>
#include <stdlib.h>

TEST(test_all_same_argsort_dedup){
  int a[5] = {3,3,3,3,3};
  TEST_ASSERT_TRUE(tg_all_same_int(a, 5));
  int b[5] = {5,1,4,2,3}; int idx[5]; tg_argsort_int(b,5,idx);
  // sorted order should be [1,2,3,4,5] => indices [1,3,4,2,0]
  int expect[5] = {1,3,4,2,0};
  TEST_ASSERT_EQUAL_INT_ARRAY(expect, idx, 5);
  char* in[6] = {"a","b","a","c","b","d"}; char** out=NULL; int n = tg_dedup_str(in,6,&out);
  TEST_ASSERT_EQUAL_INT(4, n);
  TEST_ASSERT_EQUAL_STRING("a", out[0]); TEST_ASSERT_EQUAL_STRING("b", out[1]); TEST_ASSERT_EQUAL_STRING("c", out[2]); TEST_ASSERT_EQUAL_STRING("d", out[3]);
  for(int i=0;i<n;i++) free(out[i]); free(out);
}

TEST(test_diskcache_clear){
  const char* key = "k"; const char* src = "s"; const char* data = "hello"; size_t sz;
  char* out=NULL; tg_diskcache_put(key, src, data, strlen(data));
  TEST_ASSERT_EQUAL_INT(0, tg_diskcache_get(key, src, &out, &sz)); free(out);
  tg_diskcache_clear();
  TEST_ASSERT_NOT_EQUAL(0, tg_diskcache_get(key, src, &out, &sz));
}

TEST(test_profile_events){
  tg_profile_events_clear();
  tg_profile_set_enabled(1);
  TEST_ASSERT_TRUE(tg_profile_get_enabled());
  int h = tg_cpu_profile_begin("unit", "CPU", 0);
  tg_cpu_profile_end(h, 0);
  TEST_ASSERT_TRUE(tg_profile_events_count() >= 1);
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()
