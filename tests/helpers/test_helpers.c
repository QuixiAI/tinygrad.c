#include "test_common.h"
#include "helpers/helpers.h"
#include <string.h>
#include <stdlib.h>

// math helpers
TEST(test_div_round_helpers){
  TEST_ASSERT_EQUAL_INT64(-1, tg_py_floor_div(-1, 2));
  TEST_ASSERT_EQUAL_INT64(1, tg_ceildiv(-1, -2));
  TEST_ASSERT_EQUAL_INT64(6, tg_round_up(5, 3));
  TEST_ASSERT_EQUAL_INT64(4, tg_round_down(5, 2));
  TEST_ASSERT_EQUAL_INT64(-1, tg_cdiv(-3, 2));
  TEST_ASSERT_EQUAL_INT64(1, tg_cmod(5, 2));
}

TEST(test_bit_helpers){
  TEST_ASSERT_EQUAL_UINT32(0x89ABCDEFu, tg_lo32(0x0123456789ABCDEFULL));
  TEST_ASSERT_EQUAL_UINT32(0x01234567u, tg_hi32(0x0123456789ABCDEFULL));
  uint32_t hi, lo; tg_data64(0x0102030405060708ULL, &hi, &lo);
  TEST_ASSERT_EQUAL_UINT32(0x01020304u, hi);
  TEST_ASSERT_EQUAL_UINT32(0x05060708u, lo);
  uint32_t lo2, hi2; tg_data64_le(0x0102030405060708ULL, &lo2, &hi2);
  TEST_ASSERT_EQUAL_UINT32(0x05060708u, lo2);
  TEST_ASSERT_EQUAL_UINT32(0x01020304u, hi2);
  TEST_ASSERT_EQUAL_UINT64(0x3ULL, tg_getbits(0xF3ULL, 0, 1));
  TEST_ASSERT_EQUAL_UINT64(0xF3ULL, tg_i2u(8, -13));
}

// text + ansi helpers
TEST(test_colored_and_strip){
  char* s = tg_colored("hello", "green", 0);
  TEST_ASSERT_NOT_NULL(s);
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

// seq + diskcache + profiling
TEST(test_all_same_argsort_dedup){
  int a[5] = {3,3,3,3,3};
  TEST_ASSERT_TRUE(tg_all_same_int(a, 5));
  int b[5] = {5,1,4,2,3}; int idx[5]; tg_argsort_int(b,5,idx);
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

// wait_cond
typedef struct { int counter; int target; } ctx_t;
static int inc_until(void* p){ ctx_t* c=(ctx_t*)p; if (c->counter < c->target) c->counter++; return c->counter; }
static int always_zero(void* p){ (void)p; return 0; }

TEST(test_wait_cond_success){
  ctx_t ctx = {0, 5};
  int ok = tg_wait_cond(inc_until, &ctx, /*value*/5, /*timeout_ms*/200, "");
  TEST_ASSERT_TRUE(ok);
}

TEST(test_wait_cond_timeout){
  int ok = tg_wait_cond(always_zero, NULL, /*value*/1, /*timeout_ms*/10, "");
  TEST_ASSERT_FALSE(ok);
}

TEST(test_is_ci_flag){
  const char* prev = getenv("CI");
  setenv("CI", "1", 1);
  TEST_ASSERT_TRUE(tg_is_ci());
  if (prev) setenv("CI", prev, 1); else unsetenv("CI");
}

// fetch + gunzip
#ifdef TG_HAVE_ZLIB
#include <zlib.h>
#include <stdio.h>
static void write_gz(const char* path, const char* content){
  gzFile gz = gzopen(path, "wb");
  TEST_ASSERT_TRUE(gz != NULL);
  int len = (int)strlen(content);
  TEST_ASSERT_TRUE(gzwrite(gz, content, len) == len);
  gzclose(gz);
}

TEST(test_fetch_local_gunzip){
  const char* gzpath = "build/testdata.txt.gz";
  const char* text = "hello tinygrad.c";
  write_gz(gzpath, text);
  char out[512]={0};
  int rc = tg_fetch(gzpath, NULL, NULL, /*gunzip*/1, /*allow_caching*/0, out, sizeof(out));
  TEST_ASSERT_EQUAL_INT(0, rc);
  TEST_ASSERT_TRUE(strstr(out, ".gz.gunzip") != NULL);
  FILE* f = fopen(out, "rb"); TEST_ASSERT_NOT_NULL(f);
  char buf[64]={0}; size_t n = fread(buf,1,sizeof(buf)-1,f); fclose(f); (void)n;
  TEST_ASSERT_EQUAL_STRING(text, buf);
}
#else
TEST(test_fetch_local_gunzip){
  char out[16]; int rc = tg_fetch("./doesnotmatter.gz", NULL, NULL, 1, 0, out, sizeof(out));
  TEST_ASSERT_TRUE(rc != 0);
}
#endif

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()

