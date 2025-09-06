#include "test_common.h"
#include "helpers/helpers.h"

#ifdef TG_HAVE_ZLIB
#include <zlib.h>
#include <stdio.h>
#include <string.h>

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
  // ensure dir exists via fetch logic creating outputs; we write file directly
  write_gz(gzpath, text);
  char out[512]={0};
  int rc = tg_fetch(gzpath, NULL, NULL, /*gunzip*/1, /*allow_caching*/0, out, sizeof(out));
  TEST_ASSERT_EQUAL_INT(0, rc);
  // out should be gzpath + .gunzip
  TEST_ASSERT_TRUE(strstr(out, ".gz.gunzip") != NULL);
  FILE* f = fopen(out, "rb"); TEST_ASSERT_NOT_NULL(f);
  char buf[64]={0}; size_t n = fread(buf,1,sizeof(buf)-1,f); fclose(f);
  TEST_ASSERT_EQUAL_STRING(text, buf);
}
#else
TEST(test_fetch_local_gunzip){
  // If no zlib, fetching with gunzip should fail cleanly
  char out[16]; int rc = tg_fetch("./doesnotmatter.gz", NULL, NULL, 1, 0, out, sizeof(out));
  TEST_ASSERT_TRUE(rc != 0);
}
#endif

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()

