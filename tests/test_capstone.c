#include "test_common.h"
#include "helpers/helpers.h"

TEST(test_capstone_flatdump){
#ifdef TG_HAVE_CAPSTONE
  int rc = 0;
  #if defined(__x86_64__) || defined(_M_X64)
    const unsigned char code[] = { 0xC3 }; // ret
    rc = tg_capstone_flatdump(code, sizeof(code));
    TEST_ASSERT_EQUAL_INT(0, rc);
  #elif defined(__aarch64__)
    const unsigned char code[] = { 0xC0, 0x03, 0x5F, 0xD6 }; // ret
    rc = tg_capstone_flatdump(code, sizeof(code));
    TEST_ASSERT_EQUAL_INT(0, rc);
  #else
    // Unsupported arch in this test; just assert that calling with 0 returns error
    rc = tg_capstone_flatdump(NULL, 0);
    TEST_ASSERT_TRUE(rc != 0);
  #endif
#else
  // Without capstone, function should return non-zero (unsupported)
  int rc = tg_capstone_flatdump((const unsigned char*)"\xC3", 1);
  TEST_ASSERT_TRUE(rc != 0);
#endif
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()

