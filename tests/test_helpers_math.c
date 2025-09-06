#include "test_common.h"
#include "helpers/helpers.h"

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

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()
