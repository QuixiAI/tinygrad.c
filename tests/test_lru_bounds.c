#include "test_common.h"
#include "device/buffer.h"
#include "device/allocator.h"
#include "dtype/dtype.h"
#include <stdint.h>

TEST(test_lru_cache_cap_basic){
  // Set a small cap and ensure eviction keeps cached bytes <= cap
  tg_allocator_set_cache_cap(1024);
  TEST_ASSERT_EQUAL_size_t(1024, tg_allocator_get_cache_cap());

  tg_buffer_t* b1 = tg_buffer_create("CPU", 512, &dtypes.uint8, NULL, NULL, 0, 0, NULL, 0);
  tg_buffer_t* b2 = tg_buffer_create("CPU", 512, &dtypes.uint8, NULL, NULL, 0, 0, NULL, 0);
  tg_buffer_t* b3 = tg_buffer_create("CPU", 512, &dtypes.uint8, NULL, NULL, 0, 0, NULL, 0);
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(b1));
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(b2));
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(b3));

  // Deallocate pushes into LRU (free_sized), total 1536 bytes, cap 1024 => evicted ~512
  tg_buffer_destroy(b1);
  tg_buffer_destroy(b2);
  tg_buffer_destroy(b3);

  size_t cached = tg_allocator_get_cached_bytes();
  TEST_ASSERT_TRUE(cached <= 1024);
}

TEST(test_lru_cache_cap_shrink){
  tg_allocator_set_cache_cap(2048);
  tg_buffer_t* b1 = tg_buffer_create("CPU", 1024, &dtypes.uint8, NULL, NULL, 0, 0, NULL, 0);
  tg_buffer_t* b2 = tg_buffer_create("CPU", 1024, &dtypes.uint8, NULL, NULL, 0, 0, NULL, 0);
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(b1));
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(b2));
  tg_buffer_destroy(b1);
  tg_buffer_destroy(b2);
  TEST_ASSERT_EQUAL_size_t(2048, tg_allocator_get_cached_bytes());
  // Shrink cap to 1024, should evict ~1024
  tg_allocator_set_cache_cap(1024);
  TEST_ASSERT_TRUE(tg_allocator_get_cached_bytes() <= 1024);
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()

