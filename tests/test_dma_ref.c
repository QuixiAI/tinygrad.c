#include "test_common.h"
#include "device/buffer.h"
#include "dtype/dtype.h"
#include <string.h>
#include <stdint.h>

TEST(test_dmaref_cpu_basic){
  size_t size = 8;
  uint8_t init[8] = {0,1,2,3,4,5,6,7};
  tg_buffer_t* b = tg_buffer_create("CPU", size, &dtypes.uint8, NULL, init, sizeof(init), 0, NULL, 0);
  TEST_ASSERT_NOT_NULL(b);
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(b));
  tg_dma_cpu_ref_t r = tg_buffer_as_dmaref_cpu(b);
  TEST_ASSERT_TRUE(r.addr != 0);
  TEST_ASSERT_EQUAL_size_t(tg_buffer_nbytes(b), r.size);
  // check contents
  uint8_t* p = (uint8_t*)(uintptr_t)r.addr;
  TEST_ASSERT_EQUAL_UINT8_ARRAY(init, p, 8);
  tg_buffer_destroy(b);
}

TEST(test_dmaref_cpu_view_offset){
  size_t size = 8;
  uint8_t init[8] = {10,20,30,40,50,60,70,80};
  tg_buffer_t* base = tg_buffer_create("CPU", size, &dtypes.uint8, NULL, init, sizeof(init), 0, NULL, 0);
  TEST_ASSERT_NOT_NULL(base);
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(base));
  // view of last 4 bytes starting at offset 4
  tg_buffer_t* view = tg_buffer_view(base, 4, &dtypes.uint8, 4);
  TEST_ASSERT_NOT_NULL(view);
  tg_dma_cpu_ref_t r = tg_buffer_as_dmaref_cpu(view);
  TEST_ASSERT_TRUE(r.addr != 0);
  TEST_ASSERT_EQUAL_size_t(4, r.size);
  uint8_t* p = (uint8_t*)(uintptr_t)r.addr;
  TEST_ASSERT_EQUAL_UINT8_ARRAY(&init[4], p, 4);
  tg_buffer_destroy(view);
  tg_buffer_destroy(base);
}

TEST(test_dmaref_cpu_noop_on_disk_npy){
  tg_buffer_t* d = tg_buffer_create("DISK", 4, &dtypes.uint8, NULL, NULL, 0, 0, NULL, 0);
  TEST_ASSERT_NOT_NULL(d);
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(d));
  tg_dma_cpu_ref_t rd = tg_buffer_as_dmaref_cpu(d);
  TEST_ASSERT_EQUAL_UINT(0, rd.addr);
  TEST_ASSERT_EQUAL_size_t(0, rd.size);
  tg_buffer_destroy(d);

  tg_buffer_t* n = tg_buffer_create("NPY", 4, &dtypes.uint8, NULL, NULL, 0, 0, NULL, 0);
  TEST_ASSERT_NOT_NULL(n);
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(n));
  tg_dma_cpu_ref_t rn = tg_buffer_as_dmaref_cpu(n);
  TEST_ASSERT_EQUAL_UINT(0, rn.addr);
  TEST_ASSERT_EQUAL_size_t(0, rn.size);
  tg_buffer_destroy(n);
}

TEST(test_dmaref_fd_unimplemented){
  tg_buffer_t* b = tg_buffer_create("CPU", 4, &dtypes.uint8, NULL, NULL, 0, 0, NULL, 0);
  TEST_ASSERT_NOT_NULL(b);
  TEST_ASSERT_EQUAL_INT(0, tg_buffer_allocate(b));
  tg_dma_fd_ref_t rf = tg_buffer_as_dmaref_fd(b);
  TEST_ASSERT_EQUAL_INT(0, rf.fd);
  TEST_ASSERT_EQUAL_size_t(0, rf.size);
  TEST_ASSERT_EQUAL_size_t(0, rf.offset);
  tg_buffer_destroy(b);
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()
