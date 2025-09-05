#include "test_common.h"
#include "device/device.h"
#include <string.h>

static int contains(const char** arr, int n, const char* s){ for(int i=0;i<n;i++) if (strcmp(arr[i], s)==0) return 1; return 0; }

TEST(test_llvm_backend_basic){
  // Exists in registry
  TEST_ASSERT_TRUE(tg_device_exists("LLVM"));

  // Can open device
  tg_device_t* dev = tg_device_get("LLVM");
  TEST_ASSERT_NOT_NULL(dev);

  // Listed in available devices when usage allowed
  tg_device_set_allow_usage(1);
  const char* names[16] = {0};
  int n = tg_device_get_available(names, 16);
  TEST_ASSERT_TRUE(n >= 1);
  TEST_ASSERT_TRUE(contains(names, n, "LLVM"));
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()

