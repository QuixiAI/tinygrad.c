#include "test_common.h"
#include "device/device.h"
#include <string.h>

static int contains(const char** arr, int n, const char* s){ for(int i=0;i<n;i++) if (strcmp(arr[i], s)==0) return 1; return 0; }

TEST(test_get_available_devices) {
  const char* names[8] = {0};
  int n = tg_device_get_available(names, 8);
  TEST_ASSERT(n >= 1);
  // CPU should be available when usage is allowed (default)
  TEST_ASSERT_TRUE(contains(names, n, "CPU"));

  // When usage is disallowed, DISK and NPY remain available, CPU does not
  tg_device_set_allow_usage(0);
  memset(names, 0, sizeof(names));
  int m = tg_device_get_available(names, 8);
  TEST_ASSERT(m >= 1);
  TEST_ASSERT_FALSE(contains(names, m, "CPU"));
  TEST_ASSERT_TRUE(contains(names, m, "DISK"));
  TEST_ASSERT_TRUE(contains(names, m, "NPY"));

  // restore allow
  tg_device_set_allow_usage(1);
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()

