#include "test_common.h"
#include "device/device.h"
#include <stdlib.h>
#include <string.h>

static void set_env(const char* k, const char* v){ if (v) setenv(k, v, 1); else unsetenv(k); }

TEST(test_default_from_DEV_env){
  // Save originals
  const char* orig_dev = getenv("DEV");
  const char* orig_cpu = getenv("CPU");
  const char* orig_npy = getenv("NPY");
  // Clear flags, set DEV
  set_env("CPU", NULL); set_env("NPY", NULL);
  set_env("DEV", "npy");
  const char* def = tg_device_get_default();
  TEST_ASSERT_EQUAL_STRING("NPY", def);
  // restore
  set_env("DEV", orig_dev); set_env("CPU", orig_cpu); set_env("NPY", orig_npy);
}

TEST(test_default_ignores_disk_npy_flags){
  // Save originals
  const char* orig_dev = getenv("DEV");
  const char* orig_cpu = getenv("CPU");
  const char* orig_npy = getenv("NPY");
  // Clear DEV, set NPY=1 and ensure CPU available
  set_env("DEV", NULL);
  set_env("CPU", NULL);
  set_env("NPY", "1");
  const char* def = tg_device_get_default();
  TEST_ASSERT_EQUAL_STRING("CPU", def);
  // restore
  set_env("DEV", orig_dev); set_env("CPU", orig_cpu); set_env("NPY", orig_npy);
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()

