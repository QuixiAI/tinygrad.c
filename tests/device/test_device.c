#include "test_common.h"
#include "device/device.h"
#include "helpers/helpers.h"
#include <string.h>
#include <stdlib.h>

// Consolidated device tests: one test file per source area (device)

static int contains(const char** arr, int n, const char* s){ for(int i=0;i<n;i++) if (strcmp(arr[i], s)==0) return 1; return 0; }
static void set_env(const char* k, const char* v){ if (v) setenv(k, v, 1); else unsetenv(k); }

// ---- device.c focused tests ----

TEST(test_canonicalize) {
    const char* result;

    result = tg_device_canonicalize(NULL);
    const char* default_device = tg_device_get_default();
    if (default_device != NULL && result != NULL) {
        TEST_ASSERT_EQUAL_STRING(default_device, result);
    } else {
        TEST_ASSERT_NOT_NULL(result);
    }

    result = tg_device_canonicalize("CPU");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("CPU", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("cpu");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("CPU", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("GPU");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("GPU", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("GPU:0");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("GPU", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("gpu:0");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("GPU", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("GPU:1");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("GPU:1", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("gpu:1");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("GPU:1", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("GPU:2");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("GPU:2", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("disk:/dev/shm/test");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("DISK:/dev/shm/test", result); else TEST_ASSERT_NOT_NULL(result);

    result = tg_device_canonicalize("disk:000.txt");
    if (result != NULL) TEST_ASSERT_EQUAL_STRING("DISK:000.txt", result); else TEST_ASSERT_NOT_NULL(result);
}

TEST(test_getitem_not_exist) {
    tg_device_t* device = tg_device_get("TYPO");
    TEST_ASSERT_NULL(device);
    int exists = tg_device_exists("TYPO");
    TEST_ASSERT_FALSE(exists);
}

TEST(test_lowercase_canonicalizes) {
    const char* device = tg_device_get_default();
    TEST_ASSERT_NOT_NULL(device);

    char lowercase_device[256];
    strncpy(lowercase_device, device, sizeof(lowercase_device) - 1);
    lowercase_device[sizeof(lowercase_device) - 1] = '\0';
    for (char* p = lowercase_device; *p; p++) if (*p >= 'A' && *p <= 'Z') *p = (char)(*p + 32);

    int result = tg_device_set_default(lowercase_device);
    TEST_ASSERT_EQUAL(TG_SUCCESS, result);
    const char* canonicalized = tg_device_canonicalize(NULL);
    TEST_ASSERT_NOT_NULL(canonicalized);
    TEST_ASSERT_EQUAL_STRING(device, canonicalized);
    tg_device_set_default(device);
}

TEST(test_get_available_devices) {
  const char* names[8] = {0};
  int n = tg_device_get_available(names, 8);
  TEST_ASSERT(n >= 1);
  TEST_ASSERT_TRUE(contains(names, n, "CPU"));

  tg_device_set_allow_usage(0);
  memset(names, 0, sizeof(names));
  int m = tg_device_get_available(names, 8);
  TEST_ASSERT(m >= 1);
  TEST_ASSERT_FALSE(contains(names, m, "CPU"));
  TEST_ASSERT_TRUE(contains(names, m, "DISK"));
  TEST_ASSERT_TRUE(contains(names, m, "NPY"));
  tg_device_set_allow_usage(1);
}

TEST(test_default_from_DEV_env){
  const char* orig_dev = getenv("DEV");
  const char* orig_cpu = getenv("CPU");
  const char* orig_npy = getenv("NPY");
  set_env("CPU", NULL); set_env("NPY", NULL);
  set_env("DEV", "npy");
  const char* def = tg_device_get_default();
  TEST_ASSERT_EQUAL_STRING("NPY", def);
  set_env("DEV", orig_dev); set_env("CPU", orig_cpu); set_env("NPY", orig_npy);
}

TEST(test_default_ignores_disk_npy_flags){
  const char* orig_dev = getenv("DEV");
  const char* orig_cpu = getenv("CPU");
  const char* orig_npy = getenv("NPY");
  set_env("DEV", NULL);
  set_env("CPU", NULL);
  set_env("NPY", "1");
  const char* def = tg_device_get_default();
  TEST_ASSERT_EQUAL_STRING("CPU", def);
  set_env("DEV", orig_dev); set_env("CPU", orig_cpu); set_env("NPY", orig_npy);
}

// ---- broader device compilation/cache tests (still under device for now) ----

TEST(test_compile_cached) {
    int clear_result = tg_diskcache_put("key", "123", NULL, 0);
    (void)clear_result; // TDD stubbed; don't assert here

    int cache_result = tg_context_set_compiler_cache(1); // enable
    if (cache_result == TG_SUCCESS) {
        tg_compiler_t* compiler = tg_compiler_create("key");
        if (compiler != NULL) {
            char* output = NULL; size_t output_size = 0;
            int compile_result = tg_compiler_compile_cached(compiler, "123", &output, &output_size);
            if (compile_result == TG_SUCCESS && output != NULL) {
                char* cached_output = NULL; size_t cached_size = 0;
                int cache_get_result = tg_diskcache_get("key", "123", &cached_output, &cached_size);
                if (cache_get_result == TG_SUCCESS && cached_output != NULL) {
                    free(cached_output);
                } else {
                    TEST_ASSERT_EQUAL(TG_SUCCESS, cache_get_result);
                }
                free(output);
            } else {
                TEST_ASSERT_EQUAL(TG_SUCCESS, compile_result);
            }
            tg_compiler_destroy(compiler);
        } else {
            TEST_ASSERT_NOT_NULL(compiler);
        }
    } else {
        TEST_ASSERT_EQUAL(TG_SUCCESS, cache_result);
    }
}

TEST(test_compile_cached_disabled) {
    int clear_result = tg_diskcache_put("disabled_key", "123", NULL, 0);
    (void)clear_result; // TDD stubbed; don't assert here

    int cache_result = tg_context_set_compiler_cache(0); // disable
    if (cache_result == TG_SUCCESS) {
        tg_compiler_t* compiler = tg_compiler_create("disabled_key");
        if (compiler != NULL) {
            char* output = NULL; size_t output_size = 0;
            int compile_result = tg_compiler_compile_cached(compiler, "123", &output, &output_size);
            if (compile_result == TG_SUCCESS && output != NULL) {
                char* cached_output = NULL; size_t cached_size = 0;
                int cache_get_result = tg_diskcache_get("disabled_key", "123", &cached_output, &cached_size);
                TEST_ASSERT_NOT_EQUAL(TG_SUCCESS, cache_get_result);
                free(output);
            } else {
                TEST_ASSERT_EQUAL(TG_SUCCESS, compile_result);
            }
            tg_compiler_destroy(compiler);
        } else {
            TEST_ASSERT_NOT_NULL(compiler);
        }
    } else {
        TEST_ASSERT_EQUAL(TG_SUCCESS, cache_result);
    }
}

TEST(test_device_compile) {
    int cache_result = tg_context_set_compiler_cache(0);
    TEST_ASSERT_EQUAL(TG_SUCCESS, cache_result);
    const char* default_device = tg_device_get_default();
    TEST_ASSERT_NOT_NULL(default_device);
    int compile_result = tg_device_compile_test(default_device);
    TEST_ASSERT_EQUAL(TG_SUCCESS, compile_result);
}

TEST(test_module_functionality) {
    const char* default_device = tg_device_get_default();
    TEST_ASSERT_NOT_NULL(default_device);
    const char* canonicalized = tg_device_canonicalize(default_device);
    TEST_ASSERT_NOT_NULL(canonicalized);
    const char* cpu_result = tg_device_canonicalize("CPU");
    TEST_ASSERT_NOT_NULL(cpu_result);
}

TEST(test_backend_llvm_basic){
  // If LLVM backend is registered, basic queries should succeed
  if (tg_device_exists("LLVM")) {
    tg_device_t* dev = tg_device_get("LLVM");
    TEST_ASSERT_NOT_NULL(dev);
    tg_device_set_allow_usage(1);
    const char* names[16] = {0};
    int n = tg_device_get_available(names, 16);
    TEST_ASSERT_TRUE(n >= 1);
  }
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()
