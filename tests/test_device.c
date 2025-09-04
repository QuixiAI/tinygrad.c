#include "test_common.h"
#include "device/device.h"
#include <string.h>
#include <stdlib.h>

// Faithful TDD port of reference/test/unit/test_device.py
// Tests call actual API functions and will fail naturally when product code is stubbed

// TestDevice class tests

TEST(test_canonicalize) {
    // Faithful port of Python test_canonicalize
    const char* result;
    
    // Test: Device.canonicalize(None) -> Device.DEFAULT
    result = tg_device_canonicalize(NULL);
    const char* default_device = tg_device_get_default();
    if (default_device != NULL && result != NULL) {
        TEST_ASSERT_EQUAL_STRING(default_device, result);
    } else {
        // TDD: Will fail because stubs return NULL
        TEST_ASSERT_NOT_NULL(result);
    }
    
    // Test: Device.canonicalize("CPU") -> "CPU" 
    result = tg_device_canonicalize("CPU");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("CPU", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("cpu") -> "CPU"
    result = tg_device_canonicalize("cpu");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("CPU", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("GPU") -> "GPU"
    result = tg_device_canonicalize("GPU");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("GPU", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("GPU:0") -> "GPU"
    result = tg_device_canonicalize("GPU:0");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("GPU", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("gpu:0") -> "GPU"
    result = tg_device_canonicalize("gpu:0");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("GPU", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("GPU:1") -> "GPU:1"
    result = tg_device_canonicalize("GPU:1");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("GPU:1", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("gpu:1") -> "GPU:1"
    result = tg_device_canonicalize("gpu:1");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("GPU:1", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("GPU:2") -> "GPU:2"
    result = tg_device_canonicalize("GPU:2");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("GPU:2", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("disk:/dev/shm/test") -> "DISK:/dev/shm/test"
    result = tg_device_canonicalize("disk:/dev/shm/test");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("DISK:/dev/shm/test", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
    
    // Test: Device.canonicalize("disk:000.txt") -> "DISK:000.txt"
    result = tg_device_canonicalize("disk:000.txt");
    if (result != NULL) {
        TEST_ASSERT_EQUAL_STRING("DISK:000.txt", result);
    } else {
        TEST_ASSERT_NOT_NULL(result); // Will fail in TDD
    }
}

TEST(test_getitem_not_exist) {
    // Faithful port of Python test_getitem_not_exist
    // Test: Device["TYPO"] should raise ModuleNotFoundError
    
    tg_device_t* device = tg_device_get("TYPO");
    
    // In Python this raises ModuleNotFoundError
    // In C, we expect NULL return to indicate error
    TEST_ASSERT_NULL(device); // Will pass because stub returns NULL, but for wrong reason in TDD
    
    // Also test that device doesn't exist
    int exists = tg_device_exists("TYPO");
    TEST_ASSERT_FALSE(exists); // Will pass because stub returns 0, but for wrong reason in TDD
}

TEST(test_lowercase_canonicalizes) {
    // Faithful port of Python test_lowercase_canonicalizes
    const char* device = tg_device_get_default();
    if (device == NULL) {
        // TDD: Will fail because stub returns NULL
        TEST_ASSERT_NOT_NULL(device);
        return;
    }
    
    // Create lowercase version for testing
    char lowercase_device[256];
    strncpy(lowercase_device, device, sizeof(lowercase_device) - 1);
    lowercase_device[sizeof(lowercase_device) - 1] = '\0';
    
    // Convert to lowercase
    for (char* p = lowercase_device; *p; p++) {
        if (*p >= 'A' && *p <= 'Z') {
            *p = *p + 32;
        }
    }
    
    // Set default to lowercase
    int result = tg_device_set_default(lowercase_device);
    if (result == TG_SUCCESS) {
        // Test that canonicalize(None) returns the original device
        const char* canonicalized = tg_device_canonicalize(NULL);
        if (canonicalized != NULL) {
            TEST_ASSERT_EQUAL_STRING(device, canonicalized);
        } else {
            TEST_ASSERT_NOT_NULL(canonicalized); // Will fail in TDD
        }
        
        // Restore original default
        tg_device_set_default(device);
    } else {
        // TDD: Will fail because stub returns TG_ERR_UNIMPL
        TEST_ASSERT_EQUAL(TG_SUCCESS, result);
    }
}

// TestCompiler class tests

TEST(test_compile_cached) {
    // Faithful port of Python test_compile_cached
    
    // Clear cache: diskcache_put("key", "123", None)
    int clear_result = tg_diskcache_put("key", "123", NULL, 0);
    // TDD: Will fail because stub returns TG_ERR_UNIMPL
    
    // Set compiler cache enabled: Context(DISABLE_COMPILER_CACHE=0)
    int cache_result = tg_context_set_compiler_cache(1); // 1 = enabled, 0 = disabled
    // TDD: Will fail because stub returns TG_ERR_UNIMPL
    
    if (cache_result == TG_SUCCESS) {
        // Create mock compiler
        tg_compiler_t* compiler = tg_compiler_create("key");
        if (compiler != NULL) {
            // Test compile_cached
            char* output = NULL;
            size_t output_size = 0;
            int compile_result = tg_compiler_compile_cached(compiler, "123", &output, &output_size);
            
            if (compile_result == TG_SUCCESS && output != NULL) {
                // Should return encoded "123"
                TEST_ASSERT_EQUAL_STRING("123", output);
                
                // Check that it's cached
                char* cached_output = NULL;
                size_t cached_size = 0;
                int cache_get_result = tg_diskcache_get("key", "123", &cached_output, &cached_size);
                if (cache_get_result == TG_SUCCESS && cached_output != NULL) {
                    TEST_ASSERT_EQUAL_STRING("123", cached_output);
                    free(cached_output);
                } else {
                    TEST_ASSERT_EQUAL(TG_SUCCESS, cache_get_result); // Will fail in TDD
                }
                
                free(output);
            } else {
                TEST_ASSERT_EQUAL(TG_SUCCESS, compile_result); // Will fail in TDD
            }
            
            tg_compiler_destroy(compiler);
        } else {
            TEST_ASSERT_NOT_NULL(compiler); // Will fail in TDD
        }
    } else {
        TEST_ASSERT_EQUAL(TG_SUCCESS, cache_result); // Will fail in TDD
    }
}

TEST(test_compile_cached_disabled) {
    // Faithful port of Python test_compile_cached_disabled
    
    // Clear cache
    int clear_result = tg_diskcache_put("disabled_key", "123", NULL, 0);
    // TDD: Will fail because stub returns TG_ERR_UNIMPL
    
    // Disable compiler cache: Context(DISABLE_COMPILER_CACHE=1) 
    int cache_result = tg_context_set_compiler_cache(0); // 0 = disabled
    // TDD: Will fail because stub returns TG_ERR_UNIMPL
    
    if (cache_result == TG_SUCCESS) {
        tg_compiler_t* compiler = tg_compiler_create("disabled_key");
        if (compiler != NULL) {
            char* output = NULL;
            size_t output_size = 0;
            int compile_result = tg_compiler_compile_cached(compiler, "123", &output, &output_size);
            
            if (compile_result == TG_SUCCESS && output != NULL) {
                TEST_ASSERT_EQUAL_STRING("123", output);
                
                // Should NOT be cached when disabled
                char* cached_output = NULL;
                size_t cached_size = 0;
                int cache_get_result = tg_diskcache_get("disabled_key", "123", &cached_output, &cached_size);
                // Should return error/NULL indicating not cached
                TEST_ASSERT_NOT_EQUAL(TG_SUCCESS, cache_get_result);
                
                free(output);
            } else {
                TEST_ASSERT_EQUAL(TG_SUCCESS, compile_result); // Will fail in TDD
            }
            
            tg_compiler_destroy(compiler);
        } else {
            TEST_ASSERT_NOT_NULL(compiler); // Will fail in TDD
        }
    } else {
        TEST_ASSERT_EQUAL(TG_SUCCESS, cache_result); // Will fail in TDD
    }
}

TEST(test_device_compile) {
    // Faithful port of Python test_device_compile
    
    // Disable compiler cache for this test
    int cache_result = tg_context_set_compiler_cache(0);
    if (cache_result != TG_SUCCESS) {
        // TDD: Will fail because stub returns TG_ERR_UNIMPL
        TEST_ASSERT_EQUAL(TG_SUCCESS, cache_result);
        return;
    }
    
    // Test device compilation - simplified version of Python test
    // Python creates tensors and realizes operations, we test device compile function
    const char* default_device = tg_device_get_default();
    if (default_device != NULL) {
        int compile_result = tg_device_compile_test(default_device);
        if (compile_result != TG_SUCCESS) {
            // TDD: Will fail because stub returns TG_ERR_UNIMPL
            TEST_ASSERT_EQUAL(TG_SUCCESS, compile_result);
        }
    } else {
        // TDD: Will fail because stub returns NULL
        TEST_ASSERT_NOT_NULL(default_device);
    }
}

// TestRunAsModule class test - simplified since we can't easily run subprocess in C tests

TEST(test_module_functionality) {
    // Simplified port of Python test_module_runs
    // Test that basic device functionality works instead of running as module
    
    const char* default_device = tg_device_get_default();
    if (default_device != NULL) {
        // Test that we can canonicalize the default device
        const char* canonicalized = tg_device_canonicalize(default_device);
        if (canonicalized != NULL) {
            // Should at least return something for default device
            TEST_ASSERT_NOT_NULL(canonicalized);
            // For sanity check, ensure CPU is supported (like Python test checks "CPU" in output)
            const char* cpu_result = tg_device_canonicalize("CPU");
            TEST_ASSERT_NOT_NULL(cpu_result); // Will fail in TDD
        } else {
            TEST_ASSERT_NOT_NULL(canonicalized); // Will fail in TDD
        }
    } else {
        // TDD: Will fail because stub returns NULL
        TEST_ASSERT_NOT_NULL(default_device);
    }
}

// Unity framework setup/teardown functions
void setUp(void) {
    // Set up before each test
}

void tearDown(void) {
    // Clean up after each test
}

// Unity framework test main
TEST_MAIN()