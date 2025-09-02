#ifndef TEST_COMMON_H
#define TEST_COMMON_H

/* Unity configuration - defined once for all tests */
#define UNITY_INCLUDE_DOUBLE
#define UNITY_INCLUDE_FLOAT
#define UNITY_OUTPUT_COLOR

#include "unity.h"
#include "tg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <limits.h>
#include <assert.h>

/* Additional test macros for compatibility */
#define ASSERT(condition) TEST_ASSERT(condition)
#define ASSERT_NEAR(actual, expected, tolerance) TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, actual)

/* Auto-registration of tests */
#define MAX_TESTS 256
typedef struct {
    void (*func)(void);
    const char* name;
} TestEntry;

extern TestEntry g_tests[MAX_TESTS];
extern int g_test_count;

#define TEST(fname) \
    static void fname(void); \
    static void __attribute__((constructor)) register_##fname(void) { \
        g_tests[g_test_count].func = fname; \
        g_tests[g_test_count].name = #fname; \
        g_test_count++; \
    } \
    static void fname(void)

/* Simplified main for test files */
#define TEST_MAIN() \
    TestEntry g_tests[MAX_TESTS]; \
    int g_test_count = 0; \
    int main(void) { \
        UNITY_BEGIN(); \
        for (int i = 0; i < g_test_count; i++) { \
            UnityDefaultTestRun(g_tests[i].func, g_tests[i].name, __LINE__); \
        } \
        return UNITY_END(); \
    }

/* Common test setup/teardown if needed in the future */

#endif /* TEST_COMMON_H */