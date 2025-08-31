#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "../include/tg.h"
#include "../src/dtype/dtype.h"
#include "../src/helpers/helpers.h"

// Test counter
static int tests_run = 0;
static int tests_passed = 0;

// Constants from test_dtype_spec.py
#define FP8E4M3_MAX 448.0
#define FP8E5M2_MAX 57344.0

#define ASSERT(cond) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("FAIL: %s:%d - %s\n", __FILE__, __LINE__, #cond); \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    tests_run++; \
    if ((a) != (b)) { \
        printf("FAIL: %s:%d - Expected %ld == %ld\n", __FILE__, __LINE__, (long)(a), (long)(b)); \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_STREQ(a, b) do { \
    tests_run++; \
    if (strcmp((a), (b)) != 0) { \
        printf("FAIL: %s:%d - Expected '%s' == '%s'\n", __FILE__, __LINE__, (a), (b)); \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b) do { \
    tests_run++; \
    if (fabs((a) - (b)) > 1e-6) { \
        printf("FAIL: %s:%d - Expected %f == %f (diff: %e)\n", __FILE__, __LINE__, (a), (b), fabs((a)-(b))); \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_DOUBLE_EQ(a, b) do { \
    tests_run++; \
    if (fabs((a) - (b)) > 1e-10) { \
        printf("FAIL: %s:%d - Expected %.15f == %.15f (diff: %e)\n", __FILE__, __LINE__, (a), (b), fabs((a)-(b))); \
    } else { \
        tests_passed++; \
    } \
} while(0)

// Test basic DType creation - based on reference/test/test_dtype.py
void test_dtype_creation() {
    printf("Testing DType creation...\n");
    
    // Test basic types match Python definitions exactly
    ASSERT(dtypes.bool_.priority == 0);
    ASSERT(dtypes.bool_.itemsize == 1);
    ASSERT_STREQ(dtypes.bool_.name, "bool");
    ASSERT(dtypes.bool_.fmt == '?');
    ASSERT(dtypes.bool_.count == 1);
    ASSERT(dtypes.bool_._scalar == NULL);
    
    ASSERT(dtypes.int32.priority == 5);
    ASSERT(dtypes.int32.itemsize == 4);
    ASSERT_STREQ(dtypes.int32.name, "int");
    ASSERT(dtypes.int32.fmt == 'i');
    
    ASSERT(dtypes.float32.priority == 13);
    ASSERT(dtypes.float32.itemsize == 4);
    ASSERT_STREQ(dtypes.float32.name, "float");
    ASSERT(dtypes.float32.fmt == 'f');
    
    // Test all type priorities match Python version
    ASSERT(dtypes.void_.priority == -1);
    ASSERT(dtypes.int8.priority == 1);
    ASSERT(dtypes.uint8.priority == 2);
    ASSERT(dtypes.int16.priority == 3);
    ASSERT(dtypes.uint16.priority == 4);
    ASSERT(dtypes.uint32.priority == 6);
    ASSERT(dtypes.int64.priority == 7);
    ASSERT(dtypes.uint64.priority == 8);
    ASSERT(dtypes.fp8e4m3.priority == 9);
    ASSERT(dtypes.fp8e5m2.priority == 10);
    ASSERT(dtypes.float16.priority == 11);
    ASSERT(dtypes.bfloat16.priority == 12);
    ASSERT(dtypes.float64.priority == 14);
}

// Test type checking functions based on reference/test/unit/test_dtype_spec.py
void test_dtype_type_checks() {
    printf("Testing DType type checks...\n");
    
    // Test is_float
    ASSERT(dtypes_is_float(&dtypes.float32));
    ASSERT(dtypes_is_float(&dtypes.float64));
    ASSERT(dtypes_is_float(&dtypes.float16));
    ASSERT(dtypes_is_float(&dtypes.bfloat16));
    ASSERT(dtypes_is_float(&dtypes.fp8e4m3));
    ASSERT(dtypes_is_float(&dtypes.fp8e5m2));
    ASSERT(!dtypes_is_float(&dtypes.int32));
    ASSERT(!dtypes_is_float(&dtypes.bool_));
    
    // Test is_int  
    ASSERT(dtypes_is_int(&dtypes.int32));
    ASSERT(dtypes_is_int(&dtypes.uint32));
    ASSERT(dtypes_is_int(&dtypes.int8));
    ASSERT(dtypes_is_int(&dtypes.uint8));
    ASSERT(dtypes_is_int(&dtypes.int16));
    ASSERT(dtypes_is_int(&dtypes.uint16));
    ASSERT(dtypes_is_int(&dtypes.int64));
    ASSERT(dtypes_is_int(&dtypes.uint64));
    ASSERT(!dtypes_is_int(&dtypes.float32));
    ASSERT(!dtypes_is_int(&dtypes.bool_));
    
    // Test is_unsigned
    ASSERT(dtypes_is_unsigned(&dtypes.uint32));
    ASSERT(dtypes_is_unsigned(&dtypes.uint8));
    ASSERT(dtypes_is_unsigned(&dtypes.uint16));
    ASSERT(dtypes_is_unsigned(&dtypes.uint64));
    ASSERT(!dtypes_is_unsigned(&dtypes.int32));
    ASSERT(!dtypes_is_unsigned(&dtypes.int8));
    ASSERT(!dtypes_is_unsigned(&dtypes.int16));
    ASSERT(!dtypes_is_unsigned(&dtypes.int64));
    ASSERT(!dtypes_is_unsigned(&dtypes.float32));
    ASSERT(!dtypes_is_unsigned(&dtypes.bool_));
    
    // Test is_bool
    ASSERT(dtypes_is_bool(&dtypes.bool_));
    ASSERT(!dtypes_is_bool(&dtypes.int32));
    ASSERT(!dtypes_is_bool(&dtypes.float32));
}

// Test vectorization with scalar extraction
void test_dtype_vectorization() {
    printf("Testing DType vectorization and scalar...\n");
    
    // Test vec() method
    DType vec2 = dtype_vec(&dtypes.float32, 2);
    ASSERT(vec2.count == 2);
    ASSERT(vec2.itemsize == 8); // 4 * 2
    ASSERT(vec2._scalar == &dtypes.float32);
    
    // Test vectorizing sz=1 returns original (Python: if sz == 1 or self == dtypes.void: return self)
    DType vec1 = dtype_vec(&dtypes.float32, 1);
    ASSERT(dtype_eq(&vec1, &dtypes.float32));
    
    // Test void doesn't vectorize
    DType void_vec = dtype_vec(&dtypes.void_, 4);
    ASSERT(dtype_eq(&void_vec, &dtypes.void_));
    
    // Test scalar() method (Python: return self._scalar if self._scalar is not None else self)
    DType scalar_from_vec = dtype_scalar(&vec2);
    ASSERT(dtype_eq(&scalar_from_vec, &dtypes.float32));
    
    DType scalar_from_scalar = dtype_scalar(&dtypes.float32);
    ASSERT(dtype_eq(&scalar_from_scalar, &dtypes.float32));
}

// Test dtype range values - based on test_dtype_spec.py:test_dtype_range
void test_dtype_range() {
    printf("Testing DType range values...\n");
    
    // Test float types (should be -inf/+inf)
    ASSERT(isinf(dtypes_min(&dtypes.float32)) && dtypes_min(&dtypes.float32) < 0);
    ASSERT(isinf(dtypes_max(&dtypes.float32)) && dtypes_max(&dtypes.float32) > 0);
    ASSERT(isinf(dtypes_min(&dtypes.float64)) && dtypes_min(&dtypes.float64) < 0);
    ASSERT(isinf(dtypes_max(&dtypes.float64)) && dtypes_max(&dtypes.float64) > 0);
    
    // Test integer types - must match numpy's iinfo exactly
    ASSERT_EQ((int64_t)dtypes_min(&dtypes.int8), -128);
    ASSERT_EQ((int64_t)dtypes_max(&dtypes.int8), 127);
    ASSERT_EQ((uint64_t)dtypes_min(&dtypes.uint8), 0);
    ASSERT_EQ((uint64_t)dtypes_max(&dtypes.uint8), 255);
    
    ASSERT_EQ((int64_t)dtypes_min(&dtypes.int16), -32768);
    ASSERT_EQ((int64_t)dtypes_max(&dtypes.int16), 32767);
    ASSERT_EQ((uint64_t)dtypes_min(&dtypes.uint16), 0);
    ASSERT_EQ((uint64_t)dtypes_max(&dtypes.uint16), 65535);
    
    ASSERT_EQ((int64_t)dtypes_min(&dtypes.int32), -2147483648LL);
    ASSERT_EQ((int64_t)dtypes_max(&dtypes.int32), 2147483647LL);
    ASSERT_EQ((uint64_t)dtypes_min(&dtypes.uint32), 0ULL);
    ASSERT_EQ((uint64_t)dtypes_max(&dtypes.uint32), 4294967295ULL);
    
    // Test bool (False=0, True=1)
    ASSERT_DOUBLE_EQ(dtypes_min(&dtypes.bool_), 0.0); // False
    ASSERT_DOUBLE_EQ(dtypes_max(&dtypes.bool_), 1.0); // True
}

// Test finfo function - based on test_dtype_spec.py
void test_dtype_finfo() {
    printf("Testing DType finfo...\n");
    
    // Test values from Python: dtypes.finfo() returns (exponent, mantissa)
    FInfo info_f16 = dtypes_finfo(&dtypes.float16);
    ASSERT_EQ(info_f16.exponent, 5);
    ASSERT_EQ(info_f16.mantissa, 10);
    
    FInfo info_bf16 = dtypes_finfo(&dtypes.bfloat16);
    ASSERT_EQ(info_bf16.exponent, 8);
    ASSERT_EQ(info_bf16.mantissa, 7);
    
    FInfo info_f32 = dtypes_finfo(&dtypes.float32);
    ASSERT_EQ(info_f32.exponent, 8);
    ASSERT_EQ(info_f32.mantissa, 23);
    
    FInfo info_f64 = dtypes_finfo(&dtypes.float64);
    ASSERT_EQ(info_f64.exponent, 11);
    ASSERT_EQ(info_f64.mantissa, 52);
    
    FInfo info_fp8e5m2 = dtypes_finfo(&dtypes.fp8e5m2);
    ASSERT_EQ(info_fp8e5m2.exponent, 5);
    ASSERT_EQ(info_fp8e5m2.mantissa, 2);
    
    FInfo info_fp8e4m3 = dtypes_finfo(&dtypes.fp8e4m3);
    ASSERT_EQ(info_fp8e4m3.exponent, 4);
    ASSERT_EQ(info_fp8e4m3.mantissa, 3);
}

// Test from_py functionality - based on test_dtype_spec.py:test_from_py
void test_dtype_from_py() {
    printf("Testing DType from_py...\n");
    
    // Python: assert dtypes.from_py(True) == dtypes.bool
    DType dt_bool = dtypes_from_py_bool(true);
    ASSERT(dtype_eq(&dt_bool, &dtypes.bool_));
    
    // Python: assert dtypes.from_py(2) == dtypes.default_int
    DType dt_int = dtypes_from_py_int(2);
    ASSERT(dtype_eq(&dt_int, &dtypes.default_int));
    
    // Python: assert dtypes.from_py(3.0) == dtypes.default_float
    DType dt_float = dtypes_from_py_float(3.0);
    ASSERT(dtype_eq(&dt_float, &dtypes.default_float));
}

// Test truncate_fp16 - based on test_dtype_spec.py:test_truncate_fp16
void test_truncate_fp16() {
    printf("Testing truncate_fp16...\n");
    
    // Python test cases:
    // self.assertEqual(truncate_fp16(1), 1)
    ASSERT_FLOAT_EQ(truncate_fp16(1.0), 1.0);
    
    // self.assertEqual(truncate_fp16(65504), 65504)
    ASSERT_FLOAT_EQ(truncate_fp16(65504.0), 65504.0);
    
    // self.assertEqual(truncate_fp16(65520), math.inf)
    ASSERT(isinf(truncate_fp16(65520.0)));
    
    // Test overflow to infinity
    ASSERT(isinf(truncate_fp16(1e10)));
    
    // Test sign preservation
    ASSERT(truncate_fp16(-1.0) < 0);
    ASSERT(isinf(truncate_fp16(-1e10)) && truncate_fp16(-1e10) < 0);
}

// Test type promotion - based on test_dtype_spec.py:test_dtype_promo  
void test_dtype_promotion() {
    printf("Testing DType promotion...\n");
    
    // Python test cases from test_dtype_promo():
    // assert least_upper_dtype(dtypes.bool, dtypes.int8) == dtypes.int8
    DType result = least_upper_dtype(&dtypes.bool_, &dtypes.int8);
    ASSERT(dtype_eq(&result, &dtypes.int8));
    
    // assert least_upper_dtype(dtypes.float16, dtypes.float32) == dtypes.float32
    result = least_upper_dtype(&dtypes.float16, &dtypes.float32);
    ASSERT(dtype_eq(&result, &dtypes.float32));
    
    // assert least_upper_dtype(dtypes.float32, dtypes.float64) == dtypes.float64
    result = least_upper_dtype(&dtypes.float32, &dtypes.float64);
    ASSERT(dtype_eq(&result, &dtypes.float64));
    
    // assert least_upper_dtype(dtypes.bool, dtypes.float32) == dtypes.float32
    result = least_upper_dtype(&dtypes.bool_, &dtypes.float32);
    ASSERT(dtype_eq(&result, &dtypes.float32));
}

// Test sum accumulator dtype - based on test_dtype_alu.py:test_sum
void test_sum_acc_dtype() {
    printf("Testing sum_acc_dtype...\n");
    
    // From Python test_sum():
    // assert (Tensor([0, 1], dtype=dtypes.bool)).sum().dtype == dtypes.int32
    DType result = sum_acc_dtype(&dtypes.bool_);
    ASSERT(dtype_eq(&result, &dtypes.int32));
    
    // assert (Tensor([0, 1], dtype=dtypes.int8)).sum().dtype == dtypes.int32
    result = sum_acc_dtype(&dtypes.int8);
    ASSERT(dtype_eq(&result, &dtypes.int32));
    
    // assert (Tensor([0, 1], dtype=dtypes.uint8)).sum().dtype == dtypes.uint32
    result = sum_acc_dtype(&dtypes.uint8);
    ASSERT(dtype_eq(&result, &dtypes.uint32));
    
    // assert (Tensor([0, 1], dtype=dtypes.int64)).sum().dtype == dtypes.int64
    result = sum_acc_dtype(&dtypes.int64);
    ASSERT(dtype_eq(&result, &dtypes.int64));
    
    // assert (Tensor([0, 1], dtype=dtypes.uint64)).sum().dtype == dtypes.uint64
    result = sum_acc_dtype(&dtypes.uint64);
    ASSERT(dtype_eq(&result, &dtypes.uint64));
    
    // In Python, float16 sum uses float32 as accumulator (higher priority)
    result = sum_acc_dtype(&dtypes.float16);
    ASSERT(dtype_eq(&result, &dtypes.float32));
    
    // assert (Tensor([0, 1], dtype=dtypes.float32)).sum().dtype == dtypes.float32
    result = sum_acc_dtype(&dtypes.float32);
    ASSERT(dtype_eq(&result, &dtypes.float32));
}

// Test least_upper_float - based on test_dtype_spec.py:TestAutoCastType
void test_least_upper_float() {
    printf("Testing least_upper_float...\n");
    
    // Python: if input is float, should return input unchanged
    DType result = least_upper_float(&dtypes.float32);
    ASSERT(dtype_eq(&result, &dtypes.float32));
    
    result = least_upper_float(&dtypes.float16);
    ASSERT(dtype_eq(&result, &dtypes.float16));
    
    // Python: if input is int, should return default_float
    result = least_upper_float(&dtypes.int32);
    ASSERT(dtype_eq(&result, &dtypes.default_float));
    
    result = least_upper_float(&dtypes.uint8);
    ASSERT(dtype_eq(&result, &dtypes.default_float));
}

// Test aliases work correctly
void test_dtype_aliases() {
    printf("Testing DType aliases...\n");
    
    // Test aliases match Python definitions
    ASSERT(dtype_eq(&dtypes.half, &dtypes.float16));
    ASSERT(dtype_eq(&dtypes.float_, &dtypes.float32));
    ASSERT(dtype_eq(&dtypes.double_, &dtypes.float64));
    ASSERT(dtype_eq(&dtypes.uchar, &dtypes.uint8));
    ASSERT(dtype_eq(&dtypes.ushort, &dtypes.uint16));
    ASSERT(dtype_eq(&dtypes.uint, &dtypes.uint32));
    ASSERT(dtype_eq(&dtypes.ulong, &dtypes.uint64));
    ASSERT(dtype_eq(&dtypes.char_, &dtypes.int8));
    ASSERT(dtype_eq(&dtypes.short_, &dtypes.int16));
    ASSERT(dtype_eq(&dtypes.int_, &dtypes.int32));
    ASSERT(dtype_eq(&dtypes.long_, &dtypes.int64));
}

// Test string conversion
void test_string_conversion() {
    printf("Testing string conversion...\n");
    
    // Test to_dtype function
    DType result = to_dtype("float32");
    ASSERT(dtype_eq(&result, &dtypes.float32));
    
    result = to_dtype("int32");
    ASSERT(dtype_eq(&result, &dtypes.int32));
    
    result = to_dtype("bool");
    ASSERT(dtype_eq(&result, &dtypes.bool_));
    
    // Test aliases
    result = to_dtype("float");
    ASSERT(dtype_eq(&result, &dtypes.float32));
    
    result = to_dtype("int");
    ASSERT(dtype_eq(&result, &dtypes.int32));
}

int main() {
    printf("Running DType tests based on Python reference tests...\n\n");
    
    // Initialize the dtype system
    dtypes_init();
    
    // Run tests based on Python test suite
    test_dtype_creation();
    test_dtype_type_checks();
    test_dtype_vectorization();
    test_dtype_range();
    test_dtype_finfo();
    test_dtype_from_py();
    test_truncate_fp16();
    test_dtype_promotion();
    test_sum_acc_dtype();
    test_least_upper_float();
    test_dtype_aliases();
    test_string_conversion();
    
    // Cleanup
    dtypes_cleanup();
    
    printf("\nTest Results: %d/%d passed\n", tests_passed, tests_run);
    if (tests_passed != tests_run) {
        printf("FAILED: Some tests did not pass. Implementation needs fixes.\n");
        return 1;
    }
    printf("SUCCESS: All tests passed!\n");
    return 0;
}