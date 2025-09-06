#include "test_common.h"
#include <assert.h>
#include "dtype/dtype.h"
#include "helpers/helpers.h"
#include <float.h>

// Constants from test_dtype_spec.py
#define FP8E4M3_MAX 448.0
#define FP8E5M2_MAX 57344.0

void setUp(void) {
    // Initialize the dtype system
    dtypes_init();
}

void tearDown(void) {
    // Cleanup
    dtypes_cleanup();
}

// Test basic DType creation - based on reference/test/test_dtype.py
TEST(test_dtype_creation) {
    
    // Test basic types match Python definitions exactly
    TEST_ASSERT_EQUAL(0, dtypes.bool_.priority);
    TEST_ASSERT_EQUAL(1, dtypes.bool_.itemsize);
    TEST_ASSERT_EQUAL_STRING("bool", dtypes.bool_.name);
    TEST_ASSERT_EQUAL('?', dtypes.bool_.fmt);
    TEST_ASSERT_EQUAL(1, dtypes.bool_.count);
    TEST_ASSERT_NULL(dtypes.bool_._scalar);
    
    TEST_ASSERT_EQUAL(5, dtypes.int32.priority);
    TEST_ASSERT_EQUAL(4, dtypes.int32.itemsize);
    TEST_ASSERT_EQUAL_STRING("int", dtypes.int32.name);
    TEST_ASSERT_EQUAL('i', dtypes.int32.fmt);
    
    TEST_ASSERT_EQUAL(13, dtypes.float32.priority);
    TEST_ASSERT_EQUAL(4, dtypes.float32.itemsize);
    TEST_ASSERT_EQUAL_STRING("float", dtypes.float32.name);
    TEST_ASSERT_EQUAL('f', dtypes.float32.fmt);
    
    // Test all type priorities match Python version
    TEST_ASSERT_EQUAL(-1, dtypes.void_.priority);
    TEST_ASSERT_EQUAL(1, dtypes.int8.priority);
    TEST_ASSERT_EQUAL(2, dtypes.uint8.priority);
    TEST_ASSERT_EQUAL(3, dtypes.int16.priority);
    TEST_ASSERT_EQUAL(4, dtypes.uint16.priority);
    TEST_ASSERT_EQUAL(6, dtypes.uint32.priority);
    TEST_ASSERT_EQUAL(7, dtypes.int64.priority);
    TEST_ASSERT_EQUAL(8, dtypes.uint64.priority);
    TEST_ASSERT_EQUAL(9, dtypes.fp8e4m3.priority);
    TEST_ASSERT_EQUAL(10, dtypes.fp8e5m2.priority);
    TEST_ASSERT_EQUAL(11, dtypes.float16.priority);
    TEST_ASSERT_EQUAL(12, dtypes.bfloat16.priority);
    TEST_ASSERT_EQUAL(14, dtypes.float64.priority);
}

// Test type checking functions based on reference/test/unit/test_dtype_spec.py
TEST(test_dtype_type_checks) {
    
    // Test is_float
    TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.float32));
    TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.float64));
    TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.float16));
    TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.bfloat16));
    TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.fp8e4m3));
    TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.fp8e5m2));
    TEST_ASSERT_FALSE(dtypes_is_float(&dtypes.int32));
    TEST_ASSERT_FALSE(dtypes_is_float(&dtypes.bool_));
    
    // Test is_int  
    TEST_ASSERT_TRUE(dtypes_is_int(&dtypes.int32));
    TEST_ASSERT_TRUE(dtypes_is_int(&dtypes.uint32));
    TEST_ASSERT_TRUE(dtypes_is_int(&dtypes.int8));
    TEST_ASSERT_TRUE(dtypes_is_int(&dtypes.uint8));
    TEST_ASSERT_TRUE(dtypes_is_int(&dtypes.int16));
    TEST_ASSERT_TRUE(dtypes_is_int(&dtypes.uint16));
    TEST_ASSERT_TRUE(dtypes_is_int(&dtypes.int64));
    TEST_ASSERT_TRUE(dtypes_is_int(&dtypes.uint64));
    TEST_ASSERT_FALSE(dtypes_is_int(&dtypes.float32));
    TEST_ASSERT_FALSE(dtypes_is_int(&dtypes.bool_));
    
    // Test is_unsigned
    TEST_ASSERT_TRUE(dtypes_is_unsigned(&dtypes.uint32));
    TEST_ASSERT_TRUE(dtypes_is_unsigned(&dtypes.uint8));
    TEST_ASSERT_TRUE(dtypes_is_unsigned(&dtypes.uint16));
    TEST_ASSERT_TRUE(dtypes_is_unsigned(&dtypes.uint64));
    TEST_ASSERT_FALSE(dtypes_is_unsigned(&dtypes.int32));
    TEST_ASSERT_FALSE(dtypes_is_unsigned(&dtypes.int8));
    TEST_ASSERT_FALSE(dtypes_is_unsigned(&dtypes.int16));
    TEST_ASSERT_FALSE(dtypes_is_unsigned(&dtypes.int64));
    TEST_ASSERT_FALSE(dtypes_is_unsigned(&dtypes.float32));
    TEST_ASSERT_FALSE(dtypes_is_unsigned(&dtypes.bool_));
    
    // Test is_bool
    TEST_ASSERT_TRUE(dtypes_is_bool(&dtypes.bool_));
    TEST_ASSERT_FALSE(dtypes_is_bool(&dtypes.int32));
    TEST_ASSERT_FALSE(dtypes_is_bool(&dtypes.float32));
}

// Test vectorization with scalar extraction
TEST(test_dtype_vectorization) {
    
    // Test vec() method
    DType vec2 = dtype_vec(&dtypes.float32, 2);
    TEST_ASSERT_EQUAL(2, vec2.count);
    TEST_ASSERT_EQUAL(8, vec2.itemsize); // 4 * 2
    TEST_ASSERT_EQUAL_PTR(&dtypes.float32, vec2._scalar);
    
    // Test vectorizing sz=1 returns original (Python: if sz == 1 or self == dtypes.void: return self)
    DType vec1 = dtype_vec(&dtypes.float32, 1);
    TEST_ASSERT_TRUE(dtype_eq(&vec1, &dtypes.float32));
    
    // Test void doesn't vectorize
    DType void_vec = dtype_vec(&dtypes.void_, 4);
    TEST_ASSERT_TRUE(dtype_eq(&void_vec, &dtypes.void_));
    
    // Test scalar() method (Python: return self._scalar if self._scalar is not None else self)
    DType scalar_from_vec = dtype_scalar(&vec2);
    TEST_ASSERT_TRUE(dtype_eq(&scalar_from_vec, &dtypes.float32));
    
    DType scalar_from_scalar = dtype_scalar(&dtypes.float32);
    TEST_ASSERT_TRUE(dtype_eq(&scalar_from_scalar, &dtypes.float32));
}

// Test dtype range values - based on test_dtype_spec.py:test_dtype_range
TEST(test_dtype_range) {
    
    // Test float types (should be -inf/+inf)
    TEST_ASSERT_TRUE(isinf(dtypes_min(&dtypes.float32)) && dtypes_min(&dtypes.float32) < 0);
    TEST_ASSERT_TRUE(isinf(dtypes_max(&dtypes.float32)) && dtypes_max(&dtypes.float32) > 0);
    TEST_ASSERT_TRUE(isinf(dtypes_min(&dtypes.float64)) && dtypes_min(&dtypes.float64) < 0);
    TEST_ASSERT_TRUE(isinf(dtypes_max(&dtypes.float64)) && dtypes_max(&dtypes.float64) > 0);
    
    // Test integer types - must match numpy's iinfo exactly
    TEST_ASSERT_EQUAL_INT64(-128, (int64_t)dtypes_min(&dtypes.int8));
    TEST_ASSERT_EQUAL_INT64(127, (int64_t)dtypes_max(&dtypes.int8));
    TEST_ASSERT_EQUAL_UINT64(0, (uint64_t)dtypes_min(&dtypes.uint8));
    TEST_ASSERT_EQUAL_UINT64(255, (uint64_t)dtypes_max(&dtypes.uint8));
    
    TEST_ASSERT_EQUAL_INT64(-32768, (int64_t)dtypes_min(&dtypes.int16));
    TEST_ASSERT_EQUAL_INT64(32767, (int64_t)dtypes_max(&dtypes.int16));
    TEST_ASSERT_EQUAL_UINT64(0, (uint64_t)dtypes_min(&dtypes.uint16));
    TEST_ASSERT_EQUAL_UINT64(65535, (uint64_t)dtypes_max(&dtypes.uint16));
    
    TEST_ASSERT_EQUAL_INT64(-2147483648LL, (int64_t)dtypes_min(&dtypes.int32));
    TEST_ASSERT_EQUAL_INT64(2147483647LL, (int64_t)dtypes_max(&dtypes.int32));
    TEST_ASSERT_EQUAL_UINT64(0ULL, (uint64_t)dtypes_min(&dtypes.uint32));
    TEST_ASSERT_EQUAL_UINT64(4294967295ULL, (uint64_t)dtypes_max(&dtypes.uint32));
    
    // Test bool (False=0, True=1)
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, dtypes_min(&dtypes.bool_)); // False
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, dtypes_max(&dtypes.bool_)); // True
}

// Test finfo function - based on test_dtype_spec.py
TEST(test_dtype_finfo) {
    
    // Test values from Python: dtypes.finfo() returns (exponent, mantissa)
    FInfo info_f16 = dtypes_finfo(&dtypes.float16);
    TEST_ASSERT_EQUAL(5, info_f16.exponent);
    TEST_ASSERT_EQUAL(10, info_f16.mantissa);
    
    FInfo info_bf16 = dtypes_finfo(&dtypes.bfloat16);
    TEST_ASSERT_EQUAL(8, info_bf16.exponent);
    TEST_ASSERT_EQUAL(7, info_bf16.mantissa);
    
    FInfo info_f32 = dtypes_finfo(&dtypes.float32);
    TEST_ASSERT_EQUAL(8, info_f32.exponent);
    TEST_ASSERT_EQUAL(23, info_f32.mantissa);
    
    FInfo info_f64 = dtypes_finfo(&dtypes.float64);
    TEST_ASSERT_EQUAL(11, info_f64.exponent);
    TEST_ASSERT_EQUAL(52, info_f64.mantissa);
    
    FInfo info_fp8e5m2 = dtypes_finfo(&dtypes.fp8e5m2);
    TEST_ASSERT_EQUAL(5, info_fp8e5m2.exponent);
    TEST_ASSERT_EQUAL(2, info_fp8e5m2.mantissa);
    
    FInfo info_fp8e4m3 = dtypes_finfo(&dtypes.fp8e4m3);
    TEST_ASSERT_EQUAL(4, info_fp8e4m3.exponent);
    TEST_ASSERT_EQUAL(3, info_fp8e4m3.mantissa);
}

// Test from_py functionality - based on test_dtype_spec.py:test_from_py
TEST(test_dtype_from_py) {
    
    // Python: assert dtypes.from_py(True) == dtypes.bool
    DType dt_bool = dtypes_from_py_bool(true);
    TEST_ASSERT_TRUE(dtype_eq(&dt_bool, &dtypes.bool_));
    
    // Python: assert dtypes.from_py(2) == dtypes.default_int
    DType dt_int = dtypes_from_py_int(2);
    TEST_ASSERT_TRUE(dtype_eq(&dt_int, dtypes.default_int));
    
    // Python: assert dtypes.from_py(3.0) == dtypes.default_float
    DType dt_float = dtypes_from_py_float(3.0);
    TEST_ASSERT_TRUE(dtype_eq(&dt_float, dtypes.default_float));
}

// Test truncate_fp16 - based on test_dtype_spec.py:test_truncate_fp16
TEST(test_truncate_fp16) {
    
    // Python test cases:
    // self.assertEqual(truncate_fp16(1), 1)
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0, truncate_fp16(1.0));
    
    // self.assertEqual(truncate_fp16(65504), 65504)
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 65504.0, truncate_fp16(65504.0));
    
    // self.assertEqual(truncate_fp16(65520), math.inf)
    TEST_ASSERT_TRUE(isinf(truncate_fp16(65520.0)));
    
    // Test overflow to infinity
    TEST_ASSERT_TRUE(isinf(truncate_fp16(1e10)));
    
    // Test sign preservation
    TEST_ASSERT_TRUE(truncate_fp16(-1.0) < 0);
    TEST_ASSERT_TRUE(isinf(truncate_fp16(-1e10)) && truncate_fp16(-1e10) < 0);
}

// Test type promotion - based on test_dtype_spec.py:test_dtype_promo  
TEST(test_dtype_promotion) {
    
    // Python test cases from test_dtype_promo() - these must match exactly:
    
    // assert least_upper_dtype(dtypes.bool, dtypes.int8) == dtypes.int8
    DType result = least_upper_dtype(&dtypes.bool_, &dtypes.int8);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int8));
    
    // assert least_upper_dtype(dtypes.int8, dtypes.uint8) == dtypes.int16
    result = least_upper_dtype(&dtypes.int8, &dtypes.uint8);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int16));
    
    // assert least_upper_dtype(dtypes.uint8, dtypes.int16) == dtypes.int16
    result = least_upper_dtype(&dtypes.uint8, &dtypes.int16);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int16));
    
    // assert least_upper_dtype(dtypes.int16, dtypes.uint16) == dtypes.int32
    result = least_upper_dtype(&dtypes.int16, &dtypes.uint16);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int32));
    
    // assert least_upper_dtype(dtypes.uint16, dtypes.int32) == dtypes.int32
    result = least_upper_dtype(&dtypes.uint16, &dtypes.int32);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int32));
    
    // assert least_upper_dtype(dtypes.int32, dtypes.uint32) == dtypes.int64
    result = least_upper_dtype(&dtypes.int32, &dtypes.uint32);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int64));
    
    // assert least_upper_dtype(dtypes.uint32, dtypes.int64) == dtypes.int64
    result = least_upper_dtype(&dtypes.uint32, &dtypes.int64);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int64));
    
    // assert least_upper_dtype(dtypes.int64, dtypes.uint64) == dtypes.float16
    result = least_upper_dtype(&dtypes.int64, &dtypes.uint64);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float16));
    
    // assert least_upper_dtype(dtypes.float16, dtypes.float32) == dtypes.float32
    result = least_upper_dtype(&dtypes.float16, &dtypes.float32);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float32));
    
    // assert least_upper_dtype(dtypes.float32, dtypes.float64) == dtypes.float64
    result = least_upper_dtype(&dtypes.float32, &dtypes.float64);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float64));
    
    // assert least_upper_dtype(dtypes.bool, dtypes.float32) == dtypes.float32
    result = least_upper_dtype(&dtypes.bool_, &dtypes.float32);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float32));
    
    // assert least_upper_dtype(dtypes.bool, dtypes.float64) == dtypes.float64
    result = least_upper_dtype(&dtypes.bool_, &dtypes.float64);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float64));
    
    // assert least_upper_dtype(dtypes.float16, dtypes.int64) == dtypes.float16
    result = least_upper_dtype(&dtypes.float16, &dtypes.int64);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float16));
    
    // assert least_upper_dtype(dtypes.float16, dtypes.uint64) == dtypes.float16
    result = least_upper_dtype(&dtypes.float16, &dtypes.uint64);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float16));
    
    // assert least_upper_dtype(dtypes.fp8e4m3, dtypes.fp8e5m2) == dtypes.half
    result = least_upper_dtype(&dtypes.fp8e4m3, &dtypes.fp8e5m2);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float16));
}

// Test sum accumulator dtype - based on test_dtype_alu.py:test_sum
TEST(test_sum_acc_dtype) {
    
    // From Python test_sum():
    // assert (Tensor([0, 1], dtype=dtypes.bool)).sum().dtype == dtypes.int32
    DType result = sum_acc_dtype(&dtypes.bool_);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int32));
    
    // assert (Tensor([0, 1], dtype=dtypes.int8)).sum().dtype == dtypes.int32
    result = sum_acc_dtype(&dtypes.int8);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int32));
    
    // assert (Tensor([0, 1], dtype=dtypes.uint8)).sum().dtype == dtypes.uint32
    result = sum_acc_dtype(&dtypes.uint8);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.uint32));
    
    // assert (Tensor([0, 1], dtype=dtypes.int64)).sum().dtype == dtypes.int64
    result = sum_acc_dtype(&dtypes.int64);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int64));
    
    // assert (Tensor([0, 1], dtype=dtypes.uint64)).sum().dtype == dtypes.uint64
    result = sum_acc_dtype(&dtypes.uint64);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.uint64));
    
    // In Python, float16 sum uses float32 as accumulator (higher priority)
    result = sum_acc_dtype(&dtypes.float16);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float32));
    
    // assert (Tensor([0, 1], dtype=dtypes.float32)).sum().dtype == dtypes.float32
    result = sum_acc_dtype(&dtypes.float32);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float32));
}

// Test least_upper_float - based on test_dtype_spec.py:TestAutoCastType
TEST(test_least_upper_float) {
    
    // Python: if input is float, should return input unchanged
    DType result = least_upper_float(&dtypes.float32);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float32));
    
    result = least_upper_float(&dtypes.float16);
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float16));
    
    // Python: if input is int, should return default_float
    result = least_upper_float(&dtypes.int32);
    TEST_ASSERT_TRUE(dtype_eq(&result, dtypes.default_float));
    
    result = least_upper_float(&dtypes.uint8);
    TEST_ASSERT_TRUE(dtype_eq(&result, dtypes.default_float));
}

// Test aliases work correctly
TEST(test_dtype_aliases) {
    
    // Test aliases match Python definitions
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.half, &dtypes.float16));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.float_, &dtypes.float32));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.double_, &dtypes.float64));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.uchar, &dtypes.uint8));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.ushort, &dtypes.uint16));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.uint, &dtypes.uint32));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.ulong, &dtypes.uint64));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.char_, &dtypes.int8));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.short_, &dtypes.int16));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.int_, &dtypes.int32));
    TEST_ASSERT_TRUE(dtype_eq(&dtypes.long_, &dtypes.int64));
}

// Test string conversion
TEST(test_string_conversion) {
    
    // Test to_dtype function
    DType result = to_dtype("float32");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float32));
    
    result = to_dtype("int32");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int32));
    
    result = to_dtype("bool");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.bool_));
    
    // Test aliases
    result = to_dtype("float");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float32));
    
    result = to_dtype("int");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int32));

    // Additional alias and canonical coverage
    result = to_dtype("double");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float64));
    result = to_dtype("half");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.float16));
    result = to_dtype("uchar");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.uint8));
    result = to_dtype("long");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.int64));
    result = to_dtype("bfloat16");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.bfloat16));
    result = to_dtype("fp8e4m3");
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.fp8e4m3));
    result = to_dtype("Fp8E5M2"); // mixed case
    TEST_ASSERT_TRUE(dtype_eq(&result, &dtypes.fp8e5m2));
}

TEST(test_comprehensive_dtype_features) {
    // Test ImageDType creation
    int shape[] = {224, 224, 3};
    ImageDType img_h = dtypes_imageh(shape, 3);
    TEST_ASSERT_EQUAL_STRING("imageh", img_h.ptr_base.base.name);
    TEST_ASSERT_EQUAL(224 * 224 * 3, img_h.ptr_base.size);
    TEST_ASSERT_TRUE(dtypes_is_float((const DType*)&img_h.ptr_base.base));
    
    ImageDType img_f = dtypes_imagef(shape, 3);
    TEST_ASSERT_EQUAL_STRING("imagef", img_f.ptr_base.base.name);
    TEST_ASSERT_EQUAL(224 * 224 * 3, img_f.ptr_base.size);
    TEST_ASSERT_TRUE(dtypes_is_float((const DType*)&img_f.ptr_base.base));
    
    // Test canonical names (DTYPES_DICT/INVERSE_DTYPES_DICT equivalent)
    TEST_ASSERT_EQUAL_STRING("int8", dtype_canonical_name(&dtypes.int8));
    TEST_ASSERT_EQUAL_STRING("float32", dtype_canonical_name(&dtypes.float32));
    TEST_ASSERT_EQUAL_STRING("bool", dtype_canonical_name(&dtypes.bool_));
    
    // Test as_const with truncation
    double truncated = dtypes_as_const_float(300.7, &dtypes.uint8);
    TEST_ASSERT_DOUBLE_WITHIN(0.1, 44.0, truncated); // 300 % 256 = 44
    
    truncated = dtypes_as_const_float(1.5, &dtypes.float16);
    TEST_ASSERT_DOUBLE_WITHIN(0.01, 1.5, truncated);
}

// Additional parity tests for vec-naming and PtrDType.vec behavior
TEST(test_vec_name_and_cache) {
    DType v4a = dtype_vec(&dtypes.float32, 4);
    DType v4b = dtype_vec(&dtypes.float32, 4);
    // name must be canonical base + count, like "float324"
    TEST_ASSERT_EQUAL_STRING("float324", dtype_name(&v4a));
    // equality on repeated calls
    TEST_ASSERT_TRUE(dtype_eq(&v4a, &v4b));
}

TEST(test_ptrdtype_vec_basic) {
    PtrDType p = dtype_ptr(&dtypes.float32, 16, ADDRSPACE_GLOBAL);
    PtrDType pv = ptrdtype_vec(&p, 8);
    TEST_ASSERT_EQUAL_INT(8, pv.v);
    TEST_ASSERT_EQUAL_INT(16, pv.size);
    TEST_ASSERT_TRUE(dtype_eq(&pv.base, &p.base));
}

TEST(test_ptrdtype_vec_image) {
    int shp[2] = {2,2};
    ImageDType im = dtypes_imageh(shp, 2);
    PtrDType pv = ptrdtype_vec(&im.ptr_base, 8);
    TEST_ASSERT_EQUAL_INT(8, pv.v);
    TEST_ASSERT_EQUAL_STRING("imageh", dtype_name(&pv.base));
}

// Image LUB short-circuit and multi-arg parity
TEST(test_lub_image_short_circuit) {
    int shp[1] = {4};
    ImageDType im = dtypes_imagef(shp, 1);
    DType r1 = least_upper_dtype((const DType*)&im.ptr_base.base, &dtypes.int32);
    TEST_ASSERT_EQUAL_STRING("imagef", dtype_name(&r1));
    const DType* arr[3] = { (const DType*)&im.ptr_base.base, &dtypes.uint8, &dtypes.float16 };
    DType r2 = least_upper_dtype_multi(arr, 3);
    TEST_ASSERT_EQUAL_STRING("imagef", dtype_name(&r2));
}

// FP16 rounding: verify rounding slightly above midpoint goes up
TEST(test_truncate_fp16_rounding_edges) {
    double half_ulp = 0.00048828125; // 2^-11
    double up = 0.0000001; // small epsilon above midpoint
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0009765625f, truncate_fp16(1.0 + half_ulp + up)); // 1 + 2^-10
}

// Test collections ordering/content against Python's definitions
TEST(test_collections_ordering) {
    // fp8s ordering
    TEST_ASSERT_TRUE(dtypes.fp8s[0] == &dtypes.fp8e4m3);
    TEST_ASSERT_TRUE(dtypes.fp8s[1] == &dtypes.fp8e5m2);
    // floats ordering
    TEST_ASSERT_TRUE(dtypes.floats[0] == &dtypes.fp8e4m3);
    TEST_ASSERT_TRUE(dtypes.floats[1] == &dtypes.fp8e5m2);
    TEST_ASSERT_TRUE(dtypes.floats[2] == &dtypes.float16);
    TEST_ASSERT_TRUE(dtypes.floats[3] == &dtypes.bfloat16);
    TEST_ASSERT_TRUE(dtypes.floats[4] == &dtypes.float32);
    TEST_ASSERT_TRUE(dtypes.floats[5] == &dtypes.float64);
    // ints ordering (uints then sints)
    TEST_ASSERT_TRUE(dtypes.ints[0] == &dtypes.uint8);
    TEST_ASSERT_TRUE(dtypes.ints[1] == &dtypes.uint16);
    TEST_ASSERT_TRUE(dtypes.ints[2] == &dtypes.uint32);
    TEST_ASSERT_TRUE(dtypes.ints[3] == &dtypes.uint64);
    TEST_ASSERT_TRUE(dtypes.ints[4] == &dtypes.int8);
    TEST_ASSERT_TRUE(dtypes.ints[5] == &dtypes.int16);
    TEST_ASSERT_TRUE(dtypes.ints[6] == &dtypes.int32);
    TEST_ASSERT_TRUE(dtypes.ints[7] == &dtypes.int64);
}

// (optional) can_safe_cast mapping is verified indirectly by dtype promotion tests.

// FP8 conversion sanity: round-trip and saturation behaviors
TEST(test_fp8_conversions_sanity) {
    // e4m3
    float z = (float)dtypes_truncate(0.0, &dtypes.fp8e4m3);
    TEST_ASSERT_FLOAT_WITHIN(1e-8f, 0.0f, z);
    float one = (float)dtypes_truncate(1.0, &dtypes.fp8e4m3);
    TEST_ASSERT_TRUE(fabsf(one-1.0f) < 5e-2f); // within ~5e-2 quantization
    float tiny = (float)dtypes_truncate(1e-10, &dtypes.fp8e4m3);
    TEST_ASSERT_FLOAT_WITHIN(1e-8f, 0.0f, tiny);
    float big = (float)dtypes_truncate(1e6, &dtypes.fp8e4m3);
    TEST_ASSERT_TRUE(isfinite(big));
    // e5m2
    z = (float)dtypes_truncate(0.0, &dtypes.fp8e5m2);
    TEST_ASSERT_FLOAT_WITHIN(1e-8f, 0.0f, z);
    one = (float)dtypes_truncate(1.0, &dtypes.fp8e5m2);
    TEST_ASSERT_TRUE(fabsf(one-1.0f) < 1e-1f);
    tiny = (float)dtypes_truncate(1e-10, &dtypes.fp8e5m2);
    TEST_ASSERT_FLOAT_WITHIN(1e-8f, 0.0f, tiny);
    big = (float)dtypes_truncate(1e8, &dtypes.fp8e5m2);
    TEST_ASSERT_TRUE(isfinite(big));
}


// Auto-register all test functions and run them
TEST_MAIN()
