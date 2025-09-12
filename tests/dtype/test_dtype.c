#include "test_common.h"
#include <assert.h>
#include "dtype/dtype.h"
#include "helpers/helpers.h"
#include <float.h>

#define FP8E4M3_MAX 448.0
#define FP8E5M2_MAX 57344.0

void setUp(void) { dtypes_init(); }
void tearDown(void) { dtypes_cleanup(); }

TEST(test_dtype_creation) {
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

TEST(test_dtype_type_checks) {
  TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.float32));
  TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.float64));
  TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.float16));
  TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.bfloat16));
  TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.fp8e4m3));
  TEST_ASSERT_TRUE(dtypes_is_float(&dtypes.fp8e5m2));
  TEST_ASSERT_FALSE(dtypes_is_float(&dtypes.int32));
  TEST_ASSERT_FALSE(dtypes_is_float(&dtypes.bool_));

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

  TEST_ASSERT_TRUE(dtypes_is_bool(&dtypes.bool_));
  TEST_ASSERT_FALSE(dtypes_is_bool(&dtypes.int32));
  TEST_ASSERT_FALSE(dtypes_is_bool(&dtypes.float32));
}

TEST(test_dtype_vectorization) {
  DType vec2 = dtype_vec(&dtypes.float32, 2);
  TEST_ASSERT_EQUAL(2, vec2.count);
  TEST_ASSERT_EQUAL(8, vec2.itemsize);
  TEST_ASSERT_EQUAL_PTR(&dtypes.float32, vec2._scalar);
  DType vec1 = dtype_vec(&dtypes.float32, 1);
  TEST_ASSERT_TRUE(dtype_eq(&vec1, &dtypes.float32));
  DType void_vec = dtype_vec(&dtypes.void_, 4);
  TEST_ASSERT_TRUE(dtype_eq(&void_vec, &dtypes.void_));
  DType scalar_from_vec = dtype_scalar(&vec2);
  TEST_ASSERT_TRUE(dtype_eq(&scalar_from_vec, &dtypes.float32));
  DType scalar_from_scalar = dtype_scalar(&dtypes.float32);
  TEST_ASSERT_TRUE(dtype_eq(&scalar_from_scalar, &dtypes.float32));
}

TEST(test_dtype_range) {
  TEST_ASSERT_TRUE(isinf(dtypes_min(&dtypes.float32)) && dtypes_min(&dtypes.float32) < 0);
  TEST_ASSERT_TRUE(isinf(dtypes_max(&dtypes.float32)) && dtypes_max(&dtypes.float32) > 0);
  TEST_ASSERT_TRUE(isinf(dtypes_min(&dtypes.float64)) && dtypes_min(&dtypes.float64) < 0);
  TEST_ASSERT_TRUE(isinf(dtypes_max(&dtypes.float64)) && dtypes_max(&dtypes.float64) > 0);
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
  TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, dtypes_min(&dtypes.bool_));
  TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, dtypes_max(&dtypes.bool_));
}

TEST(test_dtype_finfo) {
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

TEST(test_dtype_from_py) {
  DType dt_bool = dtypes_from_py_bool(true);
  TEST_ASSERT_TRUE(dtype_eq(&dt_bool, &dtypes.bool_));
  DType dt_int = dtypes_from_py_int(2);
  TEST_ASSERT_TRUE(dtype_eq(&dt_int, dtypes.default_int));
  DType dt_float = dtypes_from_py_float(3.0);
  TEST_ASSERT_TRUE(dtype_eq(&dt_float, dtypes.default_float));
}

TEST_MAIN()

