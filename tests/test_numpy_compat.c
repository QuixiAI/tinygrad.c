#include "test_common.h"
#include "compat/numpy.h"

TEST(test_empty_zeros_ones) {
  size_t shp[2] = {2,3};
  np_array_t* e = np_empty(2, shp, &dtypes.float32);
  TEST_ASSERT_NOT_NULL(e);
  TEST_ASSERT_EQUAL_UINT64(6, e->size);
  np_free(e);

  np_array_t* z = np_zeros(2, shp, &dtypes.float32);
  float *zd = (float*)np_data(z);
  for(size_t i=0;i<z->size;i++) TEST_ASSERT_FLOAT_WITHIN(1e-7f, 0.0f, zd[i]);
  np_free(z);

  np_array_t* o = np_ones(2, shp, &dtypes.float32);
  float *od = (float*)np_data(o);
  for(size_t i=0;i<o->size;i++) TEST_ASSERT_FLOAT_WITHIN(1e-7f, 1.0f, od[i]);
  np_free(o);
}

TEST(test_frombuffer_and_data) {
  float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  np_array_t* a = np_frombuffer(buf, sizeof(buf), &dtypes.float32);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_EQUAL_UINT64(4, a->size);
  TEST_ASSERT_EQUAL_UINT64(1, a->ndim);
  TEST_ASSERT_EQUAL_UINT64(4, a->shape[0]);
  TEST_ASSERT_EQUAL_PTR(buf, np_data(a));
  // free array metadata but not external buffer
  np_free(a);
}

TEST(test_array_copy_independent) {
  size_t shp[2] = {2,3};
  float src[6] = {0,1,2,3,4,5};
  np_array_t* a = np_array_copy(2, shp, &dtypes.float32, src);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_EQUAL_UINT64(6, a->size);
  float *ad = (float*)np_data(a);
  for (size_t i=0;i<6;i++) TEST_ASSERT_FLOAT_WITHIN(1e-7f, (float)i, ad[i]);
  src[0] = -1.0f;
  TEST_ASSERT_FLOAT_WITHIN(1e-7f, 0.0f, ad[0]);
  np_free(a);
}

TEST(test_dtype_helpers) {
  const char* n = np_dtype_name(&dtypes.float32);
  TEST_ASSERT_NOT_NULL(n);
  const DType* dt = np_dtype_from_name(n);
  TEST_ASSERT_TRUE(dtype_eq(dt, &dtypes.float32));
  const DType* di = np_dtype_from_name(np_dtype_name(&dtypes.int32));
  TEST_ASSERT_TRUE(dtype_eq(di, &dtypes.int32));
}

TEST(test_allclose_and_testing) {
  size_t shp[1] = {3};
  float av[3] = {1.0f, 2.0f, 3.0f};
  float bv[3] = {1.0f + 1e-5f, 2.0f, 3.0f};
  np_array_t* a = np_array_copy(1, shp, &dtypes.float32, av);
  np_array_t* b = np_array_copy(1, shp, &dtypes.float32, bv);
  TEST_ASSERT_TRUE(np_allclose(a,b,1e-4,1e-6));
  TEST_ASSERT_EQUAL_INT(0, np_testing.assert_allclose(a,b,1e-4,1e-6));
  ((float*)np_data(b))[0] = 10.0f;
  TEST_ASSERT_FALSE(np_allclose(a,b,1e-4,1e-6));
  TEST_ASSERT_TRUE(np_testing.assert_allclose(a,b,1e-4,1e-6) != 0);
  np_free(a); np_free(b);
}

TEST(test_require_c_contiguous) {
  size_t shp[1] = {5};
  np_array_t* a = np_zeros(1, shp, &dtypes.float32);
  np_array_t* c = np_require_c_contiguous(a);
  TEST_ASSERT_EQUAL_PTR(a, c);
  TEST_ASSERT_EQUAL_PTR(np_data(a), np_data(c));
  np_free(a);
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()
