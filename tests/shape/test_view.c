#include "test_common.h"
#include "shape/view.h"
#include <string.h>
#include <stdlib.h>

// Unity requires these functions
void setUp(void) {
    // Setup before each test
}

void tearDown(void) {
    // Cleanup after each test
}

// Test View creation and basic properties
TEST(test_view_create) {
    int32_t shape[] = {2, 2, 2};
    int32_t strides[] = {4, 2, 1};
    View *v = view_create(shape, 3, strides, 3, 0);
    TEST_ASSERT_NOT_NULL(v);
    view_free(v);
}

// Test canonicalize empty mask
TEST(test_canonicalize_empty_mask) {
    int32_t shape[] = {2, 2, 2};
    int32_t strides[] = {4, 2, 1};
    int32_t mask_start[] = {0, 0, 0};
    int32_t mask_end[] = {2, 2, 2};
    View *v = view_create_with_mask(shape, 3, strides, 3, 0, mask_start, mask_end, 3);
    TEST_ASSERT_NOT_NULL(v);
    // Mask should be None when it covers the full shape
    TEST_ASSERT_NULL(view_mask(v));
    view_free(v);
}

// Test minify zero strided dims
TEST(test_minify_zero_strided_dims) {
    // Target: (2,2) strides=(30,2) offset=7
    int32_t target_shape[] = {2, 2};
    int32_t target_strides[] = {30, 2};
    View *target = view_create(target_shape, 2, target_strides, 2, 7);
    TEST_ASSERT_NOT_NULL(target);
    
    // Test: (2,1,2) strides=(30,0,2) offset=7 -> should minify to target
    int32_t shape[] = {2, 1, 2};
    int32_t strides[] = {30, 0, 2};
    View *v = view_create(shape, 3, strides, 3, 7);
    TEST_ASSERT_NOT_NULL(v);
    
    View *minified = view_minify(v);
    TEST_ASSERT_NOT_NULL(minified);
    
    // Should have same shape as target after minify
    const int32_t *min_shape = view_shape(minified);
    TEST_ASSERT_EQUAL(2, view_ndim(minified));
    TEST_ASSERT_EQUAL(2, min_shape[0]);
    TEST_ASSERT_EQUAL(2, min_shape[1]);
    
    view_free(v);
    view_free(target);
    view_free(minified);
}

// Test empty mask contiguous
TEST(test_empty_mask_contiguous) {
    int32_t shape[] = {2, 2, 2};
    int32_t strides[] = {4, 2, 1};
    
    // View without mask
    View *v1 = view_create(shape, 3, strides, 3, 0);
    TEST_ASSERT_NOT_NULL(v1);
    
    // View with full mask (should be equivalent)
    int32_t mask_start[] = {0, 0, 0};
    int32_t mask_end[] = {2, 2, 2};
    View *v2 = view_create_with_mask(shape, 3, strides, 3, 0, mask_start, mask_end, 3);
    TEST_ASSERT_NOT_NULL(v2);
    
    // Both should have same contiguous property
    TEST_ASSERT_EQUAL(view_contiguous(v1), view_contiguous(v2));
    
    view_free(v1);
    view_free(v2);
}

// Test reshape all invalid
TEST(test_reshape_all_invalid) {
    int32_t shape[] = {4, 5};
    int32_t mask_start[] = {0, 0};
    int32_t mask_end[] = {0, 0}; // Empty mask
    View *v = view_create_with_mask(shape, 2, NULL, 0, 0, mask_start, mask_end, 2);
    TEST_ASSERT_NOT_NULL(v);
    
    int32_t new_shape[] = {20};
    View *reshaped = view_reshape(v, new_shape, 1);
    TEST_ASSERT_NOT_NULL(reshaped);
    
    // Should have empty mask after reshape
    const int32_t *r_mask = view_mask_ranges(reshaped);
    TEST_ASSERT_NOT_NULL(r_mask);
    
    view_free(v);
    view_free(reshaped);
}

// Test view addition with 0 shapes
TEST(test_add_0) {
    int32_t shape1[] = {2, 3, 4};
    View *v1 = view_create(shape1, 3, NULL, 0, 0);
    TEST_ASSERT_NOT_NULL(v1);
    
    int32_t shape2[] = {2, 0, 4}; // Has 0 dimension
    View *v2 = view_create(shape2, 3, NULL, 0, 0);
    TEST_ASSERT_NOT_NULL(v2);
    
    View *result = view_add(v1, v2);
    TEST_ASSERT_NOT_NULL(result);
    
    // Result should be v2 (the one with 0 dimension)
    const int32_t *result_shape = view_shape(result);
    TEST_ASSERT_EQUAL(0, result_shape[1]);
    
    view_free(v1);
    view_free(v2);
    view_free(result);
}

// Test view addition with masked 0
TEST(test_add_0_masked) {
    int32_t shape[] = {2, 3, 4};
    int32_t mask_start[] = {0, 0, 0};
    int32_t mask_end[] = {0, 0, 0}; // All masked out
    View *v1 = view_create_with_mask(shape, 3, NULL, 0, 0, mask_start, mask_end, 3);
    TEST_ASSERT_NOT_NULL(v1);
    
    int32_t shape2[] = {2, 0, 4};
    View *v2 = view_create(shape2, 3, NULL, 0, 0);
    TEST_ASSERT_NOT_NULL(v2);
    
    View *result = view_add(v1, v2);
    TEST_ASSERT_NOT_NULL(result);
    
    // Should equal v2
    const int32_t *result_shape = view_shape(result);
    TEST_ASSERT_EQUAL(0, result_shape[1]);
    
    view_free(v1);
    view_free(v2);
    view_free(result);
}

// Test merge views with mask
TEST(test_merge_views_with_mask_0) {
    // From test_ops.py::TestOps::test_pad_reflect_mode
    int32_t shape0[] = {1, 1, 5, 8};
    int32_t strides0[] = {0, 0, 5, 1};
    int32_t mask0_start[] = {0, 0, 0, 3};
    int32_t mask0_end[] = {1, 1, 5, 8};
    View *v0 = view_create_with_mask(shape0, 4, strides0, 4, -3, mask0_start, mask0_end, 4);
    TEST_ASSERT_NOT_NULL(v0);
    
    int32_t shape1[] = {1, 1, 2, 2};
    int32_t strides1[] = {0, 0, 8, 1};
    View *v1 = view_create(shape1, 4, strides1, 4, 3);
    TEST_ASSERT_NOT_NULL(v1);
    
    View *v = view_add(v0, v1);
    TEST_ASSERT_NOT_NULL(v);
    
    // Expected: View(shape=(1, 1, 2, 2), strides=(0, 0, 5, 1), offset=0, mask=None, contiguous=False)
    const int32_t *result_shape = view_shape(v);
    TEST_ASSERT_EQUAL(1, result_shape[0]);
    TEST_ASSERT_EQUAL(1, result_shape[1]);
    TEST_ASSERT_EQUAL(2, result_shape[2]);
    TEST_ASSERT_EQUAL(2, result_shape[3]);
    
    TEST_ASSERT_EQUAL(0, view_offset(v));
    
    view_free(v0);
    view_free(v1);
    view_free(v);
}

// Test merge views with variable
TEST(test_merge_views_variable) {
    // Uses Variable - simplified test without actual Variable implementation
    int32_t N = 100;
    int32_t shape0[] = {N, 32, 2};
    int32_t strides0[] = {32, 1, 0};
    int32_t mask0_start[] = {0, 0, 0};
    int32_t mask0_end[] = {N, 32, 1};
    View *v0 = view_create_with_mask(shape0, 3, strides0, 3, 0, mask0_start, mask0_end, 3);
    TEST_ASSERT_NOT_NULL(v0);
    
    int32_t shape1[] = {1, 8, 1, 32};
    int32_t strides1[] = {0, 0, 0, 2};
    int32_t start_pos = 10; // Simulated variable value
    View *v1 = view_create(shape1, 4, strides1, 4, start_pos * 64);
    TEST_ASSERT_NOT_NULL(v1);
    
    View *v = view_add(v0, v1);
    TEST_ASSERT_NOT_NULL(v);
    
    // Expected: View(shape=(1, 8, 1, 32), strides=(0,0,0,1), offset=start_pos*32, mask=None, contiguous=False)
    const int32_t *result_shape = view_shape(v);
    TEST_ASSERT_EQUAL(1, result_shape[0]);
    TEST_ASSERT_EQUAL(8, result_shape[1]);
    TEST_ASSERT_EQUAL(1, result_shape[2]);
    TEST_ASSERT_EQUAL(32, result_shape[3]);
    
    view_free(v0);
    view_free(v1);
    view_free(v);
}

// Test view padded area cases
TEST(test_view_padded_area1) {
    // test_multinomial case
    int32_t shape[] = {2};
    int32_t strides[] = {0};
    int32_t mask_start[] = {1};
    int32_t mask_end[] = {2};
    View *v0 = view_create_with_mask(shape, 1, strides, 1, 0, mask_start, mask_end, 1);
    TEST_ASSERT_NOT_NULL(v0);
    
    int32_t shape1[] = {1};
    int32_t strides1[] = {0};
    View *v1 = view_create(shape1, 1, strides1, 1, 0);
    TEST_ASSERT_NOT_NULL(v1);
    
    View *v = view_add(v0, v1);
    TEST_ASSERT_NOT_NULL(v);
    
    // Expected: View(shape=(1,), strides=(0,), offset=0, mask=((0,0),), contiguous=False)
    const int32_t *result_shape = view_shape(v);
    TEST_ASSERT_EQUAL(1, result_shape[0]);
    TEST_ASSERT_EQUAL(0, view_offset(v));
    
    view_free(v0);
    view_free(v1);
    view_free(v);
}

// Test empty shape view cases
TEST(test_empty_shape_view1) {
    // test_stack_slice case
    int32_t shape0[] = {3, 5};
    int32_t strides0[] = {0, 1};
    int32_t mask0_start[] = {0, 0};
    int32_t mask0_end[] = {1, 5};
    View *v0 = view_create_with_mask(shape0, 2, strides0, 2, 0, mask0_start, mask0_end, 2);
    TEST_ASSERT_NOT_NULL(v0);
    
    // Empty shape view
    View *v1 = view_create(NULL, 0, NULL, 0, 0);
    TEST_ASSERT_NOT_NULL(v1);
    
    View *v = view_add(v0, v1);
    TEST_ASSERT_NOT_NULL(v);
    
    // Expected: View(shape=(), strides=(), offset=0, mask=None, contiguous=True)
    TEST_ASSERT_EQUAL(0, view_ndim(v));
    TEST_ASSERT_EQUAL(0, view_offset(v));
    TEST_ASSERT_TRUE(view_contiguous(v));
    
    view_free(v0);
    view_free(v1);
    view_free(v);
}

TEST(test_empty_shape_view2) {
    // test_std_mean case
    int32_t shape[] = {2};
    int32_t strides[] = {0};
    int32_t mask_start[] = {1};
    int32_t mask_end[] = {2};
    View *v0 = view_create_with_mask(shape, 1, strides, 1, 0, mask_start, mask_end, 1);
    TEST_ASSERT_NOT_NULL(v0);
    
    // Empty shape view
    View *v1 = view_create(NULL, 0, NULL, 0, 0);
    TEST_ASSERT_NOT_NULL(v1);
    
    View *v = view_add(v0, v1);
    // This case should return None in Python, so NULL in C
    TEST_ASSERT_NULL(v);
    
    view_free(v0);
    view_free(v1);
}

TEST_MAIN()