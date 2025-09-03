#include "test_common.h"
#include "shape/shapetracker.h"
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

// Helper class equivalent - CheckingShapeTracker simplified
typedef struct {
    ShapeTracker *st;
    int32_t *reference_data;
    int32_t *shape;
    int32_t ndim;
} CheckingShapeTracker;

static CheckingShapeTracker* checking_shapetracker_create(const int32_t *shape, int32_t ndim) {
    CheckingShapeTracker *cst = malloc(sizeof(CheckingShapeTracker));
    cst->st = shapetracker_from_shape(shape, ndim);
    cst->shape = malloc(ndim * sizeof(int32_t));
    memcpy(cst->shape, shape, ndim * sizeof(int32_t));
    cst->ndim = ndim;
    
    // Create reference data (simulated numpy.arange)
    int32_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    cst->reference_data = malloc(total_size * sizeof(int32_t));
    for (int i = 0; i < total_size; i++) {
        cst->reference_data[i] = i;
    }
    
    return cst;
}

static void checking_shapetracker_free(CheckingShapeTracker *cst) {
    if (cst) {
        shapetracker_free(cst->st);
        free(cst->reference_data);
        free(cst->shape);
        free(cst);
    }
}

// Basic ShapeTracker tests
TEST(test_simple_equals) {
    int32_t shape[] = {10, 10};
    ShapeTracker *st1 = shapetracker_from_shape(shape, 2);
    ShapeTracker *st2 = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st1);
    TEST_ASSERT_NOT_NULL(st2);
    
    // Should be equal
    TEST_ASSERT_TRUE(shapetracker_equal(st1, st2));
    
    shapetracker_free(st1);
    shapetracker_free(st2);
}

TEST(test_other_equals) {
    int32_t shape[] = {3};
    int32_t strides[] = {1};
    View *v1 = view_create(shape, 1, strides, 1, 0);
    View *v2 = view_create(shape, 1, strides, 1, 0);
    TEST_ASSERT_NOT_NULL(v1);
    TEST_ASSERT_NOT_NULL(v2);
    
    ShapeTracker *st1 = shapetracker_from_views(&v1, 1);
    ShapeTracker *st2 = shapetracker_from_views(&v2, 1);
    TEST_ASSERT_NOT_NULL(st1);
    TEST_ASSERT_NOT_NULL(st2);
    
    TEST_ASSERT_TRUE(shapetracker_equal(st1, st2));
    
    view_free(v1);
    view_free(v2);
    shapetracker_free(st1);
    shapetracker_free(st2);
}

// Single ShapeTracker tests
TEST(test_single_reshape) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){7, 4}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t new_shape[] = {7, 1, 4};
    cst->st = shapetracker_reshape(cst->st, new_shape, 3);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    TEST_ASSERT_TRUE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

TEST(test_single_permute) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){7, 4}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t axes[] = {1, 0};
    cst->st = shapetracker_permute(cst->st, axes, 2);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    TEST_ASSERT_FALSE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

TEST(test_single_shrink) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){7, 4}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t start[] = {1, 0};
    int32_t end[] = {2, 4};
    cst->st = shapetracker_shrink(cst->st, start, end, 2);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    TEST_ASSERT_FALSE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

TEST(test_double_permute) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){7, 4}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t axes[] = {1, 0};
    cst->st = shapetracker_permute(cst->st, axes, 2);
    cst->st = shapetracker_permute(cst->st, axes, 2);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    TEST_ASSERT_TRUE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

TEST(test_reshape_permute) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){7, 4}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t new_shape[] = {7, 1, 4};
    cst->st = shapetracker_reshape(cst->st, new_shape, 3);
    
    int32_t axes[] = {0, 1, 2};
    cst->st = shapetracker_permute(cst->st, axes, 3);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    TEST_ASSERT_TRUE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

TEST(test_reshape_permute_yes) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){7, 4}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t new_shape[] = {7, 1, 4};
    cst->st = shapetracker_reshape(cst->st, new_shape, 3);
    
    int32_t axes[] = {0, 2, 1};
    cst->st = shapetracker_permute(cst->st, axes, 3);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    TEST_ASSERT_TRUE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

TEST(test_reshape_permute_no) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){7, 4}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t new_shape[] = {4, 7};
    cst->st = shapetracker_reshape(cst->st, new_shape, 2);
    
    int32_t axes[] = {1, 0};
    cst->st = shapetracker_permute(cst->st, axes, 2);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    TEST_ASSERT_FALSE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

// ShapeTracker size tests
TEST(test_simple_size) {
    int32_t shape[] = {100, 100};
    ShapeTracker *st = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_EQUAL(100 * 100, shapetracker_real_size(st));
    
    shapetracker_free(st);
}

TEST(test_0_in_shape_size) {
    int32_t shape1[] = {0, 100};
    ShapeTracker *st1 = shapetracker_from_shape(shape1, 2);
    TEST_ASSERT_NOT_NULL(st1);
    TEST_ASSERT_EQUAL(0, shapetracker_real_size(st1));
    
    int32_t shape2[] = {100, 0};
    ShapeTracker *st2 = shapetracker_from_shape(shape2, 2);
    TEST_ASSERT_NOT_NULL(st2);
    TEST_ASSERT_EQUAL(0, shapetracker_real_size(st2));
    
    shapetracker_free(st1);
    shapetracker_free(st2);
}

TEST(test_expand_size) {
    int32_t shape[] = {100, 100};
    ShapeTracker *st = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t reshape1[] = {100, 100, 1};
    st = shapetracker_reshape(st, reshape1, 3);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t expand_shape[] = {100, 100, 100};
    st = shapetracker_expand(st, expand_shape, 3);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_EQUAL(100 * 100, shapetracker_real_size(st));
    
    shapetracker_free(st);
}

TEST(test_expand_size_flatten) {
    int32_t shape[] = {100, 100};
    ShapeTracker *st = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t reshape1[] = {100, 100, 1};
    st = shapetracker_reshape(st, reshape1, 3);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t expand_shape[] = {100, 100, 100};
    st = shapetracker_expand(st, expand_shape, 3);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t flatten_shape[] = {100 * 100 * 100};
    st = shapetracker_reshape(st, flatten_shape, 1);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_EQUAL(100 * 100, shapetracker_real_size(st));
    
    shapetracker_free(st);
}

TEST(test_shrink_size_axis_0) {
    int32_t shape[] = {100, 100};
    ShapeTracker *st = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t start[] = {0, 0};
    int32_t end[] = {50, 100};
    st = shapetracker_shrink(st, start, end, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_EQUAL(50 * 100, shapetracker_real_size(st));
    
    shapetracker_free(st);
}

TEST(test_shrink_size_axis_1) {
    int32_t shape[] = {100, 100};
    ShapeTracker *st = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t start[] = {0, 0};
    int32_t end[] = {100, 50};
    st = shapetracker_shrink(st, start, end, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_EQUAL(9950, shapetracker_real_size(st)); // careful here - from Python test
    
    shapetracker_free(st);
}

TEST(test_pad_size_simple) {
    int32_t shape[] = {10};
    ShapeTracker *st = shapetracker_from_shape(shape, 1);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t pad_before[] = {2};
    int32_t pad_after[] = {4};
    st = shapetracker_pad(st, pad_before, pad_after, 1);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_EQUAL(10, shapetracker_real_size(st));
    
    shapetracker_free(st);
}

TEST(test_pad_size_multiview) {
    int32_t shape[] = {10, 10};
    ShapeTracker *st = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t pad_before[] = {2, 3};
    int32_t pad_after[] = {4, 1};
    st = shapetracker_pad(st, pad_before, pad_after, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t flatten_shape[] = {16 * 14};
    st = shapetracker_reshape(st, flatten_shape, 1);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_EQUAL(100, shapetracker_real_size(st));
    
    shapetracker_free(st);
}

TEST(test_flip_size) {
    int32_t shape[] = {10, 10};
    ShapeTracker *st = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t pad_before[] = {2, 3};
    int32_t pad_after[] = {4, 1};
    st = shapetracker_pad(st, pad_before, pad_after, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    bool flip_axes[] = {true, true};
    st = shapetracker_flip(st, flip_axes, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_EQUAL(100, shapetracker_real_size(st));
    
    shapetracker_free(st);
}

// Masked ShapeTracker tests
TEST(test_pad_1x1) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){1, 1}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t pad_before[] = {1, 1};
    int32_t pad_after[] = {1, 1};
    cst->st = shapetracker_pad(cst->st, pad_before, pad_after, 2);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    checking_shapetracker_free(cst);
}

TEST(test_pad_2x2) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){2, 2}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t pad_before[] = {1, 1};
    int32_t pad_after[] = {1, 1};
    cst->st = shapetracker_pad(cst->st, pad_before, pad_after, 2);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    checking_shapetracker_free(cst);
}

TEST(test_axis_is_masked) {
    int32_t shape[] = {100, 100, 100, 100};
    ShapeTracker *st = shapetracker_from_shape(shape, 4);
    TEST_ASSERT_NOT_NULL(st);
    
    int32_t pad_before[] = {0, 0, 2, 0};
    int32_t pad_after[] = {1, 0, 0, 0};
    st = shapetracker_pad(st, pad_before, pad_after, 4);
    TEST_ASSERT_NOT_NULL(st);
    
    TEST_ASSERT_TRUE(shapetracker_axis_is_masked(st, 0));
    TEST_ASSERT_FALSE(shapetracker_axis_is_masked(st, 1));
    TEST_ASSERT_TRUE(shapetracker_axis_is_masked(st, 2));
    TEST_ASSERT_FALSE(shapetracker_axis_is_masked(st, 3));
    
    shapetracker_free(st);
}

// Complex ShapeTracker tests
TEST(test_add_1s) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){4, 4}, 2);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t axes1[] = {1, 0};
    cst->st = shapetracker_permute(cst->st, axes1, 2);
    TEST_ASSERT_NOT_NULL(cst->st);
    
    int32_t reshape1[] = {1, 4, 1, 4, 1};
    cst->st = shapetracker_reshape(cst->st, reshape1, 5);
    TEST_ASSERT_NOT_NULL(cst->st);
    TEST_ASSERT_FALSE(shapetracker_contiguous(cst->st));
    
    int32_t axes2[] = {0, 3, 2, 1, 4};
    cst->st = shapetracker_permute(cst->st, axes2, 5);
    TEST_ASSERT_NOT_NULL(cst->st);
    TEST_ASSERT_TRUE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

TEST(test_permute_1s_simple) {
    CheckingShapeTracker *cst1 = checking_shapetracker_create((int32_t[]){1, 16, 9, 9}, 4);
    TEST_ASSERT_NOT_NULL(cst1);
    
    int32_t axes[] = {1, 0, 2, 3};
    cst1->st = shapetracker_permute(cst1->st, axes, 4);
    TEST_ASSERT_NOT_NULL(cst1->st);
    TEST_ASSERT_TRUE(shapetracker_contiguous(cst1->st));
    
    CheckingShapeTracker *cst2 = checking_shapetracker_create((int32_t[]){2, 16, 9, 9}, 4);
    TEST_ASSERT_NOT_NULL(cst2);
    
    cst2->st = shapetracker_permute(cst2->st, axes, 4);
    TEST_ASSERT_NOT_NULL(cst2->st);
    TEST_ASSERT_FALSE(shapetracker_contiguous(cst2->st));
    
    checking_shapetracker_free(cst1);
    checking_shapetracker_free(cst2);
}

TEST(test_remove_1s_simple) {
    CheckingShapeTracker *cst = checking_shapetracker_create((int32_t[]){1, 16, 1, 1}, 4);
    TEST_ASSERT_NOT_NULL(cst);
    
    int32_t new_shape[] = {16};
    cst->st = shapetracker_reshape(cst->st, new_shape, 1);
    TEST_ASSERT_NOT_NULL(cst->st);
    TEST_ASSERT_TRUE(shapetracker_contiguous(cst->st));
    
    checking_shapetracker_free(cst);
}

// Render tests (simplified)
TEST(test_render) {
    int32_t shape[] = {2, 3};
    ShapeTracker *st = shapetracker_from_shape(shape, 2);
    TEST_ASSERT_NOT_NULL(st);
    
    // In real implementation, this would create indexed UOps
    // For now, just test that the function exists and doesn't crash
    IndexedUOps *indexed = shapetracker_to_indexed_uops(st);
    TEST_ASSERT_NOT_NULL(indexed);
    
    indexed_uops_free(indexed);
    shapetracker_free(st);
}

TEST_MAIN()