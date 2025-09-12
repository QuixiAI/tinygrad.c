#include "test_common.h"
#include "shape/shapetracker.h"
#include "shape/view.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
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

// ---- Additional symbolic and math tests merged from separate files ----

static int count_op(UOp* root, Ops op) {
  size_t n=0; UOp** topo = uop_toposort(root, &n);
  int cnt=0; if (topo){ for (size_t i=0;i<n;i++) if (topo[i]->op==op) cnt++; free(topo);} return cnt;
}

TEST(test_view_to_indexed_uops_symbolic_prefers_sym_fields) {
  int32_t cshape[] = {10};
  View* v = view_create(cshape, 1, NULL, 0, 0);
  TEST_ASSERT_NOT_NULL(v);
  UOp* sym_shape[1] = { uop_const(dtypes.int_, 10) };
  UOp* sym_strides[1] = { uop_const(dtypes.int_, 2) };
  UOp* sym_offset = uop_var_with_range("off", dtypes.int_, 0, 100);
  UOp* sym_mstart[1] = { uop_const(dtypes.int_, 3) };
  UOp* sym_mend[1]   = { uop_const(dtypes.int_, 9) };
  View* vs = view_create_symbolic(sym_shape, 1, sym_strides, 1, sym_offset, sym_mstart, sym_mend, 1);
  TEST_ASSERT_NOT_NULL(vs);
  UOp *idx=NULL, *valid=NULL;
  view_to_indexed_uops(vs, NULL, 0, NULL, &idx, &valid);
  TEST_ASSERT_NOT_NULL(idx);
  TEST_ASSERT_NOT_NULL(valid);
  size_t n=0; UOp** topo = uop_toposort(idx, &n);
  TEST_ASSERT_TRUE(topo != NULL);
  bool has_mul=false, has_range=false, has_const2=false, has_var=false;
  for (size_t i=0;i<n;i++){
    if (topo[i]->op == OPS_MUL) has_mul=true;
    if (topo[i]->op == OPS_RANGE) has_range=true;
    if (topo[i]->op == OPS_CONST && topo[i]->arg.type==ARG_CONST && (int)topo[i]->arg.const_data.const_value==2) has_const2=true;
    if (topo[i]->op == OPS_DEFINE_VAR) has_var=true;
  }
  free(topo);
  TEST_ASSERT_TRUE(has_mul);
  TEST_ASSERT_TRUE(has_range);
  TEST_ASSERT_TRUE(has_const2);
  TEST_ASSERT_TRUE(has_var);
  TEST_ASSERT_TRUE(count_op(valid, OPS_CMPNE) >= 1);
  TEST_ASSERT_TRUE(count_op(valid, OPS_CMPLT) >= 1);
  view_free(v);
  view_free(vs);
}

TEST(test_shapetracker_vars_unbind_substitute) {
  UOp* varx = uop_var_with_range("x", dtypes.int_, 0, 100);
  UOp* sym_shape[1] = { uop_const(dtypes.int_, 16) };
  UOp* sym_strides[1] = { uop_const(dtypes.int_, 4) };
  UOp* sym_offset = varx;
  View* v = view_create_symbolic(sym_shape, 1, sym_strides, 1, sym_offset, NULL, NULL, 0);
  TEST_ASSERT_NOT_NULL(v);
  View* arr[1] = { v };
  ShapeTracker* st = shapetracker_from_views(arr, 1);
  TEST_ASSERT_NOT_NULL(st);
  int vcnt=0; UOp** vars = shapetracker_vars(st, &vcnt);
  bool seen=false; for (int i=0;i<vcnt;i++) if (vars[i] == varx) { seen=true; break; }
  TEST_ASSERT_TRUE(seen);
  if (vars) free(vars);
  UOp* to = uop_const(dtypes.int_, 7);
  UOp* froms[1] = { varx }; UOp* tos[1] = { to };
  ShapeTracker* st2 = shapetracker_substitute(st, froms, tos, 1);
  TEST_ASSERT_NOT_NULL(st2);
  IndexedUOps* u2 = shapetracker_to_indexed_uops(st2);
  TEST_ASSERT_NOT_NULL(u2);
  size_t n2=0; UOp** topo2 = uop_toposort(u2->idx, &n2);
  bool has7=false; for (size_t i=0;i<n2;i++) if (topo2[i]->op==OPS_CONST && topo2[i]->arg.type==ARG_CONST && (int)topo2[i]->arg.const_data.const_value==7) { has7=true; break; }
  free(topo2);
  indexed_uops_free(u2);
  TEST_ASSERT_TRUE(has7);
  UOp* bound = uop_bind(varx, uop_const(dtypes.int_, 5));
  View* vb = view_create_symbolic(sym_shape, 1, sym_strides, 1, bound, NULL, NULL, 0);
  View* arrb[1] = { vb };
  ShapeTracker* stb = shapetracker_from_views(arrb, 1);
  UOp** out_vars=NULL; int* out_vals=NULL; int out_cnt=0;
  ShapeTracker* stb2 = shapetracker_unbind(stb, &out_vars, &out_vals, &out_cnt);
  TEST_ASSERT_NOT_NULL(stb2);
  TEST_ASSERT_TRUE(out_cnt >= 1);
  bool found=false; for (int i=0;i<out_cnt;i++) if (out_vars[i]==varx && out_vals[i]==5) { found=true; break; }
  TEST_ASSERT_TRUE(found);
  if (out_vars) free(out_vars);
  if (out_vals) free(out_vals);
  view_free(v);
  view_free(vb);
  shapetracker_free(st);
  shapetracker_free(st2);
  shapetracker_free(stb);
  shapetracker_free(stb2);
}

TEST_MAIN()
