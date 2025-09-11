#include "test_common.h"
#include "shape/view.h"
#include "shape/shapetracker.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

static int count_op(UOp* root, Ops op) {
  size_t n=0; UOp** topo = uop_toposort(root, &n);
  int cnt=0; if (topo){ for (size_t i=0;i<n;i++) if (topo[i]->op==op) cnt++; free(topo);} return cnt;
}

TEST(test_view_to_indexed_uops_symbolic_prefers_sym_fields) {
  // 1D view with symbolic offset, stride, mask
  int32_t cshape[] = {10};
  View* v = view_create(cshape, 1, NULL, 0, 0);
  TEST_ASSERT_NOT_NULL(v);

  // Build symbolic arrays
  UOp* sym_shape[1] = { uop_const(dtypes.int_, 10) };
  UOp* sym_strides[1] = { uop_const(dtypes.int_, 2) };
  UOp* sym_offset = uop_var_with_range("off", dtypes.int_, 0, 100);
  UOp* sym_mstart[1] = { uop_const(dtypes.int_, 3) };
  UOp* sym_mend[1]   = { uop_const(dtypes.int_, 9) };

  // Attach symbolic fields by creating a symbolic view
  View* vs = view_create_symbolic(sym_shape, 1, sym_strides, 1, sym_offset, sym_mstart, sym_mend, 1);
  TEST_ASSERT_NOT_NULL(vs);

  // Compute idx, valid
  UOp *idx=NULL, *valid=NULL;
  view_to_indexed_uops(vs, NULL, 0, NULL, &idx, &valid);
  TEST_ASSERT_NOT_NULL(idx);
  TEST_ASSERT_NOT_NULL(valid);

  // Expect: idx includes RANGE*2 and includes sym_offset variable
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

  // valid should include GE(idx,3) and LT(idx,9)
  // GE is implemented as CMPNE(True, LT), so expect at least one CMPNE and one CMPLT
  TEST_ASSERT_TRUE(count_op(valid, OPS_CMPNE) >= 1);
  TEST_ASSERT_TRUE(count_op(valid, OPS_CMPLT) >= 1);

  view_free(v);
  view_free(vs);
}

TEST(test_shapetracker_vars_unbind_substitute) {
  // Build a view with symbolic offset var and stride const
  UOp* varx = uop_var_with_range("x", dtypes.int_, 0, 100);
  UOp* sym_shape[1] = { uop_const(dtypes.int_, 16) };
  UOp* sym_strides[1] = { uop_const(dtypes.int_, 4) };
  UOp* sym_offset = varx;
  View* v = view_create_symbolic(sym_shape, 1, sym_strides, 1, sym_offset, NULL, NULL, 0);
  TEST_ASSERT_NOT_NULL(v);

  // ST from this single view
  View* arr[1] = { v };
  ShapeTracker* st = shapetracker_from_views(arr, 1);
  TEST_ASSERT_NOT_NULL(st);

  // vars should include varx
  int vcnt=0; UOp** vars = shapetracker_vars(st, &vcnt);
  bool seen=false; for (int i=0;i<vcnt;i++) if (vars[i] == varx) { seen=true; break; }
  TEST_ASSERT_TRUE(seen);
  if (vars) free(vars);

  // substitute x -> 7 and check idx contains CONST 7
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

  // unbind: bind varx to 5 and expect mapping {varx:5}
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

  // cleanup
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
