// Thin wrappers that delegate to core UOp implementation
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gradient/gradient.h"
#include "runtime/uop_interpreter.h"
#include "compat/numpy.h"

tg_uop_t* tg_uop_variable(const char* name, float vmin, float vmax, tg_dtype dtype) {
  (void)dtype; // only TG_F32 is supported; map to float32
  return uop_var_with_range(name, dtypes.float32, (int)vmin, (int)vmax);
}

tg_uop_t* tg_uop_const(tg_dtype dtype, float value) {
  (void)dtype; // only TG_F32 is supported
  return uop_const(dtypes.float32, value);
}

tg_uop_t* tg_uop_const_like(tg_uop_t* t, float v) {
  DType dt = t ? t->dtype : dtypes.float32;
  return uop_const(dt, v);
}

// Unary
tg_uop_t* tg_uop_recip(tg_uop_t* x){ return uop_recip(x); }
tg_uop_t* tg_uop_sin(tg_uop_t* x){ return uop_sin(x); }
tg_uop_t* tg_uop_sqrt(tg_uop_t* x){ return uop_sqrt(x); }
tg_uop_t* tg_uop_log2(tg_uop_t* x){ return uop_log2(x); }
tg_uop_t* tg_uop_exp2(tg_uop_t* x){ return uop_exp2(x); }
tg_uop_t* tg_uop_neg(tg_uop_t* x){ return uop_neg(x); }
tg_uop_t* tg_uop_cast(tg_uop_t* x, const DType* dt){ return uop_cast(x, *dt); }
tg_uop_t* tg_uop_contiguous(tg_uop_t* x){ UOpArg arg={0}; UOp* src[]={x}; return uop_new(OPS_CONTIGUOUS, x->dtype, src, 1, &arg, NULL); }
tg_uop_t* tg_uop_flip(tg_uop_t* x, int axis){ return uop_flip_axis(x, axis); }

// Movement
tg_uop_t* tg_uop_reshape(tg_uop_t* x, const int32_t* new_shape, int new_ndim){ return uop_reshape(x, new_shape, new_ndim); }
tg_uop_t* tg_uop_permute(tg_uop_t* x, const int32_t* axes, int num_axes){ return uop_permute(x, axes, num_axes); }
tg_uop_t* tg_uop_expand(tg_uop_t* x, const int32_t* target_shape, int target_ndim){ return uop_expand(x, target_shape, target_ndim); }
tg_uop_t* tg_uop_pad(tg_uop_t* x, const int32_t* pad_before, const int32_t* pad_after, int ndim){ return uop_pad(x, pad_before, pad_after, ndim); }
tg_uop_t* tg_uop_shrink(tg_uop_t* x, const int32_t* start, const int32_t* end, int ndim){ return uop_shrink(x, start, end, ndim); }

// Binary
tg_uop_t* tg_uop_add(tg_uop_t* a, tg_uop_t* b){ return uop_add(a,b); }
tg_uop_t* tg_uop_mul(tg_uop_t* a, tg_uop_t* b){ return uop_mul(a,b); }
tg_uop_t* tg_uop_div(tg_uop_t* a, tg_uop_t* b){ return uop_div(a,b); }
tg_uop_t* tg_uop_sub(tg_uop_t* a, tg_uop_t* b){ return uop_sub(a,b); }
// POW follows UOp.alu rule: out dtype = last operand's dtype (b)
tg_uop_t* tg_uop_pow(tg_uop_t* a, tg_uop_t* b){ UOp* src[]={a,b}; UOpArg arg={0}; DType dt=b->dtype; return uop_new(OPS_POW, dt, src, 2, &arg, NULL); }

// Comparisons and where
tg_uop_t* tg_uop_cmplt(tg_uop_t* a, tg_uop_t* b){ return uop_lt(a,b); }
tg_uop_t* tg_uop_cmpgt(tg_uop_t* a, tg_uop_t* b){ return uop_gt(a,b); }
tg_uop_t* tg_uop_cmpne(tg_uop_t* a, tg_uop_t* b){ return uop_ne(a,b); }
tg_uop_t* tg_uop_where(tg_uop_t* c, tg_uop_t* t, tg_uop_t* f){ return uop_where(c,t,f); }

// Reduce-axis wrapper
tg_uop_t* tg_uop_reduce_axis(tg_uop_t* src, int reduce_op, int* axes, int axes_count){
  return uop_reduce_axis(src, (Ops)reduce_op, axes, axes_count);
}


tg_uop_t* tg_uop_substitute(tg_uop_t* uop, tg_substitution_t* substitutions, int count){
  if (!uop) return NULL;
  if (count <= 0 || !substitutions) return uop_ref(uop);

  UOp** from = (UOp**)malloc((size_t)count * sizeof(UOp*));
  UOp** to   = (UOp**)malloc((size_t)count * sizeof(UOp*));
  if (!from || !to) {
    if (from) free(from);
    if (to) free(to);
    return uop_ref(uop);
  }

  size_t used = 0;
  for (int i = 0; i < count; i++) {
    UOp* var = substitutions[i].variable;
    UOp* val = substitutions[i].value;
    if (!var || !val) continue;
    from[used] = var;
    to[used] = val;
    used++;
  }

  UOp* result = NULL;
  if (used == 0) {
    result = uop_ref(uop);
  } else {
    result = uop_substitute(uop, from, to, used, NULL);
  }

  free(from);
  free(to);

  if (!result) return NULL;

  if (result->op != OPS_CONST) {
    size_t topo_count = 0;
    UOp** nodes = uop_toposort(result, &topo_count);
    bool has_var = false;
    for (size_t i = 0; i < topo_count; i++) {
      if (nodes[i] && nodes[i]->op == OPS_DEFINE_VAR) {
        has_var = true;
        break;
      }
    }
    if (nodes) free(nodes);

    if (!has_var) {
      eval_context_t ctx = {0};
      np_array_t* arr = uop_interpreter_evaluate_with_context(result, &ctx);
      double scalar = 0.0;
      if (arr) {
        if (dtypes_is_float(&arr->dtype)) {
            if (dtype_eq(&arr->dtype, &dtypes.float64)) scalar = ((double*)arr->data)[0];
            else scalar = ((float*)arr->data)[0];
        } else if (dtypes_is_int(&arr->dtype)) {
            scalar = (double)((int32_t*)arr->data)[0];
        } else if (dtypes_is_unsigned(&arr->dtype)) {
            scalar = (double)((uint32_t*)arr->data)[0];
        } else {
            scalar = np_array_get_scalar(arr);
        }
      }
      if (arr) np_free(arr);
      DType dt = result->dtype.count ? result->dtype : dtypes.float32;
      uop_unref(result);
      return uop_const(dt, scalar);
    }
  }

  return result;
}

tg_uop_t* tg_uop_ssimplify(tg_uop_t* uop){ if (uop->op==OPS_CONST) { uop_ref(uop); return uop; } return uop_ssimplify(uop); }

float tg_uop_get_float(tg_uop_t* uop){ if (uop && uop->op==OPS_CONST && uop->arg.type==ARG_CONST) return (float)uop->arg.const_data.const_value; return NAN; }

void tg_uop_free(tg_uop_t* uop){ uop_unref(uop); }

tg_uop_t** tg_uop_toposort(tg_uop_t* root, int* out_count){ size_t n=0; UOp** arr = uop_toposort(root, &n); if (out_count) *out_count=(int)n; return (tg_uop_t**)arr; }
