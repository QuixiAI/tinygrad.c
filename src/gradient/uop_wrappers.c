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

// Minimal recursive evaluator for substitute
static float exec_alu_min(Ops op, float* a, int n){
  switch(op){
    case OPS_LOG2: return log2f(a[0]); case OPS_EXP2: return exp2f(a[0]); case OPS_SQRT: return sqrtf(a[0]);
    case OPS_RECIP: return 1.0f/a[0]; case OPS_SIN: return sinf(a[0]); case OPS_NEG: return -a[0];
    case OPS_ADD: return a[0]+a[1]; case OPS_SUB: return a[0]-a[1]; case OPS_MUL: return a[0]*a[1];
    case OPS_FDIV: return a[1]==0.0f? INFINITY : a[0]/a[1]; case OPS_POW: return powf(a[0], a[1]); case OPS_MAX: return fmaxf(a[0], a[1]);
    case OPS_WHERE: return a[0]!=0.0f ? a[1] : a[2]; case OPS_CMPLT: return a[0]<a[1]; case OPS_CMPNE: return a[0]!=a[1]; case OPS_CMPEQ: return a[0]==a[1];
    case OPS_CONST: return a[0];
    // treat meta/movement/casts as identity for scalar eval
    case OPS_CAST: case OPS_CONTIGUOUS: case OPS_FUSE: case OPS_CONTIGUOUS_BACKWARD:
    case OPS_RESHAPE: case OPS_PERMUTE: case OPS_EXPAND: case OPS_PAD: case OPS_SHRINK:
      return a[0];
    default: return a[0]; }
}

static float evaluate_with_subs(tg_uop_t* u, tg_uop_t** vars, tg_uop_t** vals, int nvars){
  if (!u) return NAN;
  if (u->op == OPS_CONST && u->arg.type==ARG_CONST) return (float)u->arg.const_data.const_value;
  if (u->op == OPS_DEFINE_VAR){
    for (int i=0;i<nvars;i++) if (u==vars[i]) {
      tg_uop_t* v = vals[i];
      return (v && v->op==OPS_CONST && v->arg.type==ARG_CONST) ? (float)v->arg.const_data.const_value : NAN;
    }
    return 0.0f;
  }
  float ops[3]={0};
  for (size_t i=0;i<u->src_count && i<3;i++) ops[i]=evaluate_with_subs(u->src[i], vars, vals, nvars);
  return exec_alu_min(u->op, ops, (int)u->src_count);
}

tg_uop_t* tg_uop_substitute(tg_uop_t* uop, tg_substitution_t* substitutions, int count){
  // Evaluate the expression using the interpreter with variable bindings
  if (!uop) return uop_const(dtypes.float32, NAN);
  eval_context_t ctx = {0};
  if (count > 0) {
    ctx.binding_count = count;
    ctx.bindings = (var_binding_t*)calloc(count, sizeof(var_binding_t));
    for (int i=0;i<count;i++) {
      ctx.bindings[i].var_uop = substitutions[i].variable;
      // Build scalar np_array for the value
      size_t one = 1;
      np_array_t* arr = np_ones(1, &one, &dtypes.float32);
      float val = NAN;
      if (substitutions[i].value && substitutions[i].value->op==OPS_CONST && substitutions[i].value->arg.type==ARG_CONST) {
        val = (float)substitutions[i].value->arg.const_data.const_value;
      }
      ((float*)arr->data)[0] = val;
      ctx.bindings[i].data = arr;
    }
  }
  np_array_t* result = uop_interpreter_evaluate_with_context(uop, &ctx);
  float scalar = np_array_get_scalar(result);
  // Free context arrays
  for (int i=0;i<ctx.binding_count;i++) { if (ctx.bindings[i].data) np_free(ctx.bindings[i].data); }
  free(ctx.bindings);
  return uop_const(dtypes.float32, scalar);
}

tg_uop_t* tg_uop_ssimplify(tg_uop_t* uop){ if (uop->op==OPS_CONST) { uop_ref(uop); return uop; } return uop_ssimplify(uop); }

float tg_uop_get_float(tg_uop_t* uop){ if (uop && uop->op==OPS_CONST && uop->arg.type==ARG_CONST) return (float)uop->arg.const_data.const_value; return NAN; }

void tg_uop_free(tg_uop_t* uop){ uop_unref(uop); }

tg_uop_t** tg_uop_toposort(tg_uop_t* root, int* out_count){ size_t n=0; UOp** arr = uop_toposort(root, &n); if (out_count) *out_count=(int)n; return (tg_uop_t**)arr; }
