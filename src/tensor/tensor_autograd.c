#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gradient/gradient.h"
#include "compat/numpy.h"
#include "runtime/uop_interpreter.h"
#include "tensor/tensor.h"

static int g_var_id = 0;
static char* make_var_name(void){ char buf[32]; snprintf(buf,sizeof(buf),"v%d", g_var_id++); return strdup(buf); }

static inline tg_tensor_t unbox(tg_tensor_t* h){ return h ? *h : NULL; }
static inline tg_tensor_t* box(tg_tensor_t t){ tg_tensor_t* p=(tg_tensor_t*)malloc(sizeof(tg_tensor_t)); *p=t; return p; }

static tg_tensor_t tensor_create_from_shape(int64_t* shape, int ndim, const DType* dtype){
  struct tg_ctx *ctx = NULL; // no graph ctx coupling needed
  struct tg_tensor* t = (struct tg_tensor*)calloc(1,sizeof(*t));
  if(!dtype) dtype = &dtypes.float32;
  t->ctx = ctx; t->dtype = TG_F32; t->np_dtype = dtype; t->rank = ndim; t->shape = (int64_t*)malloc(sizeof(int64_t)*ndim);
  size_t numel=1; for(int i=0;i<ndim;i++){ t->shape[i]=shape[i]; numel *= (size_t)shape[i]; }
  t->numel=numel; t->data = (float*)calloc(numel,sizeof(float)); t->uop=NULL;
  (void)dtype; return t;
}

static void tensor_attach_var_uop(tg_tensor_t t){
  char* name = make_var_name();
  tg_uop_t* var = tg_uop_variable(name, -INFINITY, INFINITY, TG_F32);
  free(name);
  int32_t* shp = (int32_t*)malloc(sizeof(int32_t)*t->rank);
  for(int i=0;i<t->rank;i++) shp[i]=(int32_t)t->shape[i];
  t->uop = tg_uop_reshape(var, shp, t->rank);
  free(shp);
}

tg_tensor_t* tg_tensor_eye(int size, const DType* dtype){
  (void)dtype; int64_t shp[2] = {size, size}; tg_tensor_t t = tensor_create_from_shape(shp, 2, &dtypes.float32);
  for(int i=0;i<size;i++) t->data[i*size+i] = 1.0f;
  tensor_attach_var_uop(t);
  return box(t);
}

tg_tensor_t* tg_tensor_from_data(int64_t* shape, int ndim, float* data, const DType* dtype){
  (void)dtype; tg_tensor_t t = tensor_create_from_shape(shape, ndim, &dtypes.float32);
  memcpy(t->data, data, t->numel*sizeof(float));
  tensor_attach_var_uop(t);
  return box(t);
}

tg_tensor_t* tg_tensor_from_data_int(int64_t* shape, int ndim, int32_t* data, const DType* dtype){
  tg_tensor_t t = tensor_create_from_shape(shape, ndim, dtype);
  for(size_t i=0;i<t->numel;i++) t->data[i] = (float)data[i];
  tensor_attach_var_uop(t);
  return box(t);
}

tg_tensor_t* tg_tensor_randn(int64_t* shape, int ndim, const DType* dtype){
  (void)dtype; tg_tensor_t t = tensor_create_from_shape(shape, ndim, &dtypes.float32);
  for(size_t i=0;i<t->numel;i++) t->data[i] = 0.1f * (float)(i%7 - 3);
  tensor_attach_var_uop(t);
  return box(t);
}

static tg_uop_t* uop_build_matmul(tg_uop_t* A, int64_t* ashp, int andim, tg_uop_t* B, int64_t* bshp, int bndim){
  (void)andim; (void)bndim;
  int32_t M = (int32_t)ashp[0], K = (int32_t)ashp[1], N = (int32_t)bshp[1];
  int32_t a_rs[3] = {M, K, 1}; int32_t a_ex[3] = {M, K, N};
  int32_t b_rs[3] = {1, K, N}; int32_t b_ex[3] = {M, K, N};
  tg_uop_t* A3 = tg_uop_expand(tg_uop_reshape(A, a_rs, 3), a_ex, 3);
  tg_uop_t* B3 = tg_uop_expand(tg_uop_reshape(B, b_rs, 3), b_ex, 3);
  tg_uop_t* MUL = tg_uop_mul(A3, B3);
  int axes[1] = {1};
  return tg_uop_reduce_axis(MUL, OPS_ADD, axes, 1);
}

tg_tensor_t* tg_tensor_matmul(tg_tensor_t* a_h, tg_tensor_t* b_h){ tg_tensor_t a=unbox(a_h), b=unbox(b_h);
  if(!a||!b||a->rank!=2||b->rank!=2) return NULL;
  if (a->shape[1] != b->shape[0]) return NULL;
  tg_tensor_t t = tensor_create_from_shape((int64_t[]){a->shape[0], b->shape[1]}, 2, &dtypes.float32);
  t->uop = uop_build_matmul(a->uop, a->shape, a->rank, b->uop, b->shape, b->rank);
  return box(t);
}

tg_tensor_t* tg_tensor_sum(tg_tensor_t* t_h){
  tg_tensor_t t = unbox(t_h); if(!t) return NULL;
  int nd=t->rank; int* axes = (int*)malloc(sizeof(int)*nd); for(int i=0;i<nd;i++) axes[i]=i;
  tg_tensor_t out = tensor_create_from_shape((int64_t[]){1}, 1, &dtypes.float32);
  out->uop = tg_uop_reduce_axis(t->uop, OPS_ADD, axes, nd);
  free(axes);
  return box(out);
}

static void compute_broadcast_shape(const int64_t* as, int an, const int64_t* bs, int bn, int32_t** out_shape, int* out_nd){
  int nd = an>bn? an: bn; int32_t* ts = (int32_t*)malloc(sizeof(int32_t)*nd);
  for(int i=0;i<nd;i++){
    int ia = i - (nd - an); int ib = i - (nd - bn);
    int da = ia<0? 1 : (int)as[ia];
    int db = ib<0? 1 : (int)bs[ib];
    ts[i] = da==1? db : da;
  }
  *out_shape = ts; *out_nd = nd;
}
tg_tensor_t* tg_tensor_add(tg_tensor_t* a_h, tg_tensor_t* b_h){ tg_tensor_t a=unbox(a_h), b=unbox(b_h); if(!a||!b) return NULL; int32_t* tsh=NULL; int nd=0; compute_broadcast_shape(a->shape,a->rank,b->shape,b->rank,&tsh,&nd); 
  int64_t* shape64 = (int64_t*)malloc(sizeof(int64_t)*nd); for(int i=0;i<nd;i++) shape64[i]=(int64_t)tsh[i];
  tg_tensor_t t=tensor_create_from_shape(shape64, nd, &dtypes.float32); 
  // expand inputs to target shape
  int32_t* a_rs = (int32_t*)calloc(nd,sizeof(int32_t)); for(int i=0;i<nd;i++){ int ia=i-(nd-a->rank); a_rs[i]= ia<0? 1 : (int32_t)a->shape[ia]; }
  int32_t* b_rs = (int32_t*)calloc(nd,sizeof(int32_t)); for(int i=0;i<nd;i++){ int ib=i-(nd-b->rank); b_rs[i]= ib<0? 1 : (int32_t)b->shape[ib]; }
  tg_uop_t* au = tg_uop_expand(tg_uop_reshape(a->uop, a_rs, nd), tsh, nd);
  tg_uop_t* bu = tg_uop_expand(tg_uop_reshape(b->uop, b_rs, nd), tsh, nd);
  t->uop = tg_uop_add(au, bu);
  free(a_rs); free(b_rs); free(tsh); free(shape64); return box(t); }
tg_tensor_t* tg_tensor_mul(tg_tensor_t* a_h, tg_tensor_t* b_h){ tg_tensor_t a=unbox(a_h), b=unbox(b_h); if(!a||!b) return NULL; int32_t* tsh=NULL; int nd=0; compute_broadcast_shape(a->shape,a->rank,b->shape,b->rank,&tsh,&nd); 
  int64_t* shape64 = (int64_t*)malloc(sizeof(int64_t)*nd); for(int i=0;i<nd;i++) shape64[i]=(int64_t)tsh[i];
  tg_tensor_t t=tensor_create_from_shape(shape64, nd, &dtypes.float32);
  int32_t* a_rs = (int32_t*)calloc(nd,sizeof(int32_t)); for(int i=0;i<nd;i++){ int ia=i-(nd-a->rank); a_rs[i]= ia<0? 1 : (int32_t)a->shape[ia]; }
  int32_t* b_rs = (int32_t*)calloc(nd,sizeof(int32_t)); for(int i=0;i<nd;i++){ int ib=i-(nd-b->rank); b_rs[i]= ib<0? 1 : (int32_t)b->shape[ib]; }
  tg_uop_t* au = tg_uop_expand(tg_uop_reshape(a->uop, a_rs, nd), tsh, nd);
  tg_uop_t* bu = tg_uop_expand(tg_uop_reshape(b->uop, b_rs, nd), tsh, nd);
  t->uop = tg_uop_mul(au, bu);
  free(a_rs); free(b_rs); free(tsh); free(shape64); return box(t); }

tg_tensor_t* tg_tensor_reshape(tg_tensor_t* t_h, int64_t* shape, int ndim){ tg_tensor_t t=unbox(t_h); if(!t) return NULL; tg_tensor_t r=tensor_create_from_shape(shape,ndim,&dtypes.float32); int32_t* s = (int32_t*)malloc(sizeof(int32_t)*ndim); for(int i=0;i<ndim;i++) s[i]=(int32_t)shape[i]; r->uop=tg_uop_reshape(t->uop,s,ndim); free(s); return box(r); }
tg_tensor_t* tg_tensor_cast(tg_tensor_t* t_h, const DType* dtype){ tg_tensor_t t=unbox(t_h); if(!t) return NULL; if(!dtype) dtype=&dtypes.float32; tg_tensor_t r=tensor_create_from_shape(t->shape,t->rank,dtype); r->uop=tg_uop_cast(t->uop,dtype); return box(r); }

tg_tensor_t* tg_tensor_mean(tg_tensor_t* t_h){
  tg_tensor_t s = unbox(tg_tensor_sum(t_h));
  float denom = (float)unbox(t_h)->numel;
  tg_uop_t* denom_u = tg_uop_const(TG_F32, denom);
  tg_tensor_t out = tensor_create_from_shape(s->shape, s->rank, &dtypes.float32);
  out->uop = tg_uop_div(s->uop, denom_u);
  return box(out);
}

bool tg_tensor_shape_equal(tg_tensor_t* t_h, int64_t* shape, int ndim){ tg_tensor_t t=unbox(t_h); if(!t) return false; if (t->rank!=ndim) return false; for(int i=0;i<ndim;i++) if(t->shape[i]!=shape[i]) return false; return true; }
float* tg_tensor_data_ptr(tg_tensor_t* t_h){ tg_tensor_t t=unbox(t_h); return t ? t->data : NULL; }
void tg_tensor_free(tg_tensor_t* t_h){ tg_tensor_t t=unbox(t_h); if(!t) { free(t_h); return; } free(t->data); free(t->shape); if(t->uop) tg_uop_free(t->uop); free(t); free(t_h); }

static np_array_t* eval_with_bindings(tg_uop_t* u, tg_tensor_t** inputs, int n){
  eval_context_t ctx = {0}; ctx.binding_count=n; ctx.bindings=(var_binding_t*)calloc(n,sizeof(var_binding_t));
  for(int i=0;i<n;i++){
    tg_tensor_t ti = unbox(inputs[i]);
    ctx.bindings[i].var_uop = ti->uop->src_count>0 ? ti->uop->src[0] : ti->uop;
    size_t* shp = (size_t*)malloc(sizeof(size_t)*ti->rank); for(int d=0; d<ti->rank; d++) shp[d]=(size_t)ti->shape[d];
    np_array_t* arr = np_empty(ti->rank, shp, &dtypes.float32); memcpy(arr->data, ti->data, ti->numel*sizeof(float)); free(shp); ctx.bindings[i].data=arr;
  }
  np_array_t* res = uop_interpreter_evaluate_with_context(u, &ctx);
  for(int i=0;i<n;i++) np_free(ctx.bindings[i].data);
  free(ctx.bindings);
  return res;
}

static tg_tensor_t** build_grad_tensors(tg_uop_t* out, tg_tensor_t** inputs, int input_count, tg_uop_t* seed, tg_tensor_t** eval_bindings, int eval_count){
  tg_uop_t* vars[input_count]; for(int i=0;i<input_count;i++) vars[i]=unbox(inputs[i])->uop;
  tg_gradient_result_t* gr = tg_compute_gradient(out, seed, vars, input_count);
  if(!gr) return NULL;
  tg_tensor_t** ret = (tg_tensor_t**)calloc(input_count, sizeof(tg_tensor_t*));
  for(int i=0;i<input_count;i++){
    tg_uop_t* gu = tg_gradient_result_get(gr, vars[i]); if(!gu){ ret[i]=NULL; continue; }
    np_array_t* arr = eval_with_bindings(gu, eval_bindings, eval_count);
    tg_tensor_t in_i = unbox(inputs[i]);
    tg_tensor_t outt = tensor_create_from_shape(in_i->shape, in_i->rank, &dtypes.float32);
    outt->uop = gu; memcpy(outt->data, arr->data, outt->numel*sizeof(float)); np_free(arr);
    ret[i]=box(outt);
  }
  tg_gradient_result_free(gr);
  return ret;
}

static bool is_float_dtype(const DType* dt){ return dt==&dtypes.float32 || dt==&dtypes.float16; }

tg_tensor_t** tg_tensor_gradient(tg_tensor_t* output_h, tg_tensor_t** inputs, int input_count){
  tg_tensor_t output = unbox(output_h);
  if(!output) return NULL;
  if(output->numel != 1) return NULL; // only scalar outputs allowed without custom seed
  for(int i=0;i<input_count;i++) if(!is_float_dtype(unbox(inputs[i])->np_dtype)) return NULL;
  tg_uop_t* seed = tg_uop_const(TG_F32, 1.0f);
  tg_tensor_t** grads = build_grad_tensors(output->uop, inputs, input_count, seed, inputs, input_count);
  if(!grads) return NULL;
  // ensure all requested grads exist
  for(int i=0;i<input_count;i++){ if(!grads[i]){ for(int j=0;j<input_count;j++) if(grads[j]) tg_tensor_free(grads[j]); free(grads); return NULL; } }
  return grads;
}

tg_tensor_t** tg_tensor_gradient_with_grad(tg_tensor_t* output_h, tg_tensor_t** inputs, int input_count, tg_tensor_t* grad_output_h){
  tg_tensor_t output = unbox(output_h); tg_tensor_t grad_output = unbox(grad_output_h);
  if(!output||!grad_output) return NULL;
  // require grad_output shape match output shape
  if(output->rank!=grad_output->rank) return NULL;
  for(int i=0;i<output->rank;i++) if(output->shape[i]!=grad_output->shape[i]) return NULL;
  for(int i=0;i<input_count;i++) if(!is_float_dtype(unbox(inputs[i])->np_dtype)) return NULL;
  // build eval bindings: inputs + grad_output
  tg_tensor_t** bindings = (tg_tensor_t**)malloc(sizeof(tg_tensor_t*)*(input_count+1));
  for(int i=0;i<input_count;i++) bindings[i]=inputs[i];
  bindings[input_count]=grad_output_h;
  tg_tensor_t** grads = build_grad_tensors(output->uop, inputs, input_count, grad_output->uop, bindings, input_count+1);
  free(bindings);
  if(!grads) return NULL;
  for(int i=0;i<input_count;i++){ if(!grads[i]){ for(int j=0;j<input_count;j++) if(grads[j]) tg_tensor_free(grads[j]); free(grads); return NULL; } }
  return grads;
}
