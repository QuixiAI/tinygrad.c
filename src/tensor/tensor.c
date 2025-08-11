#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tg.h"
#include "tensor.h"

int tg__calc_numel(const int64_t *shape, int rank, size_t *out){
  if(!shape||rank<=0) return TG_ERR_INVALID;
  size_t n=1; for(int i=0;i<rank;i++){ if(shape[i]<=0) return TG_ERR_INVALID; n *= (size_t)shape[i]; }
  *out = n; return TG_SUCCESS;
}
int tg__ensure_grad(struct tg_tensor *t){
  if(!t) return TG_ERR_INVALID;
  if(!t->grad){ t->grad = (float*)calloc(t->numel, sizeof(float)); if(!t->grad) return TG_ERR_NOMEM; }
  return TG_SUCCESS;
}
int tg__ctx_register_tensor(tg_ctx_t c, struct tg_tensor *t){
  if(!c||!t) return TG_ERR_INVALID;
  if(c->all_n+1 > c->all_cap){
    size_t nc = c->all_cap? c->all_cap*2 : 64;
    void *p = realloc(c->all, nc*sizeof(*c->all)); if(!p) return TG_ERR_NOMEM;
    c->all = (struct tg_tensor**)p; c->all_cap = nc;
  }
  c->all[c->all_n++] = t; return TG_SUCCESS;
}

/* Context */
int tgCreateContext(tg_ctx_t *out){
  if(!out) return TG_ERR_INVALID;
  struct tg_ctx *c = (struct tg_ctx*)calloc(1,sizeof(*c)); if(!c) return TG_ERR_NOMEM;
  *out = c; return TG_SUCCESS;
}
int tgDestroyContext(tg_ctx_t ctx){
  if(!ctx) return TG_ERR_INVALID;
  free(ctx->all); free(ctx); return TG_SUCCESS;
}
int tgZeroGrad(tg_ctx_t ctx){
  if(!ctx) return TG_ERR_INVALID;
  for(size_t i=0;i<ctx->all_n;i++){ struct tg_tensor *t = ctx->all[i]; if(t->grad) memset(t->grad,0,t->numel*sizeof(float)); }
  return TG_SUCCESS;
}

/* Tensor */
int tgTensorCreate(tg_ctx_t ctx, tg_dtype dtype, const int64_t *shape, int rank, tg_tensor_t *out){
  if(!ctx||!shape||rank<=0||!out) return TG_ERR_INVALID;
  if(dtype!=TG_F32) return TG_ERR_UNIMPL;
  size_t numel=0; int rc = tg__calc_numel(shape,rank,&numel); if(rc) return rc;
  struct tg_tensor *t = (struct tg_tensor*)calloc(1,sizeof(*t)); if(!t) return TG_ERR_NOMEM;
  t->ctx = ctx; t->dtype=dtype; t->rank=rank; t->numel=numel;
  t->shape=(int64_t*)malloc(sizeof(int64_t)*rank); if(!t->shape){ free(t); return TG_ERR_NOMEM; }
  memcpy(t->shape, shape, sizeof(int64_t)*rank);
  t->data=(float*)calloc(numel,sizeof(float)); if(!t->data){ free(t->shape); free(t); return TG_ERR_NOMEM; }
  t->grad_op = TG_OP_NONE;
  tg__ctx_register_tensor(ctx, t);
  *out = t; return TG_SUCCESS;
}
int tgTensorFill(tg_tensor_t t, float v){ if(!t) return TG_ERR_INVALID; for(size_t i=0;i<t->numel;i++) t->data[i]=v; return TG_SUCCESS; }
int tgTensorUpload(tg_tensor_t t, const void *data, size_t nbytes){
  if(!t||!data) return TG_ERR_INVALID; if(nbytes!=t->numel*sizeof(float)) return TG_ERR_INVALID; memcpy(t->data,data,nbytes); return TG_SUCCESS;
}
int tgTensorDownload(tg_tensor_t t, void *data, size_t nbytes){
  if(!t||!data) return TG_ERR_INVALID; if(nbytes!=t->numel*sizeof(float)) return TG_ERR_INVALID; memcpy(data,t->data,nbytes); return TG_SUCCESS;
}
int tgTensorDestroy(tg_tensor_t t){
  if(!t) return TG_ERR_INVALID; free(t->grad); free(t->data); free(t->shape); free(t); return TG_SUCCESS;
}
int tgTensorSetRequiresGrad(tg_tensor_t t, int r){ if(!t) return TG_ERR_INVALID; t->requires_grad = (r!=0); return TG_SUCCESS; }
