#include <math.h>
#include <string.h>
#include "tg.h"
#include "../../tensor/tensor.h"

/* Utility to check compatible shapes quickly (no broadcasting) */
static int same_shape(const struct tg_tensor *a, const struct tg_tensor *b){
  if(a->rank!=b->rank) return 0;
  for(int i=0;i<a->rank;i++) if(a->shape[i]!=b->shape[i]) return 0;
  return 1;
}

int tgOpAdd(tg_graph_t g, tg_tensor_t a, tg_tensor_t b, tg_tensor_t out){
  (void)g;
  if(!a||!b||!out) return TG_ERR_INVALID;
  if(!same_shape(a,b)||!same_shape(a,out)) return TG_ERR_INVALID;
  for(size_t i=0;i<out->numel;i++) out->data[i] = a->data[i] + b->data[i];
  if(a->requires_grad || b->requires_grad){
    out->requires_grad = 1;
    out->grad_op = TG_OP_ADD; out->p0=a; out->p1=b;
  } else { out->grad_op = TG_OP_NONE; out->p0=out->p1=NULL; }
  return TG_SUCCESS;
}

int tgOpReLU(tg_graph_t g, tg_tensor_t x, tg_tensor_t out){
  (void)g;
  if(!x||!out) return TG_ERR_INVALID;
  if(!same_shape(x,out)) return TG_ERR_INVALID;
  for(size_t i=0;i<out->numel;i++){ float v=x->data[i]; out->data[i] = v>0.f? v:0.f; }
  if(x->requires_grad){ out->requires_grad=1; out->grad_op=TG_OP_RELU; out->p0=x; out->p1=NULL; }
  else { out->grad_op=TG_OP_NONE; out->p0=out->p1=NULL; }
  return TG_SUCCESS;
}

int tgOpMatmul(tg_graph_t g, tg_tensor_t a, tg_tensor_t b, tg_tensor_t out){
  (void)g;
  if(!a||!b||!out) return TG_ERR_INVALID;
  if(a->rank!=2||b->rank!=2||out->rank!=2) return TG_ERR_INVALID;
  int64_t m=a->shape[0], k=a->shape[1], k2=b->shape[0], n=b->shape[1];
  if(k!=k2 || out->shape[0]!=m || out->shape[1]!=n) return TG_ERR_INVALID;
  for(int64_t i=0;i<m;i++){
    for(int64_t j=0;j<n;j++){
      float acc=0.f;
      for(int64_t kk=0;kk<k;kk++) acc += a->data[i*k+kk]*b->data[kk*n+j];
      out->data[i*n+j]=acc;
    }
  }
  if(a->requires_grad || b->requires_grad){
    out->requires_grad=1; out->grad_op=TG_OP_MATMUL; out->p0=a; out->p1=b;
    out->a_rows=m; out->a_cols=k; out->b_cols=n;
  } else { out->grad_op=TG_OP_NONE; out->p0=out->p1=NULL; }
  return TG_SUCCESS;
}

/* CrossEntropy over (N,C) logits + int labels[N] -> scalar */
int tgOpCrossEntropy(tg_graph_t g, tg_tensor_t logits, const int32_t *labels, int64_t n, tg_tensor_t out){
  (void)g;
  if(!logits||!labels||!out) return TG_ERR_INVALID;
  if(logits->rank!=2) return TG_ERR_INVALID;
  int64_t N = logits->shape[0], C = logits->shape[1];
  if(n!=N || out->numel!=1) return TG_ERR_INVALID;

  const double EPS = 1e-12;
  double loss = 0.0;

  for(int64_t i=0;i<N;i++){
    int y = labels[i];
    if(y < 0 || y >= C) return TG_ERR_INVALID;

    /* log-sum-exp in double */
    double maxv = -1e300;
    for(int64_t c=0;c<C;c++){
      double v = (double)logits->data[i*C+c];
      if(v > maxv) maxv = v;
    }
    double sum = 0.0;
    for(int64_t c=0;c<C;c++){
      sum += exp(((double)logits->data[i*C+c]) - maxv);
    }
    double lse = maxv + log(sum + EPS);

    /* CE_i = lse - logit_y  (always >= 0 in exact arithmetic) */
    double ce_i = lse - (double)logits->data[i*C + y];
    if(ce_i < 0.0 && ce_i > -1e-9) ce_i = 0.0;  /* clamp tiny negatives */
    loss += ce_i;
  }

  out->data[0] = (float)(loss / (double)N);

  /* set up backward */
  out->requires_grad = logits->requires_grad;
  out->grad_op = TG_OP_CE; out->p0 = logits; out->p1 = NULL;
  out->ce_labels = labels; out->ce_n = N; out->ce_c = C;
  return TG_SUCCESS;
}

const char* tgVersion(void){ return "0.0.1"; }
