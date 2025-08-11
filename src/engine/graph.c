#include <math.h>
#include "tg.h"
#include "../tensor/tensor.h"

int tgGraphCreate(tg_ctx_t ctx, tg_graph_t *out){ if(!out||!ctx) return TG_ERR_INVALID; *out=(tg_graph_t)ctx; return TG_SUCCESS; }
int tgGraphRun(tg_graph_t g){ (void)g; return TG_SUCCESS; }

/* backward does a simple reverse DFS starting from loss */
static void backward_recursive(struct tg_tensor *t){
  if(!t || t->grad_op==TG_OP_NONE) return;
  switch(t->grad_op){
        case TG_OP_ADD: {
      struct tg_tensor *A = t->p0;
      struct tg_tensor *B = t->p1;
      /* Pass-through gradient to A if it needs it */
      if (A && A->requires_grad) {
        tg__ensure_grad(A);
        for (size_t i = 0; i < t->numel; i++) A->grad[i] += t->grad[i];
      }
      if (B && B->requires_grad) {
        tg__ensure_grad(B);
        /* If shapes match, elementwise add */
        int same_shape = (A && A->rank == t->rank);
        if (same_shape) {
          for (int i = 0; i < t->rank; i++) if (A->shape[i] != t->shape[i]) { same_shape = 0; break; }
        }
        if (same_shape && B->rank == t->rank) {
          for (size_t i = 0; i < t->numel; i++) B->grad[i] += t->grad[i];
        } else {
          /* Bias broadcast: out is (N,C), B is (C) → dL/dB[c] = sum_i dL/dout[i,c] */
          if (t->rank == 2 && B->rank == 1 && t->shape[1] == B->shape[0]) {
            int64_t N = t->shape[0], C = t->shape[1];
            for (int64_t c = 0; c < C; c++) {
              float acc = 0.f;
              for (int64_t n = 0; n < N; n++) acc += t->grad[n*C + c];
              B->grad[c] += acc;
            }
          } else {
            /* Fallback: unsupported broadcast pattern for now */
            /* You could add more cases later (e.g., 4D NCHW bias) */
          }
        }
      }
      backward_recursive(A);
      backward_recursive(B);
    } break;
    case TG_OP_RELU: {
      if(t->p0 && t->p0->requires_grad){
        tg__ensure_grad(t->p0);
        for(size_t i=0;i<t->numel;i++) t->p0->grad[i] += (t->data[i]>0.f ? t->grad[i] : 0.f);
      }
      backward_recursive(t->p0);
    } break;
    case TG_OP_MATMUL: {
      /* shapes: p0 (A: m x k), p1 (B: k x n), out (m x n) */
      struct tg_tensor *A=t->p0, *B=t->p1;
      if(A && A->requires_grad){
        tg__ensure_grad(A);
        for(int64_t m=0;m<t->a_rows;m++)
          for(int64_t k=0;k<t->a_cols;k++){
            float acc=0.f;
            for(int64_t n=0;n<t->b_cols;n++) acc += t->grad[m*t->b_cols+n]*B->data[k*t->b_cols+n];
            A->grad[m*t->a_cols+k] += acc;
          }
      }
      if(B && B->requires_grad){
        tg__ensure_grad(B);
        for(int64_t k=0;k<t->a_cols;k++)
          for(int64_t n=0;n<t->b_cols;n++){
            float acc=0.f;
            for(int64_t m=0;m<t->a_rows;m++) acc += A->data[m*t->a_cols+k]*t->grad[m*t->b_cols+n];
            B->grad[k*t->b_cols+n] += acc;
          }
      }
      backward_recursive(A); backward_recursive(B);
    } break;
    case TG_OP_CE: {
      /* dL/dlogits = softmax(logits) - onehot(labels) / N */
      struct tg_tensor *logits = t->p0;
      if(logits && logits->requires_grad){
        tg__ensure_grad(logits);
        int64_t N = t->ce_n, C = t->ce_c;
        for(int64_t i=0;i<N;i++){
          /* compute softmax for row i (from logits->data) */
          float maxv=-1e30f;
          for(int64_t c=0;c<C;c++){ float v = logits->data[i*C+c]; if(v>maxv) maxv=v; }
          float sum=0.f;
          for(int64_t c=0;c<C;c++){ float e = (float)expf(logits->data[i*C+c]-maxv); logits->grad[i*C+c] += 0.f; sum += e; }
          for(int64_t c=0;c<C;c++){
            float p = (float)expf(logits->data[i*C+c]-maxv)/sum;
            float g = p;
            if(c == (int64_t) t->ce_labels[i]) g -= 1.f;
            logits->grad[i*C+c] += g / (float)N;   /* mean over N */
          }
        }
      }
      backward_recursive(logits);
    } break;
    default: break;
  }
  /* clear op to avoid double-backward in cycles */
  t->grad_op = TG_OP_NONE;
}

int tgBackward(tg_graph_t g, tg_tensor_t loss){
  (void)g;
  if(!loss || loss->numel!=1) return TG_ERR_INVALID;
  tg__ensure_grad(loss);
  loss->grad[0] = 1.f;
  backward_recursive(loss);
  return TG_SUCCESS;
}
int tgGraphDestroy(tg_graph_t g){ (void)g; return TG_SUCCESS; }
