#include "tg.h"
#include "../src/tensor/tensor.h"   // temporary: to touch internals for demo
#include <stdio.h>
#include <stdlib.h>

static float accuracy_of_batch(tg_tensor_t logits, const int32_t *labels, int N, int C){
  int correct = 0;
  for(int i=0;i<N;i++){
    int argmax = 0; float best = logits->data[i*C+0];
    for(int c=1;c<C;c++){
      float v = logits->data[i*C+c];
      if(v > best){ best = v; argmax = c; }
    }
    if(argmax == labels[i]) correct++;
  }
  return (float)correct / (float)N;
}

/* helper: random float in [-s, s] */
static float frand(float s){ return ((float)rand()/(float)RAND_MAX)*2.f*s - s; }

int main(void){
  srand(123);
  tg_ctx_t ctx; tgCreateContext(&ctx);
  tg_graph_t g; tgGraphCreate(ctx, &g);

  /* Toy dataset: two 2D Gaussians (class 0 at -1,-1, class 1 at +1,+1) */
  const int N=256, IN=2, H=32, C=2;
  float *X = (float*)malloc(N*IN*sizeof(float));
  int32_t *Y = (int32_t*)malloc(N*sizeof(int32_t));
  for(int i=0;i<N;i++){
    int cls = (i < N/2) ? 0 : 1;
    Y[i]=cls;
    float cx = cls? 1.0f : -1.0f;
    float cy = cls? 1.0f : -1.0f;
    X[i*IN+0] = cx + frand(0.4f);
    X[i*IN+1] = cy + frand(0.4f);
  }

  /* Params */
  tg_tensor_t W1,b1,W2,b2;
  { int64_t sW1[2]={IN,H}; tgTensorCreate(ctx,TG_F32,sW1,2,&W1);
    int64_t sb1[1]={H};    tgTensorCreate(ctx,TG_F32,sb1,1,&b1);
    int64_t sW2[2]={H,C};  tgTensorCreate(ctx,TG_F32,sW2,2,&W2);
    int64_t sb2[1]={C};    tgTensorCreate(ctx,TG_F32,sb2,1,&b2);
    for(int i=0;i<IN*H;i++) W1->data[i]=frand(0.5f);
    for(int i=0;i<H;i++)    b1->data[i]=0.f;
    for(int i=0;i<H*C;i++)  W2->data[i]=frand(0.5f);
    for(int i=0;i<C;i++)    b2->data[i]=0.f;
    tgTensorSetRequiresGrad(W1,1); tgTensorSetRequiresGrad(b1,1);
    tgTensorSetRequiresGrad(W2,1); tgTensorSetRequiresGrad(b2,1);
  }

  /* Work tensors */
  const int BS=64; int steps = 200;
  tg_tensor_t x,h,hb,h_act,logits_pre,logits,loss;
  int64_t sx[2]={BS,IN}, sh[2]={BS,H}, sc[2]={BS,C}, s1[1]={1};
  tgTensorCreate(ctx,TG_F32,sx,2,&x);
  tgTensorCreate(ctx,TG_F32,sh,2,&h);
  tgTensorCreate(ctx,TG_F32,sh,2,&hb);
  tgTensorCreate(ctx,TG_F32,sh,2,&h_act);
  tgTensorCreate(ctx,TG_F32,sc,2,&logits_pre);   // matmul output
  tgTensorCreate(ctx,TG_F32,sc,2,&logits);       // + bias lives here
  tgTensorCreate(ctx,TG_F32,s1,1,&loss);

  tg_sgd_t opt; tgSGDCreate(ctx, 0.1f, 0.0f, 1e-4f, &opt);

  for(int step=0; step<steps; step++){
    int off = (step*BS) % N;
    tgTensorUpload(x, &X[off*IN], BS*IN*sizeof(float));

    /* h = x @ W1 + b1 */
    tgOpMatmul(g, x, W1, h);
    for(int i=0;i<BS;i++) for(int j=0;j<H;j++) hb->data[i*H+j] = h->data[i*H+j] + b1->data[j];
    hb->requires_grad = (h->requires_grad || b1->requires_grad);
    hb->grad_op = TG_OP_ADD; hb->p0 = h; hb->p1 = b1;

    tgOpReLU(g, hb, h_act);

    /* logits_pre = h_act @ W2 */
    tgOpMatmul(g, h_act, W2, logits_pre);

    /* logits = logits_pre + b2  (NO self-edge!) */
    for(int i=0;i<BS;i++) for(int j=0;j<C;j++) logits->data[i*C+j] = logits_pre->data[i*C+j] + b2->data[j];
    logits->requires_grad = 1;
    logits->grad_op = TG_OP_ADD; logits->p0 = logits_pre; logits->p1 = b2;

    /* CE loss */
    tgOpCrossEntropy(g, logits, &Y[off], BS, loss);

    tgZeroGrad(ctx);
    tgBackward(g, loss);
    tgSGDStep(opt, g);

    if(step%20==0) {
      float acc = accuracy_of_batch(logits, &Y[off], BS, C);
      printf("step %3d  loss=%.4f  acc=%.2f\n", step, loss->data[0], acc);
    }

  }

  tgSGDDestroy(opt);
  tgTensorDestroy(loss); tgTensorDestroy(logits); tgTensorDestroy(logits_pre);
  tgTensorDestroy(h_act); tgTensorDestroy(hb); tgTensorDestroy(h); tgTensorDestroy(x);
  tgTensorDestroy(W1); tgTensorDestroy(b1); tgTensorDestroy(W2); tgTensorDestroy(b2);
  free(X); free(Y);
  tgGraphDestroy(g); tgDestroyContext(ctx);
  return 0;
}
