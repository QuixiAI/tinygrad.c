#include "tg.h"
#include "../tensor/tensor.h"
#include <stdlib.h>

struct tg_sgd { float lr, momentum, weight_decay; };

int tgSGDCreate(tg_ctx_t ctx, float lr, float momentum, float weight_decay, tg_sgd_t *out){
  (void)ctx; if(!out) return TG_ERR_INVALID;
  struct tg_sgd *o = (struct tg_sgd*)calloc(1,sizeof(*o)); if(!o) return TG_ERR_NOMEM;
  o->lr=lr; o->momentum=momentum; o->weight_decay=weight_decay; *out=o; return TG_SUCCESS;
}
int tgSGDStep(tg_sgd_t opt, tg_graph_t g){
  (void)g; if(!opt) return TG_ERR_INVALID;
  /* simple: walk all tensors in ctx and update params with requires_grad */
  tg_ctx_t ctx = NULL;
  /* we stored graph as ctx in tgGraphCreate */
  ctx = (tg_ctx_t)g;
  if(!ctx) return TG_ERR_INVALID;
  for (size_t i = 0; i < ctx->all_n; i++) {
    struct tg_tensor *t = ctx->all[i];
    /* Update only leaf params: require grad, have grad, and truly no producing op.
       Note: backward() clears grad_op on visited nodes, so also require p0/p1 == NULL. */
    if (t->requires_grad && t->grad && t->grad_op == TG_OP_NONE && t->p0 == NULL && t->p1 == NULL) {
      for (size_t j = 0; j < t->numel; j++) {
        float g = t->grad[j];
        if (opt->weight_decay != 0.f) g += opt->weight_decay * t->data[j];
        t->data[j] -= opt->lr * g;
      }
    }
  }
  return TG_SUCCESS;
}
int tgSGDDestroy(tg_sgd_t opt){ if(!opt) return TG_ERR_INVALID; free(opt); return TG_SUCCESS; }
