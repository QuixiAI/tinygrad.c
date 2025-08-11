#ifndef TG_TENSOR_H
#define TG_TENSOR_H
#include <stdint.h>
#include <stddef.h>
#include "tg.h"

typedef enum {
  TG_OP_NONE=0, TG_OP_ADD, TG_OP_RELU, TG_OP_MATMUL, TG_OP_CE
} tg_op_kind;

struct tg_ctx {
  /* track tensors so SGD can find params */
  struct tg_tensor **all; 
  size_t all_n, all_cap;
};

struct tg_tensor {
  tg_ctx_t ctx;
  tg_dtype dtype;
  int64_t *shape; int rank;
  size_t numel;
  float *data;
  float *grad;
  int requires_grad;

  /* autograd links */
  tg_op_kind grad_op;
  struct tg_tensor *p0, *p1;   /* parents (up to 2 for our ops) */
  /* saved context */
  int64_t a_rows, a_cols, b_cols;  /* for matmul */
  int relu_saved;                  /* not used; we recompute mask from output > 0 */
  /* for CE */
  const int32_t *ce_labels;
  int64_t ce_n, ce_c;
};

int tg__calc_numel(const int64_t *shape, int rank, size_t *out);
int tg__ctx_register_tensor(tg_ctx_t c, struct tg_tensor *t);
int tg__ensure_grad(struct tg_tensor *t);
#endif
