#ifndef TG_H
#define TG_H
#ifdef __cplusplus
extern "C" { 
#endif
#include <stddef.h>
#include <stdint.h>

#define TG_API

typedef struct tg_ctx    *tg_ctx_t;
typedef struct tg_tensor *tg_tensor_t;
typedef struct tg_graph  *tg_graph_t;

enum { TG_SUCCESS=0, TG_ERR_INVALID=1, TG_ERR_NOMEM=2, TG_ERR_UNIMPL=3, TG_ERR_RUNTIME=4 };
typedef enum { TG_F32=0 } tg_dtype;

/* Context */
TG_API int tgCreateContext(tg_ctx_t *out);
TG_API int tgDestroyContext(tg_ctx_t ctx);

/* Tensor */
TG_API int tgTensorCreate(tg_ctx_t ctx, tg_dtype dtype, const int64_t *shape, int rank, tg_tensor_t *out);
TG_API int tgTensorFill(tg_tensor_t t, float v);
TG_API int tgTensorUpload(tg_tensor_t t, const void *data, size_t nbytes);
TG_API int tgTensorDownload(tg_tensor_t t, void *data, size_t nbytes);
TG_API int tgTensorDestroy(tg_tensor_t t);
TG_API int tgTensorSetRequiresGrad(tg_tensor_t t, int requires_grad);
TG_API int tgZeroGrad(tg_ctx_t ctx);

/* Graph + Autograd (eager-ish) */
TG_API int tgGraphCreate(tg_ctx_t ctx, tg_graph_t *out);
TG_API int tgGraphRun(tg_graph_t g);                  /* no-op for eager */
TG_API int tgBackward(tg_graph_t g, tg_tensor_t loss);
TG_API int tgGraphDestroy(tg_graph_t g);

/* Ops we implement */
TG_API int tgOpAdd(tg_graph_t g, tg_tensor_t a, tg_tensor_t b, tg_tensor_t out);
TG_API int tgOpReLU(tg_graph_t g, tg_tensor_t x, tg_tensor_t out);
TG_API int tgOpMatmul(tg_graph_t g, tg_tensor_t a, tg_tensor_t b, tg_tensor_t out);
/* Combined CrossEntropyLoss: logits (N,C) + labels (N int32) -> scalar (1) */
TG_API int tgOpCrossEntropy(tg_graph_t g, tg_tensor_t logits, const int32_t *labels, int64_t n, tg_tensor_t out);

/* Optimizer: SGD */
typedef struct tg_sgd *tg_sgd_t;
TG_API int tgSGDCreate(tg_ctx_t ctx, float lr, float momentum, float weight_decay, tg_sgd_t *out);
TG_API int tgSGDStep(tg_sgd_t opt, tg_graph_t g);
TG_API int tgSGDDestroy(tg_sgd_t opt);

/* Version */
TG_API const char* tgVersion(void);

#ifdef __cplusplus
}
#endif
#endif
