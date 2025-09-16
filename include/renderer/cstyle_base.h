#ifndef SRC_RENDERER_CSTYLE_BASE_H
#define SRC_RENDERER_CSTYLE_BASE_H

#include "renderer/renderer.h"
#include "uop/uop.h"

#ifdef __cplusplus
extern "C" {
#endif

// Minimal shared C-Style rendering context utilities (names, buffers, counts)

typedef struct CStyleCtx {
  // Name map (UOp* -> name)
  UOp** nm_keys; char** nm_vals; int nm_count; int nm_cap;
  // Child counts (UOp* -> count of references)
  UOp** cc_keys; int* cc_vals; int cc_count; int cc_cap;
  // Buffers (DEFINE_GLOBAL) tracked in first-seen order
  int buf_ids[256]; const DType* buf_dtypes[256]; int buf_writable[256]; int buf_count;
} CStyleCtx;

// Initialize and free context (no-op for now; reserved for future)
void cstyle_ctx_init(CStyleCtx* ctx);
void cstyle_ctx_free(CStyleCtx* ctx);

// Collect DEFINE_GLOBAL buffers in first-seen order.
// Returns count, fills ids[] and dtypes[] up to max entries.
int cstyle_ctx_collect_bufs(CStyleCtx* ctx, UOp** uops, int n, int* ids, const DType** dtypes, int max);

// Mark buffers written via STORE (placeholder for future parity)
void cstyle_ctx_mark_writes(CStyleCtx* ctx, UOp** uops, int n);

// Query if a buffer id is writable (STORE was seen targeting it)
int cstyle_ctx_buf_writable(CStyleCtx* ctx, int id);

// Compute child counts for uops graph
void cstyle_ctx_compute_child_counts(CStyleCtx* ctx, UOp** uops, int n);

// Assign names (SSA-like) for nodes that need identifiers
void cstyle_ctx_assign_names(CStyleCtx* ctx, UOp** uops, int n);

// Lookup or get assigned name
const char* cstyle_ctx_name_for(CStyleCtx* ctx, UOp* u);

#ifdef __cplusplus
}
#endif
#endif /* SRC_RENDERER_CSTYLE_BASE_H */
