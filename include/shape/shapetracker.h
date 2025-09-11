#ifndef TG_SHAPE_SHAPETRACKER_H
#define TG_SHAPE_SHAPETRACKER_H

#include <stdint.h>
#include <stdbool.h>
#include "shape/view.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct MovementOp MovementOp;
typedef struct UOp UOp;

// Python line 86-87: to_indexed_uops returns tuple[UOp, UOp]
typedef struct IndexedUOps {
    UOp *idx;    // Index expression
    UOp *valid;  // Validity expression
} IndexedUOps;

// ShapeTracker struct definition (needed for other modules)
// Python line 56-58: @dataclass(frozen=True, order=True) class ShapeTracker: views: tuple[View, ...]
typedef struct ShapeTracker {
    void **views;      // Array of View pointers (Python: views: tuple[View, ...])
    int32_t num_views; // Number of views in the array
    // NOTE: shape, ndim, and size are NOT stored - they come from views[-1]
} ShapeTracker;

// ShapeTracker creation and destruction
ShapeTracker *shapetracker_from_shape(const int32_t *shape, int32_t ndim);
ShapeTracker *shapetracker_from_views(View **views, int32_t num_views);
void shapetracker_free(ShapeTracker *st);

// Basic properties
const int32_t *shapetracker_shape(const ShapeTracker *st);
int32_t shapetracker_ndim(const ShapeTracker *st);
int32_t shapetracker_size(const ShapeTracker *st);
bool shapetracker_contiguous(const ShapeTracker *st);
int32_t shapetracker_num_views(const ShapeTracker *st);
int32_t shapetracker_real_size(const ShapeTracker *st);
int32_t shapetracker_offset(const ShapeTracker *st);
// Compute real strides per axis. Returns a newly-allocated array of length ndim.
// For masked/unknown axes, the stride value is INT32_MIN. Caller must free.
int32_t* shapetracker_real_strides(const ShapeTracker* st, bool ignore_valid);
// Default helper: ignore_valid = false
int32_t* shapetracker_real_strides_default(const ShapeTracker* st);
// Return indices of axes with unit stride (==1). Caller must free.
int* shapetracker_unit_stride_axes(const ShapeTracker* st, bool ignore_valid, int* out_count);
// Default helper: ignore_valid = false
int* shapetracker_unit_stride_axes_default(const ShapeTracker* st, int* out_count);

// Comparison operations
bool shapetracker_equal(const ShapeTracker *st1, const ShapeTracker *st2);
bool shapetracker_shape_equal(const ShapeTracker *st1, const ShapeTracker *st2);

// Shape transformations
ShapeTracker *shapetracker_reshape(ShapeTracker *st, const int32_t *new_shape, int32_t new_ndim);
ShapeTracker *shapetracker_permute(ShapeTracker *st, const int32_t *axes, int32_t num_axes);
ShapeTracker *shapetracker_expand(ShapeTracker *st, const int32_t *new_shape, int32_t new_ndim);
ShapeTracker *shapetracker_pad(ShapeTracker *st, const int32_t *pad_before, const int32_t *pad_after, int32_t ndim);
ShapeTracker *shapetracker_shrink(ShapeTracker *st, const int32_t *start, const int32_t *end, int32_t ndim);
ShapeTracker *shapetracker_stride(ShapeTracker *st, const int32_t *strides, int32_t ndim);
ShapeTracker *shapetracker_flip(ShapeTracker *st, const bool *axes, int32_t ndim);

// Movement operations
MovementOp *shapetracker_movement_ops(const ShapeTracker *st);
int32_t shapetracker_num_movement_ops(const ShapeTracker *st);
MovementOp *shapetracker_to_movement_ops(const ShapeTracker *st);

// Advanced operations
ShapeTracker *shapetracker_simplify(ShapeTracker *st);
ShapeTracker *shapetracker_invert(ShapeTracker *st, const int32_t *target_shape, int32_t target_ndim);
bool shapetracker_axis_is_masked(const ShapeTracker *st, int32_t axis);

// View access
const int32_t *shapetracker_last_view_mask(const ShapeTracker *st);

// ShapeTracker composition
ShapeTracker *shapetracker_add(ShapeTracker *st1, ShapeTracker *st2);

// Mathematical operations
ShapeTracker *shapetracker_merge_dims(ShapeTracker *st, int32_t start_dim, int32_t end_dim);
ShapeTracker *shapetracker_split_dim(ShapeTracker *st, int32_t dim, const int32_t *split_sizes, int32_t num_splits);
ShapeTracker *shapetracker_swap_dims(ShapeTracker *st, int32_t dim1, int32_t dim2);
ShapeTracker *shapetracker_flatten(ShapeTracker *st);
ShapeTracker *shapetracker_unflatten(ShapeTracker *st, int32_t dim, const int32_t *target_shape, int32_t target_ndim);
ShapeTracker *shapetracker_squeeze(ShapeTracker *st);
ShapeTracker *shapetracker_squeeze_dim(ShapeTracker *st, int32_t dim);
ShapeTracker *shapetracker_unsqueeze_dim(ShapeTracker *st, int32_t dim);

// Broadcasting and compatibility
bool shapetracker_broadcast_compatible(const ShapeTracker *st1, const ShapeTracker *st2);
ShapeTracker *shapetracker_broadcast_with(ShapeTracker *st1, ShapeTracker *st2);

// Matrix operations
bool shapetracker_matmul_compatible(const ShapeTracker *st1, const ShapeTracker *st2);
ShapeTracker *shapetracker_matmul_result_shape(const ShapeTracker *st1, const ShapeTracker *st2);

// Reduction operations
ShapeTracker *shapetracker_reduce(ShapeTracker *st, int32_t axis, bool keepdims);

// Concatenation and stacking
ShapeTracker *shapetracker_concat(ShapeTracker **sts, int32_t num_sts, int32_t axis);
ShapeTracker *shapetracker_stack(ShapeTracker **sts, int32_t num_sts, int32_t axis);

// Repetition
ShapeTracker *shapetracker_repeat(ShapeTracker *st, int32_t axis, int32_t repeats);

// Rendering to indexed UOps
IndexedUOps *shapetracker_to_indexed_uops(const ShapeTracker *st);
void indexed_uops_free(IndexedUOps *uops);
// Capital S version for compatibility with existing tests
ShapeTracker *ShapeTracker_from_shape(const int32_t *shape, int32_t ndim);

// Python-parity helpers (stubs unless view-level support is added)
// Return set of variables referenced by views. Currently returns empty set.
UOp** shapetracker_vars(const ShapeTracker* st, int* out_count);
// Return mapping of variables to values. Currently returns empty mapping.
UOp** shapetracker_var_vals(const ShapeTracker* st, int* out_count, int** out_vals);
// Unbind variables from views, returning a new shapetracker and mapping. Currently returns a shallow copy and empty mapping.
ShapeTracker* shapetracker_unbind(const ShapeTracker* st, UOp*** out_vars, int** out_vals, int* out_count);
// Substitute variables in views with provided mapping. Currently returns a shallow copy.
ShapeTracker* shapetracker_substitute(const ShapeTracker* st, UOp** from_vars, UOp** to_vals, int count);


#ifdef __cplusplus
}
#endif

#endif // TG_SHAPE_SHAPETRACKER_H
