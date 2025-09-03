#ifndef TG_SHAPE_VIEW_H
#define TG_SHAPE_VIEW_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct View View;
typedef struct UOp UOp;

// View creation and destruction
View *view_create(const int32_t *shape, int32_t ndim, const int32_t *strides, int32_t num_strides, int32_t offset);
View *view_create_with_mask(const int32_t *shape, int32_t ndim, const int32_t *strides, int32_t num_strides, 
                            int32_t offset, const int32_t *mask_start, const int32_t *mask_end, int32_t mask_ndim);
void view_free(View *view);
View *view_copy(const View *view);

// Basic properties
int32_t view_ndim(const View *view);
int32_t view_size(const View *view);
const int32_t *view_shape(const View *view);
const int32_t *view_strides(const View *view);
int32_t view_offset(const View *view);
bool view_contiguous(const View *view);

// Mask operations
const int32_t *view_mask(const View *view);
const int32_t *view_mask_ranges(const View *view);

// View transformations
View *view_reshape(const View *view, const int32_t *new_shape, int32_t new_ndim);
View *view_permute(const View *view, const int32_t *axes, int32_t num_axes);
View *view_expand(const View *view, const int32_t *new_shape, int32_t new_ndim);
View *view_pad(const View *view, const int32_t *pad_before, const int32_t *pad_after, int32_t ndim);
View *view_shrink(const View *view, const int32_t *start, const int32_t *end, int32_t ndim);
View *view_stride(const View *view, const int32_t *strides, int32_t ndim);
View *view_flip(const View *view, const bool *flip_axes, int32_t num_axes);

// Advanced view operations
View *view_unsafe_resize(const View *view, const int32_t *new_shape, int32_t new_ndim);
View *view_invert(const View *view, const int32_t *out_shape, int32_t out_ndim);
View *view_minify(const View *view);
View *view_simplify(const View *view);

// View composition
View *view_add(const View *v1, const View *v2);

// Indexing
int32_t view_index_to_offset(const View *view, const int32_t *indices, int32_t num_indices);

// Convert View to indexed UOps (for symbolic computation)
void view_to_indexed_uops(const View* view, UOp** idxs, int idxs_count, 
                         UOp* vexpr, UOp** out_idx, UOp** out_valid);

// Merge dims utility
typedef struct {
    int32_t shape;
    int32_t stride;
    int32_t size;
} MergeDim;

MergeDim *merge_dims(const int32_t *shape, const int32_t *strides, int32_t ndim, 
                     const int32_t *mask_start, const int32_t *mask_end, int32_t *result_count);
void merge_dims_free(MergeDim *dims);

#ifdef __cplusplus
}
#endif

#endif // TG_SHAPE_VIEW_H
