#include "shape/shapetracker.h"
#include "shape/view.h"
#include "helpers/helpers.h"
#include "dtype/dtype.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "uop/symbolic.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Macro to mark unused parameters
#define UNUSED(x) ((void)(x))


// Python line 56-58: @dataclass(frozen=True, order=True) class ShapeTracker:
// ShapeTracker struct is defined in the header file

// Helper function to copy a ShapeTracker
static ShapeTracker *shapetracker_copy(const ShapeTracker *st) {
    if (!st) return NULL;
    
    ShapeTracker *copy = malloc(sizeof(ShapeTracker));
    if (!copy) return NULL;
    
    copy->num_views = st->num_views;
    copy->views = malloc(copy->num_views * sizeof(void*));
    if (!copy->views) {
        free(copy);
        return NULL;
    }
    
    for (int32_t i = 0; i < st->num_views; i++) {
        copy->views[i] = view_copy((View*)st->views[i]);
        if (!copy->views[i]) {
            for (int32_t j = 0; j < i; j++) {
                view_free((View*)copy->views[j]);
            }
            free(copy->views);
            free(copy);
            return NULL;
        }
    }
    
    return copy;
}

struct MovementOp {
    int op_type;
    int32_t *params;
    int32_t num_params;
};

// IndexedUOps struct is now defined in the header file

// Python line 72-73: @staticmethod def from_shape
ShapeTracker *shapetracker_from_shape(const int32_t *shape, int32_t ndim) {
    if (!shape || ndim <= 0) return NULL;
    
    ShapeTracker *st = malloc(sizeof(ShapeTracker));
    if (!st) return NULL;
    
    // Create a single View from the shape
    View *view = view_create(shape, ndim, NULL, 0, 0);
    if (!view) {
        free(st);
        return NULL;
    }
    
    st->views = malloc(sizeof(View*));
    if (!st->views) {
        view_free(view);
        free(st);
        return NULL;
    }
    
    st->views[0] = view;
    st->num_views = 1;
    
    return st;
}

ShapeTracker *shapetracker_from_views(View **views, int32_t num_views) {
    if (!views || num_views <= 0) return NULL;
    
    ShapeTracker *st = malloc(sizeof(ShapeTracker));
    if (!st) return NULL;
    
    st->views = malloc(num_views * sizeof(View*));
    if (!st->views) {
        free(st);
        return NULL;
    }
    
    // Copy view pointers
    for (int32_t i = 0; i < num_views; i++) {
        st->views[i] = views[i];
    }
    st->num_views = num_views;
    
    // Shape is accessed from views[-1], not stored (Python line 79: @property def shape)
    
    return st;
}

void shapetracker_free(ShapeTracker *st) {
    if (!st) return;
    
    // Free views
    if (st->views) {
        // Don't free individual views as they may be shared
        // This is a simplification - proper reference counting would be better
        free(st->views);
    }
    
    // Free the ShapeTracker itself
    free(st);
}

// Python line 78-79: @property def shape: return self.views[-1].shape
const int32_t *shapetracker_shape(const ShapeTracker *st) {
    if (!st || !st->views || st->num_views == 0) return NULL;
    View *last_view = (View*)st->views[st->num_views - 1];
    return view_shape(last_view);
}

int32_t shapetracker_ndim(const ShapeTracker *st) {
    if (!st || !st->views || st->num_views == 0) return -1;
    View *last_view = (View*)st->views[st->num_views - 1];
    return view_ndim(last_view);
}

// Python line 81-82: @property def size
// Python line 81-82: @property def size: return self.views[-1].size()
int32_t shapetracker_size(const ShapeTracker *st) {
    if (!st || !st->views || st->num_views == 0) return -1;
    View *last_view = (View*)st->views[st->num_views - 1];
    return view_size(last_view);
}

// Python line 75-76: @property def contiguous
bool shapetracker_contiguous(const ShapeTracker *st) {
    if (!st || st->num_views != 1) return false;
    View *view = (View*)st->views[0];
    return view_contiguous(view);
}

int32_t shapetracker_num_views(const ShapeTracker *st) {
    if (!st) return -1;
    return st->num_views;
}

// Python line 89-95: def real_size
int32_t shapetracker_real_size(const ShapeTracker *st) {
    if (!st) return -1;
    
    // Python: if 0 in self.shape: return 0
    const int32_t *shape = shapetracker_shape(st);
    int32_t ndim = shapetracker_ndim(st);
    for (int32_t i = 0; i < ndim; i++) {
        if (shape[i] == 0) return 0;
    }
    
    // Full implementation requires IndexedUOps, but we can handle common cases
    if (st->num_views == 0) return 0;
    
    View *first_view = (View*)st->views[0];
    
    // Calculate the real size by finding both min and max indices
    const int32_t *first_shape = view_shape(first_view);
    const int32_t *first_strides = view_strides(first_view);
    int32_t first_ndim = view_ndim(first_view);
    int32_t offset = view_offset(first_view);
    
    // Handle mask if present
    const int32_t *mask = view_mask_ranges(first_view);
    
    int32_t min_index = offset;
    int32_t max_index = offset;
    
    for (int32_t i = 0; i < first_ndim; i++) {
        int32_t dim_size = first_shape[i];
        int32_t dim_start = 0;
        
        // Apply mask if present
        if (mask) {
            dim_start = mask[i * 2];
            int32_t end = mask[i * 2 + 1];
            dim_size = end - dim_start;
        }
        
        // Only process if dimension size > 1 and stride != 0
        if (dim_size > 1 && first_strides[i] != 0) {
            int32_t stride = first_strides[i];
            if (stride > 0) {
                // Positive stride: min at start, max at end
                min_index += stride * dim_start;
                max_index += stride * (dim_start + dim_size - 1);
            } else {
                // Negative stride: max at start, min at end
                max_index += stride * dim_start;
                min_index += stride * (dim_start + dim_size - 1);
            }
        } else if (mask && first_strides[i] != 0) {
            // Single element with mask
            min_index += first_strides[i] * dim_start;
            max_index += first_strides[i] * dim_start;
        }
    }
    
    // Return the range size
    return (max_index > min_index ? max_index : min_index) + 1;
}

int32_t shapetracker_offset(const ShapeTracker *st) {
    UNUSED(st);
    return -1; // Stub - indicates unimplemented
}

bool shapetracker_equal(const ShapeTracker *st1, const ShapeTracker *st2) {
    if (!st1 || !st2) return st1 == st2;
    if (st1->num_views != st2->num_views) return false;
    
    int32_t ndim1 = shapetracker_ndim(st1);
    int32_t ndim2 = shapetracker_ndim(st2);
    if (ndim1 != ndim2) return false;
    
    // Check if shapes are equal
    const int32_t *shape1 = shapetracker_shape(st1);
    const int32_t *shape2 = shapetracker_shape(st2);
    for (int32_t i = 0; i < ndim1; i++) {
        if (shape1[i] != shape2[i]) return false;
    }
    
    // For complete equality, would need to check all views
    // For now, just check shape equality
    return true;
}

bool shapetracker_shape_equal(const ShapeTracker *st1, const ShapeTracker *st2) {
    if (!st1 || !st2) return st1 == st2;
    
    int32_t ndim1 = shapetracker_ndim(st1);
    int32_t ndim2 = shapetracker_ndim(st2);
    if (ndim1 != ndim2) return false;
    
    // Check if shapes are equal
    const int32_t *shape1 = shapetracker_shape(st1);
    const int32_t *shape2 = shapetracker_shape(st2);
    for (int32_t i = 0; i < ndim1; i++) {
        if (shape1[i] != shape2[i]) return false;
    }
    
    return true;
}

// Python line 130-132: def reshape
ShapeTracker *shapetracker_reshape(ShapeTracker *st, const int32_t *new_shape, int32_t new_ndim) {
    if (!st || !new_shape || new_ndim <= 0) return NULL;
    
    // Try to reshape the last view (Python: if getenv("MERGE_VIEW", 1) and ...)
    View *last_view = (View*)st->views[st->num_views - 1];
    View *new_view = view_reshape(last_view, new_shape, new_ndim);
    
    ShapeTracker *result = malloc(sizeof(ShapeTracker));
    if (!result) {
        if (new_view) view_free(new_view);
        return NULL;
    }
    
    if (new_view) {
        // Successfully reshaped last view
        result->views = malloc(st->num_views * sizeof(View*));
        if (!result->views) {
            view_free(new_view);
            free(result);
            return NULL;
        }
        
        // Copy all views except last
        for (int32_t i = 0; i < st->num_views - 1; i++) {
            result->views[i] = st->views[i];  // Shallow copy
        }
        result->views[st->num_views - 1] = new_view;
        result->num_views = st->num_views;
    } else {
        // Add a new view (Python: return ShapeTracker(self.views + (View.create(new_shape), )))
        result->views = malloc((st->num_views + 1) * sizeof(View*));
        if (!result->views) {
            free(result);
            return NULL;
        }
        
        // Copy all existing views
        for (int32_t i = 0; i < st->num_views; i++) {
            result->views[i] = st->views[i];  // Shallow copy
        }
        
        // Add new view
        View *created_view = view_create(new_shape, new_ndim, NULL, 0, 0);
        if (!created_view) {
            free(result->views);
            free(result);
            return NULL;
        }
        result->views[st->num_views] = created_view;
        result->num_views = st->num_views + 1;
    }
    
    // Shape comes from views[-1], no need to store
    
    return result;
}

// Python line 127: def permute
ShapeTracker *shapetracker_permute(ShapeTracker *st, const int32_t *axes, int32_t num_axes) {
    if (!st || !axes || num_axes <= 0) return NULL;
    
    View *last_view = (View*)st->views[st->num_views - 1];
    View *permuted = view_permute(last_view, axes, num_axes);
    if (!permuted) return NULL;
    
    ShapeTracker *result = malloc(sizeof(ShapeTracker));
    if (!result) {
        view_free(permuted);
        return NULL;
    }
    
    result->views = malloc(st->num_views * sizeof(View*));
    if (!result->views) {
        view_free(permuted);
        free(result);
        return NULL;
    }
    
    // Copy all views except last
    for (int32_t i = 0; i < st->num_views - 1; i++) {
        result->views[i] = st->views[i];  // Shallow copy
    }
    result->views[st->num_views - 1] = permuted;
    result->num_views = st->num_views;
    
    // Update shape
    // Shape comes from views[-1], no need to store
    
    return result;
}

// Python line 126: def expand
ShapeTracker *shapetracker_expand(ShapeTracker *st, const int32_t *new_shape, int32_t new_ndim) {
    if (!st || !new_shape || new_ndim <= 0) return NULL;
    
    View *last_view = (View*)st->views[st->num_views - 1];
    View *expanded = view_expand(last_view, new_shape, new_ndim);
    if (!expanded) return NULL;
    
    ShapeTracker *result = malloc(sizeof(ShapeTracker));
    if (!result) {
        view_free(expanded);
        return NULL;
    }
    
    result->views = malloc(st->num_views * sizeof(View*));
    if (!result->views) {
        view_free(expanded);
        free(result);
        return NULL;
    }
    
    // Copy all views except last
    for (int32_t i = 0; i < st->num_views - 1; i++) {
        result->views[i] = st->views[i];  // Shallow copy
    }
    result->views[st->num_views - 1] = expanded;
    result->num_views = st->num_views;
    
    // Update shape
    // Shape comes from views[-1], no need to store
    
    return result;
}

// Python line 124: def pad
ShapeTracker *shapetracker_pad(ShapeTracker *st, const int32_t *pad_before, const int32_t *pad_after, int32_t ndim) {
    if (!st || !pad_before || !pad_after || ndim <= 0) return NULL;
    
    View *last_view = (View*)st->views[st->num_views - 1];
    View *padded = view_pad(last_view, pad_before, pad_after, ndim);
    if (!padded) return NULL;
    
    ShapeTracker *result = malloc(sizeof(ShapeTracker));
    if (!result) {
        view_free(padded);
        return NULL;
    }
    
    result->views = malloc(st->num_views * sizeof(View*));
    if (!result->views) {
        view_free(padded);
        free(result);
        return NULL;
    }
    
    // Copy all views except last
    for (int32_t i = 0; i < st->num_views - 1; i++) {
        result->views[i] = st->views[i];  // Shallow copy
    }
    result->views[st->num_views - 1] = padded;
    result->num_views = st->num_views;
    
    // Update shape
    // Shape comes from views[-1], no need to store
    
    return result;
}

// Python line 125: def shrink
ShapeTracker *shapetracker_shrink(ShapeTracker *st, const int32_t *start, const int32_t *end, int32_t ndim) {
    if (!st || !start || !end || ndim <= 0) return NULL;
    
    View *last_view = (View*)st->views[st->num_views - 1];
    View *shrunk = view_shrink(last_view, start, end, ndim);
    if (!shrunk) return NULL;
    
    ShapeTracker *result = malloc(sizeof(ShapeTracker));
    if (!result) {
        view_free(shrunk);
        return NULL;
    }
    
    result->views = malloc(st->num_views * sizeof(View*));
    if (!result->views) {
        view_free(shrunk);
        free(result);
        return NULL;
    }
    
    // Copy all views except last
    for (int32_t i = 0; i < st->num_views - 1; i++) {
        result->views[i] = st->views[i];  // Shallow copy
    }
    result->views[st->num_views - 1] = shrunk;
    result->num_views = st->num_views;
    
    // Update shape
    // Shape comes from views[-1], no need to store
    
    return result;
}

ShapeTracker *shapetracker_stride(ShapeTracker *st, const int32_t *strides, int32_t ndim) {
    UNUSED(st);
    UNUSED(strides);
    UNUSED(ndim);
    return NULL; // Stub - not implemented
}

// Python line 128: def flip
ShapeTracker *shapetracker_flip(ShapeTracker *st, const bool *axes, int32_t ndim) {
    if (!st || !axes || ndim <= 0) return NULL;
    
    View *last_view = (View*)st->views[st->num_views - 1];
    View *flipped = view_flip(last_view, axes, ndim);
    if (!flipped) return NULL;
    
    ShapeTracker *result = malloc(sizeof(ShapeTracker));
    if (!result) {
        view_free(flipped);
        return NULL;
    }
    
    result->views = malloc(st->num_views * sizeof(View*));
    if (!result->views) {
        view_free(flipped);
        free(result);
        return NULL;
    }
    
    // Copy all views except last
    for (int32_t i = 0; i < st->num_views - 1; i++) {
        result->views[i] = st->views[i];  // Shallow copy
    }
    result->views[st->num_views - 1] = flipped;
    result->num_views = st->num_views;
    
    // Copy shape (flip doesn't change shape)
    // Shape comes from views[-1], no need to store
    
    return result;
}

MovementOp *shapetracker_movement_ops(const ShapeTracker *st) {
    UNUSED(st);
    return NULL; // Stub - not implemented
}

int32_t shapetracker_num_movement_ops(const ShapeTracker *st) {
    UNUSED(st);
    return -1; // Stub - indicates unimplemented
}

MovementOp *shapetracker_to_movement_ops(const ShapeTracker *st) {
    UNUSED(st);
    return NULL; // Stub - not implemented
}

// Python line 117-120: def simplify
// NOTE: Always returns a new ShapeTracker to avoid double-free issues
ShapeTracker *shapetracker_simplify(ShapeTracker *st) {
    if (!st || st->num_views < 2) {
        // Nothing to simplify, return a copy
        ShapeTracker *copy = malloc(sizeof(ShapeTracker));
        if (!copy) return NULL;
        
        copy->views = malloc(st->num_views * sizeof(View*));
        if (!copy->views) {
            free(copy);
            return NULL;
        }
        
        for (int32_t i = 0; i < st->num_views; i++) {
            copy->views[i] = st->views[i];  // Shallow copy
        }
        copy->num_views = st->num_views;
        
        return copy;
    }
    
    // Try to merge the last two views
    View *second_last = (View*)st->views[st->num_views - 2];
    View *last = (View*)st->views[st->num_views - 1];
    View *merged = view_add(second_last, last);
    
    if (merged) {
        // Successfully merged
        ShapeTracker *result = malloc(sizeof(ShapeTracker));
        if (!result) {
            view_free(merged);
            return NULL;
        }
        
        result->views = malloc((st->num_views - 1) * sizeof(View*));
        if (!result->views) {
            view_free(merged);
            free(result);
            return NULL;
        }
        
        // Copy all views except last two
        for (int32_t i = 0; i < st->num_views - 2; i++) {
            result->views[i] = st->views[i];  // Shallow copy
        }
        result->views[st->num_views - 2] = merged;
        result->num_views = st->num_views - 1;
        
        // Copy shape from original
    // Shape comes from views[-1], no need to store
        
        // Recursively simplify
        return shapetracker_simplify(result);
    }
    
    // Could not simplify, return a copy
    ShapeTracker *copy = malloc(sizeof(ShapeTracker));
    if (!copy) return NULL;
    
    copy->views = malloc(st->num_views * sizeof(View*));
    if (!copy->views) {
        free(copy);
        return NULL;
    }
    
    for (int32_t i = 0; i < st->num_views; i++) {
        copy->views[i] = st->views[i];  // Shallow copy
    }
    copy->num_views = st->num_views;
    
    return copy;
}

// Python lines 65-70: invert
// def invert(self, out_shape:tuple[sint, ...]) -> ShapeTracker|None:
//   inverted_views:list[View] = []
//   for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]):
//     if (inverted:= v.invert(s)) is None: return None
//     inverted_views.append(inverted)
//   return ShapeTracker(tuple(inverted_views)).reshape(out_shape)
ShapeTracker *shapetracker_invert(ShapeTracker *st, const int32_t *out_shape, int32_t out_ndim) {
    if (!st || !st->views || st->num_views == 0) return NULL;
    
    // Line 66: inverted_views:list[View] = []
    View **inverted_views = malloc(st->num_views * sizeof(View*));
    if (!inverted_views) return NULL;
    
    // Line 67: for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]):
    // Process views in reverse order
    for (int32_t i = 0; i < st->num_views; i++) {
        View *v = (View*)st->views[st->num_views - 1 - i];
        
        // Determine shape s for this view
        const int32_t *s;
        int32_t s_ndim;
        if (i < st->num_views - 1) {
            // Use shape from previous view (in reverse order)
            View *prev_view = (View*)st->views[st->num_views - 2 - i];
            s = view_shape(prev_view);
            s_ndim = view_ndim(prev_view);
        } else {
            // Last iteration, use out_shape
            s = out_shape;
            s_ndim = out_ndim;
        }
        
        // Line 68: if (inverted:= v.invert(s)) is None: return None
        View *inverted = view_invert(v, s, s_ndim);
        if (!inverted) {
            // Free previously inverted views
            for (int32_t j = 0; j < i; j++) {
                view_free(inverted_views[j]);
            }
            free(inverted_views);
            return NULL;
        }
        
        // Line 69: inverted_views.append(inverted)
        inverted_views[i] = inverted;
    }
    
    // Line 70: return ShapeTracker(tuple(inverted_views)).reshape(out_shape)
    // Create ShapeTracker from inverted views
    ShapeTracker *result = malloc(sizeof(ShapeTracker));
    if (!result) {
        for (int32_t i = 0; i < st->num_views; i++) {
            view_free(inverted_views[i]);
        }
        free(inverted_views);
        return NULL;
    }
    
    result->num_views = st->num_views;
    result->views = malloc(result->num_views * sizeof(void*));
    if (!result->views) {
        for (int32_t i = 0; i < st->num_views; i++) {
            view_free(inverted_views[i]);
        }
        free(inverted_views);
        free(result);
        return NULL;
    }
    
    // Copy inverted views to result (they're already in the right order)
    for (int32_t i = 0; i < st->num_views; i++) {
        result->views[i] = inverted_views[i];
    }
    free(inverted_views);
    
    // Apply reshape to out_shape
    ShapeTracker *reshaped = shapetracker_reshape(result, out_shape, out_ndim);
    shapetracker_free(result);
    
    return reshaped;
}

// Python line 112-115: def axis_is_masked
bool shapetracker_axis_is_masked(const ShapeTracker *st, int32_t axis) {
    if (!st || axis < 0 || axis >= shapetracker_ndim(st)) return false;
    
    // Check if the axis is masked in the last view
    if (st->num_views > 0) {
        View *last_view = (View*)st->views[st->num_views - 1];
        const int32_t *mask = view_mask_ranges(last_view);
        if (mask) {
            // Check if this axis has a non-full mask
            const int32_t *shape = view_shape(last_view);
            int32_t start = mask[axis * 2];
            int32_t end = mask[axis * 2 + 1];
            if (start != 0 || end != shape[axis]) {
                return true;
            }
        }
    }
    
    return false;
}

// Helper to get the mask of the last view
const int32_t *shapetracker_last_view_mask(const ShapeTracker *st) {
    if (!st || !st->views || st->num_views == 0) return NULL;
    
    View *last_view = (View*)st->views[st->num_views - 1];
    // Check if the last view has a mask
    // We need to check the View structure for mask information
    // In Python, this would be: self.views[-1].mask
    return view_mask(last_view);
}

// Python lines 60-63: __add__
// def __add__(self, st:ShapeTracker) -> ShapeTracker:
//   ret = self
//   for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
//   return ret
ShapeTracker *shapetracker_add(ShapeTracker *st1, ShapeTracker *st2) {
    if (!st1 || !st2) return NULL;
    
    // Line 61: ret = self
    ShapeTracker *ret = shapetracker_copy(st1);
    if (!ret) return NULL;
    
    // Line 62: for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify()
    for (int32_t i = 0; i < st2->num_views; i++) {
        // Create new ShapeTracker with ret.views + (v,)
        ShapeTracker *new_st = malloc(sizeof(ShapeTracker));
        if (!new_st) {
            shapetracker_free(ret);
            return NULL;
        }
        
        new_st->num_views = ret->num_views + 1;
        new_st->views = malloc(new_st->num_views * sizeof(void*));
        if (!new_st->views) {
            free(new_st);
            shapetracker_free(ret);
            return NULL;
        }
        
        // Copy existing views from ret
        for (int32_t j = 0; j < ret->num_views; j++) {
            new_st->views[j] = view_copy((View*)ret->views[j]);
        }
        // Add the new view
        new_st->views[ret->num_views] = view_copy((View*)st2->views[i]);
        
        // Free old ret
        shapetracker_free(ret);
        
        // Simplify and update ret
        ret = shapetracker_simplify(new_st);
        shapetracker_free(new_st);
        if (!ret) return NULL;
    }
    
    // Line 63: return ret
    return ret;
}

// Mathematical operations stubs
ShapeTracker *shapetracker_merge_dims(ShapeTracker *st, int32_t start_dim, int32_t end_dim) {
    UNUSED(st);
    UNUSED(start_dim);
    UNUSED(end_dim);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_split_dim(ShapeTracker *st, int32_t dim, const int32_t *split_sizes, int32_t num_splits) {
    UNUSED(st);
    UNUSED(dim);
    UNUSED(split_sizes);
    UNUSED(num_splits);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_swap_dims(ShapeTracker *st, int32_t dim1, int32_t dim2) {
    UNUSED(st);
    UNUSED(dim1);
    UNUSED(dim2);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_flatten(ShapeTracker *st) {
    UNUSED(st);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_unflatten(ShapeTracker *st, int32_t dim, const int32_t *target_shape, int32_t target_ndim) {
    UNUSED(st);
    UNUSED(dim);
    UNUSED(target_shape);
    UNUSED(target_ndim);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_squeeze(ShapeTracker *st) {
    UNUSED(st);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_squeeze_dim(ShapeTracker *st, int32_t dim) {
    UNUSED(st);
    UNUSED(dim);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_unsqueeze_dim(ShapeTracker *st, int32_t dim) {
    UNUSED(st);
    UNUSED(dim);
    return NULL; // Stub - not implemented
}

bool shapetracker_broadcast_compatible(const ShapeTracker *st1, const ShapeTracker *st2) {
    UNUSED(st1);
    UNUSED(st2);
    return false; // Stub - indicates unimplemented
}

ShapeTracker *shapetracker_broadcast_with(ShapeTracker *st1, ShapeTracker *st2) {
    UNUSED(st1);
    UNUSED(st2);
    return NULL; // Stub - not implemented
}

bool shapetracker_matmul_compatible(const ShapeTracker *st1, const ShapeTracker *st2) {
    UNUSED(st1);
    UNUSED(st2);
    return false; // Stub - indicates unimplemented
}

ShapeTracker *shapetracker_matmul_result_shape(const ShapeTracker *st1, const ShapeTracker *st2) {
    UNUSED(st1);
    UNUSED(st2);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_reduce(ShapeTracker *st, int32_t axis, bool keepdims) {
    UNUSED(st);
    UNUSED(axis);
    UNUSED(keepdims);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_concat(ShapeTracker **sts, int32_t num_sts, int32_t axis) {
    UNUSED(sts);
    UNUSED(num_sts);
    UNUSED(axis);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_stack(ShapeTracker **sts, int32_t num_sts, int32_t axis) {
    UNUSED(sts);
    UNUSED(num_sts);
    UNUSED(axis);
    return NULL; // Stub - not implemented
}

ShapeTracker *shapetracker_repeat(ShapeTracker *st, int32_t axis, int32_t repeats) {
    UNUSED(st);
    UNUSED(axis);
    UNUSED(repeats);
    return NULL; // Stub - not implemented
}

// Helper function to unravel a UOp offset into UOp indices based on shape
// Python: unravel(shape, offset) returns list of UOps
static UOp** unravel_uop(const int32_t* shape, int ndim, UOp* offset, int* out_count) {
    UOp** idxs = malloc(ndim * sizeof(UOp*));
    UOp* acc = sint_to_uop(1);
    
    // Python: for d in reversed(shape):
    for (int i = ndim - 1; i >= 0; i--) {
        // Python: idxs.append((offset//acc)%d)
        UOp* shape_val = sint_to_uop(shape[i]);
        UOp* div_result = uop_div(offset, acc);
        idxs[i] = uop_mod(div_result, shape_val);
        
        // Python: acc *= d
        acc = uop_mul(acc, shape_val);
    }
    
    *out_count = ndim;
    return idxs;
}

// Python lines 26-38: views_to_indexed_uops implementation
static void views_to_indexed_uops(View **views, int num_views, UOp **idxs, int idxs_count, UOp **out_idx, UOp **out_valid) {
    if (!views || num_views == 0) {
        *out_idx = sint_to_uop(0);
        *out_valid = uop_const(dtypes.bool_, 1.0);
        return;
    }
    
    // Python line 27: idx, valid = views[-1].to_indexed_uops(_idxs)
    UOp *idx = NULL;
    UOp *valid = NULL;
    view_to_indexed_uops(views[num_views - 1], idxs, idxs_count, NULL, &idx, &valid);
    
    // Python lines 28-30: for view in reversed(views[0:-1]):
    for (int i = num_views - 2; i >= 0; i--) {
        // Python line 29: view = view.minify()
        View *view = view_minify(views[i]);
        
        // Python line 30: idx, valid = view.to_indexed_uops([sint_to_uop(i) for i in unravel(view.shape, idx)], valid)
        // unravel returns UOps, not integers
        int ndim = view_ndim(view);
        const int32_t *shape = view_shape(view);
        int unraveled_count = 0;
        UOp **unraveled_uops = unravel_uop(shape, ndim, idx, &unraveled_count);
        
        // Call view.to_indexed_uops with unraveled indices and current valid
        view_to_indexed_uops(view, unraveled_uops, unraveled_count, valid, &idx, &valid);
        
        // Clean up
        free(unraveled_uops);
        if (view != views[i]) view_free(view);  // Free minified view if different
    }
    
    // Python lines 31-38: Apply graph rewrites and simplifications
    // Line 32-33: idx, valid = graph_rewrite(UOp.sink(idx, valid), symbolic_flat, name="indexing sym @ 1").src
    // For now, we'll skip the complex graph rewriting (would need full pattern matching)
    
    // Line 35: if (newvalid:=simplify_valid(valid)) is not None: valid = newvalid
    UOp *newvalid = simplify_valid(valid);
    if (newvalid) valid = newvalid;
    
    // Line 36: if (newidx:=uop_given_valid(valid, idx)) is not None: idx = newidx
    UOp *newidx = uop_given_valid(valid, idx);
    if (newidx) idx = newidx;
    
    // Line 38: return graph_rewrite(UOp.sink(idx, valid), symbolic_flat+pm_upcast, name="indexing sym @ 2").src
    // Skipping second graph rewrite for now
    
    *out_idx = idx;
    *out_valid = valid;
}

// Python lines 86-87: ShapeTracker.to_indexed_uops implementation
IndexedUOps *shapetracker_to_indexed_uops(const ShapeTracker *st) {
    if (!st || !st->views || st->num_views == 0) return NULL;
    
    IndexedUOps *result = malloc(sizeof(IndexedUOps));
    if (!result) return NULL;
    
    // Python line 87: return views_to_indexed_uops(self.views, tuple(_idxs) if _idxs is not None else None)
    views_to_indexed_uops((View**)st->views, st->num_views, NULL, 0, &result->idx, &result->valid);
    
    return result;
}

void indexed_uops_free(IndexedUOps *uops) {
    if (!uops) return;
    // UOps are managed elsewhere, we just free the container
    free(uops);
}


// Capital S version for compatibility with existing tests
ShapeTracker *ShapeTracker_from_shape(const int32_t *shape, int32_t ndim) {
    return shapetracker_from_shape(shape, ndim);
}
