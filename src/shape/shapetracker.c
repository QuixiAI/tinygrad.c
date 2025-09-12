#include "shape/shapetracker.h"
#include "shape/view.h"
#include "helpers/helpers.h"
#include "dtype/dtype.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "uop/symbolic.h"
#include "uop/uop.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <math.h>

// Macro to mark unused parameters
#define UNUSED(x) ((void)(x))


// Python line 56-58: @dataclass(frozen=True, order=True) class ShapeTracker:
// ShapeTracker struct is defined in the header file

// extern symbolic_flat pattern set from symbolic.c
extern struct PatternMatcher symbolic_flat;

// Upcast PatternMatcher (pm_upcast)
static void* cb_handle_upcast(void* ctx, void* node) {
    (void)ctx; UOp* u=(UOp*)node; if (!u) return NULL;
    if (!dtypes_is_int(&u->dtype)) return NULL;
    if (!u->vmin_vmax_valid) return NULL;
    double int_min = dtypes_min(&dtypes.int_);
    double int_max = dtypes_max(&dtypes.int_);
    bool overflow = (u->vmin < (long long)int_min) || (u->vmax > (long long)int_max);
    int vcount = u->dtype.count;
    DType int64 = dtypes.int64; if (vcount>1) int64 = dtype_vec(&dtypes.int64, vcount);
    if (overflow) {
        UOp** srcs = NULL; if (u->src_count>0){ srcs=(UOp**)malloc(sizeof(UOp*)*u->src_count); for(size_t i=0;i<u->src_count;i++) srcs[i] = uop_cast(u->src[i], int64); }
        UOp* repl = uop_replace_ex(u, u->op, int64, srcs, u->src_count, &u->arg, u->tag);
        if (srcs) free(srcs);
        return repl;
    }
    bool any_i64=false; for (size_t i=0;i<u->src_count;i++){ DType sc = dtype_scalar(&u->src[i]->dtype); if (dtype_eq(&sc, &dtypes.int64)) { any_i64=true; break; } }
    if (any_i64) {
        // Rebuild node at int64 then cast back to original dtype
        UOp** srcs = NULL; if (u->src_count>0){ srcs=(UOp**)malloc(sizeof(UOp*)*u->src_count); for(size_t i=0;i<u->src_count;i++) srcs[i] = uop_cast(u->src[i], int64); }
        UOp* widened = uop_replace_ex(u, u->op, int64, srcs, u->src_count, &u->arg, u->tag);
        if (srcs) free(srcs);
        if (!widened) return NULL;
        return uop_cast(widened, u->dtype);
    }
    return NULL;
}

static PatternMatcher* build_pm_upcast(void){
    static PatternMatcher pm; static int inited=0; if (inited) return &pm; inited=1;
    PatternMatch* entries = (PatternMatch*)calloc(1, sizeof(PatternMatch));
    UPat* alu = upat_group_ops(group_op.is_alu, NULL, 0); upat_set_dtype(alu, &dtypes.int_);
    entries[0].pattern = alu; entries[0].callback = cb_handle_upcast; entries[0].callback_ex = NULL; entries[0].user_data=NULL;
    pm.matches = entries; pm.match_count=1; pm.capacity=1; pm.compiled=false; return &pm;
}

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
// Forward declaration
static void views_to_indexed_uops(View **views, int num_views, UOp **idxs, int idxs_count, UOp **out_idx, UOp **out_valid);

bool shapetracker_axis_is_masked(const ShapeTracker *st, int32_t axis) {
    if (!st || axis < 0 || axis >= shapetracker_ndim(st)) return false;
    // Compute idx, valid via views_to_indexed_uops (includes symbolic + simplify stages)
    UOp *idx=NULL, *valid=NULL;
    views_to_indexed_uops((View**)st->views, st->num_views, NULL, 0, &idx, &valid);
    if (!valid) return false;
    // symbolic rewrite on valid
    UOp* sink_srcs[2] = { idx, valid };
    UOp* sink = uop_sink(sink_srcs, 2);
    UOp* rew = upat_graph_rewrite(sink, &symbolic_flat, "axis_is_masked");
    UOp* v = (rew && rew->src_count>=2) ? rew->src[1] : valid;
    // Search for RANGE(axis)
    size_t n=0; UOp** topo = uop_toposort(v, &n);
    bool masked=false;
    if (topo){
        for (size_t i=0;i<n;i++){
            UOp* u=topo[i]; if (u->op==OPS_RANGE){ int ax = (u->arg.type==ARG_INT)? u->arg.int_data.i : (int)u->arg.const_data.const_value; if (ax==axis){ masked=true; break; } }
        }
        free(topo);
    }
    if (masked) return true;
    // Fallback: inspect last view mask ranges
    if (st->num_views > 0) {
        View *last_view = (View*)st->views[st->num_views - 1];
        const int32_t *mask = view_mask_ranges(last_view);
        if (mask) {
            const int32_t *shape = view_shape(last_view);
            int32_t start = mask[axis * 2];
            int32_t end = mask[axis * 2 + 1];
            if (start != 0 || end != shape[axis]) return true;
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
    UOp* sink_srcs1[2] = { idx, valid };
    UOp* sink1 = uop_sink(sink_srcs1, 2);
    UOp* rew1 = upat_graph_rewrite(sink1, &symbolic_flat, "indexing sym @ 1");
    if (rew1 && rew1->src_count>=2) { idx = rew1->src[0]; valid = rew1->src[1]; }
    
    // Line 35: if (newvalid:=simplify_valid(valid)) is not None: valid = newvalid
    UOp *newvalid = simplify_valid(valid);
    if (newvalid) valid = newvalid;
    
    // Line 36: if (newidx:=uop_given_valid(valid, idx)) is not None: idx = newidx
    UOp *newidx = uop_given_valid(valid, idx);
    if (newidx) idx = newidx;
    
    // Line 38: return graph_rewrite(UOp.sink(idx, valid), symbolic_flat+pm_upcast, name="indexing sym @ 2").src
    UOp* sink_srcs2[2] = { idx, valid };
    UOp* sink2 = uop_sink(sink_srcs2, 2);
    UOp* rew2 = upat_graph_rewrite(sink2, &symbolic_flat, "indexing sym @ 2a");
    PatternMatcher* pm_upcast = build_pm_upcast();
    UOp* rew3 = upat_graph_rewrite(rew2 ? rew2 : sink2, pm_upcast, "indexing sym @ 2b");
    if (rew3 && rew3->src_count>=2) { idx = rew3->src[0]; valid = rew3->src[1]; }
    
    *out_idx = idx;
    *out_valid = valid;
}

// Compute real strides per Python views_to_real_strides
// Returns a newly-allocated array of length ndim. For masked/unknown axes, the stride is INT32_MIN.
static int32_t* views_to_real_strides(View **views, int num_views, bool ignore_valid, int* out_ndim) {
    if (!views || num_views <= 0) { if(out_ndim) *out_ndim = 0; return NULL; }
    View* last = views[num_views-1];
    int ndim = view_ndim(last);
    if (out_ndim) *out_ndim = ndim;
    // Fast path: single view and unmasked
    if (num_views == 1 && view_mask(last) == NULL) {
        const int32_t* s = view_strides(last);
        int32_t* ret = (int32_t*)malloc(sizeof(int32_t)*ndim);
        for (int i=0;i<ndim;i++) ret[i] = s[i];
        return ret;
    }
    int32_t* ret = (int32_t*)malloc(sizeof(int32_t)*ndim);
    for (int i=0;i<ndim;i++) ret[i] = INT32_MIN; // None sentinel

    // Build idx,valid with full pipeline
    UOp *idx=NULL, *valid=NULL;
    views_to_indexed_uops(views, num_views, NULL, 0, &idx, &valid);

    // Split idx on ADD terms and detect stride contributions
    UOp** terms=NULL; int nterms=0; split_uop(idx, OPS_ADD, &terms, &nterms);
    for (int i=0;i<nterms;i++) {
        UOp* c = terms[i];
        if (c->op == OPS_RANGE) {
            int ax = (c->arg.type==ARG_INT) ? c->arg.int_data.i : (int)c->arg.const_data.const_value;
            if (ax >= 0 && ax < ndim) ret[ax] = 1;
        } else if (c->op == OPS_MUL && c->src_count==2) {
            UOp* a=c->src[0], *b=c->src[1];
            int ax=-1; long long k=0; bool ok=false;
            if (a->op==OPS_RANGE && (b->arg.type==ARG_INT || b->arg.type==ARG_CONST)) {
                ax = (a->arg.type==ARG_INT) ? a->arg.int_data.i : (int)a->arg.const_data.const_value;
                k = (b->arg.type==ARG_INT) ? b->arg.int_data.i : (long long)llround(b->arg.const_data.const_value);
                ok=true;
            } else if (b->op==OPS_RANGE && (a->arg.type==ARG_INT || a->arg.type==ARG_CONST)) {
                ax = (b->arg.type==ARG_INT) ? b->arg.int_data.i : (int)b->arg.const_data.const_value;
                k = (a->arg.type==ARG_INT) ? a->arg.int_data.i : (long long)llround(a->arg.const_data.const_value);
                ok=true;
            }
            if (ok && ax>=0 && ax<ndim) ret[ax] = (int32_t)k;
        }
    }
    if (terms) free(terms);

    // used_ranges from idx toposort
    size_t tcnt=0; UOp** topo = uop_toposort(idx, &tcnt);
    bool* used = (bool*)calloc((size_t)ndim, sizeof(bool));
    if (topo) {
        for (size_t i=0;i<tcnt;i++) {
            if (topo[i]->op == OPS_RANGE) {
                int ax = (topo[i]->arg.type==ARG_INT) ? topo[i]->arg.int_data.i : (int)topo[i]->arg.const_data.const_value;
                if (ax>=0 && ax<ndim) used[ax] = true;
            }
        }
        free(topo);
    }
    // Axes not used → stride 0; keep 1/CONST where present; leave as None if untouched (masked later)
    for (int i=0;i<ndim;i++) {
        if (!used[i]) ret[i] = 0;
        else if (ret[i] == INT32_MIN) ret[i] = 1; // RANGE without CONST → unit stride
    }
    free(used);

    if (!ignore_valid && valid) {
        size_t vcnt=0; UOp** vtopo = uop_toposort(valid, &vcnt);
        if (vtopo) {
            for (size_t i=0;i<vcnt;i++) {
                if (vtopo[i]->op == OPS_RANGE) {
                    int ax = (vtopo[i]->arg.type==ARG_INT) ? vtopo[i]->arg.int_data.i : (int)vtopo[i]->arg.const_data.const_value;
                    if (ax>=0 && ax<ndim) ret[ax] = INT32_MIN; // masked → None
                }
            }
            free(vtopo);
        }
    }
    return ret;
}

int32_t* shapetracker_real_strides(const ShapeTracker* st, bool ignore_valid) {
    if (!st || !st->views || st->num_views<=0) return NULL;
    int ndim=0; return views_to_real_strides((View**)st->views, st->num_views, ignore_valid, &ndim);
}

int32_t* shapetracker_real_strides_default(const ShapeTracker* st) {
    return shapetracker_real_strides(st, false);
}

int* shapetracker_unit_stride_axes(const ShapeTracker* st, bool ignore_valid, int* out_count) {
    if (out_count) *out_count = 0;
    if (!st || !st->views || st->num_views<=0) return NULL;
    int ndim=0; int32_t* strides = views_to_real_strides((View**)st->views, st->num_views, ignore_valid, &ndim);
    if (!strides) return NULL;
    int* axes = (int*)malloc(sizeof(int)*ndim); int n=0;
    for (int i=0;i<ndim;i++) if (strides[i] == 1) axes[n++]=i;
    free(strides);
    if (out_count) *out_count = n;
    return axes;
}

int* shapetracker_unit_stride_axes_default(const ShapeTracker* st, int* out_count) {
    return shapetracker_unit_stride_axes(st, false, out_count);
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
/* Upcast PM forward declarations moved to top-level */
    free(uops);
}


// Capital S version for compatibility with existing tests
ShapeTracker *ShapeTracker_from_shape(const int32_t *shape, int32_t ndim) {
    return shapetracker_from_shape(shape, ndim);
}

// --- Python parity stubs: vars, var_vals, unbind, substitute ---
UOp** shapetracker_vars(const ShapeTracker* st, int* out_count) {
    if (out_count) *out_count = 0;
    if (!st || !st->views || st->num_views<=0) return NULL;
    size_t cap=16, n=0; UOp** acc=(UOp**)malloc(sizeof(UOp*)*cap);
    if (!acc) return NULL;
    for (int i=0;i<st->num_views;i++){
        int vcnt=0; UOp** vv = view_vars((View*)st->views[i], &vcnt);
        for (int j=0;j<vcnt;j++){
            if (n>=cap){ cap*=2; acc=(UOp**)realloc(acc, sizeof(UOp*)*cap); }
            acc[n++] = vv[j];
        }
        if (vv) free(vv);
    }
    size_t outn=0; UOp** dedup = upat_dedup(acc, n, &outn);
    free(acc);
    if (out_count) *out_count = (int)outn;
    return dedup;
}

UOp** shapetracker_var_vals(const ShapeTracker* st, int* out_count, int** out_vals) {
    if (out_count) *out_count = 0;
    if (out_vals) *out_vals = NULL;
    if (!st || !st->views || st->num_views<=0) return NULL;
    size_t cap=16, n=0; UOp** vars=(UOp**)malloc(sizeof(UOp*)*cap); int* vals=(int*)malloc(sizeof(int)*cap);
    if (!vars || !vals){ if (vars) free(vars); if (vals) free(vals); return NULL; }
    for (int i=0;i<st->num_views;i++){
        UOp** vvars=NULL; UOp** vvals_u=NULL; int vcnt=0;
        View* tmp = view_unbind((View*)st->views[i], &vvars, &vvals_u, &vcnt);
        if (tmp) view_free(tmp);
        for (int j=0;j<vcnt;j++){
            UOp* val = vvals_u[j];
            if (val && val->op==OPS_CONST && val->arg.type==ARG_CONST){
                // dedup on var pointer
                bool seen=false; for (size_t k=0;k<n;k++){ if (vars[k]==vvars[j]) { seen=true; break; } }
                if (!seen){
                    if (n>=cap){ cap*=2; vars=(UOp**)realloc(vars, sizeof(UOp*)*cap); vals=(int*)realloc(vals, sizeof(int)*cap); }
                    vars[n] = vvars[j];
                    vals[n] = (int)llround(val->arg.const_data.const_value);
                    n++;
                }
            }
        }
        if (vvars) free(vvars);
        if (vvals_u) free(vvals_u);
    }
    if (out_count) *out_count = (int)n;
    if (out_vals) *out_vals = vals; else free(vals);
    return vars;
}

ShapeTracker* shapetracker_unbind(const ShapeTracker* st, UOp*** out_vars, int** out_vals, int* out_count) {
    if (out_vars) *out_vars = NULL;
    if (out_vals) *out_vals = NULL;
    if (out_count) *out_count = 0;
    if (!st || !st->views || st->num_views<=0) return NULL;
    View** nviews = (View**)malloc(sizeof(View*)*st->num_views);
    if (!nviews) return NULL;
    size_t cap=16, n=0; UOp** vars=(UOp**)malloc(sizeof(UOp*)*cap); int* vals=(int*)malloc(sizeof(int)*cap);
    if (!vars || !vals){ if (vars) free(vars); if (vals) free(vals); free(nviews); return NULL; }
    for (int i=0;i<st->num_views;i++){
        UOp** vvars=NULL; UOp** vvals_u=NULL; int vcnt=0;
        nviews[i] = view_unbind((View*)st->views[i], &vvars, &vvals_u, &vcnt);
        if (!nviews[i]) nviews[i] = view_copy((View*)st->views[i]);
        for (int j=0;j<vcnt;j++){
            UOp* val = vvals_u[j];
            if (val && val->op==OPS_CONST && val->arg.type==ARG_CONST){
                bool seen=false; for (size_t k=0;k<n;k++){ if (vars[k]==vvars[j]) { seen=true; break; } }
                if (!seen){
                    if (n>=cap){ cap*=2; vars=(UOp**)realloc(vars, sizeof(UOp*)*cap); vals=(int*)realloc(vals, sizeof(int)*cap); }
                    vars[n] = vvars[j];
                    vals[n] = (int)llround(val->arg.const_data.const_value);
                    n++;
                }
            }
        }
        if (vvars) free(vvars);
        if (vvals_u) free(vvals_u);
    }
    ShapeTracker* out = (ShapeTracker*)malloc(sizeof(ShapeTracker));
    if (!out){ for (int i=0;i<st->num_views;i++) if (nviews[i]) view_free(nviews[i]); free(nviews); free(vars); free(vals); return NULL; }
    out->num_views = st->num_views; out->views = (void**)nviews;
    if (out_vars) *out_vars = vars; else free(vars);
    if (out_vals) *out_vals = vals; else free(vals);
    if (out_count) *out_count = (int)n;
    return out;
}

ShapeTracker* shapetracker_substitute(const ShapeTracker* st, UOp** from_vars, UOp** to_vals, int count) {
    if (!st || !st->views || st->num_views<=0) return NULL;
    View** nviews = (View**)malloc(sizeof(View*)*st->num_views);
    if (!nviews) return NULL;
    for (int i=0;i<st->num_views;i++){
        View* nv = view_substitute((View*)st->views[i], from_vars, to_vals, count);
        if (!nv) nv = view_copy((View*)st->views[i]);
        nviews[i] = nv;
    }
    ShapeTracker* out = (ShapeTracker*)malloc(sizeof(ShapeTracker));
    if (!out){ for (int i=0;i<st->num_views;i++) if (nviews[i]) view_free(nviews[i]); free(nviews); return NULL; }
    out->num_views = st->num_views; out->views = (void**)nviews;
    return out;
}
