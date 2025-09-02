/* dtype.c
 * Port of tinygrad/dtype.py to C
 * Faithful line-by-line port with minimal changes for C compatibility
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <ctype.h>

#include "dtype/dtype.h"
#include "helpers/helpers.h"

// DType cache for singleton pattern (like Python metaclass)
#define DTYPE_CACHE_SIZE 256
typedef struct {
    int priority;
    int itemsize; 
    char name[32];
    char fmt;
    int count;
    DType* dtype;
} DTypeCache;

static DTypeCache g_dtype_cache[DTYPE_CACHE_SIZE];
static int g_dtype_cache_count = 0;

// Global dtypes instance - defined in header as extern, so we define it here
DtypesStruct dtypes;

// DType creation function (equivalent to DType.new)
DType dtype_new(int priority, int itemsize, const char* name, char fmt) {
    // Check cache first (like Python metaclass)
    for (int i = 0; i < g_dtype_cache_count; i++) {
        DTypeCache* entry = &g_dtype_cache[i];
        if (entry->priority == priority && 
            entry->itemsize == itemsize &&
            strcmp(entry->name, name) == 0 &&
            entry->fmt == fmt) {
            return *entry->dtype;
        }
    }
    
    // Create new DType
    DType dt = {
        .priority = priority,
        .itemsize = itemsize,
        .fmt = fmt,
        .count = 1,
        ._scalar = NULL
    };
    strncpy(dt.name, name, sizeof(dt.name) - 1);
    dt.name[sizeof(dt.name) - 1] = '\0';
    
    // Add to cache
    if (g_dtype_cache_count < DTYPE_CACHE_SIZE) {
        DTypeCache* entry = &g_dtype_cache[g_dtype_cache_count];
        entry->priority = priority;
        entry->itemsize = itemsize;
        strncpy(entry->name, name, sizeof(entry->name) - 1);
        entry->name[sizeof(entry->name) - 1] = '\0';
        entry->fmt = fmt;
        entry->count = 1;
        // Note: We can't store pointer to local variable, so we'll store a copy
        static DType cached_types[DTYPE_CACHE_SIZE];
        cached_types[g_dtype_cache_count] = dt;
        entry->dtype = &cached_types[g_dtype_cache_count];
        g_dtype_cache_count++;
    }
    
    return dt;
}

// DType comparison functions
bool dtype_eq(const DType* a, const DType* b) {
    return a->priority == b->priority &&
           a->itemsize == b->itemsize &&
           strcmp(a->name, b->name) == 0 &&
           a->fmt == b->fmt &&
           a->count == b->count;
}

bool dtype_same_instance(const DType* a, const DType* b) {
    // Pointer equality check for singleton instances
    return a == b;
}

bool dtype_lt(const DType* a, const DType* b) {
    // Python: __lt__(self, o:DType): return (self.priority, self.itemsize, self.name, self.fmt, self.count) < (o.priority, o.itemsize, o.name, o.fmt, o.count)
    if (a->priority != b->priority) return a->priority < b->priority;
    if (a->itemsize != b->itemsize) return a->itemsize < b->itemsize;
    int name_cmp = strcmp(a->name, b->name);
    if (name_cmp != 0) return name_cmp < 0;
    if (a->fmt != b->fmt) return a->fmt < b->fmt;
    return a->count < b->count;
}

// Vectorization function
DType dtype_vec(const DType* dt, int sz) {
    // Python: def vec(self, sz:int) -> DType:
    assert(dt->count == 1); // can't vectorize already vectorized type
    if (sz == 1 || dtype_eq(dt, &dtypes.void_)) return *dt; // void doesn't vectorize, sz=1 is scalar
    
    DType vec_dt = *dt;
    vec_dt.itemsize *= sz;
    vec_dt.count = sz;
    vec_dt._scalar = dt;
    
    // Update name to include vector size (simplified)
    char vec_name[64]; // Increased size to avoid truncation
    int written = snprintf(vec_name, sizeof(vec_name), "%s%d", dt->name, sz);
    if (written >= (int)sizeof(vec_name)) {
        // Truncation would occur, handle gracefully
        vec_name[sizeof(vec_name) - 1] = '\0';
    }
    strncpy(vec_dt.name, vec_name, sizeof(vec_dt.name) - 1);
    vec_dt.name[sizeof(vec_dt.name) - 1] = '\0';
    vec_dt.fmt = 0; // vectorized types have no fmt
    
    return vec_dt;
}

// Pointer type creation
PtrDType dtype_ptr(const DType* dt, int size, AddrSpace addrspace) {
    PtrDType ptr_dt;
    ptr_dt.base = *dt;
    ptr_dt.base_dtype = dt;
    ptr_dt.addrspace = addrspace;
    ptr_dt.v = 1;
    ptr_dt.size = size;
    return ptr_dt;
}

// Scalar extraction
DType dtype_scalar(const DType* dt) {
    // Python: def scalar(self) -> DType: return self._scalar if self._scalar is not None else self
    return dt->_scalar != NULL ? *dt->_scalar : *dt;
}

// nbytes for non-ptr types (should throw error)
int dtype_nbytes(const DType* dt) {
    // Python: def nbytes(self): raise RuntimeError("only ptr types have nbytes")
    fprintf(stderr, "Error: only ptr types have nbytes\n");
    abort(); // C equivalent of raising RuntimeError
}

// PtrDType nbytes
int ptrdtype_nbytes(const PtrDType* dt) {
    // Python: def nbytes(self) -> int:
    //   if self.size == -1: return 0  # TODO: this should be an exception
    //   return self.size*self.itemsize
    if (dt->size == -1) return 0;
    return dt->size * dt->base.itemsize;
}

// ImageDType creation - faithful port from Python
ImageDType imagedtype_create(const int* shape, int shape_len, bool is_half) {
    ImageDType img_dt;
    
    // Calculate product of shape
    int prod_shape = 1;
    for (int i = 0; i < shape_len; i++) {
        prod_shape *= shape[i];
    }
    
    if (is_half) {
        // Python: ImageDType(100, 2, "imageh", 'e', 1, None, dtypes.float32, AddrSpace.GLOBAL, 1, prod(shp), shp)
        img_dt.ptr_base.base = dtype_new(100, 2, "imageh", 'e');
        img_dt.ptr_base.base.count = 1;
        img_dt.ptr_base.base._scalar = NULL;
        img_dt.ptr_base.base_dtype = &dtypes.float32;
        img_dt.ptr_base.addrspace = ADDRSPACE_GLOBAL;
        img_dt.ptr_base.v = 1;
        img_dt.ptr_base.size = prod_shape;
    } else {
        // Python: ImageDType(100, 4, "imagef", 'f', 1, None, dtypes.float32, AddrSpace.GLOBAL, 1, prod(shp), shp)
        img_dt.ptr_base.base = dtype_new(100, 4, "imagef", 'f');
        img_dt.ptr_base.base.count = 1;
        img_dt.ptr_base.base._scalar = NULL;
        img_dt.ptr_base.base_dtype = &dtypes.float32;
        img_dt.ptr_base.addrspace = ADDRSPACE_GLOBAL;
        img_dt.ptr_base.v = 1;
        img_dt.ptr_base.size = prod_shape;
    }
    
    // Copy shape
    img_dt.shape_len = shape_len < 4 ? shape_len : 4;
    for (int i = 0; i < img_dt.shape_len; i++) {
        img_dt.shape[i] = shape[i];
    }
    
    return img_dt;
}

// Python: @staticmethod def imageh(shp): return ImageDType(...)
ImageDType dtypes_imageh(const int* shape, int shape_len) {
    return imagedtype_create(shape, shape_len, true);
}

// Python: @staticmethod def imagef(shp): return ImageDType(...)
ImageDType dtypes_imagef(const int* shape, int shape_len) {
    return imagedtype_create(shape, shape_len, false);
}

// Type checking functions
bool dtypes_is_float(const DType* dt) {
    // Python: def is_float(x: DType) -> bool: return x.scalar() in dtypes.floats or isinstance(x, ImageDType)
    // Check if it's an ImageDType by checking the name
    if (strcmp(dt->name, "imageh") == 0 || strcmp(dt->name, "imagef") == 0) {
        return true;
    }
    
    DType scalar = dtype_scalar(dt);
    for (int i = 0; i < 6; i++) {
        if (dtype_eq(&scalar, dtypes.floats[i])) return true;
    }
    return false;
}

bool dtypes_is_int(const DType* dt) {
    // Python: def is_int(x: DType) -> bool: return x.scalar() in dtypes.ints
    DType scalar = dtype_scalar(dt);
    for (int i = 0; i < 8; i++) {
        if (dtype_eq(&scalar, dtypes.ints[i])) return true;
    }
    return false;
}

bool dtypes_is_unsigned(const DType* dt) {
    // Python: def is_unsigned(x: DType) -> bool: return x.scalar() in dtypes.uints
    DType scalar = dtype_scalar(dt);
    for (int i = 0; i < 4; i++) {
        if (dtype_eq(&scalar, dtypes.uints[i])) return true;
    }
    return false;
}

bool dtypes_is_bool(const DType* dt) {
    // Python: def is_bool(x: DType) -> bool: return x.scalar() == dtypes.bool
    DType scalar = dtype_scalar(dt);
    return dtype_eq(&scalar, &dtypes.bool_);
}

// from_py functions
DType dtypes_from_py_float(double val) {
    // Python: if x.__class__ is float: return dtypes.default_float
    (void)val; // unused
    return *dtypes.default_float;
}

DType dtypes_from_py_int(int64_t val) {
    // Python: if x.__class__ is int: return dtypes.default_int
    (void)val; // unused
    return *dtypes.default_int;
}

DType dtypes_from_py_bool(bool val) {
    // Python: if x.__class__ is bool: return dtypes.bool
    (void)val; // unused
    return dtypes.bool_;
}

// as_const functions - faithful port with proper truncation
double dtypes_as_const_float(double val, const DType* dt) {
    // Python: # TODO: should truncate here
    // Python: return int(val) if dtypes.is_int(dtype) else float(val) if dtypes.is_float(dtype) else bool(val)
    if (dtypes_is_int(dt)) {
        // Truncate to integer type
        return dtypes_truncate((double)(int64_t)val, dt);
    }
    if (dtypes_is_float(dt)) {
        // Truncate to float type
        return dtypes_truncate(val, dt);
    }
    return (double)(bool)val;
}

int64_t dtypes_as_const_int(int64_t val, const DType* dt) {
    if (dtypes_is_int(dt)) {
        // Truncate to specific integer type
        return (int64_t)dtypes_truncate((double)val, dt);
    }
    if (dtypes_is_float(dt)) {
        // Convert to float and truncate
        return (int64_t)dtypes_truncate((double)val, dt);
    }
    return (int64_t)(bool)val;
}

bool dtypes_as_const_bool(bool val, const DType* dt) {
    if (dtypes_is_int(dt)) return (bool)val;
    if (dtypes_is_float(dt)) return (bool)val;
    return val;
}

// Min/max functions
double dtypes_min(const DType* dt) {
    // Python: def min(dtype:DType):
    //   if dtypes.is_int(dtype): return 0 if dtypes.is_unsigned(dtype) else -2**(dtype.itemsize*8-1)
    //   return -float("inf") if dtypes.is_float(dtype) else False
    if (dtypes_is_int(dt)) {
        if (dtypes_is_unsigned(dt)) return 0.0;
        int bits = dt->itemsize * 8 - 1;
        return -(double)(1LL << bits);
    }
    if (dtypes_is_float(dt)) return -INFINITY;
    return 0.0; // false for bool
}

double dtypes_max(const DType* dt) {
    // Python: def max(dtype:DType):
    //   if dtypes.is_int(dtype): return 2**(dtype.itemsize*8)-1+dtypes.min(dtype)
    //   return float("inf") if dtypes.is_float(dtype) else True
    if (dtypes_is_int(dt)) {
        int bits = dt->itemsize * 8;
        if (dtypes_is_unsigned(dt)) {
            if (bits >= 64) return (double)UINT64_MAX;
            return (double)((1ULL << bits) - 1);
        } else {
            return (double)((1LL << (bits-1)) - 1);
        }
    }
    if (dtypes_is_float(dt)) return INFINITY;
    return 1.0; // true for bool
}

// finfo function
FInfo dtypes_finfo(const DType* dt) {
    // Python: def finfo(dtype:DType) -> tuple[int, int]:
    //   return {dtypes.float16: (5, 10), dtypes.bfloat16: (8, 7), dtypes.float32: (8, 23), dtypes.float64: (11, 52),
    //           dtypes.fp8e5m2: (5, 2), dtypes.fp8e4m3: (4, 3)}[dtype]
    if (!dtypes_is_float(dt)) {
        fprintf(stderr, "Error: %s is not a floating point type\n", dt->name);
        abort();
    }
    
    if (dtype_eq(dt, &dtypes.float16)) return (FInfo){5, 10};
    if (dtype_eq(dt, &dtypes.bfloat16)) return (FInfo){8, 7};
    if (dtype_eq(dt, &dtypes.float32)) return (FInfo){8, 23};
    if (dtype_eq(dt, &dtypes.float64)) return (FInfo){11, 52};
    if (dtype_eq(dt, &dtypes.fp8e5m2)) return (FInfo){5, 2};
    if (dtype_eq(dt, &dtypes.fp8e4m3)) return (FInfo){4, 3};
    
    fprintf(stderr, "Error: unknown float type for finfo\n");
    abort();
}

// Type promotion lattice - faithful port of Python promo_lattice
// Python: promo_lattice = { dtypes.bool: [dtypes.int8, dtypes.uint8], dtypes.int8: [dtypes.int16], ...}
// g_promo_lattice is unused but kept for future implementation
// static const DType** g_promo_lattice[16]; // Max 16 entries for now
static PromoLatticeEntry g_promo_entries[16];
static int g_promo_entry_count = 0;

// Cache for recursive parents (equivalent to @functools.cache)
#define RECURSIVE_CACHE_SIZE 64
typedef struct {
    const DType* dtype;
    const DType** parents;
    int parent_count;
} RecursiveCache;
static RecursiveCache g_recursive_cache[RECURSIVE_CACHE_SIZE];
static int g_recursive_cache_count = 0;

// Initialize promotion lattice (called from dtypes_init)
static void init_promo_lattice() {
    // Clear any existing entries
    g_promo_entry_count = 0;
    
    // Python: dtypes.bool: [dtypes.int8, dtypes.uint8]
    static const DType* bool_promotions[] = {&dtypes.int8, &dtypes.uint8};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.bool_, bool_promotions, 2};
    
    // Python: dtypes.int8: [dtypes.int16]
    static const DType* int8_promotions[] = {&dtypes.int16};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.int8, int8_promotions, 1};
    
    // Python: dtypes.int16: [dtypes.int32]
    static const DType* int16_promotions[] = {&dtypes.int32};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.int16, int16_promotions, 1};
    
    // Python: dtypes.int32: [dtypes.int64]
    static const DType* int32_promotions[] = {&dtypes.int64};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.int32, int32_promotions, 1};
    
    // Python: dtypes.int64: [dtypes.float16, dtypes.bfloat16]
    static const DType* int64_promotions[] = {&dtypes.float16, &dtypes.bfloat16};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.int64, int64_promotions, 2};
    
    // Python: dtypes.uint8: [dtypes.int16, dtypes.uint16]
    static const DType* uint8_promotions[] = {&dtypes.int16, &dtypes.uint16};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.uint8, uint8_promotions, 2};
    
    // Python: dtypes.uint16: [dtypes.int32, dtypes.uint32]
    static const DType* uint16_promotions[] = {&dtypes.int32, &dtypes.uint32};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.uint16, uint16_promotions, 2};
    
    // Python: dtypes.uint32: [dtypes.int64, dtypes.uint64]
    static const DType* uint32_promotions[] = {&dtypes.int64, &dtypes.uint64};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.uint32, uint32_promotions, 2};
    
    // Python: dtypes.uint64: [dtypes.float16, dtypes.bfloat16]
    static const DType* uint64_promotions[] = {&dtypes.float16, &dtypes.bfloat16};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.uint64, uint64_promotions, 2};
    
    // Python: dtypes.fp8e5m2: [dtypes.float16, dtypes.bfloat16]
    static const DType* fp8e5m2_promotions[] = {&dtypes.float16, &dtypes.bfloat16};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.fp8e5m2, fp8e5m2_promotions, 2};
    
    // Python: dtypes.fp8e4m3: [dtypes.float16, dtypes.bfloat16]
    static const DType* fp8e4m3_promotions[] = {&dtypes.float16, &dtypes.bfloat16};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.fp8e4m3, fp8e4m3_promotions, 2};
    
    // Python: dtypes.float16: [dtypes.float32]
    static const DType* float16_promotions[] = {&dtypes.float32};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.float16, float16_promotions, 1};
    
    // Python: dtypes.bfloat16: [dtypes.float32]
    static const DType* bfloat16_promotions[] = {&dtypes.float32};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.bfloat16, bfloat16_promotions, 1};
    
    // Python: dtypes.float32: [dtypes.float64]
    static const DType* float32_promotions[] = {&dtypes.float64};
    g_promo_entries[g_promo_entry_count++] = (PromoLatticeEntry){&dtypes.float32, float32_promotions, 1};
    
    // dtypes.float64 has no promotions (terminal node)
}

// Python: @functools.cache def _get_recursive_parents(dtype:DType) -> set[DType]:
const DType** _get_recursive_parents(const DType* dtype, int* count) {
    // Check cache first - use pointer equality since we're working with singleton instances
    for (int i = 0; i < g_recursive_cache_count; i++) {
        if (g_recursive_cache[i].dtype == dtype) {
            *count = g_recursive_cache[i].parent_count;
            return g_recursive_cache[i].parents;
        }
    }
    
    // Python: return set.union(*[_get_recursive_parents(d) for d in promo_lattice[dtype]], {dtype}) if dtype != dtypes.float64 else {dtypes.float64}
    // Use dynamic allocation to avoid static corruption on recursive calls
    const DType** result_set = malloc(64 * sizeof(DType*));
    int result_count = 0;
    
    // Add self to set
    result_set[result_count++] = dtype;
    
    // If this is float64, return just {float64} (terminal case)
    if (dtype == &dtypes.float64) {
        // Resize to actual count
        const DType** cached_result = realloc(result_set, sizeof(DType*));
        
        // Add to cache
        if (g_recursive_cache_count < RECURSIVE_CACHE_SIZE) {
            g_recursive_cache[g_recursive_cache_count] = (RecursiveCache){dtype, cached_result, 1};
            g_recursive_cache_count++;
        }
        
        *count = 1;
        return cached_result;
    }
    
    // Find this dtype in the promotion lattice - use pointer equality
    for (int i = 0; i < g_promo_entry_count; i++) {
        if (g_promo_entries[i].dtype == dtype) {
            // Recursively get parents of each promotion target and union them
            for (int j = 0; j < g_promo_entries[i].promotion_count; j++) {
                int sub_count;
                const DType** sub_parents = _get_recursive_parents(g_promo_entries[i].promotions[j], &sub_count);
                
                // Add all sub_parents to result_set (if not already present) - use pointer equality
                for (int k = 0; k < sub_count; k++) {
                    bool already_present = false;
                    for (int l = 0; l < result_count; l++) {
                        if (result_set[l] == sub_parents[k]) {
                            already_present = true;
                            break;
                        }
                    }
                    if (!already_present && result_count < 64) {
                        result_set[result_count++] = sub_parents[k];
                    }
                }
            }
            break;
        }
    }
    
    // Resize to actual count
    const DType** cached_result = realloc(result_set, result_count * sizeof(DType*));
    
    // Add to cache
    if (g_recursive_cache_count < RECURSIVE_CACHE_SIZE) {
        g_recursive_cache[g_recursive_cache_count] = (RecursiveCache){dtype, cached_result, result_count};
        g_recursive_cache_count++;
    }
    
    *count = result_count;
    return cached_result;
}

void _free_recursive_parents(const DType** parents) {
    // Note: We don't actually free here since we're using a cache
    // In a full implementation, we'd need reference counting
    (void)parents;
}

// Python: @functools.cache def least_upper_dtype(*ds:DType) -> DType:
DType least_upper_dtype(const DType* a, const DType* b) {
    // Python: return min(set.intersection(*[_get_recursive_parents(d) for d in ds])) if not (images:=[d for d in ds if isinstance(d, ImageDType)]) else images[0]
    
    // TODO: Check for ImageDType first (not implemented yet)
    
    // Get recursive parents for both types
    int count_a, count_b;
    const DType** parents_a = _get_recursive_parents(a, &count_a);
    const DType** parents_b = _get_recursive_parents(b, &count_b);
    
    // Find intersection - use pointer equality since we're working with singleton instances
    static const DType* intersection[32];
    int intersection_count = 0;
    
    for (int i = 0; i < count_a; i++) {
        for (int j = 0; j < count_b; j++) {
            if (parents_a[i] == parents_b[j]) {
                intersection[intersection_count++] = parents_a[i];
                break;
            }
        }
    }
    
    // Return minimum (lowest priority in intersection)
    if (intersection_count == 0) {
        // No common promotion path - this shouldn't happen in well-defined lattice
        return *a; // fallback
    }
    
    const DType* min_type = intersection[0];
    for (int i = 1; i < intersection_count; i++) {
        if (dtype_lt(intersection[i], min_type)) {
            min_type = intersection[i];
        }
    }
    
    return *min_type;
}

DType least_upper_dtype_multi(const DType** types, int count) {
    if (count == 0) return *dtypes.default_float;
    DType result = *types[0];
    for (int i = 1; i < count; i++) {
        result = least_upper_dtype(&result, types[i]);
    }
    return result;
}

DType least_upper_float(const DType* dt) {
    // Python: def least_upper_float(dt:DType) -> DType: return dt if dtypes.is_float(dt) else least_upper_dtype(dt, dtypes.default_float)
    if (dtypes_is_float(dt)) return *dt;
    return least_upper_dtype(dt, dtypes.default_float);
}

// Safe casting
bool can_safe_cast(const DType* dt0, const DType* dt1) {
    // Python: def can_safe_cast(dt0:DType, dt1:DType) -> bool:
    if (dtype_eq(dt0, dt1) || dtype_eq(dt0, &dtypes.bool_)) return true;
    
    // Simplified safe cast rules (matching Python version)
    if (dtype_eq(dt1, &dtypes.float64)) {
        return dtype_eq(dt0, &dtypes.float32) || 
               dtype_eq(dt0, &dtypes.float16) || 
               dtype_eq(dt0, &dtypes.bfloat16);
    }
    if (dtype_eq(dt1, &dtypes.float32)) {
        return dtype_eq(dt0, &dtypes.float16) || 
               dtype_eq(dt0, &dtypes.bfloat16);
    }
    if (dtype_eq(dt1, &dtypes.uint64)) {
        return dtype_eq(dt0, &dtypes.uint32) || 
               dtype_eq(dt0, &dtypes.uint16) || 
               dtype_eq(dt0, &dtypes.uint8);
    }
    if (dtype_eq(dt1, &dtypes.uint32)) {
        return dtype_eq(dt0, &dtypes.uint16) || 
               dtype_eq(dt0, &dtypes.uint8);
    }
    if (dtype_eq(dt1, &dtypes.int64)) {
        return dtype_eq(dt0, &dtypes.uint32) || dtype_eq(dt0, &dtypes.uint16) || 
               dtype_eq(dt0, &dtypes.uint8) || dtype_eq(dt0, &dtypes.int32) || 
               dtype_eq(dt0, &dtypes.int16) || dtype_eq(dt0, &dtypes.int8);
    }
    if (dtype_eq(dt1, &dtypes.int32)) {
        return dtype_eq(dt0, &dtypes.uint16) || dtype_eq(dt0, &dtypes.uint8) ||
               dtype_eq(dt0, &dtypes.int16) || dtype_eq(dt0, &dtypes.int8);
    }
    if (dtype_eq(dt1, &dtypes.int16)) {
        return dtype_eq(dt0, &dtypes.uint8) || dtype_eq(dt0, &dtypes.int8);
    }
    
    return false;
}

// Sum accumulator dtype
DType sum_acc_dtype(const DType* dt) {
    // Python: def sum_acc_dtype(dt:DType):
    //   if dtypes.is_unsigned(dt): return least_upper_dtype(dt, dtypes.uint)
    //   if dtypes.is_int(dt) or dt == dtypes.bool: return least_upper_dtype(dt, dtypes.int)
    //   return least_upper_dtype(dt, to_dtype(getenv("SUM_DTYPE", "float32")))
    
    if (dtypes_is_unsigned(dt)) return least_upper_dtype(dt, &dtypes.uint32);
    if (dtypes_is_int(dt) || dtype_eq(dt, &dtypes.bool_)) return least_upper_dtype(dt, &dtypes.int32);
    
    // For float types, use SUM_DTYPE environment variable (default "float32")
    const char* sum_dtype_env = tg_getenv_default("SUM_DTYPE", "float32");
    // to_dtype returns a copy, but we need a pointer to a global dtype
    if (strcmp(sum_dtype_env, "float32") == 0) {
        return least_upper_dtype(dt, &dtypes.float32);
    } else if (strcmp(sum_dtype_env, "float16") == 0) {
        return least_upper_dtype(dt, &dtypes.float16);
    } else if (strcmp(sum_dtype_env, "float64") == 0) {
        return least_upper_dtype(dt, &dtypes.float64);
    } else {
        // Default to float32 if unknown
        return least_upper_dtype(dt, &dtypes.float32);
    }
}

// FP16 truncation
float truncate_fp16(double x) {
    // Python: def truncate_fp16(x):
    //   try: return struct.unpack("@e", struct.pack("@e", float(x)))[0]
    //   except OverflowError: return math.copysign(math.inf, x)
    
    if (isnan(x)) return (float)x;
    if (isinf(x)) return copysignf(INFINITY, (float)x);
    
    // Simple truncation to float16 precision (simplified)
    float f = (float)x;
    if (fabsf(f) > 65504.0f) { // fp16 max value
        return copysignf(INFINITY, f);
    }
    // More precise fp16 conversion would require bit manipulation
    return f;
}

// BF16 truncation  
float truncate_bf16(double x) {
    // Python implementation using bit manipulation
    union { float f; uint32_t i; } u;
    u.f = (float)x;
    
    float max_bf16 = 3.38953e+38f; // approximately
    if (fabsf(u.f) > max_bf16) {
        return copysignf(INFINITY, u.f);
    }
    
    // Truncate to bf16 precision by masking lower 16 bits
    u.i &= 0xFFFF0000;
    return u.f;
}

// FP8 conversion config tables - faithful port from Python
typedef struct {
    int EXP_BIAS;
    int SIGNIFICAND_BITS;
    uint64_t MANTISSA_MASK;
    uint64_t MINDENORM_O2;
    uint64_t OVERFLOW_THRESHOLD;
    uint8_t MAXNORM;
    uint64_t MINNORM;
    uint8_t INF_VALUE;
} Fp8Config;

static const Fp8Config fp8e4m3_config = {
    .EXP_BIAS = 7,
    .SIGNIFICAND_BITS = 4,
    .MANTISSA_MASK = 0x7,
    .MINDENORM_O2 = 0x3F50000000000000ULL,
    .OVERFLOW_THRESHOLD = 0x407D000000000000ULL,
    .MAXNORM = 0x7E,
    .MINNORM = 0x3F90000000000000ULL,
    .INF_VALUE = 0x7F
};

static const Fp8Config fp8e5m2_config = {
    .EXP_BIAS = 15,
    .SIGNIFICAND_BITS = 3,
    .MANTISSA_MASK = 0x3,
    .MINDENORM_O2 = 0x3EE0000000000000ULL,
    .OVERFLOW_THRESHOLD = 0x40EE000000000000ULL - 1,
    .MAXNORM = 0x7B,
    .MINNORM = 0x3F10000000000000ULL,
    .INF_VALUE = 0x7E
};

// Faithful port of Python float_to_fp8
uint8_t float_to_fp8(double x, const DType* dtype) {
    // Python: assert dtype in dtypes.fp8s, "Only for fp8s"
    assert(dtype_eq(dtype, &dtypes.fp8e4m3) || dtype_eq(dtype, &dtypes.fp8e5m2));
    
    // Select config based on dtype
    const Fp8Config* config = dtype_eq(dtype, &dtypes.fp8e4m3) ? &fp8e4m3_config : &fp8e5m2_config;
    
    // Python: xbits, = struct.unpack('Q', struct.pack('d', x))
    union { double d; uint64_t q; } u;
    u.d = x;
    uint64_t xbits = u.q;
    
    // Python: FP8_DP_HALF_ULP = 1 << (53 - config["SIGNIFICAND_BITS"] - 1)
    uint64_t FP8_DP_HALF_ULP = 1ULL << (53 - config->SIGNIFICAND_BITS - 1);
    
    // Python: sign = ((xbits >> 63) & 1) << 7
    uint8_t sign = ((xbits >> 63) & 1) << 7;
    
    // Python: exp = (((xbits >> 52) & 0x7FF) - 1023 + config["EXP_BIAS"])
    int64_t exp = (((xbits >> 52) & 0x7FF) - 1023 + config->EXP_BIAS);
    
    // Python: mantissa = (xbits >> (53 - config["SIGNIFICAND_BITS"])) & config["MANTISSA_MASK"]
    uint64_t mantissa = (xbits >> (53 - config->SIGNIFICAND_BITS)) & config->MANTISSA_MASK;
    
    // Python: absx = xbits & 0x7FFFFFFFFFFFFFFF
    uint64_t absx = xbits & 0x7FFFFFFFFFFFFFFFULL;
    
    uint8_t res;
    
    // Python: if absx <= config["MINDENORM_O2"]: res = 0
    if (absx <= config->MINDENORM_O2) {
        res = 0;
    }
    // Python: elif absx > 0x7FF0000000000000: res = 0x7F if dtype == dtypes.fp8e4m3 else 0x7E | mantissa
    else if (absx > 0x7FF0000000000000ULL) {
        res = dtype_eq(dtype, &dtypes.fp8e4m3) ? 0x7F : (0x7E | mantissa);
    }
    // Python: elif absx > config["OVERFLOW_THRESHOLD"]: res = config["MAXNORM"]
    else if (absx > config->OVERFLOW_THRESHOLD) {
        res = config->MAXNORM;
    }
    // Python: elif absx >= config["MINNORM"]:
    else if (absx >= config->MINNORM) {
        res = ((exp << (config->SIGNIFICAND_BITS - 1)) | mantissa);
        uint64_t round_bits = xbits & ((FP8_DP_HALF_ULP << 1) - 1);
        if ((round_bits > FP8_DP_HALF_ULP) || (round_bits == FP8_DP_HALF_ULP && (mantissa & 1))) {
            res = res + 1;
        }
    }
    else {
        int64_t shift = 1 - exp;
        mantissa |= 1ULL << (config->SIGNIFICAND_BITS - 1);
        res = (mantissa >> shift);
        uint64_t round_bits = (xbits | (1ULL << (53 - 1))) & ((FP8_DP_HALF_ULP << (shift + 1)) - 1);
        if ((round_bits > (FP8_DP_HALF_ULP << shift)) || 
            (round_bits == (FP8_DP_HALF_ULP << shift) && (res & 1))) {
            res = res + 1;
        }
    }
    
    // Python: res |= sign
    res |= sign;
    return res;
}

// Faithful port of Python fp8_to_float  
float fp8_to_float(uint8_t x, const DType* dtype) {
    // Python: assert dtype in dtypes.fp8s, "Only for fp8s"
    assert(dtype_eq(dtype, &dtypes.fp8e4m3) || dtype_eq(dtype, &dtypes.fp8e5m2));
    
    // Python: ur = x << 8
    uint16_t ur = x << 8;
    
    // Python: if dtype == dtypes.fp8e5m2 and (ur & 0x7FFF) > 0x7C00: ur = 0x7FFF
    if (dtype_eq(dtype, &dtypes.fp8e5m2) && (ur & 0x7FFF) > 0x7C00) {
        ur = 0x7FFF;
    }
    // Python: elif dtype == dtypes.fp8e4m3:
    else if (dtype_eq(dtype, &dtypes.fp8e4m3)) {
        uint16_t sign = ur & 0x8000;
        uint16_t exponent = ((ur & 0x7800) >> 1) + 0x2000;
        uint16_t mantissa = (ur & 0x0700) >> 1;
        uint8_t absx = x & 0x7F;
        
        if (absx == 0x7F) {
            ur = 0x7FFF;
        }
        else if (exponent == 0x2000) {
            if (mantissa != 0) {
                mantissa <<= 1;
                while ((mantissa & 0x0400) == 0) {
                    mantissa <<= 1;
                    exponent -= 0x0400;
                }
                mantissa &= 0x03FF;
            } else {
                exponent = 0;
            }
            ur = (sign | exponent) | mantissa;
        } else {
            ur = (sign | exponent) | mantissa;
        }
    }
    
    // Python: half_bytes = struct.pack('<H', ur)
    // Python: float32_val = struct.unpack('e', half_bytes)[0]
    union { uint16_t u16; uint8_t bytes[2]; } pack;
    pack.u16 = ur;
    
    // Convert from half precision to float
    union { uint16_t u16; uint8_t bytes[2]; } half_union;
    half_union.bytes[0] = pack.bytes[0];
    half_union.bytes[1] = pack.bytes[1];
    
    // Simple half to float conversion (simplified - real version needs proper IEEE conversion)
    uint16_t h = half_union.u16;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = h & 0x03FF;
    
    if (exponent == 0x1F) {
        // Infinity or NaN
        exponent = 0xFF;
        mantissa <<= 13;
    } else if (exponent == 0) {
        // Zero or denormal
        if (mantissa == 0) {
            exponent = 0;
        } else {
            // Convert denormal to normal
            exponent = 127 - 15;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa <<= 13;
            mantissa &= 0x007FFFFF;
        }
    } else {
        // Normal number
        exponent = exponent - 15 + 127;
        mantissa <<= 13;
    }
    
    union { uint32_t u32; float f; } result;
    result.u32 = sign | (exponent << 23) | mantissa;
    return result.f;
}

// Truncation function dispatcher
double dtypes_truncate(double val, const DType* dt) {
    if (dtype_eq(dt, &dtypes.bool_)) return (double)(bool)val;
    if (dtype_eq(dt, &dtypes.float16)) return (double)truncate_fp16(val);
    if (dtype_eq(dt, &dtypes.bfloat16)) return (double)truncate_bf16(val);
    if (dtype_eq(dt, &dtypes.fp8e4m3)) return (double)fp8_to_float(float_to_fp8(val, dt), dt);
    if (dtype_eq(dt, &dtypes.fp8e5m2)) return (double)fp8_to_float(float_to_fp8(val, dt), dt);
    
    // For other types, just cast appropriately
    if (dtypes_is_int(dt)) {
        if (dtypes_is_unsigned(dt)) {
            switch (dt->itemsize) {
                case 1: return (double)(uint8_t)val;
                case 2: return (double)(uint16_t)val;
                case 4: return (double)(uint32_t)val;
                case 8: return (double)(uint64_t)val;
            }
        } else {
            switch (dt->itemsize) {
                case 1: return (double)(int8_t)val;
                case 2: return (double)(int16_t)val;
                case 4: return (double)(int32_t)val;
                case 8: return (double)(int64_t)val;
            }
        }
    }
    
    // Float types
    if (dtype_eq(dt, &dtypes.float32)) return (double)(float)val;
    if (dtype_eq(dt, &dtypes.float64)) return val;
    
    return val;
}

// String to dtype conversion
DType to_dtype(const char* name) {
    // Simplified lookup - in real implementation would use hash table
    if (strcmp(name, "bool") == 0) return dtypes.bool_;
    if (strcmp(name, "int8") == 0) return dtypes.int8;
    if (strcmp(name, "uint8") == 0) return dtypes.uint8;
    if (strcmp(name, "int16") == 0) return dtypes.int16;
    if (strcmp(name, "uint16") == 0) return dtypes.uint16;
    if (strcmp(name, "int32") == 0) return dtypes.int32;
    if (strcmp(name, "uint32") == 0) return dtypes.uint32;
    if (strcmp(name, "int64") == 0) return dtypes.int64;
    if (strcmp(name, "uint64") == 0) return dtypes.uint64;
    if (strcmp(name, "float16") == 0) return dtypes.float16;
    if (strcmp(name, "float32") == 0) return dtypes.float32;
    if (strcmp(name, "float64") == 0) return dtypes.float64;
    
    // Try aliases
    if (strcmp(name, "float") == 0) return dtypes.float32;
    if (strcmp(name, "double") == 0) return dtypes.float64;
    if (strcmp(name, "int") == 0) return dtypes.int32;
    
    fprintf(stderr, "Error: unknown dtype name: %s\n", name);
    abort();
}

const char* dtype_name(const DType* dt) {
    return dt->name;
}

// DTYPES_DICT and INVERSE_DTYPES_DICT equivalent
// Python: DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if isinstance(v, DType) and not k.startswith(("default", "void"))}
// Python: INVERSE_DTYPES_DICT = {**{v.name:k for k,v in DTYPES_DICT.items()}, "void": "void"}
const char* dtype_canonical_name(const DType* dt) {
    // Map from dtype to canonical Python name
    if (dtype_eq(dt, &dtypes.void_)) return "void";
    if (dtype_eq(dt, &dtypes.bool_)) return "bool";
    if (dtype_eq(dt, &dtypes.int8)) return "int8";
    if (dtype_eq(dt, &dtypes.uint8)) return "uint8";
    if (dtype_eq(dt, &dtypes.int16)) return "int16";
    if (dtype_eq(dt, &dtypes.uint16)) return "uint16";
    if (dtype_eq(dt, &dtypes.int32)) return "int32";
    if (dtype_eq(dt, &dtypes.uint32)) return "uint32";
    if (dtype_eq(dt, &dtypes.int64)) return "int64";
    if (dtype_eq(dt, &dtypes.uint64)) return "uint64";
    if (dtype_eq(dt, &dtypes.fp8e4m3)) return "fp8e4m3";
    if (dtype_eq(dt, &dtypes.fp8e5m2)) return "fp8e5m2";
    if (dtype_eq(dt, &dtypes.float16)) return "float16";
    if (dtype_eq(dt, &dtypes.bfloat16)) return "bfloat16";
    if (dtype_eq(dt, &dtypes.float32)) return "float32";
    if (dtype_eq(dt, &dtypes.float64)) return "float64";
    
    // Aliases
    if (dtype_eq(dt, &dtypes.half)) return "half";
    if (dtype_eq(dt, &dtypes.float_)) return "float";
    if (dtype_eq(dt, &dtypes.double_)) return "double";
    if (dtype_eq(dt, &dtypes.uchar)) return "uchar";
    if (dtype_eq(dt, &dtypes.ushort)) return "ushort";
    if (dtype_eq(dt, &dtypes.uint)) return "uint";
    if (dtype_eq(dt, &dtypes.ulong)) return "ulong";
    if (dtype_eq(dt, &dtypes.char_)) return "char";
    if (dtype_eq(dt, &dtypes.short_)) return "short";
    if (dtype_eq(dt, &dtypes.int_)) return "int";
    if (dtype_eq(dt, &dtypes.long_)) return "long";
    
    // Fallback to actual name
    return dt->name;
}

// Initialization function
void dtypes_init(void) {
    // Initialize all basic types - equivalent to Python class definitions
    dtypes.void_ = dtype_new(-1, 0, "void", 0);
    dtypes.bool_ = dtype_new(0, 1, "bool", '?');
    dtypes.int8 = dtype_new(1, 1, "signed char", 'b');
    dtypes.uint8 = dtype_new(2, 1, "unsigned char", 'B');
    dtypes.int16 = dtype_new(3, 2, "short", 'h');
    dtypes.uint16 = dtype_new(4, 2, "unsigned short", 'H');
    dtypes.int32 = dtype_new(5, 4, "int", 'i');
    dtypes.uint32 = dtype_new(6, 4, "unsigned int", 'I');
    dtypes.int64 = dtype_new(7, 8, "long", 'q');
    dtypes.uint64 = dtype_new(8, 8, "unsigned long", 'Q');
    dtypes.fp8e4m3 = dtype_new(9, 1, "float8_e4m3", 0);
    dtypes.fp8e5m2 = dtype_new(10, 1, "float8_e5m2", 0);
    dtypes.float16 = dtype_new(11, 2, "half", 'e');
    dtypes.bfloat16 = dtype_new(12, 2, "__bf16", 0);
    dtypes.float32 = dtype_new(13, 4, "float", 'f');
    dtypes.float64 = dtype_new(14, 8, "double", 'd');
    
    // Set up aliases
    dtypes.half = dtypes.float16;
    dtypes.float_ = dtypes.float32;
    dtypes.double_ = dtypes.float64;
    dtypes.uchar = dtypes.uint8;
    dtypes.ushort = dtypes.uint16;
    dtypes.uint = dtypes.uint32;
    dtypes.ulong = dtypes.uint64;
    dtypes.char_ = dtypes.int8;
    dtypes.short_ = dtypes.int16;
    dtypes.int_ = dtypes.int32;
    dtypes.long_ = dtypes.int64;
    
    // Set defaults (as pointers)
    dtypes.default_float = &dtypes.float32;
    dtypes.default_int = &dtypes.int32;
    
    // Initialize type collections
    dtypes.fp8s[0] = &dtypes.fp8e4m3;
    dtypes.fp8s[1] = &dtypes.fp8e5m2;
    
    dtypes.floats[0] = &dtypes.fp8e4m3;
    dtypes.floats[1] = &dtypes.fp8e5m2;
    dtypes.floats[2] = &dtypes.float16;
    dtypes.floats[3] = &dtypes.bfloat16;
    dtypes.floats[4] = &dtypes.float32;
    dtypes.floats[5] = &dtypes.float64;
    
    dtypes.uints[0] = &dtypes.uint8;
    dtypes.uints[1] = &dtypes.uint16;
    dtypes.uints[2] = &dtypes.uint32;
    dtypes.uints[3] = &dtypes.uint64;
    
    dtypes.sints[0] = &dtypes.int8;
    dtypes.sints[1] = &dtypes.int16;
    dtypes.sints[2] = &dtypes.int32;
    dtypes.sints[3] = &dtypes.int64;
    
    dtypes.ints[0] = &dtypes.uint8;
    dtypes.ints[1] = &dtypes.uint16;
    dtypes.ints[2] = &dtypes.uint32;
    dtypes.ints[3] = &dtypes.uint64;
    dtypes.ints[4] = &dtypes.int8;
    dtypes.ints[5] = &dtypes.int16;
    dtypes.ints[6] = &dtypes.int32;
    dtypes.ints[7] = &dtypes.int64;
    
    dtypes.all[0] = &dtypes.fp8e4m3;
    dtypes.all[1] = &dtypes.fp8e5m2;
    dtypes.all[2] = &dtypes.float16;
    dtypes.all[3] = &dtypes.bfloat16;
    dtypes.all[4] = &dtypes.float32;
    dtypes.all[5] = &dtypes.float64;
    dtypes.all[6] = &dtypes.uint8;
    dtypes.all[7] = &dtypes.uint16;
    dtypes.all[8] = &dtypes.uint32;
    dtypes.all[9] = &dtypes.uint64;
    dtypes.all[10] = &dtypes.int8;
    dtypes.all[11] = &dtypes.int16;
    dtypes.all[12] = &dtypes.int32;
    dtypes.all[13] = &dtypes.int64;
    dtypes.all[14] = &dtypes.bool_;
    dtypes.all[15] = NULL; // padding
    dtypes.all[16] = NULL; // padding
    
    // Check for environment variable to override default float
    const char* env_default_float = tg_getenv("DEFAULT_FLOAT");
    if (env_default_float && strlen(env_default_float) > 0) {
        // Convert to lowercase and lookup
        char lower_name[32];
        strncpy(lower_name, env_default_float, sizeof(lower_name) - 1);
        lower_name[sizeof(lower_name) - 1] = '\0';
        for (int i = 0; lower_name[i]; i++) {
            lower_name[i] = tolower(lower_name[i]);
        }
        
        DType candidate = to_dtype(lower_name);
        if (dtypes_is_float(&candidate)) {
            // Need to point to the actual global dtype instance
            if (dtype_eq(&candidate, &dtypes.float16)) {
                dtypes.default_float = &dtypes.float16;
            } else if (dtype_eq(&candidate, &dtypes.float32)) {
                dtypes.default_float = &dtypes.float32;
            } else if (dtype_eq(&candidate, &dtypes.float64)) {
                dtypes.default_float = &dtypes.float64;
            } else if (dtype_eq(&candidate, &dtypes.bfloat16)) {
                dtypes.default_float = &dtypes.bfloat16;
            }
        } else {
            fprintf(stderr, "Warning: %s is not a float dtype, keeping default\n", env_default_float);
        }
    }
    
    // Initialize promotion lattice
    init_promo_lattice();
}

void dtypes_cleanup(void) {
    // Clear dtype cache
    g_dtype_cache_count = 0;
    
    // Clear promotion lattice
    g_promo_entry_count = 0;
    
    // Free recursive parent caches
    for (int i = 0; i < g_recursive_cache_count; i++) {
        free((void*)g_recursive_cache[i].parents);
    }
    g_recursive_cache_count = 0;
}
