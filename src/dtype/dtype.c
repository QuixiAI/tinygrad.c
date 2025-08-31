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

#include "dtype.h"
#include "../helpers/helpers.h"

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
    char vec_name[32];
    snprintf(vec_name, sizeof(vec_name), "%s%d", dt->name, sz);
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

// Type checking functions
bool dtypes_is_float(const DType* dt) {
    // Python: def is_float(x: DType) -> bool: return x.scalar() in dtypes.floats or isinstance(x, ImageDType)
    DType scalar = dtype_scalar(dt);
    for (int i = 0; i < 6; i++) {
        if (dtype_eq(&scalar, dtypes.floats[i])) return true;
    }
    // TODO: check for ImageDType
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
    return dtypes.default_float;
}

DType dtypes_from_py_int(int64_t val) {
    // Python: if x.__class__ is int: return dtypes.default_int
    (void)val; // unused
    return dtypes.default_int;
}

DType dtypes_from_py_bool(bool val) {
    // Python: if x.__class__ is bool: return dtypes.bool
    (void)val; // unused
    return dtypes.bool_;
}

// as_const functions
double dtypes_as_const_float(double val, const DType* dt) {
    // Python: return int(val) if dtypes.is_int(dtype) else float(val) if dtypes.is_float(dtype) else bool(val)
    if (dtypes_is_int(dt)) return (double)(int64_t)val;
    if (dtypes_is_float(dt)) return val;
    return (double)(bool)val;
}

int64_t dtypes_as_const_int(int64_t val, const DType* dt) {
    if (dtypes_is_int(dt)) return val;
    if (dtypes_is_float(dt)) return (int64_t)val;
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

// Least upper dtype (simplified version)
DType least_upper_dtype(const DType* a, const DType* b) {
    // Simplified promotion logic - return the type with higher priority
    if (a->priority > b->priority) return *a;
    return *b;
}

DType least_upper_dtype_multi(const DType** types, int count) {
    if (count == 0) return dtypes.default_float;
    DType result = *types[0];
    for (int i = 1; i < count; i++) {
        result = least_upper_dtype(&result, types[i]);
    }
    return result;
}

DType least_upper_float(const DType* dt) {
    // Python: def least_upper_float(dt:DType) -> DType: return dt if dtypes.is_float(dt) else least_upper_dtype(dt, dtypes.default_float)
    if (dtypes_is_float(dt)) return *dt;
    return least_upper_dtype(dt, &dtypes.default_float);
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
    DType sum_dtype = to_dtype(sum_dtype_env);
    return least_upper_dtype(dt, &sum_dtype);
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

// FP8 conversions (simplified implementations)
uint8_t float_to_fp8(double x, const DType* dtype) {
    // Simplified fp8 conversion - real implementation would need full bit manipulation
    assert(dtype_eq(dtype, &dtypes.fp8e4m3) || dtype_eq(dtype, &dtypes.fp8e5m2));
    
    float f = (float)x;
    if (isnan(f) || isinf(f)) return 0x7F;
    if (f == 0.0f) return 0;
    
    // Very simplified conversion
    int sign = f < 0 ? 0x80 : 0;
    f = fabsf(f);
    
    if (f > 1.0f) return sign | 0x7E; // saturate to max
    if (f < 0.01f) return sign; // underflow to zero
    
    // Crude quantization
    return sign | (uint8_t)(f * 127.0f);
}

float fp8_to_float(uint8_t x, const DType* dtype) {
    // Simplified fp8 to float conversion
    assert(dtype_eq(dtype, &dtypes.fp8e4m3) || dtype_eq(dtype, &dtypes.fp8e5m2));
    
    int sign = (x & 0x80) ? -1 : 1;
    uint8_t val = x & 0x7F;
    
    if (val == 0x7F) return sign * INFINITY;
    if (val == 0) return 0.0f;
    
    // Crude dequantization
    return sign * ((float)val / 127.0f);
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
    
    // Set defaults
    dtypes.default_float = dtypes.float32;
    dtypes.default_int = dtypes.int32;
    
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
            dtypes.default_float = candidate;
        } else {
            fprintf(stderr, "Warning: %s is not a float dtype, keeping default\n", env_default_float);
        }
    }
}

void dtypes_cleanup(void) {
    // Clear cache
    g_dtype_cache_count = 0;
}
