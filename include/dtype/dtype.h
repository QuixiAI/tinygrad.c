#ifndef SRC_DTYPE_DTYPE_H
#define SRC_DTYPE_DTYPE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Address spaces
typedef enum {
    ADDRSPACE_GLOBAL = 0,
    ADDRSPACE_LOCAL = 1,
    ADDRSPACE_REG = 2
} AddrSpace;

// Forward declarations
typedef struct DType DType;
typedef struct PtrDType PtrDType;
typedef struct ImageDType ImageDType;

// Base DType structure
struct DType {
    int priority;           // determines when things get upcasted
    int itemsize;          // size in bytes
    char name[32];         // name string
    char fmt;              // format character (or 0 for None)
    int count;             // vector count
    const DType* _scalar;  // scalar type (NULL if this is scalar)
};

// Pointer DType structure  
struct PtrDType {
    DType base;            // inherits from DType
    const DType* base_dtype; // the base dtype this points to
    AddrSpace addrspace;   // address space
    int v;                 // vector size
    int size;              // size (-1 for unlimited)
};

// Image DType structure
struct ImageDType {
    PtrDType ptr_base;     // inherits from PtrDType
    int shape[4];          // shape dimensions
    int shape_len;         // number of dimensions
};

// FInfo structure for floating point info
typedef struct {
    int exponent;
    int mantissa;
} FInfo;

// Global dtypes namespace
typedef struct dtypes_struct {
    // Basic types
    DType void_;
    DType bool_;
    DType int8;
    DType uint8;
    DType int16;
    DType uint16;
    DType int32;
    DType uint32;
    DType int64;
    DType uint64;
    DType fp8e4m3;
    DType fp8e5m2;
    DType float16;
    DType bfloat16;
    DType float32;
    DType float64;
    
    // Aliases
    DType half;
    DType float_;
    DType double_;
    DType uchar;
    DType ushort;
    DType uint;
    DType ulong;
    DType char_;
    DType short_;
    DType int_;
    DType long_;
    
    // Default types (pointers to actual types)
    const DType* default_float;
    const DType* default_int;
    
    // Type collections
    const DType* fp8s[2];
    const DType* floats[6];
    const DType* uints[4];
    const DType* sints[4];
    const DType* ints[8];
    const DType* all[17];
} DtypesStruct;

extern DtypesStruct dtypes;

// DType creation functions
DType dtype_new(int priority, int itemsize, const char* name, char fmt);

// DType methods
bool dtype_eq(const DType* a, const DType* b);
bool dtype_same_instance(const DType* a, const DType* b); // pointer equality
bool dtype_lt(const DType* a, const DType* b);
DType dtype_vec(const DType* dt, int sz);
PtrDType dtype_ptr(const DType* dt, int size, AddrSpace addrspace);
DType dtype_scalar(const DType* dt);
int dtype_nbytes(const DType* dt); // throws error for non-ptr types
double dtype_min_val(const DType* dt);
double dtype_max_val(const DType* dt);
// Properties parity helpers
const DType* dtype_base(const DType* dt);
int dtype_vcount(const DType* dt);

// PtrDType methods
int ptrdtype_nbytes(const PtrDType* dt);
PtrDType ptrdtype_vec(const PtrDType* dt, int sz);
const DType* ptrdtype_base(const PtrDType* dt);
int ptrdtype_vcount(const PtrDType* dt);

// ImageDType methods
ImageDType imagedtype_create(const int* shape, int shape_len, bool is_half);
ImageDType dtypes_imageh(const int* shape, int shape_len);
ImageDType dtypes_imagef(const int* shape, int shape_len);

// Type checking functions
bool dtypes_is_float(const DType* dt);
bool dtypes_is_int(const DType* dt);
bool dtypes_is_unsigned(const DType* dt);
bool dtypes_is_bool(const DType* dt);

// Type conversion functions
DType dtypes_from_py_float(double val);
DType dtypes_from_py_int(int64_t val);
DType dtypes_from_py_bool(bool val);

// as_const functions
double dtypes_as_const_float(double val, const DType* dt);
int64_t dtypes_as_const_int(int64_t val, const DType* dt);
bool dtypes_as_const_bool(bool val, const DType* dt);

// Min/max functions
double dtypes_min(const DType* dt);
double dtypes_max(const DType* dt);

// FInfo function
FInfo dtypes_finfo(const DType* dt);

// Type promotion lattice entry
typedef struct {
    const DType* dtype;
    const DType** promotions;
    int promotion_count;
} PromoLatticeEntry;

// Type promotion functions
DType least_upper_dtype(const DType* a, const DType* b);
DType least_upper_dtype_multi(const DType** types, int count);
DType least_upper_float(const DType* dt);

// Internal promotion functions
const DType** _get_recursive_parents(const DType* dtype, int* count);
void _free_recursive_parents(const DType** parents);

// Safe casting
bool can_safe_cast(const DType* from, const DType* to);

// Accumulator dtype for sum
DType sum_acc_dtype(const DType* dt);

// FP16/BF16 truncation
float truncate_fp16(double x);
float truncate_bf16(double x);

// FP8 conversions  
uint8_t float_to_fp8(double x, const DType* dtype);
float fp8_to_float(uint8_t x, const DType* dtype);

// Truncation functions
double dtypes_truncate(double val, const DType* dt);

// String conversion
DType to_dtype(const char* name);
const char* dtype_name(const DType* dt);

// DTYPES_DICT equivalent - get canonical name for dtype
const char* dtype_canonical_name(const DType* dt);

// dtypes.fields() equivalent (iterate canonical types)
int dtypes_fields_count(void);
const DType* dtypes_field_dtype(int index);
const char* dtypes_field_name(int index);

// Initialization and cleanup
void dtypes_init(void);
void dtypes_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* SRC_DTYPE_DTYPE_H */
