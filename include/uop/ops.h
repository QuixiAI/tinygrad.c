#ifndef TINYGRAD_UOP_OPS_H
#define TINYGRAD_UOP_OPS_H

#include "uop.h"
#include "mathtraits.h"
#include "dtype/dtype.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct ShapeTracker ShapeTracker;
typedef struct Buffer Buffer;
typedef struct Metadata Metadata;

// UOp argument types
typedef enum {
    ARG_NONE = 0,
    ARG_INT,
    ARG_FLOAT,
    ARG_CONST,
    ARG_STRING,
    ARG_REDUCE,
    ARG_SHAPE_TRACKER,
    ARG_BUFFER,
} ArgType;

typedef struct {
    ArgType type;
    union {
        int i;
        float f;
        double const_value;
        const char* s;
        void* ptr;
        struct {
            int x, y, z;
        } vec3i;
        struct {
            Ops reduce_op;
            int* axes;
            int axes_count;
        } reduce_arg;
        ShapeTracker* st;
        Buffer* buf;
    };
} UOpArg;

// UOp structure - faithful port of Python UOp dataclass
typedef struct UOp {
    Ops op;
    DType dtype;
    struct UOp** src;      // Source UOps
    size_t src_count;
    UOpArg arg;
    void* tag;             // Optional tag
    
    // Weak references to children
    struct UOp** children;
    size_t children_count;
    size_t children_capacity;
    
    // Optional metadata
    Metadata* metadata;
    
    // Optional buffer reference (for BUFFER ops)
    Buffer* buffer;
    
    // Cached values
    bool cached_hash_valid;
    size_t cached_hash;
    
    // MathTrait operations
    const MathTraitOps* math_ops;
    
    // Reference counting
    int ref_count;
} UOp;

// UOp creation and management
UOp* uop_new(Ops op, DType dtype, UOp** src, size_t src_count, UOpArg* arg, void* tag);
void uop_free(UOp* uop);
UOp* uop_ref(UOp* uop);
void uop_unref(UOp* uop);

// UOp operations
UOp* uop_sink(UOp** stores, size_t count);
UOp* uop_store(UOp* buf, UOp* value);
UOp* uop_load(UOp* buf, DType dtype);
UOp* uop_const(DType dtype, double value);
UOp* uop_define_global(DType dtype, int idx);
UOp* uop_define_local(DType dtype, size_t size);
UOp* uop_define_reg(DType dtype);

// Binary operations shortcuts
UOp* uop_add(UOp* a, UOp* b);
UOp* uop_mul(UOp* a, UOp* b);
UOp* uop_sub(UOp* a, UOp* b);
UOp* uop_div(UOp* a, UOp* b);
UOp* uop_max(UOp* a, UOp* b);
UOp* uop_min(UOp* a, UOp* b);
UOp* uop_lt(UOp* a, UOp* b);
UOp* uop_eq(UOp* a, UOp* b);
UOp* uop_ne(UOp* a, UOp* b);

// Unary operations shortcuts
UOp* uop_neg(UOp* a);
UOp* uop_exp2(UOp* a);
UOp* uop_log2(UOp* a);
UOp* uop_sin(UOp* a);
UOp* uop_sqrt(UOp* a);
UOp* uop_recip(UOp* a);
UOp* uop_cast(UOp* a, DType dtype);

// Ternary operations
UOp* uop_where(UOp* cond, UOp* true_val, UOp* false_val);
UOp* uop_mulacc(UOp* a, UOp* b, UOp* c);

// Reduction operations
UOp* uop_reduce_axis(UOp* src, Ops reduce_op, int* axes, int axes_count);

// View operations
UOp* uop_view(UOp* buf, ShapeTracker* st);
UOp* uop_index(UOp* buf, UOp* idx);

// Graph operations
UOp** uop_toposort(UOp* root, size_t* count);
void uop_print(UOp* uop, int depth);
void uop_print_graph(UOp* root);
size_t uop_hash(UOp* uop);
bool uop_equals(UOp* a, UOp* b);

// Simplification
UOp* uop_simplify(UOp* uop);
UOp* uop_ssimplify(UOp* uop);

// Helper functions
bool uop_commutative(UOp* uop);
bool uop_is_zero(UOp* uop);
bool uop_is_one(UOp* uop);
int uop_divides(UOp* uop, int v);
UOp** uop_parents(UOp* uop, size_t* count);
UOp* uop_replace(UOp* uop, Ops new_op, DType* new_dtype, UOp** new_src, size_t new_src_count, UOpArg* new_arg);

// Symbolic operations
int uop_sym_infer(UOp* uop);
bool uop_resolve(UOp* uop, bool default_val);

// Cache management
typedef struct {
    size_t size;
    size_t capacity;
    struct UOpCacheEntry* entries;
} UOpCache;

extern UOpCache* global_uop_cache;

void uop_cache_init(void);
void uop_cache_cleanup(void);
UOp* uop_cache_get(Ops op, DType dtype, UOp** src, size_t src_count, UOpArg* arg, void* tag);
void uop_cache_put(UOp* uop);

// Pattern matching support
typedef struct UPat UPat;

struct UPat {
    enum {
        UPAT_OP,
        UPAT_VAR,
        UPAT_CONST,
        UPAT_ANY
    } type;
    
    union {
        Ops op;
        int var_id;
        double const_val;
    } value;
    
    DType* dtype;  // Optional dtype constraint
    UPat** src;
    size_t src_count;
};

UPat* upat_op(Ops op, UPat** src, size_t src_count);
UPat* upat_var(int id);
UPat* upat_const(double val);
UPat* upat_any(void);
bool upat_match(UPat* pattern, UOp* uop);
void upat_free(UPat* pat);

// Execution
double exec_alu(Ops op, DType dtype, double* args, size_t arg_count);
double identity_element(Ops op, DType* dtype);

// Module initialization
void uop_ops_init(void);
void uop_ops_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* TINYGRAD_UOP_OPS_H */