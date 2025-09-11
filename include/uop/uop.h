#ifndef TINYGRAD_UOP_H
#define TINYGRAD_UOP_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Forward declarations for other headers
#include "dtype/dtype.h"

// Forward declarations
typedef struct ShapeTracker ShapeTracker;
typedef struct UPat UPat;
struct UOp;  // forward declaration for mixed tuple


#ifdef __cplusplus
extern "C" {
#endif

// FastEnum equivalent - all ops have unique values
typedef enum {
    // uops that aren't rendered
    OPS_NOOP = 1,
    OPS_SINK,
    OPS_UNIQUE,
    OPS_DEVICE,
    OPS_KERNEL,
    OPS_PRECAST,
    
    // track children
    OPS_CHILD,
    
    // buffer ops
    OPS_COPY,
    OPS_BUFFER,
    OPS_BUFFER_VIEW,
    OPS_MSELECT,
    OPS_MSTACK,
    
    // ops that adjust scheduler behavior
    OPS_CONTIGUOUS,
    OPS_CONTIGUOUS_BACKWARD,
    OPS_DETACH,
    OPS_FUSE,
    
    // blocks in linearizer
    OPS_BLOCK,
    OPS_BLOCKSTART,
    OPS_BLOCKEND,
    OPS_BLOCKFINAL,
    
    // movement ops
    OPS_RESHAPE,
    OPS_PERMUTE,
    OPS_EXPAND,
    OPS_PAD,
    OPS_SHRINK,
    OPS_FLIP,
    OPS_MULTI,
    
    // view op
    OPS_VIEW,
    
    // valid op
    OPS_VALID,
    
    // memory hierarchy ops
    OPS_DEFINE_GLOBAL,
    OPS_DEFINE_LOCAL,
    OPS_DEFINE_REG,
    
    // symbolic shapes
    OPS_DEFINE_VAR,
    OPS_BIND,
    
    // GPU dimensions
    OPS_SPECIAL,
    
    // reduce ops
    OPS_REDUCE_AXIS,
    OPS_REDUCE,
    OPS_ALLREDUCE,
    
    // optimization helpers
    OPS_UNROLL,
    OPS_CONTRACT,
    OPS_GEP,
    OPS_VECTORIZE,
    OPS_CAT,
    OPS_PTRCAT,
    
    // UnaryOps
    OPS_CAST,
    OPS_BITCAST,
    OPS_EXP2,
    OPS_LOG2,
    OPS_SIN,
    OPS_SQRT,
    OPS_RECIP,
    OPS_NEG,
    
    // load/store
    OPS_LOAD,
    OPS_STORE,
    OPS_ASSIGN,
    
    // tensor core op
    OPS_WMMA,
    
    // pointer index op
    OPS_INDEX,
    
    // BinaryOps
    OPS_ADD,
    OPS_MUL,
    OPS_SHL,
    OPS_SHR,
    OPS_IDIV,
    OPS_MAX,
    OPS_MOD,
    OPS_CMPLT,
    OPS_CMPNE,
    OPS_CMPEQ,
    OPS_XOR,
    OPS_OR,
    OPS_AND,
    OPS_THREEFRY,
    OPS_SUB,
    OPS_FDIV,
    OPS_POW,
    
    // TernaryOps
    OPS_WHERE,
    OPS_MULACC,
    
    // control flow
    OPS_BARRIER,
    OPS_RANGE,
    OPS_IF,
    OPS_ENDRANGE,
    OPS_ENDIF,
    
    // constants
    OPS_VCONST,
    OPS_CONST,
    
    // custom codegen
    OPS_CUSTOM,
    OPS_CUSTOMI,
    
    OPS_MAX_VALUE  // sentinel for max value
} Ops;

// UOp argument types
typedef enum {
    ARG_NONE = 0,
    ARG_CONST,
    ARG_VCONST,
    ARG_INT,
    ARG_REDUCE,
    ARG_SHAPE_TRACKER,
    ARG_VAR,           // For variables with ranges
    ARG_PAD_PARAMS,    // For PAD: before/after per-dim
    ARG_SHRINK_PARAMS, // For SHRINK: start/end per-dim
    ARG_TUPLE2,        // Generic tuple of pairs (for WMMA/CONTRACT/UNROLL validation)
    ARG_STRING,        // Single C string (for DEVICE)
    ARG_STRLIST,       // List/tuple of C strings (for multi-device)
    ARG_TUPLE_MIXED    // Heterogeneous tuple (for BUFFER_VIEW args)
} UOpArgType;

// Tags for ARG_TUPLE_MIXED items
typedef enum {
    MIXED_INT = 1,
    MIXED_UOP = 2,
} MixedTag;

typedef struct UOpArg {
    UOpArgType type;
    union {
        struct {
            double const_value;
        } const_data;
        struct {
            double* values;  // vector constant values
            int count;
        } vconst_data;
        struct {
            int i;
        } int_data;
        struct {
            Ops reduce_op;
            int* axes;
            int axes_count;
        } reduce_data;
        struct {
            void* st;  // ShapeTracker*
        } st_data;
        struct {
            char* name;  // Variable name - Python line 422
            int64_t vmin;  // Variable min value - Python line 422
            int64_t vmax;  // Variable max value - Python line 422
        } var;  // For DEFINE_VAR arguments
        struct {
            int32_t* before;
            int32_t* after;
            int32_t ndim;
        } pad_data;
        struct {
            int32_t* start;
            int32_t* end;
            int32_t ndim;
        } shrink_data;
        struct {
            int count;
            int* first;
            int* second;
        } tuple2;
        struct {
            char* s;
        } str;
        struct {
            int count;
            char** items;  // array of C strings
        } strlist;
        struct {
            int count;
            struct MixedItem { int tag; long long ival; struct UOp* uop; } *items;
        } tmixed;
    };
} UOpArg;

// GroupOp collections - mirrors Python GroupOp class
typedef struct {
    bool is_unary[OPS_MAX_VALUE];
    bool is_binary[OPS_MAX_VALUE];
    bool is_ternary[OPS_MAX_VALUE];
    bool is_alu[OPS_MAX_VALUE];
    bool is_define[OPS_MAX_VALUE];
    bool is_irreducible[OPS_MAX_VALUE];
    bool is_movement[OPS_MAX_VALUE];
    bool is_buffer[OPS_MAX_VALUE];
    bool is_block[OPS_MAX_VALUE];
    bool is_commutative[OPS_MAX_VALUE];
    bool is_associative[OPS_MAX_VALUE];
    bool is_idempotent[OPS_MAX_VALUE];
    bool is_unsafe_pad[OPS_MAX_VALUE];
    bool is_meta[OPS_MAX_VALUE];
    bool is_all[OPS_MAX_VALUE];
} GroupOp;

// Global GroupOp instance
extern const GroupOp group_op;

// Simple metadata KV list for optional metadata map
typedef struct UOpMetaKV {
    char* key;
    void* value;
    struct UOpMetaKV* next;
} UOpMetaKV;

// UOp structure - mirrors Python UOp class
typedef struct UOp UOp;
typedef struct UOpCacheEntry UOpCacheEntry;

// Forward declaration for MathTraitOps
typedef struct MathTraitOps MathTraitOps;

// Include mathtraits after forward declarations
#include "mathtraits.h"

typedef struct UOp {
    Ops op;
    DType dtype;
    UOp** src;
    size_t src_count;
    UOpArg arg;
    struct ShapeTracker* st;  // attached ShapeTracker for movement/view ops
    const MathTraitOps* math_ops;
    void* tag;  // parity with Python cache key (op, dtype, src, arg, tag)
    int ref_count;
    // Symbolic range tracking - Python line 472-474
    int64_t vmin;  // minimum value in symbolic range
    int64_t vmax;  // maximum value in symbolic range
    bool vmin_vmax_valid;  // flag to indicate if vmin/vmax are computed
    // Optional: children tracking and metadata map
    UOp** children;
    size_t children_count;
    size_t children_cap;
    struct UOpMetaKV* meta_head;
} UOp;

// Pattern matching structures
typedef enum {
    UPAT_ANY,
    UPAT_VAR,
    UPAT_CONST,
    UPAT_OP,
    UPAT_DTYPE
} UPatType;

typedef struct UPat {
    UPatType type;
    union {
        struct {
            Ops op;
        } op_data;
        struct {
            int var_id;
        } var_data;
        struct {
            double const_val;
        } const_data;
    };
    UPat** src;
    size_t src_count;
    // Extended pattern metadata for parity with Python
    // Multi-op and multi-dtype lists
    Ops* op_list;
    size_t op_list_count;
    const DType** dtype_list;
    size_t dtype_list_count;
    bool vec_any;          // accept vectorized variants of dtype filter
    bool require_const;    // for cvar: require uop to be CONST
    // Argument equality (integer literal)
    bool has_int_arg;
    int arg_int;
    // Argument equality via binding/placeholder (rendered by renderer)
    bool has_arg_bind;
    const char* arg_bind_str;
    // Repeat and forked source forms
    bool src_is_repeat;         // src[0] is a repeat pattern
    bool src_is_fork;           // src encodes concatenated tuples
    int* fork_group_sizes;      // sizes of each tuple group in src
    size_t fork_group_count;
    // Additional fields for pattern matching
    bool strict_length;
    int required_len;
    const char* name;
    void* dtype;  // Placeholder for dtype matching
} UPat;

// Pattern compilation error
typedef struct {
    int code;
    const char* message;
} UPatCompileError;

// Pattern matcher structure
typedef struct PatternMatcher PatternMatcher;
typedef struct PatternMatch PatternMatch;

typedef struct PatternMatch {
    UPat* pattern;
    void* (*callback)(void*, void*);  // Function pointer
    void* (*callback_ex)(void* ctx, void* node, const char** names, UOp** values, size_t nbinds);  // Extended with bindings
    void* user_data;
} PatternMatch;

typedef enum {
    PM_OK = 0,
    PM_COMPILE_ERROR,
    PM_MATCH_ERROR
} PatternMatcherResult;

typedef struct PatternMatcher {
    PatternMatch* matches;
    size_t match_count;
    size_t capacity;
    bool compiled;
} PatternMatcher;

// Cache structures
typedef struct UOpCacheEntry {
    size_t key_hash;
    UOp* value;
    struct UOpCacheEntry* next;
} UOpCacheEntry;

typedef struct {
    UOpCacheEntry** buckets;
    size_t bucket_count;
    size_t size;
} UOpCacheTable;

// Helper functions
const char* ops_to_string(Ops op);
bool ops_is_valid(Ops op);
int ops_get_arity(Ops op);

// UOp operations
UOp* uop_new(Ops op, DType dtype, UOp** src, size_t src_count, UOpArg* arg, void* tag);
void uop_free(UOp* uop);
UOp* uop_ref(UOp* uop);
void uop_unref(UOp* uop);
bool uop_commutative(UOp* uop);
bool uop_is_zero(UOp* uop);
bool uop_is_one(UOp* uop);
int uop_divides(UOp* uop, int v);
UOp* uop_sink(UOp** stores, size_t count);
UOp* uop_store(UOp* buf, UOp* value);
UOp* uop_load(UOp* buf, DType dtype);
UOp* uop_const(DType dtype, double value);
UOp* uop_vconst(DType dtype, const double* vals, int count);
UOp* uop_const_like(UOp* self, double b);
// Create CONST with optional device source and explicit shape
UOp* uop_const_ex(DType dtype, double value, UOp* device_uop, const int32_t* shape, int nd);
UOp* uop_define_global(DType dtype, int idx);
UOp* uop_define_local(DType dtype, size_t size);
UOp* uop_define_reg(DType dtype);
UOp* uop_add(UOp* a, UOp* b);
UOp* uop_mul(UOp* a, UOp* b);
UOp* uop_sub(UOp* a, UOp* b);
UOp* uop_div(UOp* a, UOp* b);
UOp* uop_max(UOp* a, UOp* b);
UOp* uop_min(UOp* a, UOp* b);
UOp* uop_lt(UOp* a, UOp* b);
UOp* uop_eq(UOp* a, UOp* b);
UOp* uop_ne(UOp* a, UOp* b);
UOp* uop_neg(UOp* a);
UOp* uop_exp2(UOp* a);
UOp* uop_log2(UOp* a);
UOp* uop_sin(UOp* a);
UOp* uop_sqrt(UOp* a);
UOp* uop_recip(UOp* a);
UOp* uop_contiguous(UOp* a);
UOp* uop_contiguous_backward(UOp* a);
UOp* uop_fuse(UOp* a);
UOp* uop_detach(UOp* a);
UOp* uop_cast(UOp* a, DType dtype);
UOp* uop_cast_vec(UOp* a, DType dtype_scalar, int count);
UOp* uop_broadcast(UOp* a, int count);
// Structured emitters that populate ARG_TUPLE2 for validation-sensitive ops
UOp* uop_wmma(UOp* a, UOp* b, UOp* acc, const int* first, const int* second, int count);
UOp* uop_contract(UOp* x, const int* first, const int* second, int count);
UOp* uop_unroll(UOp* x, const int* first, const int* second, int count);

// DEVICE constructors
UOp* uop_device_str(const char* dev);
UOp* uop_device_tuple(const char* const* devs, int count);

// BUFFER_VIEW constructor with heterogeneous tuple args (two elements)
// tag0/tag1 are MIXED_INT or MIXED_UOP; provide ival/uop accordingly
UOp* uop_buffer_view(UOp* buffer, int tag0, long long ival0, struct UOp* u0,
                     int tag1, long long ival1, struct UOp* u1);
// Convenience wrappers for common BUFFER_VIEW arg combinations
UOp* uop_buffer_view_ii(UOp* buffer, long long a0, long long a1);
UOp* uop_buffer_view_iU(UOp* buffer, long long a0, struct UOp* a1);
UOp* uop_buffer_view_Ui(UOp* buffer, struct UOp* a0, long long a1);
UOp* uop_buffer_view_UU(UOp* buffer, struct UOp* a0, struct UOp* a1);
UOp* uop_where(UOp* cond, UOp* true_val, UOp* false_val);
UOp* uop_mulacc(UOp* a, UOp* b, UOp* c);
UOp* uop_reduce_axis(UOp* src, Ops reduce_op, int* axes, int axes_count);
UOp* uop_gep(UOp* base, const int* idxs, int idx_count);
int uop_axis_arg(UOp* uop, int** axes_out, int* count_out);

// Additional transcendental support functions
UOp* uop_bitcast(UOp* a, DType dtype);
UOp* uop_and(UOp* a, UOp* b);
UOp* uop_or(UOp* a, UOp* b);
UOp* uop_xor(UOp* a, UOp* b);
UOp* uop_shl(UOp* a, UOp* b);
UOp* uop_shr(UOp* a, UOp* b);
UOp* uop_ge(UOp* a, UOp* b);
UOp* uop_cmpne(UOp* a, UOp* b);
UOp* uop_abs(UOp* a);
UOp* uop_remainder(UOp* a, UOp* b);
UOp* uop_mod(UOp* a, UOp* b);
UOp* uop_gt(UOp* a, UOp* b);
UOp* uop_view(UOp* buf, struct ShapeTracker* st);
UOp* uop_index(UOp* buf, UOp* idx);
UOp* uop_assign(UOp* dst, UOp* value);
UOp* uop_barrier(UOp* after);
// Movement ops
UOp* uop_reshape(UOp* x, const int32_t* new_shape, int32_t new_ndim);
UOp* uop_permute(UOp* x, const int32_t* axes, int32_t num_axes);
UOp* uop_expand(UOp* x, const int32_t* target_shape, int32_t target_ndim);
UOp* uop_pad(UOp* x, const int32_t* pad_before, const int32_t* pad_after, int32_t ndim);
UOp* uop_shrink(UOp* x, const int32_t* start, const int32_t* end, int32_t ndim);
UOp* uop_flip_axis(UOp* x, int axis);

// Shape helper (reads from ARG_SHAPE_TRACKER if present). Returns NULL if unknown.
const int32_t* uop_shape(UOp* uop, int* ndim_out);
UOp* uop_var(const char* name, DType dtype);
UOp* uop_var_with_range(const char* name, DType dtype, int min_val, int max_val);
UOp* uop_range(UOp* n, int idx);
UOp* uop_buffer(int64_t* shape, size_t shape_count, DType dtype);
UOp* uop_reduce(UOp* src, Ops reduce_op);
UOp** uop_toposort(UOp* root, size_t* count);
UOp** uop_toposort_gate(UOp* root, size_t* count, bool (*gate)(UOp*));
void uop_print(UOp* uop, int depth);
void uop_print_graph(UOp* root);
void print_uops(UOp** uops, size_t count);
typedef const char* (*uop_rep_fn)(UOp* u, void* ctx);
void print_uops_ex(UOp** uops, size_t count, uop_rep_fn rep, void* ctx, bool color);
// Single-line pretty representation (heap-allocated; caller frees)
char* uop_pretty_str(UOp* uop, bool color);

// Replace utility (subset of Python replace semantics)
UOp* uop_replace_ex(UOp* uop, Ops new_op, DType new_dtype, UOp** new_src, size_t new_src_count, UOpArg* new_arg, void* new_tag);
// Legacy convenience (dtype pointer, no tag)
UOp* uop_replace(UOp* uop, Ops new_op, DType* new_dtype, UOp** new_src, size_t new_src_count, UOpArg* new_arg);
size_t uop_hash(UOp* uop);
bool uop_equals(UOp* a, UOp* b);
UOp* uop_simplify(UOp* uop);
UOp* uop_ssimplify(UOp* uop);
int uop_vmin(UOp* uop);
int uop_vmax(UOp* uop);
int uop_sym_infer(UOp* uop);
bool uop_resolve(UOp* uop, bool default_val);

// Additional resolve functions for TDD tests
int uop_resolve_int(UOp* uop);
float uop_resolve_float(UOp* uop);
bool uop_resolve_bool(UOp* uop);

UOp* uop_cache_get(Ops op, DType dtype, UOp** src, size_t src_count, UOpArg* arg, void* tag);
void uop_cache_put(UOp* uop);
UOp** uop_parents(UOp* uop, size_t* count);
UOp** uop_children(UOp* uop, size_t* count);
void uop_meta_set(UOp* uop, const char* key, void* value);
void* uop_meta_get(UOp* uop, const char* key);

// Optional buffer map (parity with Python's buffers WeakKeyDictionary)
void uop_buffer_map_set(UOp* uop, void* buffer_ptr);
void* uop_buffer_map_get(UOp* uop);

// Pattern matching functions
UPat* upat_op(Ops op, UPat** src, size_t src_count);
UPat* upat_var(int id);
UPat* upat_var_named(const char* name, const DType* const* dts, size_t dtype_count, bool vec_any);
UPat* upat_cvar_named(const char* name, const DType* const* dts, size_t dtype_count, bool vec_any);
UPat* upat_const(double val);
UPat* upat_any(void);
UPat* upat_not(UPat* p);
UPat* upat_and(UPat* a, UPat* b);
UPat* upat_or(UPat* a, UPat* b);
bool upat_match(UPat* pattern, UOp* uop);
void upat_free(UPat* pat);

// UPat pattern construction helpers (to mirror Python usage)
void upat_set_name(UPat* pat, const char* name);
void upat_set_required_len(UPat* pat, int required_len, bool strict);
void upat_set_dtype(UPat* pat, const DType* dtype);
void upat_set_op_list(UPat* pat, const Ops* ops, size_t count);
void upat_set_dtype_list(UPat* pat, const DType* const* dtypes, size_t count);
void upat_set_arg_int(UPat* pat, int value);
void upat_set_arg_bind(UPat* pat, const char* bind_name);
void upat_set_src(UPat* pat, UPat** src, size_t count);
void upat_set_repeat(UPat* pat, UPat* repeated);
void upat_set_fork(UPat* pat, UPat*** groups, const int* group_sizes, size_t group_count);

// GroupOp helpers
UPat* upat_group_ops(const bool* mask, UPat** src, size_t src_count);
UPat* upat_group_all_except(Ops exclude, UPat** src, size_t src_count);

// UPat compilation system
PatternMatcher* pattern_matcher_new(PatternMatch* matches, size_t match_count, bool compiled);
void pattern_matcher_free(PatternMatcher* pm);
PatternMatcherResult pattern_matcher_apply(PatternMatcher* pm, UOp* root, void* ctx, void** result);
// Extended apply that also returns named bindings captured in pattern
typedef struct {
  const char** names;
  UOp** values;
  size_t count;
} UPatBindings;
PatternMatcherResult pattern_matcher_apply_bindings(PatternMatcher* pm, UOp* root, void* ctx, void** result, UPatBindings* binds_out);

// UPat compilation functions
UPat* upat_create(void);
void upat_free(UPat* pat);
UOp* upat_get_clause(UPat* self, UOp* base, int depth);
PatternMatcherResult upat_do_process_and(UOp* a, UOp** out_result);
char* upat_final_render(UOp* x, bool has_ctx, int depth);
char* upat_get_code(UPat* self, bool has_ctx);
void* upat_compile(UPat* self, void* fxn);

// Utility functions
UOp** upat_partition(UOp** items, size_t count, bool (*pred)(UOp*), size_t* out_true);
UOp** upat_dedup(UOp** items, size_t count, size_t* out_count);
UOp* upat_graph_rewrite(UOp* root, PatternMatcher* pm, const char* name);
UOp* upat_replace(UOp* uop, Ops new_op, UOp** new_src, size_t new_src_count);
int upat_gep(UOp* uop, int index);

// Cache management
void uop_cache_init(void);
void uop_cache_cleanup(void);

// Module initialization
void uop_init(void);
void uop_ops_init(void);
void uop_ops_cleanup(void);

// Execution
double exec_alu(Ops op, DType dtype, double* args, size_t arg_count);
double identity_element(Ops op, DType* dtype);

// Transcendental operations
typedef enum {
    TRANC_NONE = 0,
    TRANC_SIN,
    TRANC_LOG2,
    TRANC_EXP2,
    TRANC_POW
} TranscendentalOp;

// Pointer to transcendental function
typedef UOp* (*TranscendentalFunc)(UOp* x, UOp* y);

// Transcendental function prototypes
UOp* transcendental_xsin(UOp* x, bool fast, float switch_over);
UOp* transcendental_xexp2(UOp* x);
UOp* transcendental_xlog2(UOp* x);
UOp* transcendental_xpow(UOp* base, UOp* exponent);

// Helper utility functions
int transcendental_mantissa_bits(DType* d);
int transcendental_exponent_bias(DType* d);
int transcendental_exponent_mask(DType* d);

// Integer division helpers
typedef struct {
    int magic;
    int shift;
    bool valid;
} DivisionMagic;

DivisionMagic transcendental_magicgu(int vmax, int d);
UOp* transcendental_fast_idiv(const char* device, UOp* x, int d);

// Symbolic simplification functions from symbolic.c
UOp* symbolic_simplify(UOp* uop);
UOp* symbolic_ssimplify(UOp* uop);

// Variable creation and binding functions
UOp* uop_variable(const char* name, int64_t min_val, int64_t max_val, DType dtype);
UOp* uop_create_variable(int min_val, int max_val);
UOp* uop_bind(UOp* var, UOp* value);

#ifdef __cplusplus
}
#endif

#endif /* TINYGRAD_UOP_H */
