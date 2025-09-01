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
    ARG_INT,
    ARG_REDUCE,
    ARG_SHAPE_TRACKER
} UOpArgType;

typedef struct UOpArg {
    UOpArgType type;
    union {
        struct {
            double const_value;
        } const_data;
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
    const MathTraitOps* math_ops;
    int ref_count;
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
UOp* uop_cast(UOp* a, DType dtype);
UOp* uop_where(UOp* cond, UOp* true_val, UOp* false_val);
UOp* uop_mulacc(UOp* a, UOp* b, UOp* c);
UOp* uop_reduce_axis(UOp* src, Ops reduce_op, int* axes, int axes_count);

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
UOp* uop_view(UOp* buf, struct ShapeTracker* st);
UOp* uop_index(UOp* buf, UOp* idx);
UOp** uop_toposort(UOp* root, size_t* count);
void uop_print(UOp* uop, int depth);
void uop_print_graph(UOp* root);
size_t uop_hash(UOp* uop);
bool uop_equals(UOp* a, UOp* b);
UOp* uop_simplify(UOp* uop);
UOp* uop_ssimplify(UOp* uop);
int uop_vmin(UOp* uop);
int uop_vmax(UOp* uop);
int uop_sym_infer(UOp* uop);
bool uop_resolve(UOp* uop, bool default_val);
UOp* uop_cache_get(Ops op, DType dtype, UOp** src, size_t src_count, UOpArg* arg, void* tag);
void uop_cache_put(UOp* uop);
UOp** uop_parents(UOp* uop, size_t* count);
UOp* uop_replace(UOp* uop, Ops new_op, DType* new_dtype, UOp** new_src, size_t new_src_count, UOpArg* new_arg);

// Pattern matching functions
UPat* upat_op(Ops op, UPat** src, size_t src_count);
UPat* upat_var(int id);
UPat* upat_const(double val);
UPat* upat_any(void);
bool upat_match(UPat* pattern, UOp* uop);
void upat_free(UPat* pat);

// UPat compilation system
PatternMatcher* pattern_matcher_new(PatternMatch* matches, size_t match_count, bool compiled);
void pattern_matcher_free(PatternMatcher* pm);
PatternMatcherResult pattern_matcher_apply(PatternMatcher* pm, UOp* root, void* ctx, void** result);

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
UOp* uop_create_variable(int min_val, int max_val);
UOp* uop_bind(UOp* var, UOp* value);

#ifdef __cplusplus
}
#endif

#endif /* TINYGRAD_UOP_H */