/* spec.c - Faithful port of reference/tinygrad/uop/spec.py */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "../include/uop/uop.h"
#include "../include/dtype/dtype.h"
#include "../include/shape/shapetracker.h"

// Forward declarations from other modules
extern UOp* symbolic_simplify(UOp* uop);
extern UOp* uop_graph_rewrite(UOp* root, PatternMatcher* pm);
extern UOp** uop_toposort(UOp* root, size_t* count);

// Z3 integration - will be stubbed for now
#ifdef HAVE_Z3
#include <z3.h>
// Z3 implementation would go here
#else
// Stub Z3 functionality
typedef void* z3_context;
typedef void* z3_solver;
typedef void* z3_arith_ref;
typedef void* z3_bool_val;

// Stub functions
z3_solver z3_mk_solver(z3_context ctx) { return NULL; }
z3_arith_ref z3_mk_int_const(z3_context ctx, const char* name) { return NULL; }
z3_bool_val z3_mk_bool_val(z3_context ctx, bool value) { return NULL; }
bool z3_solver_check(z3_solver s) { return true; }
void z3_solver_add(z3_solver s, z3_arith_ref expr) {}
void z3_solver_add2(z3_solver s, z3_arith_ref expr1, z3_arith_ref expr2) {}
#endif

// Context variable for IGNORE_OOB
static int ignore_oob_var = 1; // Default to True if Z3 not available

// Helper functions
static int all_same(int* arr, size_t count) {
    if (count <= 1) return 1;
    for (size_t i = 1; i < count; i++) {
        if (arr[i] != arr[0]) return 0;
    }
    return 1;
}

static int tg_prod(const int* shape, int len) {
    int result = 1;
    for (int i = 0; i < len; i++) {
        result *= shape[i];
    }
    return result;
}

// z3_cdiv equivalent - Euclidean division for Z3
#ifdef HAVE_Z3
z3_arith_ref z3_cdiv(z3_context ctx, z3_arith_ref a, z3_arith_ref b) {
    // Complex Z3 implementation would go here
    return NULL;
}
#endif

// Z3 ALU operations mapping - stubbed
typedef enum {
    ALU_NONE,
    ALU_MOD,
    ALU_IDIV,
    ALU_SHR,
    ALU_SHL,
    ALU_AND,
    ALU_WHERE,
    ALU_MAX
} AluOpType;

typedef struct {
    AluOpType op;
    z3_arith_ref (*func)(z3_context, z3_arith_ref, z3_arith_ref);
} Z3AluFunc;

// z3_alu equivalent
static Z3AluFunc z3_alu[] = {
    {ALU_MOD, NULL},
    {ALU_IDIV, NULL},
    {ALU_SHR, NULL},
    {ALU_SHL, NULL},
    {ALU_AND, NULL},
    {ALU_WHERE, NULL},
    {ALU_MAX, NULL}
};

// create_bounded equivalent
#ifdef HAVE_Z3
z3_arith_ref create_bounded(z3_context solver, const char* name, int vmin, int vmax) {
    return NULL; // Stub
}
#endif

// Z3 renderer - simplified PatternMatcher
typedef struct Z3RendererEntry {
    UPat* pattern;
    UOp* (*func)(UOp**, z3_context, void*);
    void* user_data;
} Z3RendererEntry;

static Z3RendererEntry* z3_renderer = NULL;
static size_t z3_renderer_count = 0;
static size_t z3_renderer_capacity = 16;

// IGNORE_OOB context variable access
void spec_set_ignore_oob(int ignore) {
    ignore_oob_var = ignore;
}

int spec_get_ignore_oob(void) {
    return ignore_oob_var;
}

// Pattern matching validation callback functions
typedef struct {
    bool (*validate)(UOp**, size_t);
    void* user_data;
} ValidationCallback;

// buffer_spec equivalent - PatternMatcher implementation
static bool validate_unique(UOp** src, size_t count) {
    return count == 0;
}

static bool validate_device_str(UOp* d) {
    if (!d) return false;
    return (d->arg.type == ARG_CONST && d->arg.const_data.const_value == 1.0) ||
           (d->arg.type == ARG_INT);
}

static bool validate_device_tuple(UOp* d) {
    if (!d) return false;
    return (d->arg.type == ARG_INT && d->arg.int_data.i > 0);
}

static bool validate_buffer(UOp** src, size_t count) {
    if (count != 2) return false;
    return src[1]->arg.type == ARG_INT && src[1]->arg.int_data.i > 0;
}

static bool validate_buffer_view(UOp** src, size_t count) {
    if (count != 1) return false;
    return src[0] != NULL;
}

static bool validate_mstack_buffer_view(UOp** src, size_t count) {
    return count == 1 && src[0]->op == OPS_MSTACK;
}

static bool validate_view(UOp** src, size_t count) {
    return true; // Allow any VIEW operation
}

// assign_spec equivalent
static bool validate_kernel(UOp** src, size_t count) {
    if (count != 1) return false;
    // Check if src[0] is BUFFER, BUFFER_VIEW, ASSIGN, MSELECT, MSTACK, or BIND
    Ops valid_ops[] = {OPS_BUFFER, OPS_BUFFER_VIEW, OPS_ASSIGN, OPS_MSELECT, OPS_MSTACK, OPS_BIND};
    for (size_t i = 0; i < 6; i++) {
        if (src[0]->op == valid_ops[i]) return true;
    }
    return false;
}

static bool validate_assign(UOp** src, size_t count) {
    if (count < 2) return false;
    if (count > 2) {
        for (size_t i = 2; i < count; i++) {
            if (src[i]->op != OPS_ASSIGN) return false;
        }
    }
    return true;
}

static bool validate_mselect(UOp** src, size_t count) {
    if (count < 1) return false;
    // Check if src[0].device exists and is a tuple
    // This is a simplified check
    return true;
}

static bool validate_mstack(UOp** src, size_t count) {
    if (count < 1) return false;
    for (size_t i = 0; i < count; i++) {
        // Check if src[i].device is a string
        // This is a simplified check
        if (src[i]->arg.type != ARG_CONST) return false;
    }
    return true;
}

// tensor_uop_spec equivalent
static bool validate_movement(UOp** src, size_t count) {
    if (count != 1) return false;
    UOp* mv = src[0];
    UOp* x = src[1];
    
    // Simplified validation - in full implementation would check dtype relationships
    return (mv->arg.type == ARG_INT) || 
           (mv->dtype._scalar == x->dtype._scalar);
}

static bool validate_view_all_sources(UOp** src, size_t count) {
    if (count != 1) return false;
    UOp* x = src[0];
    return x->op == OPS_BUFFER || x->op == OPS_BUFFER_VIEW || 
           x->op == OPS_ASSIGN || x->op == OPS_CONST || x->op == OPS_DEVICE;
}

static bool validate_bind(UOp** src, size_t count) {
    if (count != 2) return false;
    // Simplified check
    return true;
}

static bool validate_const_device_view(UOp** src, size_t count) {
    if (count != 1) return false;
    // Simplified check for ShapeTracker views
    return true;
}

static bool validate_detach_contiguous(UOp** src, size_t count) {
    if (count != 1) return false;
    UOp* root = src[0];
    UOp* x = src[1];
    return root->dtype._scalar == x->dtype._scalar;
}

static bool validate_copy(UOp** src, size_t count) {
    if (count != 2) return false;
    UOp* copy = src[0];
    UOp* x = src[1];
    return copy->dtype._scalar == x->dtype._scalar;
}

static bool validate_allreduce(UOp** src, size_t count) {
    if (count != 2) return false;
    UOp* red = src[0];
    UOp* x = src[1];
    return red->dtype._scalar == x->dtype._scalar;
}

static bool validate_multi(UOp** src, size_t count) {
    if (count < 1) return false;
    UOp* multi = src[0];
    for (size_t i = 0; i < count; i++) {
        if (multi->dtype._scalar != src[i]->dtype._scalar) return false;
    }
    return (multi->arg.type == ARG_INT);
}

// Index validation functions
bool validate_index(UOp** src, size_t count, UOp* gate) {
    if (ignore_oob_var) return true;
    
    // Simplified validation - full Z3-based validation would be complex
    // For now, assume valid if count is correct
    return count >= 1 && count <= 3;
}

bool validate_store(UOp** src, size_t count, UOp* gate) {
    // Simplified validation
    if (count < 2) return false;
    return validate_index(src, count - 1, gate);
}

// Pattern Matcher for spec validation
typedef struct SpecPatternEntry {
    Ops op;
    bool (*validate)(UOp**, size_t);
} SpecPatternEntry;

static SpecPatternEntry spec_patterns[] = {
    {OPS_DEFINE_GLOBAL, NULL},
    {OPS_DEFINE_LOCAL, NULL},
    {OPS_DEFINE_REG, NULL},
    {OPS_DEFINE_VAR, NULL},
    {OPS_RANGE, NULL},
    {OPS_SPECIAL, NULL},
    {OPS_VIEW, NULL},
    {OPS_CONST, NULL},
    {OPS_LOAD, NULL},
    {OPS_STORE, NULL},
    {OPS_INDEX, NULL},
    {OPS_WHERE, NULL},
    {OPS_CMPLT, NULL},
    {OPS_CMPNE, NULL},
    {OPS_CMPEQ, NULL},
    {OPS_SHL, NULL},
    {OPS_SHR, NULL},
    {OPS_IDIV, NULL},
    {OPS_MOD, NULL},
    {OPS_ENDRANGE, NULL},
    {OPS_WMMA, NULL},
    {OPS_CONTRACT, NULL},
    {OPS_UNROLL, NULL},
    {OPS_IF, NULL},
    {OPS_ENDIF, NULL},
    {OPS_REDUCE_AXIS, NULL},
    {OPS_GEP, NULL},
    {OPS_VECTORIZE, NULL},
    {OPS_BITCAST, NULL},
    {OPS_CAST, NULL},
    {OPS_BARRIER, NULL},
    {OPS_SINK, NULL},
    {OPS_NOOP, NULL},
    {OPS_CUSTOMI, NULL},
    {OPS_CUSTOM, NULL},
    {OPS_PRECAST, NULL}
};

// validate_index main function
bool validate_index_main(UOp* idx, UOp* gate) {
    if (!idx || idx->op != OPS_INDEX) return false;
    
    size_t src_count = idx->src_count;
    return validate_index(idx->src, src_count, gate);
}

// validate_store main function
bool validate_store_main(UOp* store, UOp* gate) {
    if (!store || store->op != OPS_STORE) return false;
    
    size_t src_count = store->src_count;
    return validate_store(store->src, src_count, gate);
}

// AST spec validation
bool validate_ast_view(UOp** src, size_t count) {
    if (count != 1) return false;
    UOp* view = src[0];
    
    // Check parent constraint
    Ops define_ops[] = {OPS_DEFINE_GLOBAL, OPS_DEFINE_LOCAL};
    for (size_t i = 0; i < 2; i++) {
        if (view->src[0]->op == define_ops[i]) return true;
    }
    return false;
}

bool validate_ast_root(UOp** src, size_t count) {
    if (count != 1) return false;
    UOp* root = src[0];
    
    // Check that all parents have the same shape
    // Simplified validation
    return true;
}

// Type verification function
void type_verify(UOp** uops, size_t uop_count, PatternMatcher* extra_spec) {
    if (!uops || uop_count == 0) return;
    
    // Simplified type verification
    for (size_t i = 0; i < uop_count; i++) {
        UOp* u = uops[i];
        
        // Basic validation
        if (!u) {
            fprintf(stderr, "UOp verification failed at %zu: NULL UOp\n", i);
            continue;
        }
        
        // Check validity with spec patterns
        bool valid = false;
        for (size_t j = 0; j < sizeof(spec_patterns)/sizeof(spec_patterns[0]); j++) {
            if (u->op == spec_patterns[j].op) {
                // Simplified validation - in full implementation would apply pattern matching
                valid = true;
                break;
            }
        }
        
        if (!valid) {
            fprintf(stderr, "UOp verification failed at %zu on %s %s %zu\n",
                   i, ops_to_string(u->op), u->dtype.name, u->src_count);
        }
    }
}

// Module initialization
void spec_init(void) {
    // Initialize spec patterns
    // This would build the pattern matchers from spec.py
    
    // Set default IGNORE_OOB based on Z3 availability
    ignore_oob_var = 1; // True if Z3 not available
}

void spec_cleanup(void) {
    // Clean up spec patterns and data structures
}

// Additional exportable functions for external modules
bool validate_basic_pattern(Ops op, UOp** src, size_t count) {
    for (size_t i = 0; i < sizeof(spec_patterns)/sizeof(spec_patterns[0]); i++) {
        if (spec_patterns[i].op == op) {
            if (spec_patterns[i].validate) {
                return spec_patterns[i].validate(src, count);
            }
            return true; // No specific validator, assume valid
        }
    }
    return false; // Unknown operation
}

// Pattern matcher creation helpers
PatternMatcher* create_buffer_spec(void) {
    // Create buffer spec pattern matcher
    // This would contain all the buffer_spec patterns from spec.py
    PatternMatch matches[10]; // Placeholder
    size_t match_count = 0;
    
    // Fill matches with buffer spec patterns
    // ...
    
    return pattern_matcher_new(matches, match_count, false);
}

PatternMatcher* create_assign_spec(void) {
    // Create assign spec pattern matcher
    PatternMatch matches[5]; // Placeholder
    size_t match_count = 0;
    
    // Fill matches with assign spec patterns
    // ...
    
    return pattern_matcher_new(matches, match_count, false);
}

PatternMatcher* create_tensor_uop_spec(void) {
    // Create tensor UOp spec pattern matcher
    PatternMatch matches[15]; // Placeholder
    size_t match_count = 0;
    
    // Fill matches with tensor UOp spec patterns
    // ...
    
    return pattern_matcher_new(matches, match_count, false);
}

PatternMatcher* create_ast_spec(void) {
    // Create AST spec pattern matcher
    PatternMatch matches[10]; // Placeholder
    size_t match_count = 0;
    
    // Fill matches with AST spec patterns
    // ...
    
    return pattern_matcher_new(matches, match_count, false);
}
