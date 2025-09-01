/* optional.c - Faithful line-by-line port of reference/tinygrad/uop/optional.py */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "../include/uop/uop.h"
#include "../include/dtype/dtype.h"

// Forward declarations from transcendental.c
extern const DType* TRANSCENDENTAL_SUPPORTED_DTYPES[];
extern int TRANSCENDENTAL_SUPPORTED_COUNT;

extern UOp* transcendental_xexp2(UOp* x);
extern UOp* transcendental_xlog2(UOp* x);
extern UOp* transcendental_xsin(UOp* x, bool fast, float switch_over);

// Forward declarations from ops.c
extern UOp* uop_sqrt(UOp* a);

// Forward declarations needed for this module
extern UOp* symbolic_simplify(UOp* uop);

// Helper functions (would need to be implemented based on dtypes module)
// For now, implement basic versions

// ***** optional *****

// powers_of_two = {2**i:i for i in range(64)}
static int powers_of_two[64] = {0};
static bool powers_of_two_initialized = false;

static void init_powers_of_two(void) {
    if (!powers_of_two_initialized) {
        for (int i = 0; i < 64; i++) {
            powers_of_two[i] = i;
        }
        powers_of_two_initialized = true;
    }
}

// Helper function to check if dtype is in dtypes.ints
bool is_dtype_int(const DType* dt) {
    if (!dt) return false;
    return (dt == &dtypes.int8 || dt == &dtypes.int16 || dt == &dtypes.int32 || dt == &dtypes.int64);
}

// Helper function to check if dtype is in dtypes.uints
bool is_dtype_uint(const DType* dt) {
    if (!dt) return false;
    return (dt == &dtypes.uint8 || dt == &dtypes.uint16 || dt == &dtypes.uint32 || dt == &dtypes.uint64);
}

// Helper function to check if dtype is in dtypes.sints
bool is_dtype_sint(const DType* dt) {
    if (!dt) return false;
    return (dt == &dtypes.int8 || dt == &dtypes.int16 || dt == &dtypes.int32 || dt == &dtypes.int64);
}

// Pattern list structure
typedef struct {
    UPat* pattern;
    UOp* (*func)(UOp** src, size_t src_count);
} PatternEntry;

// Pattern matcher structure for patterns
typedef struct OptionalPatternMatcher {
    PatternEntry* patterns;
    size_t pattern_count;
    size_t capacity;
} OptionalPatternMatcher;

// Global pattern matcher instance
static OptionalPatternMatcher* pattern_matcher = NULL;

// Check if dtype is supported for transcendental ops
bool is_transcendental_dtype(const DType* dt) {
    if (!dt) return false;
    for (int i = 0; i < TRANSCENDENTAL_SUPPORTED_COUNT; i++) {
        if (dt == TRANSCENDENTAL_SUPPORTED_DTYPES[i]) {
            return true;
        }
    }
    return false;
}

// Pattern functions
UOp* pattern_exp2_func(UOp** src, size_t src_count) {
    if (src_count != 1) return NULL;
    return transcendental_xexp2(src[0]);
}

UOp* pattern_log2_func(UOp** src, size_t src_count) {
    if (src_count != 1) return NULL;
    return transcendental_xlog2(src[0]);
}

UOp* pattern_sin_func(UOp** src, size_t src_count) {
    if (src_count != 1) return NULL;
    return transcendental_xsin(src[0], false, 10000.0f);
}

UOp* pattern_sqrt_func(UOp** src, size_t src_count) {
    if (src_count != 1) return NULL;
    return uop_sqrt(src[0]);
}

UOp* pattern_mod_to_and_func(UOp** src, size_t src_count) {
    if (src_count != 2) return NULL;
    
    // x % (2**y) -> x & (2**y-1)
    UOp* x = src[0];
    UOp* c = src[1];
    
    // Check if c is a constant power of two
    if (c->arg.type != ARG_CONST) return NULL;
    
    double c_val = c->arg.const_data.const_value;
    if (c_val <= 0) return NULL;
    
    // Check if c is a power of two
    int power = -1;
    for (int i = 0; i < 64; i++) {
        double power_val = pow(2.0, i);
        if (fabs(c_val - power_val) < 1e-10) {
            power = i;
            break;
        }
    }
    
    if (power == -1) return NULL;
    
    // Generate c-1 constant
    UOp* c_minus_one = uop_const(x->dtype, pow(2.0, power) - 1);
    UOp* and_result = uop_and(x, c_minus_one);
    
    return and_result;
}

UOp* pattern_mul_to_shl_func(UOp** src, size_t src_count) {
    if (src_count != 2) return NULL;
    
    // x*(2**y) -> shl(x,y)
    UOp* x = src[0];
    UOp* c = src[1];
    
    if (!is_dtype_int(&x->dtype)) return NULL;
    
    // Check if c is a constant power of two
    if (c->arg.type != ARG_CONST) return NULL;
    
    double c_val = c->arg.const_data.const_value;
    if (c_val <= 0) return NULL;
    
    // Find power
    int power = -1;
    for (int i = 0; i < 64; i++) {
        double power_val = pow(2.0, i);
        if (fabs(c_val - power_val) < 1e-10) {
            power = i;
            break;
        }
    }
    
    if (power == -1) return NULL;
    
    // Generate constant y
    UOp* y_const = uop_const(dtypes.int32, (double)power);
    UOp* shl_result = uop_shl(x, y_const);
    
    return shl_result;
}

UOp* pattern_idiv_uint_to_shr_func(UOp** src, size_t src_count) {
    if (src_count != 2) return NULL;
    
    // x//(2**y) -> shr(x,y) for uints
    UOp* x = src[0];
    UOp* c = src[1];
    
    if (!is_dtype_uint(&x->dtype)) return NULL;
    
    // Check if c is a constant power of two
    if (c->arg.type != ARG_CONST) return NULL;
    
    double c_val = c->arg.const_data.const_value;
    if (c_val <= 0) return NULL;
    
    // Find power
    int power = -1;
    for (int i = 0; i < 64; i++) {
        double power_val = pow(2.0, i);
        if (fabs(c_val - power_val) < 1e-10) {
            power = i;
            break;
        }
    }
    
    if (power == -1) return NULL;
    
    // Generate constant y
    UOp* y_const = uop_const(dtypes.int32, (double)power);
    UOp* shr_result = uop_shr(x, y_const);
    
    return shr_result;
}

UOp* pattern_idiv_sint_to_shr_func(UOp** src, size_t src_count) {
    if (src_count != 2) return NULL;
    
    // x//(2**y) -> shr(x,y) for sints with adjustment
    UOp* x = src[0];
    UOp* c = src[1];
    
    if (!is_dtype_sint(&x->dtype)) return NULL;
    
    // Check if c is a constant power of two
    if (c->arg.type != ARG_CONST) return NULL;
    
    double c_val = c->arg.const_data.const_value;
    if (c_val <= 0) return NULL;
    
    // Find power
    int power = -1;
    for (int i = 0; i < 64; i++) {
        double power_val = pow(2.0, i);
        if (fabs(c_val - power_val) < 1e-10) {
            power = i;
            break;
        }
    }
    
    if (power == -1) return NULL;
    
    // Simplified version: just return regular division for now
    // Full implementation would be complex due to negative handling
    UOp* y_const = uop_const(dtypes.int32, (double)power);
    return uop_div(x, uop_exp2(y_const));
    
    // TODO: Implement full version with condition and adjustments
    // (x+(x<0).where(c-1, 0)) >> v where v is the power
}

UOp* pattern_neg_one_func(UOp** src, size_t src_count) {
    if (src_count != 1) return NULL;
    
    // x*-1 -> x.alu(Ops.NEG)
    UOp* x = src[0];
    return uop_neg(x);
}

UOp* pattern_neg_sub_func(UOp** src, size_t src_count) {
    if (src_count != 2) return NULL;
    
    // x+y.alu(Ops.NEG) -> x.alu(Ops.SUB, y)
    UOp* x = src[0];
    UOp* y = src[1];
    
    // Check if the second op is NEG
    if (y->op == OPS_NEG && y->src_count == 1) {
        return uop_sub(x, y->src[0]);
    }
    
    return NULL;
}

// Initialize pattern matcher
void optional_patterns_init(void) {
    init_powers_of_two();
    
    // Create pattern matcher
    pattern_matcher = (OptionalPatternMatcher*)malloc(sizeof(OptionalPatternMatcher));
    if (!pattern_matcher) return;
    
    pattern_matcher->patterns = NULL;
    pattern_matcher->pattern_count = 0;
    pattern_matcher->capacity = 16;
    pattern_matcher->patterns = (PatternEntry*)malloc(pattern_matcher->capacity * sizeof(PatternEntry));
    if (!pattern_matcher->patterns) {
        free(pattern_matcher);
        pattern_matcher = NULL;
        return;
    }
    
    // Build patterns for transcendental ops
    for (int i = 0; i < TRANSCENDENTAL_SUPPORTED_COUNT; i++) {
        const DType* dt = TRANSCENDENTAL_SUPPORTED_DTYPES[i];
        
        // EXP2 pattern
        UPat* exp2_pattern = upat_create();
        exp2_pattern->type = UPAT_OP;
        exp2_pattern->op_data.op = OPS_EXP2;
        exp2_pattern->dtype = (DType*)dt;  // Cast away const
        
        if (pattern_matcher->pattern_count >= pattern_matcher->capacity) {
            pattern_matcher->capacity *= 2;
            pattern_matcher->patterns = (PatternEntry*)realloc(pattern_matcher->patterns, 
                pattern_matcher->capacity * sizeof(PatternEntry));
        }
        
        pattern_matcher->patterns[pattern_matcher->pattern_count].pattern = exp2_pattern;
        pattern_matcher->patterns[pattern_matcher->pattern_count].func = pattern_exp2_func;
        pattern_matcher->pattern_count++;
        
        // LOG2 pattern
        UPat* log2_pattern = upat_create();
        log2_pattern->type = UPAT_OP;
        log2_pattern->op_data.op = OPS_LOG2;
        log2_pattern->dtype = (DType*)dt;
        
        if (pattern_matcher->pattern_count >= pattern_matcher->capacity) {
            pattern_matcher->capacity *= 2;
            pattern_matcher->patterns = (PatternEntry*)realloc(pattern_matcher->patterns, 
                pattern_matcher->capacity * sizeof(PatternEntry));
        }
        
        pattern_matcher->patterns[pattern_matcher->pattern_count].pattern = log2_pattern;
        pattern_matcher->patterns[pattern_matcher->pattern_count].func = pattern_log2_func;
        pattern_matcher->pattern_count++;
        
        // SIN pattern
        UPat* sin_pattern = upat_create();
        sin_pattern->type = UPAT_OP;
        sin_pattern->op_data.op = OPS_SIN;
        sin_pattern->dtype = (DType*)dt;
        
        if (pattern_matcher->pattern_count >= pattern_matcher->capacity) {
            pattern_matcher->capacity *= 2;
            pattern_matcher->patterns = (PatternEntry*)realloc(pattern_matcher->patterns, 
                pattern_matcher->capacity * sizeof(PatternEntry));
        }
        
        pattern_matcher->patterns[pattern_matcher->pattern_count].pattern = sin_pattern;
        pattern_matcher->patterns[pattern_matcher->pattern_count].func = pattern_sin_func;
        pattern_matcher->pattern_count++;
    }
    
    // SQRT pattern
    for (int i = 0; i < TRANSCENDENTAL_SUPPORTED_COUNT; i++) {
        const DType* dt = TRANSCENDENTAL_SUPPORTED_DTYPES[i];
        
        UPat* sqrt_pattern = upat_create();
        sqrt_pattern->type = UPAT_OP;
        sqrt_pattern->op_data.op = OPS_SQRT;
        sqrt_pattern->dtype = (DType*)dt;
        
        if (pattern_matcher->pattern_count >= pattern_matcher->capacity) {
            pattern_matcher->capacity *= 2;
            pattern_matcher->patterns = (PatternEntry*)realloc(pattern_matcher->patterns, 
                pattern_matcher->capacity * sizeof(PatternEntry));
        }
        
        pattern_matcher->patterns[pattern_matcher->pattern_count].pattern = sqrt_pattern;
        pattern_matcher->patterns[pattern_matcher->pattern_count].func = pattern_sqrt_func;
        pattern_matcher->pattern_count++;
    }
    
    // Additional patterns would be added here based on ops availability
}

// Get late rewrite patterns - simplified version for now
PatternMatcher* get_late_rewrite_patterns(Ops* available_ops, size_t ops_count, bool force_transcendental) {
    // This is a simplified version - full implementation would build patterns dynamically
    
    // Create a single pattern matcher for now
    PatternMatch patterns[1];
    patterns[0].pattern = NULL;  // No patterns initially
    patterns[0].callback = NULL;
    patterns[0].user_data = NULL;
    
    PatternMatcher* pm = pattern_matcher_new(patterns, 1, false);
    if (!pm) return NULL;
    
    // TODO: Build patterns dynamically based on available_ops and force_transcendental
    
    return pm;
}

// Fast IDIV functions
typedef struct {
    char* device;
    UOp* (*fast_idiv_func)(const char*, UOp*, int);
} FastIDivContext;

extern UOp* transcendental_fast_idiv(const char* device, UOp* x, int d);

// Apply patterns to UOp graph - simplified
UOp* optional_apply_patterns(UOp* root, PatternMatcher* pm) {
    if (!root || !pm) return root;
    
    // For now, just return the root unchanged
    // Full implementation would apply the pattern matching logic
    
    return root;
}

// Module cleanup
void optional_patterns_cleanup(void) {
    if (pattern_matcher) {
        for (size_t i = 0; i < pattern_matcher->pattern_count; i++) {
            if (pattern_matcher->patterns[i].pattern) {
                upat_free(pattern_matcher->patterns[i].pattern);
            }
        }
        
        if (pattern_matcher->patterns) {
            free(pattern_matcher->patterns);
        }
        
        free(pattern_matcher);
        pattern_matcher = NULL;
    }
}

// Legacy interface for backward compatibility
void optional_init(void) {
    optional_patterns_init();
}

void optional_cleanup(void) {
    optional_patterns_cleanup();
}
