/* mathtraits.h - Faithful port of tinygrad/uop/mathtraits.py */

#ifndef TINYGRAD_MATHTRAITS_H
#define TINYGRAD_MATHTRAITS_H

#include <stdbool.h>
#include "uop/uop.h"
#include "dtype/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct UOp UOp;

// MathTrait interface - C implementation of Python MathTrait class
// This is implemented as a struct of function pointers to mimic Python's class methods
typedef struct MathTraitOps {
    // Core methods that must be implemented (like Python's abstract methods)
    UOp* (*alu)(UOp* self, Ops op, UOp** src, size_t src_count);
    UOp* (*const_like)(UOp* self, double value);
    
    // Helper function - equivalent to Python's ufix
    UOp* (*ufix)(UOp* self, void* x);
    
    // Binary operation helper - equivalent to Python's _binop
    UOp* (*binop)(UOp* self, Ops op, UOp* x, bool reverse);
    
    // Logical operations
    UOp* (*logical_not)(UOp* self);
    UOp* (*neg)(UOp* self);
    
    // Type checking
    void (*check_dtype)(UOp* self);
    
    // Mathematical operations - direct ports of Python methods
    UOp* (*add)(UOp* self, UOp* x, bool reverse);
    UOp* (*mul)(UOp* self, UOp* x, bool reverse);
    UOp* (*sub)(UOp* self, UOp* x, bool reverse);
    UOp* (*div)(UOp* self, UOp* x, bool reverse);
    UOp* (*idiv)(UOp* self, UOp* x, bool reverse);
    UOp* (*mod)(UOp* self, UOp* x, bool reverse);
    
    // Bitwise operations
    UOp* (*bitwise_and)(UOp* self, UOp* x, bool reverse);
    UOp* (*bitwise_or)(UOp* self, UOp* x, bool reverse);
    UOp* (*bitwise_xor)(UOp* self, UOp* x, bool reverse);
    
    // Shift operations
    UOp* (*lshift)(UOp* self, UOp* x, bool reverse);
    UOp* (*rshift)(UOp* self, UOp* x, bool reverse);
    
    // Comparison operations
    UOp* (*lt)(UOp* self, UOp* x);
    UOp* (*gt)(UOp* self, UOp* x);
    UOp* (*ge)(UOp* self, UOp* x);
    UOp* (*le)(UOp* self, UOp* x);
    UOp* (*ne)(UOp* self, UOp* x);
    UOp* (*eq)(UOp* self, UOp* x);
    
    // Min/max operations
    UOp* (*maximum)(UOp* self, UOp* x);
    UOp* (*minimum)(UOp* self, UOp* x);
    
    // Special operations
    UOp* (*where)(UOp* self, UOp* x, UOp* y);
    UOp* (*threefry)(UOp* self, UOp* seed);
    
    // Unary math operations
    UOp* (*reciprocal)(UOp* self);
    UOp* (*sqrt)(UOp* self);
    UOp* (*sin)(UOp* self);
    UOp* (*log2)(UOp* self);
    UOp* (*exp2)(UOp* self);
    UOp* (*pow)(UOp* self, UOp* x);
} MathTraitOps;

// Global MathTrait operations instance
extern const MathTraitOps math_ops;

// Initialize the MathTrait module
void mathtraits_init(void);
void mathtraits_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* TINYGRAD_MATHTRAITS_H */