/* mathtraits.c - Faithful port of tinygrad/uop/mathtraits.py */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "uop/mathtraits.h"
#include "uop/ops.h"
#include "dtype/dtype.h"

// Forward declarations
static UOp* alu_impl(UOp* self, Ops op, UOp** src, size_t src_count);
static UOp* const_like_impl(UOp* self, double value);

// Line 11: def ufix(self, x): return self.const_like(x) if not isinstance(x, MathTrait) else x
static UOp* ufix_impl(UOp* self, void* x) {
    // In C, we check if x is a UOp pointer (has MathTrait)
    // If x is a number (passed as void*), convert to const_like
    // Otherwise return x as UOp*
    UOp* x_uop = (UOp*)x;
    if (x_uop && x_uop->math_ops) {
        return x_uop;  // x is already a UOp with MathTrait
    } else {
        // x is a constant value, create const_like
        double val = x ? *(double*)x : 0.0;
        return const_like_impl(self, val);
    }
}

// Line 12: def _binop(self, op, x, reverse): return self.ufix(x).alu(op, self) if reverse else self.alu(op, self.ufix(x))
static UOp* binop_impl(UOp* self, Ops op, void* x, bool reverse) {
    UOp* x_fixed = ufix_impl(self, x);
    if (reverse) {
        // reverse: x.alu(op, self)
        UOp* src[] = {self};
        return alu_impl(x_fixed, op, src, 1);
    } else {
        // normal: self.alu(op, x)
        UOp* src[] = {x_fixed};
        return alu_impl(self, op, src, 1);
    }
}

// Line 13: def logical_not(self): return self.ne(True)
static UOp* logical_not_impl(UOp* self) {
    // self.ne(True) - not equal to true
    double true_val = 1.0;
    return binop_impl(self, OPS_CMPNE, &true_val, false);
}

// Line 14-16: def neg(self):
static UOp* neg_impl(UOp* self) {
    // if (dtype:=getattr(self, 'dtype')) is None: raise TypeError(f"MathTraits __neg__ requires a dtype, {self=}")
    if (!self->dtype.count) {
        fprintf(stderr, "MathTraits __neg__ requires a dtype\n");
        return NULL;
    }
    
    // return self.logical_not() if dtype.scalar() == dtypes.bool else self*(-1)
    if (self->dtype._scalar == dtypes.bool_._scalar) {
        return logical_not_impl(self);
    } else {
        double neg_one = -1.0;
        return binop_impl(self, OPS_MUL, &neg_one, false);
    }
}

// Line 17-20: def _check_dtype(self):
static void check_dtype_impl(UOp* self) {
    if (self->dtype.count) {
        DType dtype = self->dtype;
        // if isinstance(dtype, tuple): dtype = dtype[0] - in C, we just use the dtype directly
        // if not (dtypes.is_bool(dtype) or dtypes.is_int(dtype)): raise RuntimeError
        if (!dtypes_is_bool(&dtype) && !dtypes_is_int(&dtype)) {
            fprintf(stderr, "%s is not supported\n", dtype.name);
        }
    }
}

// Line 21-38: def add(self, x, reverse=False):
static UOp* add_impl(UOp* self, void* x, bool reverse) {
    return binop_impl(self, OPS_ADD, x, reverse);
}

// Line 39-57: def mul(self, x, reverse=False):
static UOp* mul_impl(UOp* self, void* x, bool reverse) {
    return binop_impl(self, OPS_MUL, x, reverse);
}

// Line 58-71: def bitwise_and(self, x, reverse=False):
static UOp* bitwise_and_impl(UOp* self, void* x, bool reverse) {
    check_dtype_impl(self);
    return binop_impl(self, OPS_AND, x, reverse);
}

// Line 72-85: def bitwise_or(self, x, reverse=False):
static UOp* bitwise_or_impl(UOp* self, void* x, bool reverse) {
    check_dtype_impl(self);
    return binop_impl(self, OPS_OR, x, reverse);
}

// Line 86-100: def bitwise_xor(self, x, reverse=False):
static UOp* bitwise_xor_impl(UOp* self, void* x, bool reverse) {
    check_dtype_impl(self);
    return binop_impl(self, OPS_XOR, x, reverse);
}

// Line 101-112: def idiv(self, x, reverse=False):
static UOp* idiv_impl(UOp* self, void* x, bool reverse) {
    return binop_impl(self, OPS_IDIV, x, reverse);
}

// Line 113: def mod(self, x, reverse=False): return self._binop(Ops.MOD, x, reverse)
static UOp* mod_impl(UOp* self, void* x, bool reverse) {
    return binop_impl(self, OPS_MOD, x, reverse);
}

// Line 114: def sub(self, x, reverse=False): return self.ufix(x).alu(Ops.ADD, -self) if reverse else self.alu(Ops.ADD, self.ufix(-x))
static UOp* sub_impl(UOp* self, void* x, bool reverse) {
    if (reverse) {
        // x.alu(Ops.ADD, -self)
        UOp* neg_self = neg_impl(self);
        UOp* x_fixed = ufix_impl(self, x);
        UOp* src[] = {neg_self};
        return alu_impl(x_fixed, OPS_ADD, src, 1);
    } else {
        // self.alu(Ops.ADD, -x)
        UOp* x_fixed = ufix_impl(self, x);
        UOp* neg_x = neg_impl(x_fixed);
        UOp* src[] = {neg_x};
        return alu_impl(self, OPS_ADD, src, 1);
    }
}

// Line 115: def div(self, x, reverse=False): return (self.ufix(x)*self.alu(Ops.RECIP)) if reverse else (self*self.ufix(x).alu(Ops.RECIP))
static UOp* div_impl(UOp* self, void* x, bool reverse) {
    if (reverse) {
        // x * self.reciprocal()
        UOp* x_fixed = ufix_impl(self, x);
        UOp* recip_self = alu_impl(self, OPS_RECIP, NULL, 0);
        UOp* src[] = {recip_self};
        return alu_impl(x_fixed, OPS_MUL, src, 1);
    } else {
        // self * x.reciprocal()
        UOp* x_fixed = ufix_impl(self, x);
        UOp* recip_x = alu_impl(x_fixed, OPS_RECIP, NULL, 0);
        UOp* src[] = {recip_x};
        return alu_impl(self, OPS_MUL, src, 1);
    }
}

// Line 139-142: Comparison operations
// def __lt__(self, x): return self.alu(Ops.CMPLT, self.ufix(x))
static UOp* lt_impl(UOp* self, void* x) {
    UOp* x_fixed = ufix_impl(self, x);
    UOp* src[] = {x_fixed};
    return alu_impl(self, OPS_CMPLT, src, 1);
}

// def __gt__(self, x): return self.ufix(x).alu(Ops.CMPLT, self)
static UOp* gt_impl(UOp* self, void* x) {
    UOp* x_fixed = ufix_impl(self, x);
    UOp* src[] = {self};
    return alu_impl(x_fixed, OPS_CMPLT, src, 1);
}

// def __ge__(self, x): return (self < x).logical_not()
static UOp* ge_impl(UOp* self, void* x) {
    UOp* lt_result = lt_impl(self, x);
    return logical_not_impl(lt_result);
}

// def __le__(self, x): return (self > x).logical_not()
static UOp* le_impl(UOp* self, void* x) {
    UOp* gt_result = gt_impl(self, x);
    return logical_not_impl(gt_result);
}

// Line 144-145: def ne(self, x): return self.alu(Ops.CMPNE, self.ufix(x))
static UOp* ne_impl(UOp* self, void* x) {
    UOp* x_fixed = ufix_impl(self, x);
    UOp* src[] = {x_fixed};
    return alu_impl(self, OPS_CMPNE, src, 1);
}

// def eq(self, x): return self.ne(x).logical_not()
static UOp* eq_impl(UOp* self, void* x) {
    UOp* ne_result = ne_impl(self, x);
    return logical_not_impl(ne_result);
}

// Line 149-150: Shift operations
static UOp* lshift_impl(UOp* self, void* x, bool reverse) {
    return binop_impl(self, OPS_SHL, x, reverse);
}

static UOp* rshift_impl(UOp* self, void* x, bool reverse) {
    return binop_impl(self, OPS_SHR, x, reverse);
}

// Line 156-157: Min/max operations
// def maximum(self, x): return self.alu(Ops.MAX, self.ufix(x))
static UOp* maximum_impl(UOp* self, void* x) {
    UOp* x_fixed = ufix_impl(self, x);
    UOp* src[] = {x_fixed};
    return alu_impl(self, OPS_MAX, src, 1);
}

// def minimum(self, x): return -(-self).maximum(-x)
static UOp* minimum_impl(UOp* self, void* x) {
    UOp* neg_self = neg_impl(self);
    UOp* x_fixed = ufix_impl(self, x);
    UOp* neg_x = neg_impl(x_fixed);
    UOp* max_result = maximum_impl(neg_self, neg_x);
    return neg_impl(max_result);
}

// Line 158-161: def where(self, x, y):
static UOp* where_impl(UOp* self, void* x, void* y) {
    // if type(self) is type(x): return self.alu(Ops.WHERE, x, x.ufix(y))
    // if type(self) is type(y): return self.alu(Ops.WHERE, y.ufix(x), y)
    // In C, we check if they're both UOps
    UOp* x_uop = (UOp*)x;
    UOp* y_uop = (UOp*)y;
    
    if (x_uop && x_uop->math_ops) {
        // self and x are both UOps
        UOp* y_fixed = ufix_impl(x_uop, y);
        UOp* src[] = {x_uop, y_fixed};
        return alu_impl(self, OPS_WHERE, src, 2);
    } else if (y_uop && y_uop->math_ops) {
        // self and y are both UOps
        UOp* x_fixed = ufix_impl(y_uop, x);
        UOp* src[] = {x_fixed, y_uop};
        return alu_impl(self, OPS_WHERE, src, 2);
    } else {
        fprintf(stderr, "where needs at least one UOp arg\n");
        return NULL;
    }
}

// Line 162: def threefry(self, seed): return self.alu(Ops.THREEFRY, seed)
static UOp* threefry_impl(UOp* self, UOp* seed) {
    UOp* src[] = {seed};
    return alu_impl(self, OPS_THREEFRY, src, 1);
}

// Line 163-168: Unary math operations
static UOp* reciprocal_impl(UOp* self) {
    return alu_impl(self, OPS_RECIP, NULL, 0);
}

static UOp* sqrt_impl(UOp* self) {
    return alu_impl(self, OPS_SQRT, NULL, 0);
}

static UOp* sin_impl(UOp* self) {
    return alu_impl(self, OPS_SIN, NULL, 0);
}

static UOp* log2_impl(UOp* self) {
    return alu_impl(self, OPS_LOG2, NULL, 0);
}

static UOp* exp2_impl(UOp* self) {
    return alu_impl(self, OPS_EXP2, NULL, 0);
}

// Line 168: def pow(self, x): return self.alu(Ops.POW, self.ufix(x))
static UOp* pow_impl(UOp* self, void* x) {
    UOp* x_fixed = ufix_impl(self, x);
    UOp* src[] = {x_fixed};
    return alu_impl(self, OPS_POW, src, 1);
}

// Core required methods - these would be implemented by the actual UOp struct
// Line 7: def alu(self:T, op:Ops, *src) -> T: raise NotImplementedError
static UOp* alu_impl(UOp* self, Ops op, UOp** src, size_t src_count) {
    // This will be implemented by the actual UOp implementation
    // For now, return a stub
    (void)self; (void)op; (void)src; (void)src_count;
    static UOp stub = {.op = OPS_NOOP};
    stub.math_ops = &math_ops;
    return &stub;
}

// Line 8: def const_like(self:T, b) -> T: raise NotImplementedError
static UOp* const_like_impl(UOp* self, double value) {
    // This will be implemented by the actual UOp implementation
    // For now, return a stub
    (void)self; (void)value;
    static UOp stub = {.op = OPS_CONST};
    stub.math_ops = &math_ops;
    return &stub;
}

// Global MathTraitOps instance - faithful port of Python MathTrait class
const MathTraitOps math_ops = {
    // Core required methods
    .alu = alu_impl,
    .const_like = const_like_impl,
    
    // Helper methods
    .ufix = ufix_impl,
    .binop = binop_impl,
    .logical_not = logical_not_impl,
    .neg = neg_impl,
    .check_dtype = check_dtype_impl,
    
    // Mathematical operations
    .add = add_impl,
    .mul = mul_impl,
    .sub = sub_impl,
    .div = div_impl,
    .idiv = idiv_impl,
    .mod = mod_impl,
    
    // Bitwise operations
    .bitwise_and = bitwise_and_impl,
    .bitwise_or = bitwise_or_impl,
    .bitwise_xor = bitwise_xor_impl,
    
    // Shift operations
    .lshift = lshift_impl,
    .rshift = rshift_impl,
    
    // Comparison operations
    .lt = lt_impl,
    .gt = gt_impl,
    .ge = ge_impl,
    .le = le_impl,
    .ne = ne_impl,
    .eq = eq_impl,
    
    // Min/max operations
    .maximum = maximum_impl,
    .minimum = minimum_impl,
    
    // Special operations
    .where = where_impl,
    .threefry = threefry_impl,
    
    // Unary math operations
    .reciprocal = reciprocal_impl,
    .sqrt = sqrt_impl,
    .sin = sin_impl,
    .log2 = log2_impl,
    .exp2 = exp2_impl,
    .pow = pow_impl
};

// Module initialization
void mathtraits_init(void) {
    // Nothing to initialize for now
}

void mathtraits_cleanup(void) {
    // Nothing to cleanup for now
}
