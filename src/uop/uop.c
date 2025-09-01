/* uop.c - UOp system implementation */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uop/uop.h"

// Global GroupOp instance with proper initialization
const GroupOp group_op = {
    .is_unary = {
        [OPS_NEG] = true, [OPS_EXP2] = true, [OPS_LOG2] = true, [OPS_SIN] = true,
        [OPS_SQRT] = true, [OPS_RECIP] = true, [OPS_CAST] = true, [OPS_BITCAST] = true
    },
    .is_binary = {
        [OPS_ADD] = true, [OPS_MUL] = true, [OPS_SHL] = true, [OPS_SHR] = true,
        [OPS_IDIV] = true, [OPS_MAX] = true, [OPS_MOD] = true, [OPS_CMPLT] = true,
        [OPS_CMPNE] = true, [OPS_CMPEQ] = true, [OPS_XOR] = true, [OPS_OR] = true,
        [OPS_AND] = true, [OPS_SUB] = true, [OPS_FDIV] = true, [OPS_POW] = true
    },
    .is_ternary = {
        [OPS_WHERE] = true, [OPS_MULACC] = true
    },
    .is_alu = {
        [OPS_ADD] = true, [OPS_MUL] = true, [OPS_SUB] = true, [OPS_FDIV] = true,
        [OPS_IDIV] = true, [OPS_MOD] = true, [OPS_MAX] = true, [OPS_CMPLT] = true,
        [OPS_CMPNE] = true, [OPS_CMPEQ] = true, [OPS_XOR] = true, [OPS_OR] = true,
        [OPS_AND] = true, [OPS_SHL] = true, [OPS_SHR] = true, [OPS_POW] = true,
        [OPS_NEG] = true, [OPS_EXP2] = true, [OPS_LOG2] = true, [OPS_SIN] = true,
        [OPS_SQRT] = true, [OPS_RECIP] = true, [OPS_CAST] = true, [OPS_BITCAST] = true,
        [OPS_WHERE] = true, [OPS_MULACC] = true
    },
    .is_define = {
        [OPS_DEFINE_GLOBAL] = true, [OPS_DEFINE_LOCAL] = true, [OPS_DEFINE_REG] = true,
        [OPS_DEFINE_VAR] = true, [OPS_BIND] = true
    },
    .is_movement = {
        [OPS_RESHAPE] = true, [OPS_PERMUTE] = true, [OPS_EXPAND] = true,
        [OPS_PAD] = true, [OPS_SHRINK] = true, [OPS_FLIP] = true, [OPS_MULTI] = true,
        [OPS_VIEW] = true, [OPS_VALID] = true, [OPS_COPY] = true, [OPS_BUFFER] = true,
        [OPS_BUFFER_VIEW] = true, [OPS_MSELECT] = true, [OPS_MSTACK] = true,
        [OPS_CONTIGUOUS] = true, [OPS_CONTIGUOUS_BACKWARD] = true
    },
    .is_commutative = {
        [OPS_ADD] = true, [OPS_MUL] = true, [OPS_MAX] = true, [OPS_XOR] = true,
        [OPS_OR] = true, [OPS_AND] = true
    },
    .is_associative = {
        [OPS_ADD] = true, [OPS_MUL] = true, [OPS_MAX] = true, [OPS_AND] = true, [OPS_OR] = true
    }
};

// Array of operation names for ops_to_string
static const char* op_names[OPS_MAX_VALUE] = {
    [OPS_NOOP] = "NOOP",
    [OPS_SINK] = "SINK",
    [OPS_UNIQUE] = "UNIQUE",
    [OPS_DEVICE] = "DEVICE",
    [OPS_KERNEL] = "KERNEL",
    [OPS_PRECAST] = "PRECAST",
    [OPS_CHILD] = "CHILD",
    [OPS_COPY] = "COPY",
    [OPS_BUFFER] = "BUFFER",
    [OPS_BUFFER_VIEW] = "BUFFER_VIEW",
    [OPS_MSELECT] = "MSELECT",
    [OPS_MSTACK] = "MSTACK",
    [OPS_CONTIGUOUS] = "CONTIGUOUS",
    [OPS_CONTIGUOUS_BACKWARD] = "CONTIGUOUS_BACKWARD",
    [OPS_DETACH] = "DETACH",
    [OPS_FUSE] = "FUSE",
    [OPS_BLOCK] = "BLOCK",
    [OPS_BLOCKSTART] = "BLOCKSTART",
    [OPS_BLOCKEND] = "BLOCKEND",
    [OPS_BLOCKFINAL] = "BLOCKFINAL",
    [OPS_RESHAPE] = "RESHAPE",
    [OPS_PERMUTE] = "PERMUTE",
    [OPS_EXPAND] = "EXPAND",
    [OPS_PAD] = "PAD",
    [OPS_SHRINK] = "SHRINK",
    [OPS_FLIP] = "FLIP",
    [OPS_MULTI] = "MULTI",
    [OPS_VIEW] = "VIEW",
    [OPS_VALID] = "VALID",
    [OPS_DEFINE_GLOBAL] = "DEFINE_GLOBAL",
    [OPS_DEFINE_LOCAL] = "DEFINE_LOCAL",
    [OPS_DEFINE_REG] = "DEFINE_REG",
    [OPS_DEFINE_VAR] = "DEFINE_VAR",
    [OPS_BIND] = "BIND",
    [OPS_SPECIAL] = "SPECIAL",
    [OPS_REDUCE_AXIS] = "REDUCE_AXIS",
    [OPS_REDUCE] = "REDUCE",
    [OPS_ALLREDUCE] = "ALLREDUCE",
    [OPS_UNROLL] = "UNROLL",
    [OPS_CONTRACT] = "CONTRACT",
    [OPS_GEP] = "GEP",
    [OPS_VECTORIZE] = "VECTORIZE",
    [OPS_CAT] = "CAT",
    [OPS_PTRCAT] = "PTRCAT",
    [OPS_CAST] = "CAST",
    [OPS_BITCAST] = "BITCAST",
    [OPS_EXP2] = "EXP2",
    [OPS_LOG2] = "LOG2",
    [OPS_SIN] = "SIN",
    [OPS_SQRT] = "SQRT",
    [OPS_RECIP] = "RECIP",
    [OPS_NEG] = "NEG",
    [OPS_LOAD] = "LOAD",
    [OPS_STORE] = "STORE",
    [OPS_ASSIGN] = "ASSIGN",
    [OPS_WMMA] = "WMMA",
    [OPS_INDEX] = "INDEX",
    [OPS_ADD] = "ADD",
    [OPS_MUL] = "MUL",
    [OPS_SHL] = "SHL",
    [OPS_SHR] = "SHR",
    [OPS_IDIV] = "IDIV",
    [OPS_MAX] = "MAX",
    [OPS_MOD] = "MOD",
    [OPS_CMPLT] = "CMPLT",
    [OPS_CMPNE] = "CMPNE",
    [OPS_CMPEQ] = "CMPEQ",
    [OPS_XOR] = "XOR",
    [OPS_OR] = "OR",
    [OPS_AND] = "AND",
    [OPS_THREEFRY] = "THREEFRY",
    [OPS_SUB] = "SUB",
    [OPS_FDIV] = "FDIV",
    [OPS_POW] = "POW",
    [OPS_WHERE] = "WHERE",
    [OPS_MULACC] = "MULACC",
    [OPS_BARRIER] = "BARRIER",
    [OPS_RANGE] = "RANGE",
    [OPS_IF] = "IF",
    [OPS_ENDRANGE] = "ENDRANGE",
    [OPS_ENDIF] = "ENDIF",
    [OPS_VCONST] = "VCONST",
    [OPS_CONST] = "CONST",
    [OPS_CUSTOM] = "CUSTOM",
    [OPS_CUSTOMI] = "CUSTOMI"
};

// Array of operation arity for ops_get_arity
static const int op_arity[OPS_MAX_VALUE] = {
    [OPS_NOOP] = 1, [OPS_SINK] = 1, [OPS_UNIQUE] = 0, [OPS_DEVICE] = 0,
    [OPS_KERNEL] = 0, [OPS_PRECAST] = 1, [OPS_CHILD] = 1, [OPS_COPY] = 2,
    [OPS_BUFFER] = 1, [OPS_BUFFER_VIEW] = 1, [OPS_MSELECT] = 2, [OPS_MSTACK] = 2,
    [OPS_CONTIGUOUS] = 1, [OPS_CONTIGUOUS_BACKWARD] = 1, [OPS_DETACH] = 1, [OPS_FUSE] = 2,
    [OPS_BLOCK] = 0, [OPS_BLOCKSTART] = 0, [OPS_BLOCKEND] = 0, [OPS_BLOCKFINAL] = 1,
    [OPS_RESHAPE] = 1, [OPS_PERMUTE] = 1, [OPS_EXPAND] = 1, [OPS_PAD] = 1,
    [OPS_SHRINK] = 1, [OPS_FLIP] = 1, [OPS_MULTI] = 1, [OPS_VIEW] = 1,
    [OPS_VALID] = 1, [OPS_DEFINE_GLOBAL] = 0, [OPS_DEFINE_LOCAL] = 0, [OPS_DEFINE_REG] = 0,
    [OPS_DEFINE_VAR] = 0, [OPS_BIND] = 1, [OPS_SPECIAL] = 0, [OPS_REDUCE_AXIS] = 1,
    [OPS_REDUCE] = 1, [OPS_ALLREDUCE] = 1, [OPS_UNROLL] = 1, [OPS_CONTRACT] = 1,
    [OPS_GEP] = 1, [OPS_VECTORIZE] = 1, [OPS_CAT] = 1, [OPS_PTRCAT] = 1,
    [OPS_CAST] = 1, [OPS_BITCAST] = 1, [OPS_EXP2] = 1, [OPS_LOG2] = 1,
    [OPS_SIN] = 1, [OPS_SQRT] = 1, [OPS_RECIP] = 1, [OPS_NEG] = 1,
    [OPS_LOAD] = 1, [OPS_STORE] = 2, [OPS_ASSIGN] = 2, [OPS_WMMA] = 3,
    [OPS_INDEX] = 2, [OPS_ADD] = 2, [OPS_MUL] = 2, [OPS_SHL] = 2,
    [OPS_SHR] = 2, [OPS_IDIV] = 2, [OPS_MAX] = 2, [OPS_MOD] = 2,
    [OPS_CMPLT] = 2, [OPS_CMPNE] = 2, [OPS_CMPEQ] = 2, [OPS_XOR] = 2,
    [OPS_OR] = 2, [OPS_AND] = 2, [OPS_THREEFRY] = 2, [OPS_SUB] = 2,
    [OPS_FDIV] = 2, [OPS_POW] = 2, [OPS_WHERE] = 3, [OPS_MULACC] = 3,
    [OPS_BARRIER] = 1, [OPS_RANGE] = 0, [OPS_IF] = 1, [OPS_ENDRANGE] = 1,
    [OPS_ENDIF] = 1, [OPS_VCONST] = 0, [OPS_CONST] = 0, [OPS_CUSTOM] = 1,
    [OPS_CUSTOMI] = 1
};

// Real implementations
const char* ops_to_string(Ops op) {
    if (op >= 0 && op < OPS_MAX_VALUE && op_names[op] != NULL) {
        return op_names[op];
    }
    return "UNKNOWN";
}

bool ops_is_valid(Ops op) {
    return op >= OPS_NOOP && op < OPS_MAX_VALUE;
}

int ops_get_arity(Ops op) {
    if (op >= 0 && op < OPS_MAX_VALUE) {
        return op_arity[op];
    }
    return 0;
}

void uop_init(void) {
    // Nothing needed for now
}

void uop_cleanup(void) {
    // Nothing needed for now
}