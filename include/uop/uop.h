#ifndef TINYGRAD_UOP_H
#define TINYGRAD_UOP_H

#include <stdint.h>
#include <stdbool.h>

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

// Helper functions
const char* ops_to_string(Ops op);
bool ops_is_valid(Ops op);
int ops_get_arity(Ops op);

// Initialize module
void uop_init(void);
void uop_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* TINYGRAD_UOP_H */