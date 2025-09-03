#ifndef SRC_UOP_SYMBOLIC_H
#define SRC_UOP_SYMBOLIC_H
/* symbolic.h
 * Symbolic operations for UOp
 */

#include "uop/uop.h"
#include "uop/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// Split a UOp tree on a specific operation
// Returns list of UOps that are not the separator operation
void split_uop(UOp* x, Ops sep, UOp*** result, int* count);

// Simplify a UOp given that valid is true
// Returns NULL if valid is always false, otherwise simplified uop
UOp* uop_given_valid(UOp* valid, UOp* uop);

// Simplify a valid expression
// Returns NULL if no simplification possible
UOp* simplify_valid(UOp* valid);

// Convert sint (int64_t) to UOp constant
UOp* sint_to_uop(int64_t val);

#ifdef __cplusplus
}
#endif
#endif /* SRC_UOP_SYMBOLIC_H */
