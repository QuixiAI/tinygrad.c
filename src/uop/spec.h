#ifndef SRC_UOP_SPEC_H
#define SRC_UOP_SPEC_H
/* spec.h
 * TODO: port from tinygrad/spec.py to C.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct UOp;

/* Test validation function - stub for TDD */
#define SPEC_ERR_UNIMPL -9999  /* Indicates function not implemented */
#define SPEC_ERR_INVALID -1     /* Indicates validation failed */
#define SPEC_OK 0               /* Indicates validation passed */

int helper_test_verify_ast(struct UOp* store);

#ifdef __cplusplus
}
#endif
#endif /* SRC_UOP_SPEC_H */
