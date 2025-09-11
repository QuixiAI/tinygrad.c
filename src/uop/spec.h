#ifndef SRC_UOP_SPEC_H
#define SRC_UOP_SPEC_H
/* spec.h
 * C port of reference/tinygrad/uop/spec.py (always using Z3).
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

// --- Spec dtype metadata shims for parity with Python ---
// Attach pointer/image + addrspace info to DEFINE_* nodes so spec checks can
// enforce PtrDType/ImageDType + AddrSpace like Python.
typedef struct SpecDTypeMeta {
  int is_ptr;
  int is_image;
  int v;
  int size;
  const struct DType* base_dtype;
  int addrspace; // AddrSpace
} SpecDTypeMeta;

// Attach metadata to a DEFINE_* node
void spec_attach_define_meta(struct UOp* u, int addrspace);
// Get metadata from a DEFINE_* node (NULL if unavailable)
const SpecDTypeMeta* spec_get_define_meta(const struct UOp* u);

#ifdef __cplusplus
}
#endif
#endif /* SRC_UOP_SPEC_H */
