#ifndef SRC_UOP_OPTIONAL_H
#define SRC_UOP_OPTIONAL_H
/* optional.h
 * TODO: port from tinygrad/optional.py to C.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdbool.h>
#include "uop/uop.h"

// Build late-rewrite patterns based on available ops and flags.
// - available_ops: list of Ops supported natively by backend
// - ops_count: length of available_ops
// - force_transcendental: when true, always rewrite EXP2/LOG2/SIN to transcendental
struct PatternMatcher* get_late_rewrite_patterns(const Ops* available_ops, size_t ops_count, bool force_transcendental);

// Apply rewrite patterns to a UOp tree. Optional device context influences fast_idiv.
UOp* optional_apply_patterns_ex(UOp* root, struct PatternMatcher* pm, const char* device);
static inline UOp* optional_apply_patterns(UOp* root, struct PatternMatcher* pm) { return optional_apply_patterns_ex(root, pm, NULL); }

#ifdef __cplusplus
}
#endif
#endif /* SRC_UOP_OPTIONAL_H */
