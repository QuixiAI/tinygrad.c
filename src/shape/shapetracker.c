/* shapetracker.c
 * Auto-generated unimplemented stub.
 * This file is a placeholder until shapetracker.h gets a real implementation.
 * Return TG_ERR_UNIMPL so callers fail loudly instead of silently succeeding.
 */
#include "tg.h"
#include "shape/shapetracker.h"

int tg_unimpl_stub_shapetracker(void) {
  return TG_ERR_UNIMPL;
}

/* Stub implementation for TDD */
struct ShapeTracker* ShapeTracker_from_shape(int* shape, int ndim) {
    static struct ShapeTracker st;
    st.ndim = ndim;
    if (shape) {
        for (int i = 0; i < ndim && i < 8; i++) {
            st.shape[i] = shape[i];
        }
    }
    return &st;
}

/* NOTE:
 * Replace this file with a real implementation when porting tinygrad/shapetracker.py.
 */
