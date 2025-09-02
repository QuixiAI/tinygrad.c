#ifndef SRC_SHAPE_SHAPETRACKER_H
#define SRC_SHAPE_SHAPETRACKER_H
/* shapetracker.h
 * TODO: port from tinygrad/shapetracker.py to C.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ShapeTracker stub for TDD - will be fully implemented later */
struct ShapeTracker {
    int shape[8];
    int ndim;
    // TODO: Add views, strides, offset, mask, contiguous flags, etc.
};

// Constructor function for ShapeTracker
struct ShapeTracker* ShapeTracker_from_shape(int* shape, int ndim);

#ifdef __cplusplus
}
#endif
#endif /* SRC_SHAPE_SHAPETRACKER_H */
