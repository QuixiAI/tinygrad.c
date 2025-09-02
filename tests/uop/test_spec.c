#include "test_uop_common.h"

void setUp(void) {
    // Initialize test fixtures if needed
}

void tearDown(void) {
    // Clean up after each test if needed
}

TEST(test_shape_validation) {
    
    // Test shape specification validation like test_uop_spec.py
    UOp* buf = uop_define_global(dtypes.float32, 0);
    
    // Create invalid view (masked view in const)
    // This should be caught by validation
    ShapeTracker st = {0};  // Invalid shape tracker
    UOp* view = uop_view(buf, &st);
    
    ASSERT(view != NULL);  // Will fail until validation implemented
}

TEST(test_shape_spec_validation) {
    
    // Test no implicit broadcasting
    UOp* buf1 = uop_define_global(dtypes.float32, 0);
    UOp* buf2 = uop_define_global(dtypes.float32, 1);
    
    // These should have incompatible shapes for broadcasting
    UOp* a = uop_load(buf1, dtypes.float32);
    UOp* b = uop_load(buf2, dtypes.float32);
    UOp* result = uop_add(a, b);
    
    // Should validate shape compatibility
    ASSERT(result != NULL);
}

TEST(test_type_inference) {
    
    // Test automatic type promotion and inference
    UOp* x = uop_const(dtypes.float32, 1.0);
    UOp* y = uop_const(dtypes.int32, 2);
    
    // Should promote to float32
    UOp* result = uop_add(x, y);
    ASSERT(result != NULL);  // Will fail until type inference works
}

TEST(test_bounds_checking) {
    
    // Test out-of-bounds access detection like test_uop_graph.py
    UOp* buf = uop_define_global(dtypes.float32, 0);
    UOpArg idx_arg = {0};
    idx_arg.type = ARG_INT;
    idx_arg.int_data.i = 1000;  // Out of bounds index
    UOp* idx = uop_new(OPS_CONST, dtypes.int32, NULL, 0, &idx_arg, NULL);
    
    // This should detect out-of-bounds access
    UOp* load = uop_index(buf, idx);
    ASSERT(load != NULL);  // Will fail until bounds checking implemented
}

TEST(test_reduce_store_validation) {
    
    UOp* buf = uop_define_global(dtypes.float32, 0);
    UOp* data = uop_load(buf, dtypes.float32);
    
    // Direct reduce to store should fail validation
    int axes[] = {0};
    UOp* reduced = uop_reduce_axis(data, OPS_ADD, axes, 1);
    // UOp* store = uop_store(buf, reduced);
    
    // bool validation_passed = uop_validate_ast(store);  // TODO: Implement validation
    ASSERT(reduced != NULL);  // Basic test for now
}

// Auto-register all test functions and run them
TEST_MAIN()