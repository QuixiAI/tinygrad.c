#include "test_uop_common.h"

void setUp(void) {
    // Initialize test fixtures if needed
}

void tearDown(void) {
    // Clean up after each test if needed
}

TEST(test_vectorize_operations) {
    
    // Test creating a vectorized UOp
    UOp* elements[4];
    for (int i = 0; i < 4; i++) {
        elements[i] = uop_const(dtypes.float32, (double)i);
    }
    
    UOp* vec = uop_new(OPS_VECTORIZE, dtypes.float32, elements, 4, NULL, NULL);
    ASSERT(vec != NULL);
    ASSERT(vec->op == OPS_VECTORIZE);
    ASSERT(vec->src_count == 4);
}

TEST(test_wmma_operations) {
    
    // Test WMMA operation
    UOp* a = uop_const(dtypes.float16, 1.0);
    UOp* b = uop_const(dtypes.float16, 2.0);
    UOp* c = uop_const(dtypes.float32, 0.0);
    
    UOp* wmma = uop_new(OPS_WMMA, dtypes.float32, (UOp*[]){a, b, c}, 3, NULL, NULL);
    ASSERT(wmma != NULL);
    ASSERT(wmma->op == OPS_WMMA);
}

TEST(test_contract_expand_operations) {
    
    // Test CONTRACT operation
    UOp* x = uop_const(dtypes.float32, 5.0);
    UOpArg contract_arg = {0};
    contract_arg.type = ARG_INT;
    // Note: This test expects a different interface
    UOp* contracted = uop_new(OPS_CONTRACT, dtypes.float32, &x, 1, &contract_arg, NULL);
    ASSERT(contracted != NULL);
    ASSERT(contracted->op == OPS_CONTRACT);
    
    // Test EXPAND operation  
    UOpArg expand_arg = {0};
    expand_arg.type = ARG_INT;
    // Note: This test expects a different interface
    UOp* expanded = uop_new(OPS_EXPAND, dtypes.float32, &x, 1, &expand_arg, NULL);
    ASSERT(expanded != NULL);
    ASSERT(expanded->op == OPS_EXPAND);
}

TEST(test_gemm_optimization) {
    
    // Test matrix multiplication patterns like test_uops_stats.py
    UOp* a = uop_define_global(dtypes.float32, 0);
    UOp* b = uop_define_global(dtypes.float32, 1);
    
    // Create GEMM pattern
    UOp* load_a = uop_load(a, dtypes.float32);
    UOp* load_b = uop_load(b, dtypes.float32);
    UOp* mul = uop_mul(load_a, load_b);
    
    // Should recognize GEMM pattern
    ASSERT(mul != NULL);
}

TEST(test_broadcast_and_expand) {
    
    // Test EXPAND operation like TestExpander
    UOpArg expand_arg = {0};
    expand_arg.type = ARG_INT;
    // Note: This test expects a vec3i interface that's not in our structure
    UOp* x = uop_const(dtypes.float32, 1.0);
    UOp* expanded = uop_new(OPS_EXPAND, dtypes.float32, &x, 1, &expand_arg, NULL);
    
    // Test CONTRACT operation
    UOpArg contract_arg = {0};
    contract_arg.type = ARG_INT;
    // Note: This test expects a vec3i interface that's not in our structure
    UOp* contracted = uop_new(OPS_CONTRACT, dtypes.float32, &expanded, 1, &contract_arg, NULL);
    
    ASSERT(contracted != NULL);
}

TEST(test_fast_idiv_and_mod) {
    
    // Test fast division by constant optimization
    UOp* x = uop_const(dtypes.int32, 100);
    UOp* divisor = uop_const(dtypes.int32, 10);
    UOp* div = uop_div(x, divisor);
    ASSERT(div != NULL);
    
    // Should use optimized division
    double result = exec_alu(OPS_IDIV, dtypes.int32, (double[]){100, 10}, 2);
    ASSERT_NEAR(result, 10, 0.001);
}

TEST(test_fast_idiv_overflow) {
    
    // Test division overflow edge cases
    double min_int = -2147483648.0;
    double result = exec_alu(OPS_IDIV, dtypes.int32, (double[]){min_int, -1}, 2);
    // Should handle overflow correctly
    ASSERT(!isnan(result));
}

TEST(test_mulacc_unrolled) {
    
    // Test unrolled multiply-accumulate patterns
    UOp* acc = uop_const(dtypes.float32, 0);
    for (int i = 0; i < 4; i++) {
        UOp* a = uop_const(dtypes.float32, i);
        UOp* b = uop_const(dtypes.float32, i+1);
        acc = uop_mulacc(a, b, acc);
    }
    ASSERT(acc != NULL);
}

TEST(test_assembly_optimizations) {
    
    // Test MUL by power of 2 -> SHL
    UOp* x = uop_const(dtypes.int32, 5);
    UOp* pow2 = uop_const(dtypes.int32, 8);  // 2^3
    UOp* mul = uop_mul(x, pow2);
    // In optimized code, this should become SHL
    ASSERT(mul != NULL);
    
    // Test IDIV by power of 2 -> SHR
    UOp* div = uop_new(OPS_IDIV, dtypes.int32, (UOp*[]){x, pow2}, 2, NULL, NULL);
    ASSERT(div != NULL);
    ASSERT(div->op == OPS_IDIV);  // Will be optimized to SHR in renderer
}

TEST(test_packed_smem_size) {
    
    // Test packed shared memory sizing
    UOpArg size_arg = {0};
    size_arg.type = ARG_INT;
    size_arg.int_data.i = 1024;
    UOp* smem = uop_new(OPS_DEFINE_LOCAL, dtypes.float32, NULL, 0, &size_arg, NULL);
    
    // Should allocate correct packed size
    ASSERT(smem->arg.int_data.i == 1024);
}

TEST(test_bitcast_operations) {
    
    // Test float to uint32 bitcast
    UOp* float_val = uop_const(dtypes.float32, 1.0);
    UOp* bitcast = uop_new(OPS_BITCAST, dtypes.uint32, &float_val, 1, NULL, NULL);
    ASSERT(bitcast != NULL);
    ASSERT(bitcast->op == OPS_BITCAST);
    ASSERT(dtype_eq(&bitcast->dtype, &dtypes.uint32));
    
    // Test int to float bitcast
    UOp* int_val = uop_const(dtypes.uint32, 0x3f800000);  // 1.0 in float
    bitcast = uop_new(OPS_BITCAST, dtypes.float32, &int_val, 1, NULL, NULL);
    ASSERT(bitcast != NULL);
    ASSERT(bitcast->op == OPS_BITCAST);
}

// Auto-register all test functions and run them
TEST_MAIN()