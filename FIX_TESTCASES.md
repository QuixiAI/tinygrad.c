# Test Case Remediation Plan

## Problem Statement

The current test suite contains multiple placeholder tests that violate Test-Driven Development (TDD) principles:
- Tests that always pass without testing anything (`TEST_ASSERT(1)`)
- Tests for unimplemented features that don't actually test
- Commented-out assertions with TODOs
- False reporting of success when no real testing occurs

This creates false confidence in code coverage and prevents proper regression detection.

## Identified Issues

### Critical: Empty Test Files
1. **test_ops.c** - Contains only `test_ops_stubs_build_sanity` with `TEST_ASSERT(1)`
2. **test_resnet18.c** - Contains only `test_resnet18_placeholder` with `TEST_ASSERT(1)`

### High Priority: Unimplemented Feature Tests
3. **test_uop.c::test_phi_operations** - PHI operations not in ops enum
4. **test_uop.c::test_acc_operations** - ACC operations not in ops enum

### Medium Priority: Commented TODOs in test_uop.c
5. Lines 1131, 1140 - GIDX/LIDX arg validation
6. Lines 2213, 2228 - Graph rewrite operations
7. Lines 2242, 2257 - Bounds checking
8. Line 2271 - Gated indexing
9. Lines 2291, 2307 - AST validation
10. Line 2441 - Cleanup function

## Remediation Strategy

### Phase 1: Immediate Actions (Do Now)

#### 1.1 Convert Empty Tests to TEST_IGNORE
```c
// Instead of:
void test_ops_stubs_build_sanity(void) {
    TEST_ASSERT(1);
}

// Change to:
void test_ops_basic_operations(void) {
    TEST_IGNORE_MESSAGE("Ops tests not yet implemented - waiting for ops.c implementation");
}
```

#### 1.2 Remove or Ignore Unimplemented Feature Tests
```c
void test_phi_operations(void) {
    TEST_IGNORE_MESSAGE("PHI operations not yet in ops enum - implement when OPS_PHI added");
}

void test_acc_operations(void) {
    TEST_IGNORE_MESSAGE("ACC operations not yet in ops enum - implement when OPS_ACC added");
}
```

### Phase 2: Write Real Tests (Per Feature)

#### 2.1 test_ops.c - CPU Operations Tests
Write actual tests for implemented operations:
```c
void test_binary_ops_cpu(void) {
    // Test ADD
    float a = 5.0f, b = 3.0f;
    float result = op_add_cpu(&a, &b);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 8.0f, result);
    
    // Test MUL
    result = op_mul_cpu(&a, &b);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 15.0f, result);
}

void test_unary_ops_cpu(void) {
    // Test NEG
    float x = 5.0f;
    float result = op_neg_cpu(&x);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -5.0f, result);
    
    // Test ABS
    float y = -3.0f;
    result = op_abs_cpu(&y);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 3.0f, result);
}

void test_reduce_ops_cpu(void) {
    // Test SUM reduction
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float result = op_reduce_sum_cpu(data, 4);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 10.0f, result);
}
```

#### 2.2 test_resnet18.c - Neural Network Tests
Write tests for actual ResNet18 components:
```c
void test_conv2d_layer(void) {
    // Test basic convolution
    tg_tensor_t input, weight, output;
    // ... setup tensors ...
    int status = resnet18_conv2d(&input, &weight, &output);
    TEST_ASSERT_EQUAL(0, status);
    // ... verify output dimensions ...
}

void test_batchnorm_layer(void) {
    // Test batch normalization
    tg_tensor_t input, output;
    // ... setup ...
    int status = resnet18_batchnorm(&input, &output);
    TEST_ASSERT_EQUAL(0, status);
    // ... verify normalization ...
}

void test_resnet_block(void) {
    // Test a complete ResNet block
    TEST_IGNORE_MESSAGE("Implement when ResNet block structure is defined");
}
```

### Phase 3: Fix TODOs in test_uop.c

#### 3.1 Uncomment and Fix Assertions
```c
// Line 1131 - Fix GIDX arg validation
TEST_ASSERT_EQUAL_STRING("gidx0", gidx0->arg.s);

// Line 1140 - Fix LIDX arg validation  
TEST_ASSERT_EQUAL_STRING("lidx0", lidx0->arg.s);
```

#### 3.2 Implement Missing Validation Functions
Create stubs that return expected failure until implemented:
```c
bool uop_check_bounds(UOp* op) {
    // TODO: Implement actual bounds checking
    return false;  // Conservative: assume bounds invalid until implemented
}

bool uop_validate_ast(UOp* op) {
    // TODO: Implement AST validation
    return false;  // Conservative: assume invalid until implemented
}
```

## Implementation Order

1. **Week 1**: Fix all TEST_ASSERT(1) placeholders (Phase 1)
2. **Week 2**: Write real tests for ops.c (Phase 2.1)
3. **Week 3**: Write real tests for resnet18.c (Phase 2.2)
4. **Week 4**: Fix all TODOs in test_uop.c (Phase 3)

## Success Criteria

- [ ] No tests that always pass without testing anything
- [ ] All unimplemented features use TEST_IGNORE with clear messages
- [ ] Each test file has at least 3 meaningful test cases
- [ ] All TODOs are either implemented or converted to TEST_IGNORE
- [ ] Test output clearly distinguishes between:
  - PASS: Feature works correctly
  - FAIL: Feature is broken
  - IGNORE: Feature not yet implemented

## Testing the Tests

To verify our test improvements:
1. Intentionally break each feature and confirm tests fail
2. Run coverage analysis to ensure code is actually exercised
3. Review that each test has clear assertions about expected behavior

## Long-term Guidelines

Going forward, follow strict TDD:
1. **Never** write placeholder tests that always pass
2. **Always** write the failing test first
3. **Use** TEST_IGNORE for unimplemented features
4. **Document** why a test is ignored in the message
5. **Remove** TEST_IGNORE only when implementing the feature

## Tracking Progress

Use this checklist to track remediation:
- [ ] test_ops.c - Convert to real tests or TEST_IGNORE
- [ ] test_resnet18.c - Convert to real tests or TEST_IGNORE
- [ ] test_uop.c::test_phi_operations - Convert to TEST_IGNORE
- [ ] test_uop.c::test_acc_operations - Convert to TEST_IGNORE
- [ ] test_uop.c TODOs - Fix or convert to TEST_IGNORE
- [ ] Documentation - Update README with test guidelines
- [ ] CI/CD - Add check for TEST_ASSERT(1) anti-pattern