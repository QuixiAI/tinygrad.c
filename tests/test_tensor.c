#include "test_common.h"

void setUp(void) {
    // Initialize test fixtures if needed
}

void tearDown(void) {
    // Clean up after each test if needed
}

void test_tensor_basic_operations(void) {
    tg_ctx_t ctx;
    TEST_ASSERT_EQUAL(0, tgCreateContext(&ctx));
    
    int64_t shape[2] = {2, 3};
    tg_tensor_t t;
    TEST_ASSERT_EQUAL(0, tgTensorCreate(ctx, TG_F32, shape, 2, &t));
    
    float data[6] = {1, 2, 3, 4, 5, 6};
    TEST_ASSERT_EQUAL(0, tgTensorUpload(t, data, sizeof(data)));
    
    float out[6] = {0};
    TEST_ASSERT_EQUAL(0, tgTensorDownload(t, out, sizeof(out)));
    
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data[i], out[i]);
    }
    
    TEST_ASSERT_EQUAL(0, tgTensorDestroy(t));
    TEST_ASSERT_EQUAL(0, tgDestroyContext(ctx));
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_tensor_basic_operations);
    
    return UNITY_END();
}
