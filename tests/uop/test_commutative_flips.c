#include "test_uop_common.h"

void setUp(void) { static int init=0; if(!init){ dtypes_init(); init=1; } }
void tearDown(void) {}

TEST(test_add_const_flips_right)
{
  UOp* two = uop_const(dtypes.int32, 2.0);
  UOp* x = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, NULL, "x");
  UOp* sum = uop_add(two, x);
  UOp* s = uop_ssimplify(sum);
  ASSERT(s->op == OPS_ADD);
  ASSERT(s->src_count == 2);
  ASSERT(s->src[1]->op == OPS_CONST);
}

TEST(test_mul_const_flips_right)
{
  UOp* three = uop_const(dtypes.int32, 3.0);
  UOp* x = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, NULL, "x");
  UOp* prod = uop_mul(three, x);
  UOp* s = uop_ssimplify(prod);
  ASSERT(s->op == OPS_MUL);
  ASSERT(s->src_count == 2);
  ASSERT(s->src[1]->op == OPS_CONST);
}

TEST_MAIN()

