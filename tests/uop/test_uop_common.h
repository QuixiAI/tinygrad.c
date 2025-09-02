#ifndef TEST_UOP_COMMON_H
#define TEST_UOP_COMMON_H

#include "../test_common.h"

/* UOp test specific includes */
#include "uop/uop.h"
#include "uop/ops.h"
#include "uop/mathtraits.h"
#include "uop/optional.h"
#include "uop/spec.h"
#include "uop/symbolic.h"
#include "uop/transcendental.h"
#include "dtype/dtype.h"

/* Additional macros for UOp tests */
#define ASSERT_FLOAT_EQ(actual, expected, tolerance) TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, actual)

#endif /* TEST_UOP_COMMON_H */