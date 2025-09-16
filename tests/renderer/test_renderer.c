#include "test_common.h"
#include "renderer/renderer.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "dtype/dtype.h"
#include "engine/realize.h"
#include <stdlib.h>

TEST(test_renderer_to_function_name_basic) {
  char* s = renderer_to_function_name("kernel name!");
  TEST_ASSERT_NOT_NULL(s);
  TEST_ASSERT_EQUAL_STRING("kernel20name21", s);
  free(s);
}

static void cleanup_basic_graph(UOp** nodes, int count) {
  if (!nodes) return;
  for (int i = count-1; i >= 0; i--) {
    if (nodes[i]) uop_unref(nodes[i]);
  }
}

TEST(test_renderer_estimates_simple) {
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* ptr = uop_index(buf, idx);
  UOp* load = uop_load(ptr, dtypes.float32);
  UOp* ctwo = uop_const(dtypes.float32, 2.0);
  UOp* add = uop_add(load, ctwo);
  UOp* store = uop_store(ptr, add);
  UOp* nodes[] = {buf, idx, ptr, load, ctwo, add, store};

  Estimates est = renderer_estimates_from_uops(nodes, (int)(sizeof(nodes)/sizeof(nodes[0])), 0);
  TEST_ASSERT_EQUAL_INT64(1, est.ops);
  TEST_ASSERT_EQUAL_INT64(8, est.lds);
  TEST_ASSERT_EQUAL_INT64(est.lds, est.mem);

  cleanup_basic_graph(nodes, (int)(sizeof(nodes)/sizeof(nodes[0])));
}

TEST(test_renderer_estimates_ignore_indexing) {
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* base = uop_const(dtypes.int32, 2);
  UOp* off = uop_const(dtypes.int32, 1);
  UOp* idx = uop_add(base, off);
  UOp* ptr = uop_index(buf, idx);
  UOp* load = uop_load(ptr, dtypes.float32);
  UOp* ctwo = uop_const(dtypes.float32, 2.0);
  UOp* add = uop_add(load, ctwo);
  UOp* store = uop_store(ptr, add);
  UOp* nodes[] = {buf, base, off, idx, ptr, load, ctwo, add, store};

  Estimates est_all = renderer_estimates_from_uops(nodes, (int)(sizeof(nodes)/sizeof(nodes[0])), 0);
  TEST_ASSERT_EQUAL_INT64(2, est_all.ops);

  Estimates est_ignore = renderer_estimates_from_uops(nodes, (int)(sizeof(nodes)/sizeof(nodes[0])), 1);
  TEST_ASSERT_EQUAL_INT64(1, est_ignore.ops);

  cleanup_basic_graph(nodes, (int)(sizeof(nodes)/sizeof(nodes[0])));
}

TEST(test_renderer_estimates_range_multiplier) {
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* ptr = uop_index(buf, idx);
  UOp* extent = uop_const(dtypes.int32, 4);
  UOp* range = uop_range(extent, 0);
  UOp* load = uop_load(ptr, dtypes.float32);
  UOp* ctwo = uop_const(dtypes.float32, 2.0);
  UOp* add = uop_add(load, ctwo);
  UOp* store = uop_store(ptr, add);
  UOpArg arg = {0};
  UOp* endrange = uop_new(OPS_ENDRANGE, dtypes.void_, NULL, 0, &arg, NULL);
  UOp* nodes[] = {buf, idx, ptr, extent, range, load, ctwo, add, store, endrange};

  Estimates est = renderer_estimates_from_uops(nodes, (int)(sizeof(nodes)/sizeof(nodes[0])), 0);
  TEST_ASSERT_EQUAL_INT64(4, est.ops);
  TEST_ASSERT_EQUAL_INT64(32, est.lds);

  cleanup_basic_graph(nodes, (int)(sizeof(nodes)/sizeof(nodes[0])));
}

void setUp(void) {}
void tearDown(void) {}

static void cleanup_nodes(UOp** nodes, size_t count) {
  if (!nodes) return;
  for (size_t i = count; i > 0; i--) {
    if (nodes[i-1]) uop_unref(nodes[i-1]);
  }
}

TEST(test_programspec_field_extraction) {
  UOp* buf0 = uop_define_global(dtypes.float32, 0);
  UOp* buf1 = uop_define_global(dtypes.float32, 1);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* ptr0 = uop_index(buf0, idx);
  UOp* load = uop_load(ptr0, dtypes.float32);
  UOp* one = uop_const(dtypes.float32, 1.0);
  UOp* add0 = uop_add(load, one);
  UOp* ptr1 = uop_index(buf1, idx);
  UOp* store0 = uop_store(ptr1, add0);
  UOp* three = uop_const(dtypes.float32, 3.0);
  UOp* add1 = uop_add(load, three);
  UOp* store1 = uop_store(ptr1, add1);
  UOp* stores[] = {store0, store1};
  UOp* sink = uop_sink(stores, 2);

  UOp* uops[] = {buf0, buf1, idx, ptr0, load, one, add0, ptr1, store0, three, add1, store1, sink};
  ProgramSpec spec = {0};
  spec.ast = sink;
  spec.uops = uops;
  spec.uops_count = (int)(sizeof(uops) / sizeof(uops[0]));
  spec.estimates = renderer_estimates_from_uops(uops, spec.uops_count, 1);
  programspec_finalize(&spec);

  TEST_ASSERT_EQUAL_INT(2, spec.globals_count);
  TEST_ASSERT_EQUAL_INT(0, spec.globals[0]);
  TEST_ASSERT_EQUAL_INT(1, spec.globals[1]);

  TEST_ASSERT_EQUAL_INT(1, spec.ins_count);
  TEST_ASSERT_EQUAL_INT(0, spec.ins[0]);

  TEST_ASSERT_EQUAL_INT(1, spec.outs_count);
  TEST_ASSERT_EQUAL_INT(1, spec.outs[0]);

  TEST_ASSERT_EQUAL_INT64(2, spec.estimates.ops);
  TEST_ASSERT_EQUAL_INT64(8, spec.estimates.lds);
  TEST_ASSERT_EQUAL_INT64(8, spec.estimates.mem);

  free(spec.globals);
  free(spec.ins);
  free(spec.outs);
  if (spec.vars) free(spec.vars);
  cleanup_nodes(uops, sizeof(uops) / sizeof(uops[0]));
}

TEST_MAIN()
