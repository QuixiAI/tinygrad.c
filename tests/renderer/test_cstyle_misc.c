#include "test_common.h"
#include "renderer/cstyle.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_renderer_cstyle_precast_passthrough) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* x = uop_const(dtypes.float32, 3.0);
  UOp* arr0[] = {x};
  UOpArg a = {0};
  // create PRECAST node: same dtype as dst (int), but pass-through semantics
  UOp* pc = uop_new(OPS_PRECAST, dtypes.int32, arr0, 1, &a, NULL);
  UOp* arr[] = {x, pc};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(src);
  // Expect simple assignment without builtin_convertvector or pointer casts
  TEST_ASSERT_NULL(strstr(src, "__builtin_convertvector"));
  TEST_ASSERT_NULL(strstr(src, "*&"));
  TEST_ASSERT_NOT_NULL(strstr(src, "="));
  free(src);
  uop_unref(pc); uop_unref(x);
  free(r);
}

TEST(test_renderer_cstyle_index_add_strips_parens) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* a = uop_const(dtypes.int32, 1);
  UOp* b = uop_const(dtypes.int32, 2);
  UOp* sum = uop_add(a, b);
  UOp* bidx = uop_index(buf, sum);
  UOp* load = uop_load(bidx, dtypes.float32);
  UOp* arr[] = {buf,a,b,sum,bidx,load};
  char* src = r->render(r, arr, 6);
  TEST_ASSERT_NOT_NULL(src);
  // should contain + without parens around the sum
  TEST_ASSERT_NULL(strstr(src, "+(1+2)"));
  free(src);
  uop_unref(load); uop_unref(bidx); uop_unref(sum); uop_unref(b); uop_unref(a); uop_unref(buf);
  free(r);
}

TEST(test_renderer_cstyle_unary_builtins_clang_default) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* x = uop_const(dtypes.float32, 4.0);
  UOp* e = uop_exp2(x);
  UOp* l = uop_log2(x);
  UOp* s = uop_sin(x);
  UOp* arr[] = {x,e,l,s};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "exp2("));
  TEST_ASSERT_NOT_NULL(strstr(src, "log2("));
  TEST_ASSERT_NOT_NULL(strstr(src, "sin("));
  TEST_ASSERT_NULL(strstr(src, "exp2f"));
  TEST_ASSERT_NULL(strstr(src, "log2f"));
  TEST_ASSERT_NULL(strstr(src, "sinf"));
  free(src);
  uop_unref(s); uop_unref(l); uop_unref(e); uop_unref(x);
  free(r);
}

TEST(test_renderer_cstyle_max_rewrite_like_where) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* a = uop_const(dtypes.int32, 3);
  UOp* b = uop_const(dtypes.int32, 4);
  UOp* m = uop_max(a,b);
  UOp* arr[] = {a,b,m};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "?"));
  TEST_ASSERT_NOT_NULL(strstr(src, ":"));
  free(src);
  uop_unref(m); uop_unref(b); uop_unref(a);
  free(r);
}

TEST(test_renderer_cstyle_custom_formatting) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* a = uop_const(dtypes.int32, 1);
  UOp* b = uop_const(dtypes.int32, 2);
  UOp* srcs[] = {a,b};
  UOpArg arg = { .type = ARG_STRING };
  arg.str.s = "add({}, {})";
  UOp* c = uop_new(OPS_CUSTOM, dtypes.int32, srcs, 2, &arg, NULL);
  UOp* arr[] = {a,b,c};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "add(1, 2)"));
  free(src);
  uop_unref(c); uop_unref(b); uop_unref(a);
  free(r);
}

TEST(test_renderer_cstyle_gep_clang_bracket_index) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* scalar = uop_const(dtypes.float32, 1.0f);
  UOp* vec = uop_cast_vec(scalar, dtypes.float32, 4);
  int axes[1] = {2};
  UOp* gep = uop_gep(vec, axes, 1);
  UOp* arr[] = {scalar, vec, gep};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "[2]"));
  free(src);
  uop_unref(gep); uop_unref(vec); uop_unref(scalar);
  free(r);
}

TEST(test_renderer_cstyle_gep_opencl_swizzle) {
  Renderer* r = renderer_cstyle_opencl();
  TEST_ASSERT_NOT_NULL(r);
  UOp* scalar = uop_const(dtypes.float32, 1.0f);
  UOp* vec = uop_cast_vec(scalar, dtypes.float32, 4);
  int axes[1] = {1};
  UOp* gep = uop_gep(vec, axes, 1);
  UOp* arr[] = {scalar, vec, gep};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, ".y"));
  free(src);
  uop_unref(gep); uop_unref(vec); uop_unref(scalar);
  free(r);
}

TEST(test_renderer_cstyle_cmplt_operator) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* a = uop_const(dtypes.int32, 1);
  UOp* b = uop_const(dtypes.int32, 2);
  UOp* lt = uop_new(OPS_CMPLT, dtypes.bool_, (UOp*[]){a,b}, 2, NULL, NULL);
  UOp* arr[] = {a,b,lt};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "<"));
  free(src);
  uop_unref(lt); uop_unref(b); uop_unref(a);
  free(r);
}

TEST_MAIN()
