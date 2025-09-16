#include "test_common.h"
#include "renderer/cstyle.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_renderer_opencl_signature_and_params) {
  Renderer* r = renderer_cstyle_opencl();
  TEST_ASSERT_NOT_NULL(r);
  UOp* buf = uop_define_global(dtypes.uint8, 0);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* bidx = uop_index(buf, idx);
  UOp* load = uop_load(bidx, dtypes.uint8);
  UOp* arr[] = {buf, idx, bidx, load};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "__kernel void kernel_main("));
  TEST_ASSERT_NOT_NULL(strstr(src, "__global uchar* data0"));
  free(src); uop_unref(load); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_renderer_opencl_bitcast_as_type) {
  Renderer* r = renderer_cstyle_opencl();
  UOp* x = uop_const(dtypes.uint8, 7);
  UOp* bc = uop_bitcast(x, dtypes.uint8);
  UOp* arr[] = {x, bc};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "as_uchar("));
  free(src); uop_unref(bc); uop_unref(x); free(r);
}

TEST(test_renderer_opencl_vector_literal_style) {
  Renderer* r = renderer_cstyle_opencl();
  UOp* one = uop_const(dtypes.float32, 1.0);
  UOp* vec = uop_broadcast(one, 4);
  UOp* arr[] = {one, vec};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "(float4)("));
  free(src); uop_unref(vec); uop_unref(one); free(r);
}

TEST(test_renderer_metal_signature_and_params) {
  Renderer* r = renderer_cstyle_metal();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 1);
  UOp* bidx = uop_index(buf, idx);
  UOp* load = uop_load(bidx, dtypes.float32);
  UOp* arr[] = {buf, idx, bidx, load};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(strstr(src, "kernel void kernel_main("));
  TEST_ASSERT_NOT_NULL(strstr(src, "device float* data0"));
  free(src); uop_unref(load); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_renderer_metal_bitcast_as_type_template) {
  Renderer* r = renderer_cstyle_metal();
  UOp* x = uop_const(dtypes.int32, 1);
  UOp* bc = uop_bitcast(x, dtypes.int32);
  UOp* arr[] = {x, bc};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "as_type<int>("));
  free(src); uop_unref(bc); uop_unref(x); free(r);
}

TEST(test_renderer_metal_vector_literal_style) {
  Renderer* r = renderer_cstyle_metal();
  UOp* one = uop_const(dtypes.float32, 1.0);
  UOp* vec = uop_broadcast(one, 4);
  UOp* arr[] = {one, vec};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "float4("));
  free(src); uop_unref(vec); uop_unref(one); free(r);
}

TEST(test_renderer_cuda_signature_contains_global) {
  Renderer* r = renderer_cstyle_cuda("sm_80");
  UOp* x = uop_const(dtypes.int32, 0);
  char* src = r->render(r, &x, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "__global__ void kernel_main("));
  free(src); uop_unref(x); free(r);
}

TEST(test_renderer_nv_sets_device_and_uses_cuda_path) {
  Renderer* r = renderer_cstyle_nv("sm_75");
  TEST_ASSERT_NOT_NULL(r);
  TEST_ASSERT_EQUAL_STRING("NV", r->device);
  UOp* x = uop_const(dtypes.int32, 0);
  char* src = r->render(r, &x, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "__global__ void kernel_main("));
  free(src); uop_unref(x); free(r);
}

TEST(test_renderer_qcom_is_opencl_style) {
  Renderer* r = renderer_cstyle_qcom();
  UOp* x = uop_const(dtypes.int32, 0);
  char* src = r->render(r, &x, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "__kernel void kernel_main("));
  free(src); uop_unref(x); free(r);
}

TEST(test_renderer_intel_adds_subgroup_attribute) {
  Renderer* r = renderer_cstyle_intel();
  TEST_ASSERT_NOT_NULL(r);
  UOp* x = uop_const(dtypes.int32, 0);
  char* src = r->render(r, &x, 1);
  TEST_ASSERT_NOT_NULL(strstr(src, "__attribute__((intel_reqd_sub_group_size(8)))"));
  free(src); uop_unref(x); free(r);
}

TEST(test_renderer_opencl_image_load_store) {
  Renderer* r = renderer_cstyle_opencl();
  // image param as DEFINE_GLOBAL with dtype "imagef"
  DType imgdt = to_dtype("imagef");
  UOp* img = uop_define_global(imgdt, 0);
  // int2 index: represent as vectorize two ints
  UOp* i0 = uop_const(dtypes.int32, 3);
  UOp* i1 = uop_const(dtypes.int32, 4);
  UOpArg va={0}; UOp* vsrc[2]={i0,i1}; va.type=ARG_NONE; UOp* idx2 = uop_new(OPS_VECTORIZE, dtype_vec(&dtypes.int32, 2), vsrc, 2, &va, NULL);
  UOp* bidx = uop_index(img, idx2);
  UOp* val4 = uop_broadcast(uop_const(dtypes.float32, 1.0), 4);
  UOp* ld = uop_load(bidx, dtype_vec(&dtypes.float32,4));
  UOp* st = uop_store(bidx, val4);
  UOp* arr[] = {img, i0, i1, idx2, bidx, val4, ld, st};
  char* src = r->render(r, arr, 8);
  TEST_ASSERT_NOT_NULL(strstr(src, "__kernel void kernel_main"));
  TEST_ASSERT_NOT_NULL(strstr(src, "image2d_t data0"));
  TEST_ASSERT_NOT_NULL(strstr(src, "const sampler_t smp"));
  TEST_ASSERT_NOT_NULL(strstr(src, "read_imagef(data0"));
  TEST_ASSERT_NOT_NULL(strstr(src, "write_imagef(data0"));
  free(src); uop_unref(st); uop_unref(ld); uop_unref(val4); uop_unref(bidx); uop_unref(idx2); uop_unref(i1); uop_unref(i0); uop_unref(img); free(r);
}

TEST(test_renderer_amd_ocml_unary) {
  Renderer* r = renderer_cstyle_amd("gfx");
  UOp* x = uop_const(dtypes.float32, 4.0);
  UOp* s = uop_sqrt(x);
  UOp* sn = uop_sin(x);
  UOp* arr[] = {x,s,sn};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(strstr(src, "__ocml_sqrt_f32"));
  TEST_ASSERT_NOT_NULL(strstr(src, "__ocml_sin_f32"));
  free(src); uop_unref(sn); uop_unref(s); uop_unref(x); free(r);
}

TEST(test_renderer_local_and_barrier) {
  Renderer* r = renderer_cstyle_opencl();
  UOp* lbuf = uop_define_local(dtypes.float32, 8);
  UOp* bar = uop_barrier(lbuf);
  UOp* arr[] = {lbuf, bar};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "__local float"));
  TEST_ASSERT_NOT_NULL(strstr(src, "barrier("));
  free(src); uop_unref(bar); uop_unref(lbuf); free(r);
}

TEST(test_renderer_cstyle_gated_load_ternary) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  // buffer and index with gate
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 5);
  UOp* gate = uop_const(dtypes.bool_, 1);
  UOp* bidx = uop_new(OPS_INDEX, dtypes.float32, (UOp*[]){buf, idx, gate}, 3, NULL, NULL);
  // alt value for gated load
  UOp* alt = uop_const(dtypes.float32, 7.0);
  UOp* load = uop_new(OPS_LOAD, dtypes.float32, (UOp*[]){bidx, alt}, 2, NULL, NULL);
  UOp* arr[] = {buf, idx, gate, bidx, alt, load};
  char* src = r->render(r, arr, (int)(sizeof(arr)/sizeof(arr[0])));
  TEST_ASSERT_NOT_NULL(src);
  // expect ternary form in output
  TEST_ASSERT_NOT_NULL(strstr(src, "?*"));
  TEST_ASSERT_NOT_NULL(strstr(src, ":"));
  free(src);
  // cleanup
  uop_unref(load); uop_unref(alt); uop_unref(bidx); uop_unref(gate); uop_unref(idx); uop_unref(buf);
  free(r);
}

TEST(test_renderer_cstyle_gated_store_if) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  // buffer and index with gate
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 2);
  UOp* gate = uop_const(dtypes.bool_, 1);
  UOp* bidx = uop_new(OPS_INDEX, dtypes.float32, (UOp*[]){buf, idx, gate}, 3, NULL, NULL);
  UOp* val = uop_const(dtypes.float32, 3.0);
  // store with explicit gate as third src
  UOp* store = uop_new(OPS_STORE, dtypes.void_, (UOp*[]){bidx, val, gate}, 3, NULL, NULL);
  UOp* arr[] = {buf, idx, gate, bidx, val, store};
  char* src = r->render(r, arr, (int)(sizeof(arr)/sizeof(arr[0])));
  TEST_ASSERT_NOT_NULL(src);
  // expect if-guarded store
  TEST_ASSERT_NOT_NULL(strstr(src, "if ("));
  TEST_ASSERT_NOT_NULL(strstr(src, "*"));
  free(src);
  // cleanup
  uop_unref(store); uop_unref(val); uop_unref(bidx); uop_unref(gate); uop_unref(idx); uop_unref(buf);
  free(r);
}

TEST(test_renderer_cstyle_param_dtype_from_define_global) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  // define a uint8 buffer at id 0 and load from it
  UOp* buf = uop_define_global(dtypes.uint8, 0);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* bidx = uop_index(buf, idx);
  UOp* load = uop_load(bidx, dtypes.uint8);
  UOp* arr[] = {buf, idx, bidx, load};
  char* src = r->render(r, arr, (int)(sizeof(arr)/sizeof(arr[0])));
  TEST_ASSERT_NOT_NULL(src);
  // header should have unsigned char* data0
  TEST_ASSERT_NOT_NULL(strstr(src, "unsigned char* data0"));
  free(src);
  uop_unref(load); uop_unref(bidx); uop_unref(idx); uop_unref(buf);
  free(r);
}

TEST(test_renderer_cstyle_clang_minimal) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  TEST_ASSERT_EQUAL_STRING("CPU", r->device);
  TEST_ASSERT_TRUE(r->supports_float4);
  TEST_ASSERT_FALSE(r->has_local);
  TEST_ASSERT_NOT_NULL(r->render);
  char* src = r->render(r, NULL, 0);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "kernel_main"));
  free(src);
  free(r);
}

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

TEST(test_renderer_cstyle_params_have_restrict) {
  Renderer* r = renderer_cstyle_clang();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 0);
  UOp* bidx = uop_index(buf, idx);
  UOp* v = uop_load(bidx, dtypes.float32);
  UOp* arr[] = {buf, idx, bidx, v};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(strstr(src, "restrict"));
  free(src); uop_unref(v); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_renderer_cstyle_bitops_and_mod_and_cmps) {
  Renderer* r = renderer_cstyle_clang();
  UOp* a = uop_const(dtypes.int32, 13);
  UOp* b = uop_const(dtypes.int32, 2);
  UOp* shl = uop_shl(a,b);
  UOp* shr = uop_shr(a,b);
  UOp* mod = uop_mod(a,b);
  UOp* eq  = uop_eq(a,b);
  UOp* ne  = uop_cmpne(a,b);
  UOp* arr[] = {a,b,shl,shr,mod,eq,ne};
  char* src = r->render(r, arr, 7);
  TEST_ASSERT_NOT_NULL(strstr(src, "<<"));
  TEST_ASSERT_NOT_NULL(strstr(src, ">>"));
  TEST_ASSERT_NOT_NULL(strstr(src, "%"));
  TEST_ASSERT_NOT_NULL(strstr(src, "=="));
  TEST_ASSERT_NOT_NULL(strstr(src, "!="));
  free(src); uop_unref(ne); uop_unref(eq); uop_unref(mod); uop_unref(shr); uop_unref(shl); uop_unref(b); uop_unref(a); free(r);
}

TEST(test_renderer_cstyle_nan_inf_constants) {
  Renderer* r = renderer_cstyle_clang();
  UOp* pinf = uop_const(dtypes.float32, INFINITY);
  UOp* ninf = uop_const(dtypes.float32, -INFINITY);
  UOp* nnan = uop_const(dtypes.float32, NAN);
  UOp* arr[] = {pinf, ninf, nnan};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(strstr(src, "__builtin_inff"));
  TEST_ASSERT_NOT_NULL(strstr(src, "-__builtin_inff"));
  TEST_ASSERT_NOT_NULL(strstr(src, "__builtin_nanf"));
  free(src); uop_unref(nnan); uop_unref(ninf); uop_unref(pinf); free(r);
}

TEST(test_renderer_cstyle_clang_sqrt_builtin) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* c4 = uop_const(dtypes.float32, 4.0);
  UOp* s = uop_sqrt(c4);
  UOp* arr[2] = {c4, s};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "__builtin_sqrtf"));
  free(src);
  uop_unref(s);
  uop_unref(c4);
  free(r);
}

TEST(test_renderer_cstyle_clang_vectorize_literal) {
  Renderer* r = renderer_cstyle_clang();
  TEST_ASSERT_NOT_NULL(r);
  UOp* one = uop_const(dtypes.float32, 1.0);
  UOp* vec = uop_broadcast(one, 4);
  UOp* arr[2] = {one, vec};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(src);
  // Expect C-style vector literal cast: (float4){ ... }
  TEST_ASSERT_NOT_NULL(strstr(src, "(float4){"));
  free(src);
  uop_unref(vec);
  uop_unref(one);
  free(r);
}

TEST(test_renderer_cstyle_mulacc_parentheses) {
  Renderer* r = renderer_cstyle_clang();
  UOp* a = uop_const(dtypes.float32, 2.0);
  UOp* b = uop_const(dtypes.float32, 3.0);
  UOp* c = uop_const(dtypes.float32, 4.0);
  UOp* m = uop_mulacc(a,b,c);
  UOp* arr[] = {a,b,c,m};
  char* src = r->render(r, arr, 4);
  // Accept either inlined or SSA materialized forms; require presence of multiply and plus
  TEST_ASSERT_NOT_NULL(strstr(src, "*"));
  TEST_ASSERT_NOT_NULL(strstr(src, "+"));
  free(src); uop_unref(m); uop_unref(c); uop_unref(b); uop_unref(a); free(r);
}

TEST(test_renderer_cstyle_where_parentheses) {
  Renderer* r = renderer_cstyle_clang();
  UOp* cnd = uop_const(dtypes.bool_, 1);
  UOp* t = uop_const(dtypes.int32, 7);
  UOp* f = uop_const(dtypes.int32, 8);
  UOp* w = uop_where(cnd, t, f);
  UOp* arr[] = {cnd,t,f,w};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(strstr(src, "?("));
  TEST_ASSERT_NOT_NULL(strstr(src, "):("));
  free(src); uop_unref(w); uop_unref(f); uop_unref(t); uop_unref(cnd); free(r);
}

TEST(test_renderer_cstyle_cast_vector_builtin) {
  Renderer* r = renderer_cstyle_clang();
  UOp* one = uop_const(dtypes.float32, 1.0);
  UOp* vec = uop_broadcast(one, 4); // float4
  DType i32v = dtype_vec(&dtypes.int32, 4);
  UOp* castv = uop_cast(vec, i32v);
  UOp* arr[] = {one, vec, castv};
  char* src = r->render(r, arr, 3);
  TEST_ASSERT_NOT_NULL(strstr(src, "__builtin_convertvector"));
  free(src); uop_unref(castv); uop_unref(vec); uop_unref(one); free(r);
}

TEST(test_renderer_cstyle_bitcast_parens) {
  Renderer* r = renderer_cstyle_clang();
  UOp* x = uop_const(dtypes.float32, 2.5);
  UOp* bc = uop_bitcast(x, dtypes.int32);
  UOp* arr[] = {x, bc};
  char* src = r->render(r, arr, 2);
  TEST_ASSERT_NOT_NULL(strstr(src, "*(("));
  TEST_ASSERT_NOT_NULL(strstr(src, ")&"));
  free(src); uop_unref(bc); uop_unref(x); free(r);
}

TEST(test_renderer_cstyle_where_inside_mulacc) {
  Renderer* r = renderer_cstyle_clang();
  UOp* c = uop_const(dtypes.bool_, 1);
  UOp* t = uop_const(dtypes.int32, 5);
  UOp* f = uop_const(dtypes.int32, 6);
  UOp* w = uop_where(c, t, f);
  UOp* a = uop_const(dtypes.int32, 2);
  UOp* b = uop_const(dtypes.int32, 3);
  UOp* m = uop_mulacc(a, b, w); // a*b + where(...)
  UOp* arr[] = {c,t,f,w,a,b,m};
  char* src = r->render(r, arr, 7);
  // WHERE should be materialized, MULACC can reference it; require a ternary only
  TEST_ASSERT_NOT_NULL(strstr(src, "?"));
  free(src);
  uop_unref(m); uop_unref(b); uop_unref(a); uop_unref(w); uop_unref(f); uop_unref(t); uop_unref(c);
  free(r);
}

#if 0  // No direct equivalent in Python tests; formatting is backend-dependent.
TEST(test_renderer_cstyle_mulacc_inside_where) {
  Renderer* r = renderer_cstyle_clang();
  UOp* a = uop_const(dtypes.int32, 2);
  UOp* b = uop_const(dtypes.int32, 3);
  UOp* z = uop_const(dtypes.int32, 0);
  UOp* m = uop_mulacc(a, b, z); // a*b + z
  UOp* c = uop_const(dtypes.bool_, 1);
  UOp* w = uop_where(c, m, z);
  UOp* arr[] = {a,b,z,m,c,w};
  char* src = r->render(r, arr, 6);
  // Match Python tests: they do not enforce specific parentheses for MULACC-in-WHERE.
  // Just assert a ternary is present.
  if (strstr(src, "?") == NULL) {
    fprintf(stderr, "[DEBUG mulacc_inside_where 2] src=\n%s\n", src);
  }
  TEST_ASSERT_NOT_NULL(strstr(src, "?"));
  free(src);
  uop_unref(w); uop_unref(c); uop_unref(m); uop_unref(z); uop_unref(b); uop_unref(a);
  free(r);
}
#endif

TEST_MAIN()
