#include "test_common.h"
#include "renderer/llvmir.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

TEST(test_llvmir_amd_define_local_addrspace) {
  Renderer* r = renderer_llvm_amd("gfx942");
  UOp* loc = uop_define_local(dtypes.float32, 4);
  UOp* arr[] = {loc};
  char* src = r->render(r, arr, 1);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "addrspace(3) global"));
  TEST_ASSERT_NOT_NULL(strstr(src, "addrspacecast"));
  free(src); uop_unref(loc); free(r);
}

TEST(test_llvmir_min_signature) {
  Renderer* r = renderer_llvm_generic();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 4);
  UOp* bidx = uop_index(buf, idx);
  UOp* v = uop_load(bidx, dtypes.float32);
  UOp* arr[] = {buf, idx, bidx, v};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "define void @kernel_main("));
  // parameter list contains data0 pointer
  TEST_ASSERT_NOT_NULL(strstr(src, "data0"));
  TEST_ASSERT_NOT_NULL(strstr(src, "ret void"));
  free(src); uop_unref(v); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_llvmir_load_store_gep) {
  Renderer* r = renderer_llvm_generic();
  UOp* buf = uop_define_global(dtypes.float32, 0);
  UOp* idx = uop_const(dtypes.int32, 2);
  UOp* bidx = uop_index(buf, idx);
  UOp* v = uop_load(bidx, dtypes.float32);
  UOp* st = uop_store(bidx, v);
  UOp* arr[] = {buf, idx, bidx, v, st};
  char* src = r->render(r, arr, 5);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "getelementptr"));
  TEST_ASSERT_NOT_NULL(strstr(src, "load float"));
  TEST_ASSERT_NOT_NULL(strstr(src, "store float"));
  free(src); uop_unref(st); uop_unref(v); uop_unref(bidx); uop_unref(idx); uop_unref(buf); free(r);
}

TEST(test_llvmir_alu_and_cmps) {
  Renderer* r = renderer_llvm_generic();
  // float add
  UOp* fa = uop_const(dtypes.float32, 1.0);
  UOp* fb = uop_const(dtypes.float32, 2.0);
  UOp* fsum = uop_add(fa, fb);
  // int add and cmp
  UOp* ia = uop_const(dtypes.int32, 3);
  UOp* ib = uop_const(dtypes.int32, 4);
  UOp* isum = uop_add(ia, ib);
  UOp* ilt = uop_lt(ia, ib);
  // float cmp
  UOp* flt = uop_lt(fa, fb);
  UOp* arr[] = {fa, fb, fsum, ia, ib, isum, ilt, flt};
  char* src = r->render(r, arr, 8);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "fadd"));
  TEST_ASSERT_NOT_NULL(strstr(src, "add i32"));
  TEST_ASSERT_NOT_NULL(strstr(src, "icmp slt"));
  TEST_ASSERT_NOT_NULL(strstr(src, "fcmp olt"));
  free(src);
  uop_unref(flt); uop_unref(ilt); uop_unref(isum); uop_unref(ib); uop_unref(ia); uop_unref(fsum); uop_unref(fb); uop_unref(fa);
  free(r);
}

TEST(test_llvmir_div_shift_casts_ctrlflow) {
  Renderer* r = renderer_llvm_generic();
  // int div and shifts
  UOp* a = uop_const(dtypes.int32, 8);
  UOp* b = uop_const(dtypes.int32, 2);
  UOp* d = uop_div(a, b);           // sdiv
  UOp* shl = uop_shl(a, b);
  UOp* shr = uop_shr(a, b);         // ashr
  // float div
  UOp* fa = uop_const(dtypes.float32, 4.0);
  UOp* fb = uop_const(dtypes.float32, 2.0);
  UOp* fd = uop_div(fa, fb);        // fdiv
  // casts
  UOp* itof = uop_cast(a, dtypes.float32); // sitofp
  UOp* ftoi = uop_cast(fa, dtypes.int32);  // fptosi
  UOp* trunc8 = uop_cast(a, dtypes.int8);  // trunc
  UOp* ext32 = uop_cast(trunc8, dtypes.int32); // sext
  UOp* bc = uop_bitcast(a, dtypes.float32);    // bitcast
  // control flow: IF + RANGE
  UOp* cond = uop_lt(a, b);
  UOp* If = uop_new(OPS_IF, dtypes.void_, &cond, 1, NULL, NULL);
  UOp* EndIf = uop_new(OPS_ENDIF, dtypes.void_, NULL, 0, NULL, NULL);
  UOp* rngN = uop_const(dtypes.int32, 3);
  UOp* R = uop_range(rngN, 0);
  UOp* ER = uop_new(OPS_ENDRANGE, dtypes.void_, NULL, 0, NULL, NULL);
  UOp* arr[] = {a,b,d,shl,shr,fa,fb,fd,itof,ftoi,trunc8,ext32,bc,cond,If,EndIf,R,ER};
  char* src = r->render(r, arr, (int)(sizeof(arr)/sizeof(arr[0])));
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "sdiv"));
  TEST_ASSERT_NOT_NULL(strstr(src, "fdiv"));
  TEST_ASSERT_NOT_NULL(strstr(src, "shl"));
  TEST_ASSERT_NOT_NULL(strstr(src, "ashr"));
  TEST_ASSERT_NOT_NULL(strstr(src, "sitofp"));
  TEST_ASSERT_NOT_NULL(strstr(src, "fptosi"));
  TEST_ASSERT_NOT_NULL(strstr(src, "trunc"));
  TEST_ASSERT_NOT_NULL(strstr(src, "sext"));
  TEST_ASSERT_NOT_NULL(strstr(src, "bitcast"));
  TEST_ASSERT_NOT_NULL(strstr(src, "br i1"));
  TEST_ASSERT_NOT_NULL(strstr(src, "phi i32"));
  TEST_ASSERT_NOT_NULL(strstr(src, "br label %loop"));
  free(src);
  // cleanup
  uop_unref(ER); uop_unref(R); uop_unref(EndIf); uop_unref(If); uop_unref(cond);
  uop_unref(bc); uop_unref(ext32); uop_unref(trunc8); uop_unref(ftoi); uop_unref(itof);
  uop_unref(fd); uop_unref(fb); uop_unref(fa); uop_unref(shr); uop_unref(shl); uop_unref(d); uop_unref(b); uop_unref(a);
  free(r);
}

TEST(test_llvmir_wmma_mfma_suffixes) {
  Renderer* r = renderer_llvm_amd("gfx942");
  // Construct a minimal WMMA: a: <16 x half>, b: <16 x half>, acc: <8 x float>
  DType a_dt = dtype_vec(&dtypes.float16, 16);
  DType b_dt = dtype_vec(&dtypes.float16, 16);
  DType c_dt = dtype_vec(&dtypes.float32, 8);
  UOp* a = uop_vconst(a_dt, NULL, 0);
  UOp* b = uop_vconst(b_dt, NULL, 0);
  UOp* acc = uop_vconst(c_dt, NULL, 0);
  int first[1] = {0}; int second[1] = {0};
  UOp* w = uop_wmma(a, b, acc, first, second, 1);
  UOp* arr[] = {a,b,acc,w};
  char* src = r->render(r, arr, 4);
  TEST_ASSERT_NOT_NULL(src);
  TEST_ASSERT_NOT_NULL(strstr(src, "llvm.amdgcn.mfma.f32.16x16x16.f16"));
  free(src); uop_unref(w); uop_unref(acc); uop_unref(b); uop_unref(a); free(r);
}

TEST_MAIN()

