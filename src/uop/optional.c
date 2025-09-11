/* optional.c - Faithful port of reference/tinygrad/uop/optional.py */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "uop/uop.h"
#include "dtype/dtype.h"
#include "helpers/helpers.h"
#include "uop/optional.h"

// transcendental functions
extern UOp* transcendental_xexp2(UOp* x);
extern UOp* transcendental_xlog2(UOp* x);
extern UOp* transcendental_xsin(UOp* x, bool fast, float switch_over);
extern UOp* transcendental_xpow(UOp* base, UOp* exponent);
extern UOp* transcendental_fast_idiv(const char* device, UOp* x, int d);

// Utilities
static bool ops_has(const Ops* ops, size_t n, Ops op){ for (size_t i=0;i<n;i++) if (ops[i]==op) return true; return false; }
static bool is_power_of_two_double(double x, int* out_shift){ if (x<=0) return false; double y; double ip = modf(log2(x), &y); if (fabs(ip) < 1e-12){ if(out_shift) *out_shift=(int)llround(y); return true; } return false; }

// Callbacks (PatternMatch -> callback)
static void* cb_to_xexp2(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->src_count!=1) return NULL; return transcendental_xexp2(n->src[0]); }
static void* cb_to_xlog2(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->src_count!=1) return NULL; return transcendental_xlog2(n->src[0]); }
static void* cb_to_xsin(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->src_count!=1) return NULL; return transcendental_xsin(n->src[0], false, 10000.0f); }
static void* cb_sqrt_to_xpow(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->src_count!=1) return NULL; UOp* half = uop_const_like(n->src[0], 0.5); return transcendental_xpow(n->src[0], half); }

static void* cb_mod_to_and(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->src_count!=2) return NULL; UOp* x=n->src[0]; UOp* c=n->src[1]; if (c->arg.type!=ARG_CONST) return NULL; double v=c->arg.const_data.const_value; int sh=0; if (!is_power_of_two_double(v,&sh)) return NULL; UOp* cmo = uop_const(x->dtype, v-1.0); return uop_and(x, cmo); }

static void* cb_mul_to_shl(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->src_count!=2) return NULL; UOp* x=n->src[0]; UOp* c=n->src[1]; if (!dtypes_is_int(&x->dtype)) return NULL; if (c->arg.type!=ARG_CONST) return NULL; double v=c->arg.const_data.const_value; int sh=0; if (!is_power_of_two_double(v,&sh)) return NULL; UOp* y=uop_const(dtypes.uint, (double)sh); return uop_shl(x, y); }

static void* cb_idiv_uint_to_shr(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->src_count!=2) return NULL; UOp* x=n->src[0]; UOp* c=n->src[1]; if (!dtypes_is_unsigned(&x->dtype)) return NULL; if (c->arg.type!=ARG_CONST) return NULL; double v=c->arg.const_data.const_value; int sh=0; if (!is_power_of_two_double(v,&sh)) return NULL; UOp* y=uop_const(dtypes.uint, (double)sh); return uop_shr(x, y); }

static void* cb_idiv_sint_to_shr(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->src_count!=2) return NULL; UOp* x=n->src[0]; UOp* c=n->src[1]; if (!dtypes_is_int(&x->dtype)) return NULL; if (c->arg.type!=ARG_CONST) return NULL; double v=c->arg.const_data.const_value; int sh=0; if (!is_power_of_two_double(v,&sh)) return NULL; UOp* cm1 = uop_const(x->dtype, v-1.0); UOp* zero = uop_const(x->dtype, 0.0); UOp* adj = uop_where(uop_lt(x, zero), cm1, zero); UOp* y=uop_const(dtypes.uint, (double)sh); return uop_shr(uop_add(x, adj), y); }

typedef struct { const char* device; } OptionalCtx;
static void* cb_fast_idiv(void* ctx, void* node){ OptionalCtx* oc=(OptionalCtx*)ctx; UOp* n=(UOp*)node; if (n->src_count!=2) return NULL; UOp* x=n->src[0]; UOp* d=n->src[1]; if (d->arg.type!=ARG_INT && d->arg.type!=ARG_CONST) return NULL; int di = (d->arg.type==ARG_INT)? d->arg.int_data.i : (int)d->arg.const_data.const_value; return transcendental_fast_idiv(oc?oc->device:NULL, x, di); }
static void* cb_fast_mod(void* ctx, void* node){ OptionalCtx* oc=(OptionalCtx*)ctx; UOp* n=(UOp*)node; if (n->src_count!=2) return NULL; UOp* x=n->src[0]; UOp* d=n->src[1]; if (d->arg.type!=ARG_INT && d->arg.type!=ARG_CONST) return NULL; int di = (d->arg.type==ARG_INT)? d->arg.int_data.i : (int)d->arg.const_data.const_value; UOp* f = transcendental_fast_idiv(oc?oc->device:NULL, x, di); if (!f) return NULL; return uop_sub(x, uop_mul(d, f)); }

static void* cb_neg_one(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->op!=OPS_MUL || n->src_count!=2) return NULL; UOp* x=n->src[0]; UOp* y=n->src[1]; if (y->op==OPS_CONST && y->arg.type==ARG_CONST && y->arg.const_data.const_value==-1.0) return uop_neg(x); return NULL; }
static void* cb_neg_sub(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->op!=OPS_ADD || n->src_count!=2) return NULL; UOp* x=n->src[0]; UOp* y=n->src[1]; if (y->op==OPS_NEG && y->src_count==1) return uop_sub(x, y->src[0]); return NULL; }
static void* cb_mulacc_fold(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->op!=OPS_ADD || n->src_count!=2) return NULL; UOp* left=n->src[0]; UOp* right=n->src[1]; if (left->op==OPS_MUL && left->src_count==2) return uop_mulacc(left->src[0], left->src[1], right); if (right->op==OPS_MUL && right->src_count==2) return uop_mulacc(right->src[0], right->src[1], left); return NULL; }

// (c1<x & x<c2) -> x.eq(c) when c1+1==c2-1
static void* cb_range_to_eq(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->op!=OPS_AND || n->src_count!=2) return NULL; UOp* a=n->src[0]; UOp* b=n->src[1];
  if (a->op!=OPS_CMPLT || b->op!=OPS_CMPLT || a->src_count!=2 || b->src_count!=2) return NULL;
  UOp* c1=a->src[0]; UOp* x1=a->src[1]; UOp* x2=b->src[0]; UOp* c2=b->src[1];
  // match forms (c1 < x) & (x < c2)
  if (x1!=x2) return NULL;
  if (c1->op!=OPS_CONST || c2->op!=OPS_CONST) return NULL;
  // signed ints only
  DType sx = dtype_scalar(&x1->dtype); if (!dtypes_is_int(&sx)) return NULL;
  // extract integer consts
  double v1 = (c1->arg.type==ARG_CONST)? c1->arg.const_data.const_value : (double)c1->arg.int_data.i;
  double v2 = (c2->arg.type==ARG_CONST)? c2->arg.const_data.const_value : (double)c2->arg.int_data.i;
  long long i1 = (long long)llround(v1); long long i2 = (long long)llround(v2);
  if (i1+1 != i2-1) return NULL;
  long long c = i1+1;
  UOp* cconst = uop_const(x1->dtype, (double)c);
  return uop_eq(x1, cconst);
}

// Helpers
static bool is_false_const(UOp* u){ if (!u) return false; if (u->op!=OPS_CONST) return false; if (u->arg.type==ARG_CONST) return u->arg.const_data.const_value==0.0; if (u->arg.type==ARG_INT) return u->arg.int_data.i==0; return false; }
static bool is_sint_dtype(const DType* dt){ return dtypes_is_int(dt) && !dtypes_is_unsigned(dt); }

// (! (x<sint<c<sint>)) -> (c-1) < x ; (! (c<x)) -> x < (c+1)
static void* cb_not_cmplts(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->op!=OPS_CMPEQ || n->src_count<2) return NULL; UOp* cmp=n->src[0]; UOp* rhs=n->src[1]; if (!is_false_const(rhs)) return NULL; if (!cmp || cmp->op!=OPS_CMPLT || cmp->src_count<3) return NULL; UOp* a=cmp->src[0]; UOp* b=cmp->src[1];
  DType sa = dtype_scalar(&a->dtype), sb = dtype_scalar(&b->dtype);
  if (!(is_sint_dtype(&sa) && is_sint_dtype(&sb))) return NULL;
  if (a->op!=OPS_CONST && b->op!=OPS_CONST) return NULL;
  if (a->op==OPS_CONST){ // !(c < x) -> x < (c+1)
    double vc = (a->arg.type==ARG_CONST)? a->arg.const_data.const_value : (double)a->arg.int_data.i; long long ic=(long long)llround(vc); UOp* cadd = uop_const(b->dtype, (double)(ic+1)); return uop_lt(b, cadd);
  } else { // !(x < c) -> (c-1) < x
    double vc = (b->arg.type==ARG_CONST)? b->arg.const_data.const_value : (double)b->arg.int_data.i; long long ic=(long long)llround(vc); UOp* csub = uop_const(a->dtype, (double)(ic-1)); return uop_lt(csub, a);
  }
}

// (x*sints*-1 < y*sints*c) -> (y*(-c) < x)
static void* cb_cmplts_negmulmul(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->op!=OPS_CMPLT || n->src_count<3) return NULL; UOp* l=n->src[0]; UOp* r=n->src[1];
  DType sl = dtype_scalar(&l->dtype), sr = dtype_scalar(&r->dtype); if(!(is_sint_dtype(&sl) && is_sint_dtype(&sr))) return NULL;
  // left must be x * -1
  if (l->op!=OPS_MUL || l->src_count<3) return NULL; UOp* lx=l->src[0]; UOp* lm=l->src[1]; if (!(lm->op==OPS_CONST && lm->arg.type==ARG_CONST && lm->arg.const_data.const_value==-1.0)) return NULL;
  // right must be y * c (const)
  if (r->op!=OPS_MUL || r->src_count<3) return NULL; UOp* ry=r->src[0]; UOp* rc=r->src[1]; if (rc->op!=OPS_CONST) return NULL; double vc=(rc->arg.type==ARG_CONST)? rc->arg.const_data.const_value : (double)rc->arg.int_data.i;
  UOp* negc = uop_const(rc->dtype, -vc);
  UOp* ynegc = uop_mul(ry, negc);
  return uop_lt(ynegc, lx);
}

// (x*sints*-1 < c) -> (-c < x)
static void* cb_cmplts_negmulc(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->op!=OPS_CMPLT || n->src_count<3) return NULL; UOp* l=n->src[0]; UOp* r=n->src[1]; DType sl = dtype_scalar(&l->dtype), sr = dtype_scalar(&r->dtype); if(!(is_sint_dtype(&sl) && is_sint_dtype(&sr))) return NULL; if (l->op!=OPS_MUL || l->src_count<3) return NULL; UOp* lx=l->src[0]; UOp* lm=l->src[1]; if (!(lm->op==OPS_CONST && lm->arg.type==ARG_CONST && lm->arg.const_data.const_value==-1.0)) return NULL; if (r->op!=OPS_CONST) return NULL; double vc=(r->arg.type==ARG_CONST)? r->arg.const_data.const_value : (double)r->arg.int_data.i; UOp* negc = uop_const(r->dtype, -vc); return uop_lt(negc, lx); }

// (! (x.ne(y))) -> x.cmpeq(y)
static void* cb_not_ne(void* ctx, void* node){ (void)ctx; UOp* n=(UOp*)node; if (n->op!=OPS_CMPEQ || n->src_count<2) return NULL; UOp* inner=n->src[0]; UOp* rhs=n->src[1]; if (!is_false_const(rhs)) return NULL; if (!inner || inner->op!=OPS_CMPNE || inner->src_count<3) return NULL; UOp* a=inner->src[0]; UOp* b=inner->src[1]; return uop_eq(a,b); }

// Named-binding callbacks (ex)
static void* cb_not_lt_xc_ex(void* ctx, void* node, const char** names, UOp** values, size_t nbinds) {
    (void)ctx; (void)node; UOp* x=NULL; UOp* c=NULL; for (size_t i=0;i<nbinds;i++){ if (names[i] && strcmp(names[i],"x")==0) x=values[i]; if (names[i] && strcmp(names[i],"c")==0) c=values[i]; }
    if (!x||!c) return NULL; DType sx=dtype_scalar(&x->dtype); if (!dtypes_is_int(&sx)) return NULL; double vc=(c->arg.type==ARG_CONST)? c->arg.const_data.const_value : (double)c->arg.int_data.i; long long ic=(long long)llround(vc); UOp* csub=uop_const(x->dtype, (double)(ic-1)); return uop_lt(csub, x);
}
static void* cb_not_lt_cx_ex(void* ctx, void* node, const char** names, UOp** values, size_t nbinds) {
    (void)ctx; (void)node; UOp* x=NULL; UOp* c=NULL; for (size_t i=0;i<nbinds;i++){ if (names[i] && strcmp(names[i],"x")==0) x=values[i]; if (names[i] && strcmp(names[i],"c")==0) c=values[i]; }
    if (!x||!c) return NULL; DType sx=dtype_scalar(&x->dtype); if (!dtypes_is_int(&sx)) return NULL; double vc=(c->arg.type==ARG_CONST)? c->arg.const_data.const_value : (double)c->arg.int_data.i; long long ic=(long long)llround(vc); UOp* cadd=uop_const(x->dtype, (double)(ic+1)); return uop_lt(x, cadd);
}

// Build PatternMatcher
// Simple cache for built pattern matchers
typedef struct { unsigned long long key1; unsigned long long key2; bool force; PatternMatcher* pm; } PMCacheEntry;
static PMCacheEntry _pm_cache[8]; static int _pm_cache_used=0;

static unsigned long long _hash_ops(const Ops* ops, size_t n){ unsigned long long h=1469598103934665603ULL; for(size_t i=0;i<n;i++){ h ^= (unsigned long long)ops[i]; h *= 1099511628211ULL; } return h; }

PatternMatcher* get_late_rewrite_patterns(const Ops* available_ops, size_t ops_count, bool force_transcendental) {
    unsigned long long h = _hash_ops(available_ops, ops_count);
    // check cache
    for (int i=0;i<_pm_cache_used;i++) if (_pm_cache[i].key1==h && _pm_cache[i].force==force_transcendental) return _pm_cache[i].pm;

    PatternMatch* entries = NULL; size_t m=0, cap=16; entries=(PatternMatch*)malloc(sizeof(PatternMatch)*cap);

    // helper macro
    #define ADD_ENTRY(PAT, CB) do { if (m>=cap){ cap*=2; entries=(PatternMatch*)realloc(entries,sizeof(PatternMatch)*cap);} entries[m].pattern=(PAT); entries[m].callback=(CB); entries[m].callback_ex=NULL; entries[m].user_data=NULL; m++; } while(0)
    #define ADD_ENTRY_EX(PAT, CBEX) do { if (m>=cap){ cap*=2; entries=(PatternMatch*)realloc(entries,sizeof(PatternMatch)*cap);} entries[m].pattern=(PAT); entries[m].callback=NULL; entries[m].callback_ex=(CBEX); entries[m].user_data=NULL; m++; } while(0)

    // Transcendentals, only for supported float dtypes
    const DType* floats3[3] = { &dtypes.float16, &dtypes.float32, &dtypes.float64 };
    if (force_transcendental || !ops_has(available_ops, ops_count, OPS_EXP2)) { UPat* d=upat_var_named("d", floats3, 3, true); UPat* src[1]={d}; UPat* p=upat_op(OPS_EXP2, src, 1); ADD_ENTRY(p, cb_to_xexp2); }
    if (force_transcendental || !ops_has(available_ops, ops_count, OPS_LOG2)) { UPat* d=upat_var_named("d", floats3, 3, true); UPat* src[1]={d}; UPat* p=upat_op(OPS_LOG2, src, 1); ADD_ENTRY(p, cb_to_xlog2); }
    if (force_transcendental || !ops_has(available_ops, ops_count, OPS_SIN))  { UPat* d=upat_var_named("d", floats3, 3, true); UPat* src[1]={d}; UPat* p=upat_op(OPS_SIN,  src, 1); ADD_ENTRY(p, cb_to_xsin); }
    // SQRT->xpow if SQRT not present
    if (!ops_has(available_ops, ops_count, OPS_SQRT)) { UPat* d=upat_var_named("d", floats3, 3, true); UPat* src[1]={d}; UPat* p=upat_op(OPS_SQRT, src, 1); ADD_ENTRY(p, cb_sqrt_to_xpow); }
    // MOD->AND and MUL->SHL with int dtype filters
    if (ops_has(available_ops, ops_count, OPS_AND)) {
        const DType* ints8[8] = { &dtypes.int8,&dtypes.int16,&dtypes.int32,&dtypes.int64,&dtypes.uint8,&dtypes.uint16,&dtypes.uint32,&dtypes.uint64 };
        UPat* xint = upat_var_named("x", ints8, 8, false);
        UPat* cmod = upat_cvar_named("c", NULL, 0, false);
        UPat* msrc[2] = { xint, cmod }; UPat* p = upat_op(OPS_MOD, msrc, 2); ADD_ENTRY(p, cb_mod_to_and);
    }
    if (ops_has(available_ops, ops_count, OPS_SHL)) {
        const DType* ints8[8] = { &dtypes.int8,&dtypes.int16,&dtypes.int32,&dtypes.int64,&dtypes.uint8,&dtypes.uint16,&dtypes.uint32,&dtypes.uint64 };
        UPat* xim = upat_var_named("x", ints8, 8, false);
        UPat* cm = upat_cvar_named("c", NULL, 0, false);
        UPat* msrc2[2] = { xim, cm }; UPat* p2 = upat_op(OPS_MUL, msrc2, 2); ADD_ENTRY(p2, cb_mul_to_shl);
    }
    // IDIV->SHR (uint, int)
    if (ops_has(available_ops, ops_count, OPS_SHR)) {
        const DType* uints[4] = { &dtypes.uint8, &dtypes.uint16, &dtypes.uint32, &dtypes.uint64 };
        const DType* sints[4] = { &dtypes.int8, &dtypes.int16, &dtypes.int32, &dtypes.int64 };
        UPat* xu = upat_var_named("x", uints, 4, false); UPat* cu = upat_cvar_named("c", NULL, 0, false); UPat* usrc[2] = { xu, cu }; UPat* pidu = upat_op(OPS_IDIV, usrc, 2); ADD_ENTRY(pidu, cb_idiv_uint_to_shr);
        UPat* xs = upat_var_named("x", sints, 4, false); UPat* cs = upat_cvar_named("c", NULL, 0, false); UPat* ssrc2[2] = { xs, cs }; UPat* pid  = upat_op(OPS_IDIV, ssrc2, 2); ADD_ENTRY(pid,  cb_idiv_sint_to_shr);
        // fast_idiv patterns (if not disabled)
        const char* dis = tg_getenv("DISABLE_FAST_IDIV");
        if (!(dis && *dis)) {
            UPat* pfi = upat_op(OPS_IDIV, ssrc2, 2); ADD_ENTRY(pfi, cb_fast_idiv);
            UPat* pfm = upat_op(OPS_MOD,  ssrc2, 2); ADD_ENTRY(pfm, cb_fast_mod);
        }
    }
    // NEG and SUB rewrites
    if (ops_has(available_ops, ops_count, OPS_NEG)) { UPat* p=upat_op(OPS_MUL, NULL, 0); ADD_ENTRY(p, cb_neg_one); }
    if (ops_has(available_ops, ops_count, OPS_SUB)) { UPat* p=upat_op(OPS_ADD, NULL, 0); ADD_ENTRY(p, cb_neg_sub); }
    if (ops_has(available_ops, ops_count, OPS_CMPLT)) {
        // named range-to-eq: (c1 < x) & (x < c2)
        const DType* sints[4] = { &dtypes.int8, &dtypes.int16, &dtypes.int32, &dtypes.int64 };
        UPat* x = upat_var_named("x", sints, 4, false);
        UPat* c1 = upat_cvar_named("c1", sints, 4, false);
        UPat* c2 = upat_cvar_named("c2", sints, 4, false);
        UPat* lt_c1x_src[2] = { c1, x }; UPat* lt_c1x = upat_op(OPS_CMPLT, lt_c1x_src, 2);
        UPat* lt_xc2_src[2] = { x, c2 }; UPat* lt_xc2 = upat_op(OPS_CMPLT, lt_xc2_src, 2);
        UPat* pand = upat_and(lt_c1x, lt_xc2);
        // callback_ex for range-to-eq
        void* cb_range_to_eq_ex(void* ctx, void* node, const char** names, UOp** values, size_t nbinds) {
          (void)ctx; (void)node; UOp* xv=NULL; UOp* c1v=NULL; UOp* c2v=NULL;
          for (size_t i=0;i<nbinds;i++){ if (!names[i]) continue; if (strcmp(names[i],"x")==0) xv=values[i]; else if (strcmp(names[i],"c1")==0) c1v=values[i]; else if (strcmp(names[i],"c2")==0) c2v=values[i]; }
          if (!xv||!c1v||!c2v) return NULL; DType sx=dtype_scalar(&xv->dtype); if (!dtypes_is_int(&sx)) return NULL;
          double v1=(c1v->arg.type==ARG_CONST)? c1v->arg.const_data.const_value : (double)c1v->arg.int_data.i;
          double v2=(c2v->arg.type==ARG_CONST)? c2v->arg.const_data.const_value : (double)c2v->arg.int_data.i;
          long long i1=(long long)llround(v1), i2=(long long)llround(v2); if (i1+1 != i2-1) return NULL;
          long long c=i1+1; UOp* cconst=uop_const(xv->dtype, (double)c); return uop_eq(xv, cconst);
        }
        ADD_ENTRY_EX(pand, cb_range_to_eq_ex);
        // named captures for not (x < c)
        UPat* c = upat_cvar_named("c", sints, 4, false);
        UPat* lt_xc_src[2] = { x, c }; UPat* lt_xc = upat_op(OPS_CMPLT, lt_xc_src, 2); UPat* not_lt_xc = upat_not(lt_xc); ADD_ENTRY_EX(not_lt_xc, cb_not_lt_xc_ex);
        UPat* lt_cx_src[2] = { c, x }; UPat* lt_cx = upat_op(OPS_CMPLT, lt_cx_src, 2); UPat* not_lt_cx = upat_not(lt_cx); ADD_ENTRY_EX(not_lt_cx, cb_not_lt_cx_ex);
        // (x*-1 < y*c) and (x*-1 < c)
        UPat* m_xneg_src[2] = { x, upat_const(-1.0) }; UPat* m_xneg = upat_op(OPS_MUL, m_xneg_src, 2);
        UPat* y = upat_var_named("y", sints, 4, false);
        UPat* m_yc_src[2] = { y, c }; UPat* m_yc = upat_op(OPS_MUL, m_yc_src, 2);
        UPat* lt_muls_src[2] = { m_xneg, m_yc }; UPat* lt_muls = upat_op(OPS_CMPLT, lt_muls_src, 2);
        void* cb_cmplts_negmulmul_ex(void* ctx, void* node, const char** names, UOp** values, size_t nbinds){ (void)ctx; (void)node; UOp* xv=NULL; UOp* yv=NULL; UOp* cv=NULL; for(size_t i=0;i<nbinds;i++){ if (!names[i]) continue; if (strcmp(names[i],"x")==0) xv=values[i]; else if (strcmp(names[i],"y")==0) yv=values[i]; else if (strcmp(names[i],"c")==0) cv=values[i]; }
          if(!xv||!yv||!cv) return NULL; DType sx=dtype_scalar(&xv->dtype); if(!dtypes_is_int(&sx)) return NULL; double vc=(cv->arg.type==ARG_CONST)? cv->arg.const_data.const_value : (double)cv->arg.int_data.i; UOp* negc=uop_const(cv->dtype, -vc); UOp* ynegc=uop_mul(yv, negc); return uop_lt(ynegc, xv); }
        ADD_ENTRY_EX(lt_muls, cb_cmplts_negmulmul_ex);
        UPat* lt_mulc_src[2] = { m_xneg, c }; UPat* lt_mulc = upat_op(OPS_CMPLT, lt_mulc_src, 2);
        void* cb_cmplts_negmulc_ex(void* ctx, void* node, const char** names, UOp** values, size_t nbinds){ (void)ctx; (void)node; UOp* xv=NULL; UOp* cv=NULL; for(size_t i=0;i<nbinds;i++){ if (!names[i]) continue; if (strcmp(names[i],"x")==0) xv=values[i]; else if (strcmp(names[i],"c")==0) cv=values[i]; }
          if(!xv||!cv) return NULL; DType sx=dtype_scalar(&xv->dtype); if(!dtypes_is_int(&sx)) return NULL; double vc=(cv->arg.type==ARG_CONST)? cv->arg.const_data.const_value : (double)cv->arg.int_data.i; UOp* negc=uop_const(cv->dtype, -vc); return uop_lt(negc, xv); }
        ADD_ENTRY_EX(lt_mulc, cb_cmplts_negmulc_ex);
    }
    if (ops_has(available_ops, ops_count, OPS_CMPEQ)) {
        // (! (x.ne(y))) -> x.cmpeq(y)
        UPat* xeq = upat_var_named("x", NULL, 0, false);
        UPat* yeq = upat_var_named("y", NULL, 0, false);
        UPat* ne_src[2] = { xeq, yeq }; UPat* ne_xy = upat_op(OPS_CMPNE, ne_src, 2);
        UPat* not_ne = upat_not(ne_xy);
        void* cb_not_ne_ex(void* ctx, void* node, const char** names, UOp** values, size_t nbinds){ (void)ctx; (void)node; UOp* xv=NULL; UOp* yv=NULL; for(size_t i=0;i<nbinds;i++){ if (!names[i]) continue; if (strcmp(names[i],"x")==0) xv=values[i]; else if (strcmp(names[i],"y")==0) yv=values[i]; } if(!xv||!yv) return NULL; return uop_eq(xv,yv); }
        ADD_ENTRY_EX(not_ne, cb_not_ne_ex);
    }

    // MULACC folding
    if (ops_has(available_ops, ops_count, OPS_MULACC)) { UPat* p=upat_op(OPS_ADD, NULL, 0); ADD_ENTRY(p, cb_mulacc_fold); }

    PatternMatcher* pm = pattern_matcher_new(entries, m, false);
    free(entries);
    // add to cache (simple FIFO replace)
    if (_pm_cache_used < 8) { _pm_cache[_pm_cache_used++] = (PMCacheEntry){ .key1=h, .key2=0, .force=force_transcendental, .pm=pm }; }
    else { _pm_cache[_pm_cache_used%8] = (PMCacheEntry){ .key1=h, .key2=0, .force=force_transcendental, .pm=pm }; _pm_cache_used++; }
    return pm;
    #undef ADD_ENTRY
}

static UOp* optional_apply_node(UOp* node, PatternMatcher* pm, OptionalCtx* ctx){
    if (!node || !pm) return node;
    // rewrite children first
    for (size_t i=0;i<node->src_count;i++) node->src[i] = optional_apply_node(node->src[i], pm, ctx);
    void* repl=NULL; if (pattern_matcher_apply(pm, node, ctx, &repl)==PM_OK && repl) return (UOp*)repl; return node;
}

UOp* optional_apply_patterns_ex(UOp* root, PatternMatcher* pm, const char* device) {
    OptionalCtx ctx = { .device = device };
    return optional_apply_node(root, pm, &ctx);
}
