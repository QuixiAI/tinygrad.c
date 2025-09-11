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

// Build PatternMatcher
PatternMatcher* get_late_rewrite_patterns(const Ops* available_ops, size_t ops_count, bool force_transcendental) {
    PatternMatch* entries = NULL; size_t m=0, cap=16; entries=(PatternMatch*)malloc(sizeof(PatternMatch)*cap);

    // helper macro
    #define ADD_ENTRY(PAT, CB) do { if (m>=cap){ cap*=2; entries=(PatternMatch*)realloc(entries,sizeof(PatternMatch)*cap);} entries[m].pattern=(PAT); entries[m].callback=(CB); entries[m].user_data=NULL; m++; } while(0)

    // Transcendentals
    if (force_transcendental || !ops_has(available_ops, ops_count, OPS_EXP2)) { UPat* p=upat_op(OPS_EXP2, NULL, 0); ADD_ENTRY(p, cb_to_xexp2); }
    if (force_transcendental || !ops_has(available_ops, ops_count, OPS_LOG2)) { UPat* p=upat_op(OPS_LOG2, NULL, 0); ADD_ENTRY(p, cb_to_xlog2); }
    if (force_transcendental || !ops_has(available_ops, ops_count, OPS_SIN))  { UPat* p=upat_op(OPS_SIN,  NULL, 0); ADD_ENTRY(p, cb_to_xsin); }
    // SQRT->xpow if SQRT not present
    if (!ops_has(available_ops, ops_count, OPS_SQRT)) { UPat* p=upat_op(OPS_SQRT, NULL, 0); ADD_ENTRY(p, cb_sqrt_to_xpow); }
    // MOD->AND
    if (ops_has(available_ops, ops_count, OPS_AND)) { UPat* p=upat_op(OPS_MOD, NULL, 0); ADD_ENTRY(p, cb_mod_to_and); }
    // MUL->SHL
    if (ops_has(available_ops, ops_count, OPS_SHL)) { UPat* p=upat_op(OPS_MUL, NULL, 0); ADD_ENTRY(p, cb_mul_to_shl); }
    // IDIV->SHR (uint, int)
    if (ops_has(available_ops, ops_count, OPS_SHR)) {
        UPat* pidu = upat_op(OPS_IDIV, NULL, 0); ADD_ENTRY(pidu, cb_idiv_uint_to_shr);
        UPat* pid  = upat_op(OPS_IDIV, NULL, 0); ADD_ENTRY(pid,  cb_idiv_sint_to_shr);
        // fast_idiv patterns (if not disabled)
        const char* dis = tg_getenv("DISABLE_FAST_IDIV");
        if (!(dis && *dis)) {
            UPat* pfi = upat_op(OPS_IDIV, NULL, 0); ADD_ENTRY(pfi, cb_fast_idiv);
            UPat* pfm = upat_op(OPS_MOD,  NULL, 0); ADD_ENTRY(pfm, cb_fast_mod);
        }
    }
    // NEG and SUB rewrites
    if (ops_has(available_ops, ops_count, OPS_NEG)) { UPat* p=upat_op(OPS_MUL, NULL, 0); ADD_ENTRY(p, cb_neg_one); }
    if (ops_has(available_ops, ops_count, OPS_SUB)) { UPat* p=upat_op(OPS_ADD, NULL, 0); ADD_ENTRY(p, cb_neg_sub); }

    PatternMatcher* pm = pattern_matcher_new(entries, m, false);
    free(entries);
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
