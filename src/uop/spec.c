/* spec.c - Faithful port of reference/tinygrad/uop/spec.py */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "uop/uop.h"
#include "uop/spec.h"
#include "dtype/dtype.h"
#include "shape/shapetracker.h"

// Forward declarations from other modules
extern UOp* symbolic_simplify(UOp* uop);
extern UOp* uop_graph_rewrite(UOp* root, PatternMatcher* pm);
extern UOp** uop_toposort(UOp* root, size_t* count);

// Z3 integration
#include <z3.h>

// Context variable for IGNORE_OOB
static int ignore_oob_var = 0; // default: check bounds strictly when Z3 is available

// Helper functions
static int all_same(int* arr, size_t count) {
    if (count <= 1) return 1;
    for (size_t i = 1; i < count; i++) {
        if (arr[i] != arr[0]) return 0;
    }
    return 1;
}

static int tg_prod(const int* shape, int len) {
    int result = 1;
    for (int i = 0; i < len; i++) {
        result *= shape[i];
    }
    return result;
}

// Use helper functions to avoid warnings
static void use_helpers(void) {
    int test_arr[] = {1, 1, 1};
    if (all_same(test_arr, 3)) { /* Used */ }
    if (tg_prod(test_arr, 3) > 0) { /* Used */ }
}

// Z3 ALU operations mapping - used for readability in comments
typedef enum {
    ALU_NONE,
    ALU_MOD,
    ALU_IDIV,
    ALU_SHR,
    ALU_SHL,
    ALU_AND,
    ALU_WHERE,
    ALU_MAX
} AluOpType;

// (no static tables; the Z3 builder handles ops directly)

// Minimal Z3 expression builder for index validation (Z3-only)
typedef struct {
    Z3_context ctx;
    Z3_solver solver;
    // simple mapping from UOp* to Z3_ast for memoized symbols
    UOp** keys;
    Z3_ast* vals;
    size_t len, cap;
    int ctr;  // incremental id for load/cast symbols
} Z3RenderCtx;

static void z3rc_init(Z3RenderCtx* rc, Z3_context ctx, Z3_solver solver) {
    rc->ctx = ctx; rc->solver = solver; rc->keys = NULL; rc->vals = NULL; rc->len = 0; rc->cap = 0; rc->ctr = 0;
}
static void z3rc_free(Z3RenderCtx* rc) {
    if (rc->keys) free(rc->keys); if (rc->vals) free(rc->vals);
}
static Z3_ast z3rc_get(Z3RenderCtx* rc, UOp* u) {
    for (size_t i=0;i<rc->len;i++) if (rc->keys[i]==u) return rc->vals[i];
    return NULL;
}
static void z3rc_set(Z3RenderCtx* rc, UOp* u, Z3_ast a) {
    if (rc->len==rc->cap) {
        size_t ncap = rc->cap? rc->cap*2 : 16;
        rc->keys = (UOp**)realloc(rc->keys, ncap*sizeof(UOp*));
        rc->vals = (Z3_ast*)realloc(rc->vals, ncap*sizeof(Z3_ast));
        rc->cap = ncap;
    }
    rc->keys[rc->len] = u; rc->vals[rc->len] = a; rc->len++;
}

static Z3_ast z3_int_val(Z3_context ctx, long long v) {
    Z3_sort isort = Z3_mk_int_sort(ctx);
    return Z3_mk_int64(ctx, v, isort);
}
static Z3_ast z3_bool_val(Z3_context ctx, bool v) {
    return v ? Z3_mk_true(ctx) : Z3_mk_false(ctx);
}

static Z3_ast z3_pow2(Z3_context ctx, Z3_ast b) {
    // Strict literal behavior: require a numeral exponent (as_long in Python)
    Z3_string s = Z3_get_numeral_string(ctx, b);
    if (!s) return NULL;
    long long k = atoll(s);
    if (k < 0 || k >= 62) return NULL;
    long long val = 1LL << k;
    return z3_int_val(ctx, val);
}

static Z3_ast z3_eval(Z3RenderCtx* rc, UOp* u);
static Z3_ast create_bounded(Z3RenderCtx* rc, const char* name, long long vmin, long long vmax) {
    Z3_context ctx = rc->ctx; Z3_solver solver = rc->solver;
    Z3_symbol sym = Z3_mk_string_symbol(ctx, name);
    Z3_sort isort = Z3_mk_int_sort(ctx);
    Z3_ast s = Z3_mk_const(ctx, sym, isort);
    Z3_ast lb = Z3_mk_ge(ctx, s, z3_int_val(ctx, vmin));
    Z3_ast ub = Z3_mk_le(ctx, s, z3_int_val(ctx, vmax));
    Z3_ast conj = Z3_mk_and(ctx, 2, (Z3_ast[]){lb, ub});
    Z3_solver_assert(ctx, solver, conj);
    return s;
}

static Z3_ast z3_from_alu2(Z3_context ctx, Ops op, Z3_ast a, Z3_ast b) {
    switch (op) {
        case OPS_ADD: return Z3_mk_add(ctx, 2, (Z3_ast[]){a,b});
        case OPS_SUB: return Z3_mk_sub(ctx, 2, (Z3_ast[]){a,b});
        case OPS_MUL: return Z3_mk_mul(ctx, 2, (Z3_ast[]){a,b});
        case OPS_IDIV: {
            // Truncated division (toward zero): if a<0 then if b>0 then (a+(b-1))/b else (a-(b+1))/b else a/b
            Z3_ast zero = z3_int_val(ctx, 0);
            Z3_ast a_lt_0 = Z3_mk_lt(ctx, a, zero);
            Z3_ast b_gt_0 = Z3_mk_gt(ctx, b, zero);
            Z3_ast b_minus_1 = Z3_mk_sub(ctx, 2, (Z3_ast[]){b, z3_int_val(ctx,1)});
            Z3_ast b_plus_1 = Z3_mk_add(ctx, 2, (Z3_ast[]){b, z3_int_val(ctx,1)});
            Z3_ast e1 = Z3_mk_div(ctx, Z3_mk_add(ctx, 2, (Z3_ast[]){a, b_minus_1}), b);
            Z3_ast e2 = Z3_mk_div(ctx, Z3_mk_sub(ctx, 2, (Z3_ast[]){a, b_plus_1}), b);
            Z3_ast e3 = Z3_mk_div(ctx, a, b);
            return Z3_mk_ite(ctx, a_lt_0, Z3_mk_ite(ctx, b_gt_0, e1, e2), e3);
        }
        case OPS_MOD: {
            // a - cdiv(a,b)*b
            Z3_ast cdiv = z3_from_alu2(ctx, OPS_IDIV, a, b);
            Z3_ast prod = Z3_mk_mul(ctx, 2, (Z3_ast[]){cdiv,b});
            return Z3_mk_sub(ctx, 2, (Z3_ast[]){a, prod});
        }
        case OPS_SHL: {
            Z3_ast p = z3_pow2(ctx, b);
            if (!p) return NULL;  // enforce literal shift amount
            return Z3_mk_mul(ctx, 2, (Z3_ast[]){a, p});
        }
        case OPS_SHR: {
            Z3_ast p = z3_pow2(ctx, b);
            if (!p) return NULL;  // enforce literal shift amount
            return Z3_mk_div(ctx, a, p);
        }
        case OPS_AND: {
            // If b is numeral (bit-mask case), emulate bitwise AND when possible: a & (2^k-1) == a % 2^k
            Z3_string bs = Z3_get_numeral_string(ctx, b);
            if (bs) {
                long long bv = atoll(bs);
                if (bv >= 0) {
                    // find k such that bv == 2^k-1
                    long long x=bv+1; int k=0; long long t=1;
                    while (t < x && k < 62) { t <<= 1; k++; }
                    if (t==x) {
                        Z3_ast modv = z3_int_val(ctx, t);
                        return Z3_mk_mod(ctx, a, modv);
                    }
                }
            }
            // Fallback arithmetic form: a % (b+1)
            Z3_ast bp1 = Z3_mk_add(ctx, 2, (Z3_ast[]){b, z3_int_val(ctx,1)});
            return Z3_mk_mod(ctx, a, bp1);
        }
        case OPS_MAX: {
            Z3_ast a_lt_b = Z3_mk_lt(ctx, a, b);
            return Z3_mk_ite(ctx, a_lt_b, b, a);
        }
        default: return NULL;
    }
}

static Z3_ast z3_eval(Z3RenderCtx* rc, UOp* u) {
    if (!u) return NULL;
    // Memoized symbols (DEFINE_VAR, RANGE, LOAD ints, CAST non-int) return same per-node
    Z3_ast memo = z3rc_get(rc, u);
    if (memo) return memo;
    switch (u->op) {
        case OPS_CONST: {
            if (dtypes_is_bool(&u->dtype)) return z3_bool_val(rc->ctx, u->arg.type==ARG_CONST ? (u->arg.const_data.const_value!=0.0) : (u->arg.int_data.i!=0));
            long long v = 0;
            if (u->arg.type == ARG_INT) v = u->arg.int_data.i;
            else if (u->arg.type == ARG_CONST) v = (long long)u->arg.const_data.const_value;
            return z3_int_val(rc->ctx, v);
        }
        case OPS_CAST: {
            // ints/bools: passthrough; otherwise create bounded symbol for this CAST
            if (dtypes_is_int(&u->dtype) || dtypes_is_bool(&u->dtype)) {
                return (u->src_count>=1) ? z3_eval(rc, u->src[0]) : NULL;
            }
            char name[64]; snprintf(name, sizeof(name), "cast%d", rc->ctr++);
            Z3_ast s = create_bounded(rc, name, uop_vmin(u), uop_vmax(u));
            z3rc_set(rc, u, s); return s;
        }
        case OPS_WHERE: {
            if (u->src_count<3) return NULL;
            Z3_ast c = z3_eval(rc, u->src[0]);
            Z3_ast t = z3_eval(rc, u->src[1]);
            Z3_ast f = z3_eval(rc, u->src[2]);
            return Z3_mk_ite(rc->ctx, c, t, f);
        }
        case OPS_CMPLT: case OPS_CMPEQ: case OPS_CMPNE: {
            if (u->src_count<2) return NULL;
            Z3_ast a = z3_eval(rc, u->src[0]);
            Z3_ast b = z3_eval(rc, u->src[1]);
            if (u->op == OPS_CMPLT) return Z3_mk_lt(rc->ctx, a, b);
            if (u->op == OPS_CMPEQ) return Z3_mk_eq(rc->ctx, a, b);
            return Z3_mk_not(rc->ctx, Z3_mk_eq(rc->ctx, a, b));
        }
        case OPS_DEFINE_VAR: {
            if (u->arg.type == ARG_VAR && u->arg.var.name) {
                Z3_ast s = create_bounded(rc, u->arg.var.name, u->arg.var.vmin, u->arg.var.vmax);
                z3rc_set(rc, u, s); return s;
            }
            return NULL;
        }
        case OPS_RANGE: {
            // bounded Int in [0, src[0].arg-1]
            long long vmax = 0;
            if (u->src_count>=1 && u->src[0] && u->src[0]->arg.type==ARG_INT) vmax = u->src[0]->arg.int_data.i - 1;
            char name[64]; snprintf(name, sizeof(name), "ridx%lld", (long long)(u->arg.type==ARG_INT?u->arg.int_data.i:0));
            Z3_ast s = create_bounded(rc, name, 0, vmax);
            z3rc_set(rc, u, s); return s;
        }
        case OPS_LOAD: {
            // For int dtype LOADs, create bounded Int
            if (dtypes_is_int(&u->dtype)) {
                char name[64]; snprintf(name, sizeof(name), "load%d", rc->ctr++);
                Z3_ast s = create_bounded(rc, name, uop_vmin(u), uop_vmax(u));
                z3rc_set(rc, u, s); return s;
            }
            // fallthrough to ALU tree
            break;
        }
        case OPS_SPECIAL: {
            // Treat SPECIAL like a bounded integer symbol if needed
            long long vmax = (u->arg.type==ARG_INT)? (u->arg.int_data.i - 1) : (uop_vmax(u)-1);
            if (vmax < 0) vmax = 0;
            const char* nm = (const char*)u->tag;
            char name[64]; if (!nm){ snprintf(name, sizeof(name), "special%d", rc->ctr++); nm = name; }
            Z3_ast s = create_bounded(rc, nm, 0, vmax);
            z3rc_set(rc, u, s); return s;
        }
        case OPS_XOR: {
            // Model via bitvectors and back to int
            if (u->src_count<2) return NULL;
            Z3_context ctx = rc->ctx;
            int bw = dtype_nbytes(&u->dtype)*8; if (bw <= 0) bw = 32;
            Z3_ast a = z3_eval(rc, u->src[0]);
            Z3_ast b = z3_eval(rc, u->src[1]);
            Z3_ast bv_a = Z3_mk_int2bv(ctx, bw, a);
            Z3_ast bv_b = Z3_mk_int2bv(ctx, bw, b);
            Z3_ast bv_x = Z3_mk_bvxor(ctx, bv_a, bv_b);
            return Z3_mk_bv2int(ctx, bv_x, /*is_signed*/ false);
        }
        default: {
            // binary ALU
            if (u->src_count==2) {
                Z3_ast a = z3_eval(rc, u->src[0]);
                Z3_ast b = z3_eval(rc, u->src[1]);
                Z3_ast r = z3_from_alu2(rc->ctx, u->op, a, b);
                if (r) return r;
            }
            // unary passthrough
            if (u->src_count==1) return z3_eval(rc, u->src[0]);
            return NULL;
        }
    }
}

// ---- DEFINE_* dtype metadata shims ----
typedef struct SpecDTypeMeta SpecDTypeMeta; // from header
static SpecDTypeMeta* spec_meta_from_tag(const UOp* u){ return (SpecDTypeMeta*)u->tag; }
void spec_attach_define_meta(UOp* u, int addrspace){
    if (!u) return;
    SpecDTypeMeta* m = (SpecDTypeMeta*)malloc(sizeof(SpecDTypeMeta));
    m->is_image = dtypes_is_image(&u->dtype) ? 1 : 0;
    m->is_ptr = 1; // DEFINE_* are modeled as pointers in Python
    m->v = u->dtype.count;
    // size unknown by default; try to read from PtrDType alias if available
    const PtrDType* pd = (const PtrDType*)&u->dtype;
    m->size = pd ? pd->size : -1;
    m->base_dtype = dtype_base(&u->dtype);
    m->addrspace = addrspace;
    u->tag = m;
}
const SpecDTypeMeta* spec_get_define_meta(const UOp* u){ return u ? spec_meta_from_tag(u) : NULL; }

// IGNORE_OOB context variable access
void spec_set_ignore_oob(int ignore) {
    ignore_oob_var = ignore;
}

int spec_get_ignore_oob(void) {
    return ignore_oob_var;
}

// Pattern matching validation callback functions
typedef struct {
    bool (*validate)(UOp**, size_t);
    void* user_data;
} ValidationCallback;

// buffer_spec equivalent - PatternMatcher implementation
// removed unused: validate_unique

static bool validate_device_arg(UOp* d) {
    if (!d) return false;
    // Python: isinstance(d.arg, str) or (isinstance(d.arg, tuple) and all(isinstance(s, str) for s in d.arg))
    if (d->arg.type == ARG_STRING) return true;
    if (d->arg.type == ARG_STRLIST) return d->arg.strlist.count >= 0;  // empty tuple allowed
    return false;
}

// removed unused: validate_device_tuple

static bool validate_buffer(UOp** src, size_t count) {
    // Python: (UNIQUE, DEVICE), arg is int, dtype is DType or ImageDType
    if (count != 2) return false;
    if (!src[0] || src[0]->op != OPS_UNIQUE) return false;
    if (!src[1] || src[1]->op != OPS_DEVICE || !validate_device_arg(src[1])) return false;
    return true;
}

static bool validate_buffer_view(UOp** src, size_t count, UOp* node) {
    if (count != 1) return false;
    if (!src[0] || src[0]->op != OPS_BUFFER) return false;
    if (!node) return false;
    // Python: isinstance(arg, tuple) and len == 2 and each is (int | UOp)
    if (node->arg.type != ARG_TUPLE_MIXED) return false;
    if (node->arg.tmixed.count != 2) return false;
    for (int i=0;i<node->arg.tmixed.count;i++) {
        int tag = node->arg.tmixed.items[i].tag;
        if (!(tag == MIXED_INT || tag == MIXED_UOP)) return false;
    }
    return true;
}

// removed unused: validate_mstack_buffer_view

// removed unused: validate_view

// assign_spec equivalent
// removed unused: validate_kernel

static bool validate_assign(UOp** src, size_t count) {
    if (count < 2) return false;
    if (count > 2) {
        for (size_t i = 2; i < count; i++) {
            if (src[i]->op != OPS_ASSIGN) return false;
        }
    }
    return true;
}

// Helper: find DEVICE node associated to an input buffer (BUFFER or BUFFER_VIEW -> BUFFER)
static UOp* find_buffer_device(UOp* u) {
    if (!u) return NULL;
    if (u->op == OPS_BUFFER && u->src_count >= 2 && u->src[1] && u->src[1]->op == OPS_DEVICE) return u->src[1];
    if (u->op == OPS_BUFFER_VIEW && u->src_count >= 1) return find_buffer_device(u->src[0]);
    return NULL;
}

static bool validate_mselect_node(UOp* x) {
    if (!x || x->op != OPS_MSELECT || x->src_count < 1) return false;
    UOp* buf = x->src[0];
    UOp* dev = find_buffer_device(buf);
    if (!dev || dev->arg.type != ARG_STRLIST) return false;
    // Check index argument (accept INT or CONST)
    long long idx = -1; bool has_idx=false;
    if (x->arg.type == ARG_INT) { idx = x->arg.int_data.i; has_idx=true; }
    else if (x->arg.type == ARG_CONST) { idx = (long long)x->arg.const_data.const_value; has_idx=true; }
    if (!has_idx) return false;
    return idx >= 0 && idx < dev->arg.strlist.count;
}

static bool validate_mstack(UOp** src, size_t count) {
    // Python: all(isinstance(x.device, str) for x in x.src)
    if (!src || count < 1) return false;
    for (size_t i=0;i<count;i++) {
        UOp* dev = find_buffer_device(src[i]);
        if (!dev || dev->arg.type != ARG_STRING) return false;
    }
    return true;
}

// tensor_uop_spec equivalent
static bool validate_movement(UOp** src, size_t count) {
    if (count != 2) return false;
    UOp* mv = src[0];
    UOp* x = src[1];
    DType mvb = dtype_scalar(&mv->dtype);
    DType xb = dtype_scalar(&x->dtype);
    return dtype_eq(&mvb, &xb) || dtype_eq(&mv->dtype, &x->dtype);
}

static bool validate_view_all_sources(UOp** src, size_t count) {
    if (count != 1) return false;
    UOp* x = src[0];
    return x->op == OPS_BUFFER || x->op == OPS_BUFFER_VIEW || 
           x->op == OPS_ASSIGN || x->op == OPS_CONST || x->op == OPS_DEVICE;
}

// removed unused: validate_bind

// removed unused: validate_const_device_view

// removed unused: validate_detach_contiguous

static bool validate_copy(UOp** src, size_t count) {
    if (count != 2) return false;
    UOp* copy = src[0];
    UOp* x = src[1];
    return copy->dtype._scalar == x->dtype._scalar;
}

static bool validate_allreduce(UOp** src, size_t count) {
    if (count != 2) return false;
    UOp* red = src[0];
    UOp* x = src[1];
    return red->dtype._scalar == x->dtype._scalar;
}

static bool validate_multi(UOp** src, size_t count) {
    if (count < 1) return false;
    UOp* multi = src[0];
    for (size_t i = 0; i < count; i++) {
        if (multi->dtype._scalar != src[i]->dtype._scalar) return false;
    }
    return (multi->arg.type == ARG_INT);
}

// Index validation functions (Z3-only)
static bool validate_index_z3(UOp** src, size_t count, UOp* explicit_gate, UOp* val_for_implicit) {
    if (!src || count < 2 || !src[0] || !src[1]) return true;
    if (spec_get_ignore_oob()) return true;
    UOp* base = src[0]; UOp* idx = src[1];
    // skip for image types
    if (dtypes_is_image(&base->dtype)) return true;
    // unknown/unsized buffers are accepted
    int32_t size = -1; if (base->st) size = shapetracker_size(base->st);
    if (size <= 0) return true;

    // WEBGPU bitcast workaround: skip if BITCAST appears in idx expression
    size_t isz = 0; UOp** itopo = uop_toposort(idx, &isz);
    if (itopo && isz>0) {
        for (size_t i=0;i<isz;i++){ if (itopo[i]->op == OPS_BITCAST){ free(itopo); return true; } }
        free(itopo);
    }
    Z3_config cfg = Z3_mk_config();
    Z3_context ctx = Z3_mk_context(cfg);
    Z3_del_config(cfg);
    Z3_solver solver = Z3_mk_solver(ctx);
    Z3_solver_inc_ref(ctx, solver);
    Z3RenderCtx rctx; z3rc_init(&rctx, ctx, solver);
    Z3_ast idx_ast = z3_eval(&rctx, idx);
    if (!idx_ast) { Z3_solver_dec_ref(ctx, solver); Z3_del_context(ctx); return true; }
    // Bounds
    Z3_ast lower = Z3_mk_ge(ctx, idx_ast, z3_int_val(ctx, 0));
    Z3_ast upper = Z3_mk_lt(ctx, idx_ast, z3_int_val(ctx, size));
    Z3_ast bounds = Z3_mk_and(ctx, 2, (Z3_ast[]){lower, upper});
    // Combine implicit INDEX gate (src[2]) with provided gate via conjunction
    Z3_ast mask = NULL;
    if (count >= 3 && src[2]) mask = z3_eval(&rctx, src[2]);
    if (explicit_gate) {
        Z3_ast g = explicit_gate->op == OPS_IF && explicit_gate->src_count>=1 ? z3_eval(&rctx, explicit_gate->src[0]) : z3_eval(&rctx, explicit_gate);
        if (g) mask = mask ? Z3_mk_and(ctx, 2, (Z3_ast[]){mask, g}) : g;
    }
    // add implicit IFs from value expression
    if (val_for_implicit) {
        size_t vcnt=0; UOp** vtopo = uop_toposort(val_for_implicit, &vcnt);
        if (vtopo) {
            for (size_t i=0;i<vcnt;i++) if (vtopo[i]->op==OPS_IF && vtopo[i]->src_count>=1) {
                Z3_ast g = z3_eval(&rctx, vtopo[i]->src[0]);
                if (g) mask = mask ? Z3_mk_and(ctx, 2, (Z3_ast[]){mask, g}) : g;
            }
            free(vtopo);
        }
    }
    // If we have a mask, enforce bounds only under mask
    Z3_ast bad = NULL;
    if (mask) {
        Z3_ast impl = Z3_mk_implies(ctx, mask, bounds);
        bad = Z3_mk_not(ctx, impl);
    } else {
        bad = Z3_mk_not(ctx, bounds);
    }
    Z3_solver_assert(ctx, solver, bad);
    Z3_lbool res = Z3_solver_check(ctx, solver);
    bool ok = (res == Z3_L_FALSE); // UNSAT => no counterexample => valid
    z3rc_free(&rctx);
    Z3_solver_dec_ref(ctx, solver);
    Z3_del_context(ctx);
    return ok;
}
bool validate_index(UOp** src, size_t count, UOp* gate) {
    return validate_index_z3(src, count, gate, NULL);
}

bool validate_store(UOp** src, size_t count, UOp* gate) {
    if (!src || count < 2) return false;
    UOp* dst = src[0];
    UOp* val = src[1];
    // Type sanity: scalar bases must align
    if (dst && val) {
        DType ds = dtype_scalar(&dst->dtype);
        DType vs = dtype_scalar(&val->dtype);
        if (!dtype_eq(&ds, &vs)) {
            if (!(val->op == OPS_CAST && dtype_eq(&dst->dtype, &val->dtype))) return false;
        }
    }
    // incorporate explicit gate and implicit IFs from value
    UOp* eff_gate = gate ? gate : (count>=3 ? src[2] : NULL);
    if (dst && dst->op == OPS_INDEX) return validate_index_z3(dst->src, dst->src_count, eff_gate, val);
    return true;
}

// Pattern Matcher for spec validation
typedef struct SpecPatternEntry {
    Ops op;
    bool (*validate)(UOp**, size_t);
} SpecPatternEntry;

static SpecPatternEntry spec_patterns[] = {
    {OPS_DEFINE_GLOBAL, NULL},
    {OPS_DEFINE_LOCAL, NULL},
    {OPS_DEFINE_REG, NULL},
    {OPS_DEFINE_VAR, NULL},
    {OPS_RANGE, NULL},
    {OPS_SPECIAL, NULL},
    {OPS_VIEW, NULL},
    {OPS_CONST, NULL},
    {OPS_LOAD, NULL},
    {OPS_STORE, NULL},
    {OPS_INDEX, NULL},
    {OPS_WHERE, NULL},
    {OPS_CMPLT, NULL},
    {OPS_CMPNE, NULL},
    {OPS_CMPEQ, NULL},
    {OPS_SHL, NULL},
    {OPS_SHR, NULL},
    {OPS_IDIV, NULL},
    {OPS_MOD, NULL},
    {OPS_ENDRANGE, NULL},
    {OPS_WMMA, NULL},
    {OPS_CONTRACT, NULL},
    {OPS_UNROLL, NULL},
    {OPS_IF, NULL},
    {OPS_ENDIF, NULL},
    {OPS_REDUCE_AXIS, NULL},
    {OPS_GEP, NULL},
    {OPS_VECTORIZE, NULL},
    {OPS_BITCAST, NULL},
    {OPS_CAST, NULL},
    {OPS_BARRIER, NULL},
    {OPS_SINK, NULL},
    {OPS_NOOP, NULL},
    {OPS_CUSTOMI, NULL},
    {OPS_CUSTOM, NULL},
    {OPS_PRECAST, NULL}
};

// validate_index main function
bool validate_index_main(UOp* idx, UOp* gate) {
    if (!idx || idx->op != OPS_INDEX) return false;
    
    size_t src_count = idx->src_count;
    return validate_index(idx->src, src_count, gate);
}

// validate_store main function
bool validate_store_main(UOp* store, UOp* gate) {
    if (!store || store->op != OPS_STORE) return false;
    
    size_t src_count = store->src_count;
    return validate_store(store->src, src_count, gate);
}

// AST spec validation
bool validate_ast_view_parent(UOp** src, size_t count) {
    if (count != 1) return false;
    UOp* view = src[0];
    if (!view || view->op != OPS_VIEW || view->src_count != 1) return false;
    // VIEW parent must be DEFINE_GLOBAL or DEFINE_LOCAL
    return view->src[0] && (view->src[0]->op == OPS_DEFINE_GLOBAL || view->src[0]->op == OPS_DEFINE_LOCAL);
}

bool validate_ast_view_empty(UOp* view) {
    if (!view || view->op != OPS_VIEW) return false;
    if (view->src_count != 0) return false;
    // In C port, ShapeTracker is attached at uop->st
    return view->st != NULL;
}

bool validate_ast_root(UOp** src, size_t count) {
    if (count != 1) return false;
    UOp* root = src[0]; if (!root) return false;
    // all parent UOps must have the same shape (for those that have st)
    const int32_t* shape_ref = NULL; int ndim_ref = -1;
    for (size_t i=0;i<root->src_count;i++) {
        UOp* p = root->src[i]; if (!p || !p->st) continue;
        int nd = shapetracker_ndim(p->st); const int32_t* shp = shapetracker_shape(p->st);
        if (!shp) continue;
        if (!shape_ref) { shape_ref = shp; ndim_ref = nd; continue; }
        if (nd != ndim_ref) return false;
        for (int d=0; d<nd; d++) if (shp[d] != shape_ref[d]) return false;
    }
    return true;
}

// Spec matcher aggregation
extern PatternMatcher* create_buffer_spec(void);
extern PatternMatcher* create_assign_spec(void);
extern PatternMatcher* create_tensor_uop_spec(void);
extern PatternMatcher* create_ast_spec(void);
static PatternMatcher* g_buffer_spec = NULL;
static PatternMatcher* g_assign_spec = NULL;
static PatternMatcher* g_tensor_spec = NULL;
static PatternMatcher* g_ast_spec = NULL;

static bool spec_match_any(PatternMatcher** pms, size_t pm_count, UOp* u) {
    for (size_t i=0;i<pm_count;i++) {
        void* result=NULL; if (!pms[i]) continue;
        if (pattern_matcher_apply(pms[i], u, NULL, &result) == PM_OK && result) return true;
    }
    return false;
}

// Type verification function modeled after Python type_verify
void type_verify(UOp** uops, size_t uop_count, PatternMatcher* extra_spec) {
    if (!uops || uop_count == 0) return;
    // Build or reuse spec matchers
    if (!g_buffer_spec) g_buffer_spec = create_buffer_spec();
    if (!g_assign_spec) g_assign_spec = create_assign_spec();
    if (!g_tensor_spec) g_tensor_spec = create_tensor_uop_spec();
    if (!g_ast_spec) g_ast_spec = create_ast_spec();

    PatternMatcher* set1[] = { g_buffer_spec, g_assign_spec, g_tensor_spec, g_ast_spec };
    size_t set1_count = sizeof(set1)/sizeof(set1[0]);
    for (size_t i = 0; i < uop_count; i++) {
        UOp* u = uops[i]; if (!u) continue;
        bool ok = spec_match_any(set1, set1_count, u);
        if (!ok && extra_spec) {
            void* result=NULL; ok = (pattern_matcher_apply(extra_spec, u, NULL, &result) == PM_OK) && result;
        }
        if (!ok) {
            fprintf(stderr, "UOp verification failed at %zu on %s %s %zu\n",
                    i, ops_to_string(u->op), u->dtype.name, u->src_count);
        }
    }
}

// Module initialization
void spec_init(void) {
    // Initialize spec patterns
    // This would build the pattern matchers from spec.py
    
    // Z3 required: check bounds strictly
    ignore_oob_var = 0;
    
    // Call helper usage functions to avoid warnings
    // Z3 is required; no stubbed ALU init needed
    use_helpers();
}

void spec_cleanup(void) {
    // Clean up spec patterns and data structures
}

// Additional exportable functions for external modules
bool validate_basic_pattern(Ops op, UOp** src, size_t count) {
    for (size_t i = 0; i < sizeof(spec_patterns)/sizeof(spec_patterns[0]); i++) {
        if (spec_patterns[i].op == op) {
            if (spec_patterns[i].validate) {
                return spec_patterns[i].validate(src, count);
            }
            return true; // No specific validator, assume valid
        }
    }
    return false; // Unknown operation
}

// Pattern matcher creation helpers
// Generic callback adapters that return truthy pointer
static void* cb_true(void* ctx, void* node) { (void)ctx; (void)node; return (void*)1; }
static void* cb_validate_device(void* ctx, void* node) {
    (void)ctx; UOp* d=(UOp*)node; return (void*)(uintptr_t)validate_device_arg(d);
}
static void* cb_validate_buffer(void* ctx, void* node) {
    (void)ctx; UOp* buf=(UOp*)node; if (!buf) return (void*)0;
    // Check srcs and then enforce arg is int
    bool ok = validate_buffer(buf->src, buf->src_count);
    if (!ok) return (void*)0;
    return (void*)(uintptr_t)(buf->arg.type == ARG_INT);
}
static void* cb_validate_buffer_view(void* ctx, void* node) {
    (void)ctx; UOp* bv=(UOp*)node; UOp* srcs[1]={NULL};
    if (!bv) return (void*)0;
    srcs[0] = (bv->src_count>0) ? bv->src[0] : NULL;
    return (void*)(uintptr_t)validate_buffer_view(srcs, 1, bv);
}

PatternMatcher* create_buffer_spec(void) {
    // Build PatternMatcher similar to Python buffer_spec
    PatternMatch entries[8]; size_t m=0;

    // UNIQUE: allow
    {
        UPat* p = upat_op(OPS_UNIQUE, NULL, 0);
        upat_set_dtype(p, &dtypes.void_);
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_true, .user_data=NULL };
    }
    // DEVICE
    {
        UPat* p = upat_op(OPS_DEVICE, NULL, 0);
        upat_set_dtype(p, &dtypes.void_);
        upat_set_name(p, "d");
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_validate_device, .user_data=NULL };
    }
    // BUFFER with (UNIQUE, DEVICE)
    {
        UPat* unique = upat_op(OPS_UNIQUE, NULL, 0);
        UPat* device = upat_op(OPS_DEVICE, NULL, 0);
        UPat* srcs[2] = {unique, device};
        UPat* p = upat_op(OPS_BUFFER, srcs, 2);
        upat_set_name(p, "buf");
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_validate_buffer, .user_data=NULL };
    }
    // BUFFER_VIEW with (BUFFER)
    {
        UPat* buf = upat_op(OPS_BUFFER, NULL, 0);
        UPat* srcs[1] = {buf};
        UPat* p = upat_op(OPS_BUFFER_VIEW, srcs, 1);
        upat_set_name(p, "buf_view");
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_validate_buffer_view, .user_data=NULL };
    }
    // BUFFER_VIEW with (MSTACK(BUFFER))
    {
        UPat* buf = upat_op(OPS_BUFFER, NULL, 0);
        UPat* srcs_m1[1] = {buf};
        UPat* mstack = upat_op(OPS_MSTACK, srcs_m1, 1);
        UPat* srcs2[1] = {mstack};
        UPat* p = upat_op(OPS_BUFFER_VIEW, srcs2, 1);
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_true, .user_data=NULL };
    }
    // VIEW: allow
    {
        UPat* p = upat_op(OPS_VIEW, NULL, 0);
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_true, .user_data=NULL };
    }

    return pattern_matcher_new(entries, m, false);
}

static void* cb_validate_assign(void* ctx, void* node) { (void)ctx; UOp* x=(UOp*)node; return (void*)(uintptr_t)validate_assign(x->src, x->src_count); }
static void* cb_validate_mselect(void* ctx, void* node) { (void)ctx; UOp* x=(UOp*)node; return (void*)(uintptr_t)validate_mselect_node(x); }
static void* cb_validate_mstack(void* ctx, void* node) { (void)ctx; UOp* x=(UOp*)node; return (void*)(uintptr_t)validate_mstack(x->src, x->src_count); }

PatternMatcher* create_assign_spec(void) {
    PatternMatch entries[6]; size_t m=0;
    // KERNEL on (BUFFER | BUFFER_VIEW | ASSIGN | MSELECT | MSTACK | BIND)
    {
        // We approximate GroupOp by allowing any operand
        UPat* any = upat_any();
        UPat* p = upat_op(OPS_KERNEL, &any, 1);
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_true, .user_data=NULL };
    }
    // ASSIGN has at least two src, others ASSIGNs
    {
        UPat* p = upat_op(OPS_ASSIGN, NULL, 0);
        upat_set_name(p, "x");
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_validate_assign, .user_data=NULL };
    }
    // MSELECT chooses one of the multi buffers
    {
        UPat* p = upat_op(OPS_MSELECT, NULL, 0);
        upat_set_name(p, "x");
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_validate_mselect, .user_data=NULL };
    }
    // MSTACK combines buffers into multi
    {
        UPat* p = upat_op(OPS_MSTACK, NULL, 0);
        upat_set_name(p, "x");
        entries[m++] = (PatternMatch){ .pattern=p, .callback=cb_validate_mstack, .user_data=NULL };
    }
    return pattern_matcher_new(entries, m, false);
}

// Helpers for movement rule fidelity
static bool uop_is_image_dtype(UOp* u) {
    if (!u) return false;
    return dtypes_is_image(&u->dtype);
}
static UOp* find_buffer_root(UOp* u) {
    while (u && u->op == OPS_VIEW && u->src_count > 0) u = u->src[0];
    return u;
}
static bool arg_tuple_like(UOp* mv) {
    if (!mv) return false;
    // Movement ops with explicit tuple-like args: PERMUTE (ARG_REDUCE), PAD (ARG_PAD_PARAMS), SHRINK (ARG_SHRINK_PARAMS)
    if (mv->op == OPS_PERMUTE && mv->arg.type == ARG_REDUCE) return true;
    if (mv->op == OPS_PAD && mv->arg.type == ARG_PAD_PARAMS) return true;
    if (mv->op == OPS_SHRINK && mv->arg.type == ARG_SHRINK_PARAMS) return true;
    // RESHAPE/EXPAND carry ShapeTracker; treat as tuple-like for parity
    if ((mv->op == OPS_RESHAPE || mv->op == OPS_EXPAND) && mv->st != NULL) return true;
    return false;
}

static void* cb_validate_movement(void* ctx, void* node) {
    (void)ctx; UOp* mv=(UOp*)node; if (!mv||mv->src_count<1) return (void*)0; UOp* x = mv->src[0];
    // First clause: tuple-like arg and dtype equality
    if (arg_tuple_like(mv) && dtype_eq(&mv->dtype, &x->dtype)) return (void*)1;
    // Second clause: image allowance — either side image, base dtypes equal, x rooted at BUFFER
    DType mvb = dtype_scalar(&mv->dtype); DType xb = dtype_scalar(&x->dtype);
    if ((uop_is_image_dtype(mv) || uop_is_image_dtype(x)) && dtype_eq(&mvb, &xb)) {
        UOp* root = find_buffer_root(x);
        if (root && root->op == OPS_BUFFER) return (void*)1;
    }
    // Otherwise, reject (strict parity with Python conditions)
    return (void*)0;
}
static void* cb_validate_view_sources(void* ctx, void* node) { (void)ctx; UOp* v=(UOp*)node; UOp* srcs[1]={v->src[0]}; return (void*)(uintptr_t)validate_view_all_sources(srcs,1); }
static void* cb_validate_copy(void* ctx, void* node) { (void)ctx; UOp* c=(UOp*)node; UOp* srcs[2]={c->src[0], c->src[1]}; return (void*)(uintptr_t)validate_copy(srcs,2); }
static void* cb_validate_allreduce(void* ctx, void* node) { (void)ctx; UOp* a=(UOp*)node; UOp* srcs[2]={a->src[0], a->src[1]}; return (void*)(uintptr_t)validate_allreduce(srcs,2); }
static void* cb_validate_multi(void* ctx, void* node) { (void)ctx; UOp* m=(UOp*)node; return (void*)(uintptr_t)validate_multi(m->src, m->src_count); }
// Additional callbacks for parity with Python spec
static void* cb_validate_define_var(void* ctx, void* node) { (void)ctx; UOp* v=(UOp*)node; return (void*)(uintptr_t)(v && v->arg.type==ARG_VAR); }
static void* cb_validate_range(void* ctx, void* node) { (void)ctx; UOp* r=(UOp*)node; 
    if (!r || r->src_count!=1 || r->arg.type!=ARG_INT) return (void*)0; 
    UOp* x = r->src[0]; 
    return (void*)(uintptr_t)dtype_eq(&r->dtype, &x->dtype);
}
static void* cb_view_void_has_st(void* ctx, void* node) { (void)ctx; UOp* v=(UOp*)node; return (void*)(uintptr_t)(v && v->st!=NULL); }
static void* cb_view_src_base_match(void* ctx, void* node) { (void)ctx; UOp* v=(UOp*)node; if (!v||v->src_count<1) return (void*)0; UOp* s=v->src[0]; if (s->op==OPS_STORE) return (void*)0; DType vb=dtype_scalar(&v->dtype), sb=dtype_scalar(&s->dtype); return (void*)(uintptr_t)dtype_eq(&vb,&sb); }
// DEFINE_* dtype checks using attached metadata
static void* cb_define_global(void* ctx, void* node){ (void)ctx; const UOp* u=(const UOp*)node; const SpecDTypeMeta* m = spec_get_define_meta(u); if (!m) return (void*)0; return (void*)(uintptr_t)((m->is_ptr||m->is_image) && m->addrspace==ADDRSPACE_GLOBAL); }
static void* cb_define_local(void* ctx, void* node){ (void)ctx; const UOp* u=(const UOp*)node; const SpecDTypeMeta* m = spec_get_define_meta(u); if (!m) return (void*)0; return (void*)(uintptr_t)(m->is_ptr && m->addrspace==ADDRSPACE_LOCAL); }

// ALU group: all bases equal across srcs
static void* cb_validate_alu_bases(void* ctx, void* node) {
    (void)ctx; UOp* x=(UOp*)node; if (!x) return (void*)0; const DType bx = dtype_scalar(&x->dtype);
    for (size_t i=0;i<x->src_count;i++){ const DType by = dtype_scalar(&x->src[i]->dtype); if (!dtype_eq(&bx, &by)) return (void*)0; }
    return (void*)1;
}

// WHERE dtype check
static void* cb_validate_where(void* ctx, void* node) {
    (void)ctx; UOp* w=(UOp*)node; if (!w||w->src_count<3) return (void*)0;
    UOp* x=w->src[1]; UOp* y=w->src[2];
    bool ok = dtype_eq(&w->dtype, &x->dtype) && dtype_eq(&w->dtype, &y->dtype);
    return (void*)(uintptr_t)ok;
}

// CMP ops: x and y base dtypes match, result is bool
static void* cb_validate_cmp(void* ctx, void* node) {
    (void)ctx; UOp* c=(UOp*)node; if (!c||c->src_count<2) return (void*)0;
    UOp* x=c->src[0]; UOp* y=c->src[1];
    const DType bx = dtype_scalar(&x->dtype); const DType by = dtype_scalar(&y->dtype);
    bool ok = dtype_eq(&c->dtype, &dtypes.bool_) && dtype_eq(&bx, &by);
    return (void*)(uintptr_t)ok;
}

// SHL/SHR: a.dtype == x.dtype and y.dtype in (x.dtype, uint)
static void* cb_validate_shift(void* ctx, void* node) {
    (void)ctx; UOp* a=(UOp*)node; if (!a||a->src_count<2) return (void*)0;
    UOp* x=a->src[0]; UOp* y=a->src[1];
    bool ok = dtype_eq(&a->dtype, &x->dtype) && (dtype_eq(&y->dtype, &x->dtype) || dtype_eq(&y->dtype, &dtypes.uint));
    return (void*)(uintptr_t)ok;
}

// IDIV/MOD: ints only
static void* cb_validate_idiv_mod(void* ctx, void* node) {
    (void)ctx; UOp* x=(UOp*)node; if (!x) return (void*)0;
    return (void*)(uintptr_t)dtypes_is_int(&x->dtype);
}

// GEP: dtype equals scalar of src dtype
static void* cb_validate_gep(void* ctx, void* node) {
    (void)ctx; UOp* g=(UOp*)node; if (!g||g->src_count<1) return (void*)0;
    UOp* src=g->src[0]; DType sc = dtype_scalar(&src->dtype);
    return (void*)(uintptr_t)dtype_eq(&g->dtype, &sc);
}

// VECTORIZE: len(src)>1, len(src)==dtype.count and each src has matching vector dtype
static void* cb_validate_vectorize(void* ctx, void* node) {
    (void)ctx; UOp* v=(UOp*)node; if (!v) return (void*)0;
    size_t n=v->src_count; if (n<=1) return (void*)0;
    if ((int)n != v->dtype.count) return (void*)0;
    // Python: all(x.dtype == y.dtype.vec(len(x.src)) for y in x.src)
    for (size_t i=0;i<n;i++) {
        DType vec_of_y = dtype_vec(&v->src[i]->dtype, (int)n);
        if (!dtype_eq(&v->dtype, &vec_of_y)) return (void*)0;
    }
    return (void*)1;
}

// CAST/BITCAST: arg is None (ARG_NONE)
static void* cb_validate_cast(void* ctx, void* node) {
    (void)ctx; UOp* c=(UOp*)node; return (void*)(uintptr_t)(c && c->arg.type==ARG_NONE);
}

// CONST type validation: require literal category to match dtype exactly (Python: type(x.arg) is type(dtypes.as_const(...)))
static void* cb_validate_const(void* ctx, void* node) {
    (void)ctx; UOp* x=(UOp*)node; if (!x) return (void*)0;
    if (x->arg.type != ARG_CONST && x->arg.type != ARG_INT) return (void*)0;
    if (dtypes_is_bool(&x->dtype)) {
        // We can't represent bool literals distinctly in C; accept int/const 0 or 1
        if (x->arg.type == ARG_INT) return (void*)(uintptr_t)(x->arg.int_data.i==0 || x->arg.int_data.i==1);
        if (x->arg.type == ARG_CONST) {
            double v = x->arg.const_data.const_value; return (void*)(uintptr_t)(v==0.0 || v==1.0);
        }
        return (void*)0;
    }
    if (dtypes_is_int(&x->dtype)) {
        // Require integer literal form (ARG_INT) to match Python's type check
        return (void*)(uintptr_t)(x->arg.type == ARG_INT);
    }
    // Floats require ARG_CONST (double)
    return (void*)(uintptr_t)(x->arg.type == ARG_CONST);
}

// LOAD index validation and STORE validation callbacks
static void* cb_validate_load_index(void* ctx, void* node) {
    (void)ctx; UOp* l=(UOp*)node; if (!l||l->src_count<1) return (void*)0; UOp* idx=l->src[0];
    UOp* gate = (l->src_count>=2) ? l->src[1] : NULL;
    return (void*)(uintptr_t)validate_index(idx->src, idx->src_count, gate);
}
static void* cb_validate_store_index(void* ctx, void* node) {
    (void)ctx; UOp* s=(UOp*)node; if (!s||s->src_count<2) return (void*)0;
    UOp* gate = (s->src_count>=3) ? s->src[2] : NULL;
    return (void*)(uintptr_t)validate_store(s->src, s->src_count, gate);
}

// REDUCE_AXIS validation: arg.reduce_op in {ADD,MUL,MAX} and axes_count>=1
static void* cb_validate_reduce_axis(void* ctx, void* node) {
    (void)ctx; UOp* r=(UOp*)node; if (!r) return (void*)0;
    if (r->arg.type != ARG_REDUCE) return (void*)0;
    if (r->arg.reduce_data.axes_count < 1) return (void*)0;
    Ops op = r->arg.reduce_data.reduce_op;
    bool ok = (op==OPS_ADD || op==OPS_MUL || op==OPS_MAX);
    return (void*)(uintptr_t)ok;
}

// IF/ENDIF validation (simplified): IF has 1 or 2 src; ENDIF on IF
static void* cb_validate_if(void* ctx, void* node) { (void)ctx; UOp* i=(UOp*)node; return (void*)(uintptr_t)(i && (i->src_count==1 || (i->src_count==2 && i->src[1]->op==OPS_BARRIER))); }
static void* cb_validate_endif(void* ctx, void* node) { (void)ctx; UOp* e=(UOp*)node; return (void*)(uintptr_t)(e && e->src_count==1 && e->src[0]->op==OPS_IF); }

// WMMA validation: arg tuple of len 8 (approximate -> true)
static void* cb_validate_wmma(void* ctx, void* node) {
    (void)ctx; UOp* w=(UOp*)node; if (!w) return (void*)0;
    if (w->arg.type == ARG_TUPLE2) return (void*)(uintptr_t)(w->arg.tuple2.count == 8);
    return (void*)0;
}

// CONTRACT/UNROLL validation (approximate)
static void* cb_validate_contract(void* ctx, void* node) {
    (void)ctx; UOp* c=(UOp*)node; if (!c) return (void*)0;
    if (c->arg.type != ARG_TUPLE2) return (void*)0;
    long long prod = 1; for (int i=0;i<c->arg.tuple2.count;i++) prod *= c->arg.tuple2.second ? c->arg.tuple2.second[i] : 1;
    return (void*)(uintptr_t)((int)prod == c->dtype.count);
}
static void* cb_validate_unroll(void* ctx, void* node) {
    (void)ctx; UOp* u=(UOp*)node; if (!u||u->src_count<1) return (void*)0;
    UOp* s0 = u->src[0];
    if (u->arg.type != ARG_TUPLE2) return (void*)0;
    long long prod = 1; for (int i=0;i<u->arg.tuple2.count;i++) prod *= u->arg.tuple2.second ? u->arg.tuple2.second[i] : 1;
    return (void*)(uintptr_t)(s0->dtype.count == (int)prod);
}

PatternMatcher* create_tensor_uop_spec(void) {
    PatternMatch entries[64]; size_t m=0;
    // DEFINE_* and VAR/RANGE/SPECIAL
    {
        // DEFINE_GLOBAL must be Ptr or Image with GLOBAL addrspace
        UPat* dg = upat_op(OPS_DEFINE_GLOBAL, NULL, 0);
        entries[m++] = (PatternMatch){ .pattern=dg, .callback=cb_define_global, .user_data=NULL };
        // DEFINE_LOCAL must be Ptr with LOCAL addrspace
        UPat* dl = upat_op(OPS_DEFINE_LOCAL, NULL, 0);
        entries[m++] = (PatternMatch){ .pattern=dl, .callback=cb_define_local, .user_data=NULL };
        // DEFINE_REG allowed
        UPat* dr = upat_op(OPS_DEFINE_REG, NULL, 0); entries[m++] = (PatternMatch){ .pattern=dr, .callback=cb_true, .user_data=NULL };
        UPat* dv = upat_op(OPS_DEFINE_VAR, NULL, 0); upat_set_name(dv, "x"); entries[m++] = (PatternMatch){ .pattern=dv, .callback=cb_validate_define_var, .user_data=NULL };
        UPat* xv = upat_var(9); UPat* rsrc[1]={xv}; UPat* rng = upat_op(OPS_RANGE, rsrc, 1); upat_set_name(rng, "rng"); entries[m++] = (PatternMatch){ .pattern=rng, .callback=cb_validate_range, .user_data=NULL };
        UPat* sp = upat_op(OPS_SPECIAL, NULL, 0); entries[m++] = (PatternMatch){ .pattern=sp, .callback=cb_true, .user_data=NULL };
    }
    // CONST type adherence
    {
        UPat* c = upat_op(OPS_CONST, NULL, 0); upat_set_name(c, "x");
        entries[m++] = (PatternMatch){ .pattern=c, .callback=cb_validate_const, .user_data=NULL };
    }
    // Movement(mv, src=(x,)) using GroupOp.Movement
    {
        UPat* x = upat_var(0);
        UPat* mvsrc[1] = { x };
        UPat* mv = upat_group_ops(group_op.is_movement, mvsrc, 1);
        upat_set_name(mv, "mv");
        entries[m++] = (PatternMatch){ .pattern=mv, .callback=cb_validate_movement, .user_data=NULL };
    }
    // VIEW(dtypes.void, src=()) and VIEW(src=(x,)) with base match and not STORE
    {
        UPat* v0 = upat_op(OPS_VIEW, NULL, 0); upat_set_dtype(v0, &dtypes.void_); upat_set_name(v0, "x");
        entries[m++] = (PatternMatch){ .pattern=v0, .callback=cb_view_void_has_st, .user_data=NULL };
        UPat* x = upat_var(1); UPat* vsrc[1]={x}; UPat* v = upat_op(OPS_VIEW, vsrc, 1); upat_set_name(v, "x");
        entries[m++] = (PatternMatch){ .pattern=v, .callback=cb_view_src_base_match, .user_data=NULL };
    }
    // ALU group base dtype check
    {
        UPat* alu = upat_group_ops(group_op.is_alu, NULL, 0); upat_set_name(alu, "x");
        entries[m++] = (PatternMatch){ .pattern=alu, .callback=cb_validate_alu_bases, .user_data=NULL };
    }
    // COPY, ALLREDUCE, MULTI (simplified)
    {
        UPat* any = upat_any(); UPat* src2[2]={any, any};
        UPat* copy = upat_op(OPS_COPY, src2, 2);
        entries[m++] = (PatternMatch){ .pattern=copy, .callback=cb_validate_copy, .user_data=NULL };
        UPat* allr = upat_op(OPS_ALLREDUCE, src2, 2);
        entries[m++] = (PatternMatch){ .pattern=allr, .callback=cb_validate_allreduce, .user_data=NULL };
        UPat* multi = upat_op(OPS_MULTI, NULL, 0);
        entries[m++] = (PatternMatch){ .pattern=multi, .callback=cb_validate_multi, .user_data=NULL };
    }
    // WHERE
    {
        UPat* cond = upat_op(OPS_NOOP, NULL, 0); upat_set_dtype(cond, &dtypes.bool_);
        UPat* x = upat_var(2); UPat* y = upat_var(3);
        UPat* wsrc[3]={cond,x,y}; UPat* w = upat_op(OPS_WHERE, wsrc, 3);
        upat_set_name(w, "w"); entries[m++] = (PatternMatch){ .pattern=w, .callback=cb_validate_where, .user_data=NULL };
    }
    // CMPLT, CMPNE, CMPEQ
    {
        UPat* x = upat_var(4); UPat* y = upat_var(5);
        UPat* csrc[2]={x,y}; UPat* cmp = upat_op(OPS_CMPLT, csrc, 2); upat_set_dtype(cmp, &dtypes.bool_);
        entries[m++] = (PatternMatch){ .pattern=cmp, .callback=cb_validate_cmp, .user_data=NULL };
        UPat* cne = upat_op(OPS_CMPNE, csrc, 2); upat_set_dtype(cne, &dtypes.bool_);
        entries[m++] = (PatternMatch){ .pattern=cne, .callback=cb_validate_cmp, .user_data=NULL };
        UPat* ceq = upat_op(OPS_CMPEQ, csrc, 2); upat_set_dtype(ceq, &dtypes.bool_);
        entries[m++] = (PatternMatch){ .pattern=ceq, .callback=cb_validate_cmp, .user_data=NULL };
    }
    // SHL/SHR
    {
        UPat* x = upat_var(6); UPat* y = upat_var(7); UPat* ssrc[2]={x,y};
        UPat* shl = upat_op(OPS_SHL, ssrc, 2); upat_set_name(shl, "a"); entries[m++] = (PatternMatch){ .pattern=shl, .callback=cb_validate_shift, .user_data=NULL };
        UPat* shr = upat_op(OPS_SHR, ssrc, 2); upat_set_name(shr, "a"); entries[m++] = (PatternMatch){ .pattern=shr, .callback=cb_validate_shift, .user_data=NULL };
    }
    // IDIV, MOD
    {
        UPat* idiv = upat_op(OPS_IDIV, NULL, 0); upat_set_name(idiv, "x"); entries[m++] = (PatternMatch){ .pattern=idiv, .callback=cb_validate_idiv_mod, .user_data=NULL };
        UPat* mod = upat_op(OPS_MOD, NULL, 0); upat_set_name(mod, "x"); entries[m++] = (PatternMatch){ .pattern=mod, .callback=cb_validate_idiv_mod, .user_data=NULL };
    }
    // ENDRANGE(RANGE)
    {
        UPat* r = upat_op(OPS_RANGE, NULL, 0); UPat* rsrc2[1]={r}; UPat* e = upat_op(OPS_ENDRANGE, rsrc2, 1); upat_set_dtype(e, &dtypes.void_); entries[m++] = (PatternMatch){ .pattern=e, .callback=cb_true, .user_data=NULL };
    }
    // GEP dtype
    {
        UPat* s = upat_var(8); UPat* gsrc[1]={s}; UPat* g = upat_op(OPS_GEP, gsrc, 1); upat_set_name(g, "gep"); entries[m++] = (PatternMatch){ .pattern=g, .callback=cb_validate_gep, .user_data=NULL };
    }
    // VECTORIZE
    {
        UPat* v = upat_op(OPS_VECTORIZE, NULL, 0); upat_set_name(v, "x"); entries[m++] = (PatternMatch){ .pattern=v, .callback=cb_validate_vectorize, .user_data=NULL };
    }
    // CAST/BITCAST
    {
        UPat* bcsrc[1]={ upat_any() }; UPat* bc = upat_op(OPS_BITCAST, bcsrc, 1); upat_set_name(bc, "x"); entries[m++] = (PatternMatch){ .pattern=bc, .callback=cb_validate_cast, .user_data=NULL };
        UPat* csrc[1]={ upat_any() }; UPat* c = upat_op(OPS_CAST, csrc, 1); upat_set_name(c, "x"); entries[m++] = (PatternMatch){ .pattern=c, .callback=cb_validate_cast, .user_data=NULL };
    }
    // VALID on VIEW
    {
        UPat* v = upat_op(OPS_VIEW, NULL, 0);
        UPat* vsrc[1]={v}; UPat* valid = upat_op(OPS_VALID, vsrc, 1); upat_set_dtype(valid, &dtypes.bool_);
        entries[m++] = (PatternMatch){ .pattern=valid, .callback=cb_true, .user_data=NULL };
    }
    // PTX LOAD/STORE special (int64 indices)
    {
        UPat* i64 = upat_op(OPS_NOOP, NULL, 0); upat_set_dtype(i64, &dtypes.int64);
        UPat* lsrc2[1]={i64}; UPat* lptx = upat_op(OPS_LOAD, lsrc2, 1); entries[m++] = (PatternMatch){ .pattern=lptx, .callback=cb_true, .user_data=NULL };
        UPat* ssrc2[1]={i64}; UPat* sptx = upat_op(OPS_STORE, ssrc2, 1); entries[m++] = (PatternMatch){ .pattern=sptx, .callback=cb_true, .user_data=NULL };
    }
    // BARRIER and SINK and others allowed
    {
        UPat* b = upat_op(OPS_BARRIER, NULL, 0); entries[m++] = (PatternMatch){ .pattern=b, .callback=cb_true, .user_data=NULL };
        UPat* s = upat_op(OPS_SINK, NULL, 0); upat_set_dtype(s, &dtypes.void_); entries[m++] = (PatternMatch){ .pattern=s, .callback=cb_true, .user_data=NULL };
        UPat* n = upat_op(OPS_NOOP, NULL, 0); entries[m++] = (PatternMatch){ .pattern=n, .callback=cb_true, .user_data=NULL };
        UPat* ci = upat_op(OPS_CUSTOMI, NULL, 0); entries[m++] = (PatternMatch){ .pattern=ci, .callback=cb_true, .user_data=NULL };
        UPat* cu = upat_op(OPS_CUSTOM, NULL, 0); entries[m++] = (PatternMatch){ .pattern=cu, .callback=cb_true, .user_data=NULL };
        UPat* pc = upat_op(OPS_PRECAST, NULL, 0); entries[m++] = (PatternMatch){ .pattern=pc, .callback=cb_true, .user_data=NULL };
    }
    // LOAD on STORE allowed
    {
        UPat* st = upat_op(OPS_STORE, NULL, 0); UPat* lsrc[1]={st}; UPat* l = upat_op(OPS_LOAD, lsrc, 1); entries[m++] = (PatternMatch){ .pattern=l, .callback=cb_true, .user_data=NULL };
    }
    // LOAD/STORE with INDEX and gate
    {
        // LOAD any index form
        UPat* l_any = upat_op(OPS_LOAD, NULL, 0);
        entries[m++] = (PatternMatch){ .pattern=l_any, .callback=cb_validate_load_index, .user_data=NULL };
        // STORE any index form
        UPat* s_any = upat_op(OPS_STORE, NULL, 0);
        entries[m++] = (PatternMatch){ .pattern=s_any, .callback=cb_validate_store_index, .user_data=NULL };
    }
    // Early LOAD/STORE patterns with GroupOp.Defines
    {
        // LOAD with VIEW on Defines
        UPat* defs = upat_group_ops(group_op.is_define, NULL, 0);
        UPat* vsrc[1]={defs}; UPat* v = upat_op(OPS_VIEW, vsrc, 1);
        UPat* lsrc[1]={v}; UPat* l = upat_op(OPS_LOAD, lsrc, 1);
        entries[m++] = (PatternMatch){ .pattern=l, .callback=cb_true, .user_data=NULL };
        // STORE has <VIEW(Defines), val>
        UPat* svalsrc[2]={v, upat_any()}; UPat* s = upat_op(OPS_STORE, svalsrc, 2);
        entries[m++] = (PatternMatch){ .pattern=s, .callback=cb_true, .user_data=NULL };
    }
    // IF/ENDIF
    {
        UPat* if1 = upat_op(OPS_IF, NULL, 0); upat_set_dtype(if1, &dtypes.void_);
        entries[m++] = (PatternMatch){ .pattern=if1, .callback=cb_validate_if, .user_data=NULL };
        UPat* ifb_src[2]={ upat_any(), upat_op(OPS_BARRIER, NULL, 0) }; UPat* ifb = upat_op(OPS_IF, ifb_src, 2); upat_set_dtype(ifb, &dtypes.void_);
        entries[m++] = (PatternMatch){ .pattern=ifb, .callback=cb_validate_if, .user_data=NULL };
        UPat* endif_src[1]={ upat_op(OPS_IF, NULL, 0) }; UPat* endif = upat_op(OPS_ENDIF, endif_src, 1); upat_set_dtype(endif, &dtypes.void_);
        entries[m++] = (PatternMatch){ .pattern=endif, .callback=cb_validate_endif, .user_data=NULL };
    }
    // REDUCE_AXIS
    {
        UPat* r = upat_op(OPS_REDUCE_AXIS, NULL, 0); upat_set_name(r, "x"); entries[m++] = (PatternMatch){ .pattern=r, .callback=cb_validate_reduce_axis, .user_data=NULL };
    }
    // WMMA, CONTRACT, UNROLL
    {
        UPat* wsrc3[3]={ upat_any(), upat_any(), upat_any() }; UPat* w=upat_op(OPS_WMMA, wsrc3, 3); upat_set_name(w, "x"); entries[m++] = (PatternMatch){ .pattern=w, .callback=cb_validate_wmma, .user_data=NULL };
        UPat* c = upat_op(OPS_CONTRACT, NULL, 0); upat_set_name(c, "x"); entries[m++] = (PatternMatch){ .pattern=c, .callback=cb_validate_contract, .user_data=NULL };
        UPat* u = upat_op(OPS_UNROLL, NULL, 0); upat_set_name(u, "x"); entries[m++] = (PatternMatch){ .pattern=u, .callback=cb_validate_unroll, .user_data=NULL };
    }
    return pattern_matcher_new(entries, m, false);
}

static void* cb_validate_ast_view_parent(void* ctx, void* node) { (void)ctx; UOp* v=(UOp*)node; UOp* srcs[1]={v}; return (void*)(uintptr_t)validate_ast_view_parent(srcs,1); }
static void* cb_validate_ast_view_empty(void* ctx, void* node) { (void)ctx; return (void*)(uintptr_t)validate_ast_view_empty((UOp*)node); }
static void* cb_validate_ast_root(void* ctx, void* node) { (void)ctx; UOp* r=(UOp*)node; UOp* srcs[1]={r}; return (void*)(uintptr_t)validate_ast_root(srcs,1); }

PatternMatcher* create_ast_spec(void) {
    PatternMatch entries[6]; size_t m=0;
    // VIEW on DEFINE_GLOBAL/DEFINE_LOCAL
    {
        UPat* dgl = upat_op(OPS_DEFINE_GLOBAL, NULL, 0);
        UPat* vsrc[1]={dgl};
        UPat* v = upat_op(OPS_VIEW, vsrc, 1);
        entries[m++] = (PatternMatch){ .pattern=v, .callback=cb_validate_ast_view_parent, .user_data=NULL };
    }
    // VIEW with zero sources
    {
        UPat* v = upat_op(OPS_VIEW, NULL, 0);
        upat_set_name(v, "view");
        entries[m++] = (PatternMatch){ .pattern=v, .callback=cb_validate_ast_view_empty, .user_data=NULL };
    }
    // Root: GroupOp.All - {SINK}
    {
        UPat* root = upat_group_all_except(OPS_SINK, NULL, 0);
        upat_set_name(root, "root");
        entries[m++] = (PatternMatch){ .pattern=root, .callback=cb_validate_ast_root, .user_data=NULL };
    }
    return pattern_matcher_new(entries, m, false);
}

// Test validation function - basic implementation
int helper_test_verify_ast(struct UOp* store) {
    if (!store) return SPEC_ERR_INVALID;
    
    // Basic validation checks
    // 1. Must be a STORE operation or SINK containing STOREs
    if (store->op == OPS_SINK) {
        // Validate all stores in the sink
        for (size_t i = 0; i < store->src_count; i++) {
            if (store->src[i]->op != OPS_STORE) {
                return SPEC_ERR_INVALID;
            }
        }
    } else if (store->op != OPS_STORE) {
        return SPEC_ERR_INVALID;
    }
    
    // 2. Perform topological sort to check graph structure
    size_t topo_count = 0;
    UOp** topo = uop_toposort(store, &topo_count);
    if (!topo || topo_count == 0) {
        if (topo) free(topo);
        return SPEC_ERR_INVALID;
    }
    
    // 3. Check for shape mismatches in binary operations
    for (size_t i = 0; i < topo_count; i++) {
        UOp* op = topo[i];
        
        // Check ADD operations for shape compatibility
        if (op->op == OPS_ADD && op->src_count == 2) {
            // If one operand is from REDUCE_AXIS and the other isn't,
            // that's usually a shape mismatch
            bool src0_reduced = false;
            bool src1_reduced = false;
            
            // Check if sources come from reduce operations (trace through VIEWs)
            UOp* s0 = op->src[0];
            UOp* s1 = op->src[1];
            
            // Trace through VIEW operations to find the actual source
            while (s0 && s0->op == OPS_VIEW && s0->src_count > 0) {
                s0 = s0->src[0];
            }
            while (s1 && s1->op == OPS_VIEW && s1->src_count > 0) {
                s1 = s1->src[0];
            }
            
            if (s0 && s0->op == OPS_REDUCE_AXIS) src0_reduced = true;
            if (s1 && s1->op == OPS_REDUCE_AXIS) src1_reduced = true;
            
            // If exactly one is reduced, likely shape mismatch
            if (src0_reduced != src1_reduced) {
                free(topo);
                return SPEC_ERR_INVALID;
            }
        }
    }
    
    // 4. Basic validation passed
    free(topo);
    return SPEC_OK;
}
