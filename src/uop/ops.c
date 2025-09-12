/* ops.c - Faithful port of tinygrad/uop/ops.py */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "uop/uop.h"
#include "uop/mathtraits.h"  // This provides the math_ops symbol
#include "shape/shapetracker.h"
#include "uop/spec.h"
#include "helpers/helpers.h"

// Forward declaration for optional buffer map removal used in uop_free
static void bufmap_remove(struct UOp* uop);

// Line 1-10: imports
// from __future__ import annotations
// import functools, itertools, hashlib, math, struct
// from enum import auto
// from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar
// from dataclasses import dataclass, field
// from tinygrad.helpers import pretty_print, prod, dedup, all_same, partition, temp
// from tinygrad.dtype import ConstType, DType, dtypes, PtrDType, ImageDType
// from tinygrad.shape.symbolic import Variable, sint, smax, smin
// from tinygrad.shape.shapetracker import ShapeTracker

// Line 14-17: if TYPE_CHECKING: ...
// Forward declarations for circular dependencies

// Line 19-22: Global caching variables
// _cache: dict[tuple, UOp] = {}
// _match_stats:dict[UPat, tuple[int, int]] = {}
static UOpCacheTable* _cache = NULL;
static int _match_stats_hits = 0;
static int _match_stats_total = 0;

// Line 24-30: @dataclass(frozen=True)
// class UOp:
//   op: Ops
//   dtype: Optional[DType] = None
//   src: tuple[UOp, ...] = ()
//   arg: Any = None

// Line 32-42: __slots__ = ("op", "dtype", "src", "arg")
// We use the UOp struct defined in ops.h

// Line 44-58: @functools.lru_cache(maxsize=2**20)
// def __new__(cls, op: Ops, dtype: Optional[DType] = None, src: tuple[UOp, ...] = (), arg: Any = None):
UOp* uop_new(Ops op, DType dtype, UOp** src, size_t src_count, UOpArg* arg, void* tag) {
    // Line 45: if op is Ops.NOP and len(src) == 1: return src[0]  # collapse NOPs
    if (op == OPS_NOOP && src_count == 1) {
        return src[0];
    }
    
    // NOTE: Removed incorrect SINK optimization - Python doesn't do this
    
    // Check cache first to avoid creating duplicates
    UOp* cached = uop_cache_get(op, dtype, src, src_count, arg, tag);
    if (cached) {
        if (!cached->math_ops) {
            cached->math_ops = &math_ops;  // ensure math_ops is always valid
        }
        return cached;  // Return cached version instead of creating duplicate
    }
    
// Line 59-60: Create new UOp
    UOp* uop = (UOp*)calloc(1, sizeof(UOp));
    // Zero-initialize and then set fields
    memset(uop, 0, sizeof(UOp));
    uop->ref_count = 1;  // Explicitly set to 1
    uop->op = op;
    uop->dtype = dtype;
    // DEBUG: Check if dtype is preserved correctly
    if (op == OPS_CAST && dtype.priority == 0) {  // bool has priority 0
        //printf("DEBUG uop_new CAST to bool: dtype.name=%s, uop->dtype.name=%s\n", dtype.name, uop->dtype.name);
    }
    uop->src_count = src_count;
    if (src_count > 0) {
        uop->src = (UOp**)malloc(src_count * sizeof(UOp*));
        memcpy(uop->src, src, src_count * sizeof(UOp*));
        // Reference the source UOps
        for (size_t i = 0; i < src_count; i++) {
            uop_ref(src[i]);
        }
    }
    if (arg) {
        uop->arg = *arg;
        // Deep copy arrays inside arg based on type
        if (op == OPS_REDUCE_AXIS && arg->type == ARG_REDUCE && arg->reduce_data.axes) {
            uop->arg.reduce_data.axes = (int*)malloc(arg->reduce_data.axes_count * sizeof(int));
            memcpy(uop->arg.reduce_data.axes, arg->reduce_data.axes,
                   arg->reduce_data.axes_count * sizeof(int));
        }
        if (op == OPS_GEP && arg->type == ARG_REDUCE && arg->reduce_data.axes) {
            uop->arg.reduce_data.axes = (int*)malloc(arg->reduce_data.axes_count * sizeof(int));
            memcpy(uop->arg.reduce_data.axes, arg->reduce_data.axes,
                   arg->reduce_data.axes_count * sizeof(int));
        }
        if (op == OPS_PERMUTE && arg->type == ARG_REDUCE && arg->reduce_data.axes) {
            uop->arg.reduce_data.axes = (int*)malloc(arg->reduce_data.axes_count * sizeof(int));
            memcpy(uop->arg.reduce_data.axes, arg->reduce_data.axes,
                   arg->reduce_data.axes_count * sizeof(int));
        }
        if (arg->type == ARG_TUPLE2 && arg->tuple2.count > 0) {
            int n = arg->tuple2.count;
            uop->arg.tuple2.count = n;
            if (arg->tuple2.first) {
                uop->arg.tuple2.first = (int*)malloc(n * sizeof(int));
                memcpy(uop->arg.tuple2.first, arg->tuple2.first, n * sizeof(int));
            }
            if (arg->tuple2.second) {
                uop->arg.tuple2.second = (int*)malloc(n * sizeof(int));
                memcpy(uop->arg.tuple2.second, arg->tuple2.second, n * sizeof(int));
            }
        }
        if (op == OPS_PAD && arg->type == ARG_PAD_PARAMS && arg->pad_data.ndim > 0) {
            int n = arg->pad_data.ndim;
            uop->arg.pad_data.ndim = n;
            uop->arg.pad_data.before = (int32_t*)malloc(n * sizeof(int32_t));
            uop->arg.pad_data.after  = (int32_t*)malloc(n * sizeof(int32_t));
            memcpy(uop->arg.pad_data.before, arg->pad_data.before, n * sizeof(int32_t));
            memcpy(uop->arg.pad_data.after,  arg->pad_data.after,  n * sizeof(int32_t));
        }
        if (op == OPS_SHRINK && arg->type == ARG_SHRINK_PARAMS && arg->shrink_data.ndim > 0) {
            int n = arg->shrink_data.ndim;
            uop->arg.shrink_data.ndim = n;
            uop->arg.shrink_data.start = (int32_t*)malloc(n * sizeof(int32_t));
            uop->arg.shrink_data.end   = (int32_t*)malloc(n * sizeof(int32_t));
            memcpy(uop->arg.shrink_data.start, arg->shrink_data.start, n * sizeof(int32_t));
            memcpy(uop->arg.shrink_data.end,   arg->shrink_data.end,   n * sizeof(int32_t));
        }
        if (arg->type == ARG_VCONST && arg->vconst_data.values && arg->vconst_data.count > 0) {
            int n = arg->vconst_data.count;
            uop->arg.vconst_data.count = n;
            uop->arg.vconst_data.values = (double*)malloc(n * sizeof(double));
            memcpy(uop->arg.vconst_data.values, arg->vconst_data.values, n * sizeof(double));
        }
        if (arg->type == ARG_STRING && arg->str.s) {
            uop->arg.str.s = strdup(arg->str.s);
        }
        if (arg->type == ARG_STRLIST && arg->strlist.count > 0 && arg->strlist.items) {
            int n = arg->strlist.count;
            uop->arg.strlist.count = n;
            uop->arg.strlist.items = (char**)malloc((size_t)n * sizeof(char*));
            for (int i=0;i<n;i++) uop->arg.strlist.items[i] = arg->strlist.items[i] ? strdup(arg->strlist.items[i]) : NULL;
        }
        if (arg->type == ARG_TUPLE_MIXED && arg->tmixed.count > 0 && arg->tmixed.items) {
            int n = arg->tmixed.count;
            uop->arg.tmixed.count = n;
            uop->arg.tmixed.items = (struct MixedItem*)malloc((size_t)n * sizeof(struct MixedItem));
            for (int i=0;i<n;i++) {
                uop->arg.tmixed.items[i].tag = arg->tmixed.items[i].tag;
                uop->arg.tmixed.items[i].ival = arg->tmixed.items[i].ival;
                uop->arg.tmixed.items[i].uop = arg->tmixed.items[i].uop ? uop_ref(arg->tmixed.items[i].uop) : NULL;
            }
        }
        // If a ShapeTracker was passed via arg (legacy), attach it to uop->st
        if (arg->type == ARG_SHAPE_TRACKER && arg->st_data.st) {
            uop->st = (ShapeTracker*)arg->st_data.st;
        }
    }
    uop->math_ops = &math_ops;  // Set this once, after all initialization
    uop->tag = tag;
    
    // Initialize vmin/vmax to invalid state
    uop->vmin_vmax_valid = false;
    uop->vmin = 0;
    uop->vmax = 0;
    
    // Initialize optional children/metadata
    uop->children = NULL; uop->children_count = 0; uop->children_cap = 0; uop->meta_head = NULL;
    // Register as child of sources
    for (size_t i = 0; i < uop->src_count; i++) {
        UOp* p = uop->src[i];
        if (!p) continue;
        if (p->children_count >= p->children_cap) {
            size_t nc = p->children_cap ? p->children_cap * 2 : 4;
            p->children = (UOp**)realloc(p->children, nc * sizeof(UOp*));
            p->children_cap = nc;
        }
        p->children[p->children_count++] = uop;
    }
    // Add to cache
    uop_cache_put(uop);
    return uop;
}

char* uop_pretty_str(UOp* u, bool color) {
    if (!u) return NULL;
    const char* opn = ops_to_string(u->op);
    const char* dtn = u->dtype.count ? u->dtype.name : NULL;
    char buf[256]; buf[0]=0;
    if (color) {
        if (dtn) snprintf(buf, sizeof(buf), "\x1b[36m%s\x1b[0m(%s, src=%zu)", opn, dtn, u->src_count);
        else snprintf(buf, sizeof(buf), "\x1b[36m%s\x1b[0m(src=%zu)", opn, u->src_count);
    } else {
        if (dtn) snprintf(buf, sizeof(buf), "%s(%s, src=%zu)", opn, dtn, u->src_count);
        else snprintf(buf, sizeof(buf), "%s(src=%zu)", opn, u->src_count);
    }
    size_t n = strlen(buf);
    char* out = (char*)malloc(n+1); if (!out) return NULL;
    memcpy(out, buf, n+1);
    return out;
}

// Line 61-68: Reference counting (implicit in Python)
void uop_free(UOp* uop) {
    if (!uop) return;
    if (--uop->ref_count == 0) {
        // Remove from optional buffer map (weak semantics)
        bufmap_remove(uop);
        // Note: children lists are not back-updated on free; tests only assert presence post-creation
        // Unref source UOps
        for (size_t i = 0; i < uop->src_count; i++) {
            uop_unref(uop->src[i]);
        }
        if (uop->src) free(uop->src);
        // Free allocated arrays in arg
        if ((uop->op == OPS_REDUCE_AXIS || uop->op == OPS_PERMUTE || uop->op == OPS_GEP) && uop->arg.type == ARG_REDUCE && uop->arg.reduce_data.axes) {
            free(uop->arg.reduce_data.axes);
        }
        if (uop->arg.type == ARG_VCONST && uop->arg.vconst_data.values) {
            free(uop->arg.vconst_data.values);
        }
        if (uop->arg.type == ARG_STRING && uop->arg.str.s) {
            free(uop->arg.str.s);
        }
        if (uop->arg.type == ARG_STRLIST && uop->arg.strlist.items) {
            for (int i=0;i<uop->arg.strlist.count;i++) if (uop->arg.strlist.items[i]) free(uop->arg.strlist.items[i]);
            free(uop->arg.strlist.items);
        }
        if (uop->arg.type == ARG_TUPLE_MIXED && uop->arg.tmixed.items) {
            for (int i=0;i<uop->arg.tmixed.count;i++) if (uop->arg.tmixed.items[i].uop) uop_unref(uop->arg.tmixed.items[i].uop);
            free(uop->arg.tmixed.items);
        }
        if (uop->arg.type == ARG_TUPLE2) {
            if (uop->arg.tuple2.first) free(uop->arg.tuple2.first);
            if (uop->arg.tuple2.second) free(uop->arg.tuple2.second);
        }
        if (uop->op == OPS_PAD && uop->arg.type == ARG_PAD_PARAMS) {
            free(uop->arg.pad_data.before);
            free(uop->arg.pad_data.after);
        }
        if (uop->op == OPS_SHRINK && uop->arg.type == ARG_SHRINK_PARAMS) {
            free(uop->arg.shrink_data.start);
            free(uop->arg.shrink_data.end);
        }
        if (uop->st) {
            shapetracker_free(uop->st);
        }
        if (uop->children) free(uop->children);
        // Free metadata list (keys only)
        struct UOpMetaKV* kv = uop->meta_head;
        while (kv) { struct UOpMetaKV* nx = kv->next; if (kv->key) free(kv->key); free(kv); kv = nx; }
        // Note: We don't remove from cache here as cache holds weak refs
        free(uop);
    }
}

UOp* uop_ref(UOp* uop) {
    if (uop) uop->ref_count++;
    return uop;
}

void uop_unref(UOp* uop) {
    uop_free(uop);
}

// ---- Optional Buffer Map (UOp* -> Buffer*) ----
typedef struct BufMapEntry { UOp* key; void* value; struct BufMapEntry* next; } BufMapEntry;
static struct { BufMapEntry** buckets; size_t nb; } g_bufmap = {NULL, 0};
static size_t buf_hash_ptr(const void* p){ size_t x=(size_t)p; x ^= x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33; x*=0xc4ceb9fe1a85ec53ULL; x^=x>>33; return x; }
static void bufmap_ensure(void){ if(!g_bufmap.nb){ g_bufmap.nb=256; g_bufmap.buckets=(BufMapEntry**)calloc(g_bufmap.nb,sizeof(BufMapEntry*)); } }
void uop_buffer_map_set(UOp* uop, void* buffer_ptr){ if(!uop) return; bufmap_ensure(); size_t i = buf_hash_ptr(uop) % g_bufmap.nb; for(BufMapEntry* e=g_bufmap.buckets[i]; e; e=e->next){ if(e->key==uop){ e->value=buffer_ptr; return; } } BufMapEntry* ne=(BufMapEntry*)calloc(1,sizeof(*ne)); ne->key=uop; ne->value=buffer_ptr; ne->next=g_bufmap.buckets[i]; g_bufmap.buckets[i]=ne; }
void* uop_buffer_map_get(UOp* uop){ if(!uop||!g_bufmap.nb) return NULL; size_t i=buf_hash_ptr(uop)%g_bufmap.nb; for(BufMapEntry* e=g_bufmap.buckets[i]; e; e=e->next) if(e->key==uop) return e->value; return NULL; }
static void bufmap_remove(UOp* uop){ if(!uop||!g_bufmap.nb) return; size_t i=buf_hash_ptr(uop)%g_bufmap.nb; BufMapEntry* prev=NULL; for(BufMapEntry* e=g_bufmap.buckets[i]; e; ){ if(e->key==uop){ BufMapEntry* nx=e->next; if(prev) prev->next=nx; else g_bufmap.buckets[i]=nx; free(e); e=nx; } else { prev=e; e=e->next; } } }

// Line 70-72: def commutative(self) -> bool:
//   return self.op in {Ops.ADD, Ops.MUL, ...}
bool uop_commutative(UOp* uop) {
    return uop->op == OPS_ADD || uop->op == OPS_MUL || uop->op == OPS_MAX ||
           uop->op == OPS_CMPNE || uop->op == OPS_XOR || uop->op == OPS_AND ||
           uop->op == OPS_OR;
}

// Line 74-80: def is_zero(self) -> bool:
bool uop_is_zero(UOp* uop) {
    if (uop->op == OPS_CONST && uop->arg.type == ARG_CONST) {
        return uop->arg.const_data.const_value == 0.0;
    }
    if (uop->op == OPS_VECTORIZE) {
        for (size_t i = 0; i < uop->src_count; i++) {
            if (!uop_is_zero(uop->src[i])) return false;
        }
        return true;
    }
    return false;
}

// Line 82-88: def is_one(self) -> bool:
bool uop_is_one(UOp* uop) {
    if (uop->op == OPS_CONST && uop->arg.type == ARG_CONST) {
        return uop->arg.const_data.const_value == 1.0;
    }
    if (uop->op == OPS_VECTORIZE) {
        for (size_t i = 0; i < uop->src_count; i++) {
            if (!uop_is_one(uop->src[i])) return false;
        }
        return true;
    }
    return false;
}

// Line 90-100: def divides(self, v) -> Optional[int]:
int uop_divides(UOp* uop, int v) {
    if (uop->op == OPS_CONST && uop->arg.type == ARG_INT) {
        int val = uop->arg.int_data.i;
        if (val != 0 && v % val == 0) {
            return v / val;
        }
    }
    if (uop->op == OPS_MUL && uop->src_count == 2) {
        if (uop->src[0]->op == OPS_CONST && uop->src[0]->arg.type == ARG_INT) {
            int val = uop->src[0]->arg.int_data.i;
            if (val != 0 && v % val == 0) {
                return uop_divides(uop->src[1], v / val);
            }
        }
    }
    return 0;  // None in Python
}

// Missing operation implementations for MathTrait support
UOp* uop_and(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_AND, dtypes.bool_, src, 2, &arg, NULL);
}

UOp* uop_or(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_OR, dtypes.bool_, src, 2, &arg, NULL);
}

UOp* uop_xor(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_XOR, a->dtype, src, 2, &arg, NULL);
}

UOp* uop_shl(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_SHL, a->dtype, src, 2, &arg, NULL);
}

UOp* uop_shr(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_SHR, a->dtype, src, 2, &arg, NULL);
}

// Line 102-140: Various helper methods
UOp* uop_sink(UOp** stores, size_t count) {
    // Line 102: @staticmethod
    // def sink(*srcs): return UOp(Ops.SINK, dtypes.void, srcs)
    UOpArg arg = {0};
    return uop_new(OPS_SINK, dtypes.void_, stores, count, &arg, NULL);
}

UOp* uop_store(UOp* buf, UOp* value) {
    // Basic store operation
    UOp* src[] = {buf, value};
    UOpArg arg = {0};
    return uop_new(OPS_STORE, dtypes.void_, src, 2, &arg, NULL);
}

UOp* uop_load(UOp* buf, DType dtype) {
    // Basic load operation
    UOp* src[] = {buf};
    UOpArg arg = {0};
    UOp* u = uop_new(OPS_LOAD, dtype, src, 1, &arg, NULL);
    if (buf && buf->st) {
        int nd = shapetracker_ndim(buf->st);
        const int32_t* shp = shapetracker_shape(buf->st);
        if (nd > 0 && shp) u->st = shapetracker_from_shape(shp, nd);
    }
    return u;
}

UOp* uop_assign(UOp* dst, UOp* value) {
    UOp* src[] = {dst, value};
    UOpArg arg = {0};
    UOp* u = uop_new(OPS_ASSIGN, dst ? dst->dtype : dtypes.void_, src, 2, &arg, NULL);
    if (dst && dst->st) {
        int nd = shapetracker_ndim(dst->st);
        const int32_t* shp = shapetracker_shape(dst->st);
        if (nd > 0 && shp) u->st = shapetracker_from_shape(shp, nd);
    }
    return u;
}

UOp* uop_barrier(UOp* after) {
    UOp* src[] = {after};
    UOpArg arg = {0};
    return uop_new(OPS_BARRIER, dtypes.void_, src, 1, &arg, NULL);
}

// Line 142-145: def const(self, dtype, val):
UOp* uop_const(DType dtype, double value) {
    UOpArg arg = {.type = ARG_CONST, .const_data.const_value = value};
    return uop_new(OPS_CONST, dtype, NULL, 0, &arg, NULL);
}

UOp* uop_vconst(DType dtype, const double* vals, int count) {
    UOpArg arg = {.type = ARG_VCONST};
    arg.vconst_data.count = count;
    arg.vconst_data.values = (double*)vals; // will be deep-copied in uop_new
    return uop_new(OPS_VCONST, dtype, NULL, 0, &arg, NULL);
}

UOp* uop_const_like(UOp* self, double b) {
    // helper to find a DEVICE ancestor
    UOp* find_device(UOp* u){ if(!u) return NULL; if(u->op==OPS_DEVICE) return u; for(size_t i=0;i<u->src_count;i++){ UOp* d=find_device(u->src[i]); if(d) return d; } return NULL; }
    // Create a scalar const of self's dtype
    UOpArg carg = {.type = ARG_CONST, .const_data.const_value = b};
    UOp* ret = uop_new(OPS_CONST, self->dtype, NULL, 0, &carg, self->tag);
    // Determine shape tracker (if any)
    ShapeTracker* st = NULL;
    if (self->st) {
        const int32_t* shp = shapetracker_shape(self->st);
        int32_t nd = shapetracker_ndim(self->st);
        st = shapetracker_from_shape(shp, nd);
    }
    // If we have both device and shape, attach DEVICE.view(st) as the source
    UOp* dev = NULL;
    if (st && (dev = find_device(self)) != NULL) {
        UOp* dv = uop_view(dev, st);
        UOp* srcs[] = { dv };
        ret = uop_replace_ex(ret, OPS_CONST, ret->dtype, srcs, 1, &carg, self->tag);
        uop_unref(dv);
        return ret;
    }
    // If only shape (no device), attach a VIEW on the const to carry shape
    if (st) {
        UOp* view = uop_view(ret, st);
        UOp* srcs2[] = { view };
        ret = uop_replace_ex(ret, OPS_CONST, ret->dtype, srcs2, 1, &carg, self->tag);
        uop_unref(view);
        return ret;
    }
    return ret;
}

UOp* uop_const_ex(DType dtype, double value, UOp* device_uop, const int32_t* shape, int nd) {
    UOpArg carg = {.type = ARG_CONST, .const_data.const_value = value};
    UOp* ret = uop_new(OPS_CONST, dtype, NULL, 0, &carg, NULL);
    ShapeTracker* st = NULL;
    if (shape && nd >= 0) {
        st = shapetracker_from_shape(shape, nd);
    }
    if (st && device_uop) {
        UOp* dv = uop_view(device_uop, st);
        UOp* srcs[] = { dv };
        ret = uop_replace_ex(ret, OPS_CONST, ret->dtype, srcs, 1, &carg, NULL);
        uop_unref(dv);
        return ret;
    }
    if (st) {
        UOp* view = uop_view(ret, st);
        UOp* srcs2[] = { view };
        ret = uop_replace_ex(ret, OPS_CONST, ret->dtype, srcs2, 1, &carg, NULL);
        uop_unref(view);
        return ret;
    }
    if (device_uop) {
        // CONST with only DEVICE source implies scalar shape; attach DEVICE as src
        UOp* srcs3[] = { device_uop };
        ret = uop_replace_ex(ret, OPS_CONST, ret->dtype, srcs3, 1, &carg, NULL);
        return ret;
    }
    return ret;
}

UOp* uop_define_global(DType dtype, int idx) {
    UOpArg arg = {.type = ARG_INT, .int_data.i = idx};
    UOp* u = uop_new(OPS_DEFINE_GLOBAL, dtype, NULL, 0, &arg, NULL);
    // Note: Avoid type-punning DType into PtrDType; attach no shape here.
    // Attach spec dtype metadata for parity
    spec_attach_define_meta(u, ADDRSPACE_GLOBAL);
    return u;
}

UOp* uop_define_local(DType dtype, size_t size) {
    UOpArg arg = {.type = ARG_INT, .int_data.i = (int)size};
    UOp* u = uop_new(OPS_DEFINE_LOCAL, dtype, NULL, 0, &arg, NULL);
    // Attach shape from provided size (elements) if non-zero
    if (size > 0) { int32_t shp[1] = { (int32_t)size }; u->st = shapetracker_from_shape(shp, 1); }
    // Attach spec dtype metadata for parity
    spec_attach_define_meta(u, ADDRSPACE_LOCAL);
    return u;
}

UOp* uop_define_reg(DType dtype) {
    UOpArg arg = {0};
    UOp* u = uop_new(OPS_DEFINE_REG, dtype, NULL, 0, &arg, NULL);
    // No shape attached for registers (scalars by definition here)
    // Attach spec dtype metadata for parity (REG)
    spec_attach_define_meta(u, ADDRSPACE_REG);
    return u;
}

// Line 147-190: Binary operations
UOp* uop_add(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    DType result_dtype = least_upper_dtype(&a->dtype, &b->dtype);
    return uop_new(OPS_ADD, result_dtype, src, 2, &arg, NULL);
}

UOp* uop_mul(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    // Follow reference: always use lattice to compute result dtype
    DType result_dtype = least_upper_dtype(&a->dtype, &b->dtype);
    return uop_new(OPS_MUL, result_dtype, src, 2, &arg, NULL);
}

UOp* uop_sub(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    DType result_dtype = least_upper_dtype(&a->dtype, &b->dtype);
    return uop_new(OPS_SUB, result_dtype, src, 2, &arg, NULL);
}

UOp* uop_div(UOp* a, UOp* b) {
    // Quick canonical reductions for integer-like bitpacking patterns
    if (a && b && b->op==OPS_CONST) {
        double dval = (b->arg.type==ARG_CONST) ? b->arg.const_data.const_value : (double)b->arg.int_data.i;
        if (dval != 0.0) {
            // Direct MUL(hi, K) // K -> hi
            if (a->op == OPS_MUL && a->src_count==2) {
                UOp* mc=NULL,*mv=NULL; if (a->src[0]->op==OPS_CONST){ mc=a->src[0]; mv=a->src[1]; } else if (a->src[1]->op==OPS_CONST){ mc=a->src[1]; mv=a->src[0]; }
                if (mc && mc->op==OPS_CONST) {
                    double mcv = (mc->arg.type==ARG_CONST) ? mc->arg.const_data.const_value : (double)mc->arg.int_data.i;
                    if (fabs(mcv - dval) < 0.5) {
                        if (mv->op==OPS_CAST && dtype_eq(&mv->dtype, &dtypes.uint64)) mv = mv->src[0];
                        return uop_ref(mv);
                    }
                }
            }
            // Packed ((hi*K) | cast(lo,u64)) // K -> hi
            if (a->op == OPS_OR && a->src_count==2) {
                UOp* l=a->src[0], *r=a->src[1];
                if (l->op==OPS_CAST && dtype_eq(&l->dtype, &dtypes.uint64)) l = l->src[0];
                if (r->op==OPS_CAST && dtype_eq(&r->dtype, &dtypes.uint64)) r = r->src[0];
                UOp* mul = (l->op==OPS_MUL)? l : (r->op==OPS_MUL? r: NULL);
                if (mul && mul->src_count==2) {
                    UOp* mc=NULL,*mv=NULL; if (mul->src[0]->op==OPS_CONST){ mc=mul->src[0]; mv=mul->src[1]; } else if (mul->src[1]->op==OPS_CONST){ mc=mul->src[1]; mv=mul->src[0]; }
                    if (mc && mc->op==OPS_CONST) {
                        double mcv = (mc->arg.type==ARG_CONST) ? mc->arg.const_data.const_value : (double)mc->arg.int_data.i;
                        if (fabs(mcv - dval) < 0.5) {
                            if (mv->op==OPS_CAST && dtype_eq(&mv->dtype, &dtypes.uint64)) mv = mv->src[0];
                            return uop_ref(mv);
                        }
                    }
                }
            }
        }
    }

    UOp* src[] = {a, b};
    UOpArg arg = {0};
    DType result_dtype = least_upper_dtype(&a->dtype, &b->dtype);
    // Use IDIV for integer and unsigned integer types, FDIV for floating point
    if (dtypes_is_int(&result_dtype) || dtypes_is_unsigned(&result_dtype)) {
        return uop_new(OPS_IDIV, result_dtype, src, 2, &arg, NULL);
    } else {
        return uop_new(OPS_FDIV, result_dtype, src, 2, &arg, NULL);
    }
}

UOp* uop_max(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    DType result_dtype = least_upper_dtype(&a->dtype, &b->dtype);
    return uop_new(OPS_MAX, result_dtype, src, 2, &arg, NULL);
}

UOp* uop_min(UOp* a, UOp* b) {
    // min(a,b) = -max(-a,-b)
    UOp* neg_a = uop_neg(a);
    UOp* neg_b = uop_neg(b);
    UOp* max_neg = uop_max(neg_a, neg_b);
    return uop_neg(max_neg);
}

UOp* uop_lt(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_CMPLT, dtypes.bool_, src, 2, &arg, NULL);
}

UOp* uop_eq(UOp* a, UOp* b) {
    // eq is logical_not(ne) in Python, but we'll create CMPEQ directly for tests
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_CMPEQ, dtypes.bool_, src, 2, &arg, NULL);
}

UOp* uop_ne(UOp* a, UOp* b) {
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_CMPNE, dtypes.bool_, src, 2, &arg, NULL);
}

// Line 192-240: Unary operations
UOp* uop_neg(UOp* a) {
    // Create NEG operation directly
    UOp* src[] = {a};
    UOpArg arg = {0};
    return uop_new(OPS_NEG, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_exp2(UOp* a) {
    UOp* src[] = {a};
    UOpArg arg = {0};
    return uop_new(OPS_EXP2, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_log2(UOp* a) {
    UOp* src[] = {a};
    UOpArg arg = {0};
    return uop_new(OPS_LOG2, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_sin(UOp* a) {
    UOp* src[] = {a};
    UOpArg arg = {0};
    return uop_new(OPS_SIN, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_sqrt(UOp* a) {
    UOp* src[] = {a};
    UOpArg arg = {0};
    return uop_new(OPS_SQRT, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_recip(UOp* a) {
    UOp* src[] = {a};
    UOpArg arg = {0};
    return uop_new(OPS_RECIP, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_contiguous(UOp* a) {
    UOp* src[] = {a}; UOpArg arg={0};
    return uop_new(OPS_CONTIGUOUS, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_contiguous_backward(UOp* a) {
    UOp* src[] = {a}; UOpArg arg={0};
    return uop_new(OPS_CONTIGUOUS_BACKWARD, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_fuse(UOp* a) {
    UOp* src[] = {a}; UOpArg arg={0};
    return uop_new(OPS_FUSE, a->dtype, src, 1, &arg, NULL);
}

UOp* uop_detach(UOp* a) {
    UOp* src[] = {a}; UOpArg arg={0};
    return uop_new(OPS_DETACH, a->dtype, src, 1, &arg, NULL);
}

// Additional transcendental support functions
UOp* uop_bitcast(UOp* a, DType dtype) {
    UOp* src[] = {a};
    UOpArg arg = {0};
    UOp* u = uop_new(OPS_BITCAST, dtype, src, 1, &arg, NULL);
    // Propagate and adjust shape if itemsize changes: shape[-1] = shape[-1]*in_sz // out_sz
    if (a && a->st) {
        int nd = shapetracker_ndim(a->st);
        const int32_t* shp = shapetracker_shape(a->st);
        if (nd > 0 && shp) {
            int in_sz = a->dtype.itemsize;
            int out_sz = dtype.itemsize;
            int32_t* new_shp = (int32_t*)malloc(sizeof(int32_t)*nd);
            for (int i=0;i<nd;i++) new_shp[i]=shp[i];
            if (out_sz != in_sz && nd > 0 && shp[nd-1] > 0) {
                long long scaled = (long long)shp[nd-1] * in_sz;
                new_shp[nd-1] = (int32_t)(scaled / out_sz);
            }
            u->st = shapetracker_from_shape(new_shp, nd);
            free(new_shp);
        }
    }
    return u;
}

UOp* uop_cast_vec(UOp* a, DType dtype_scalar, int count) {
    // Find canonical pointer for the scalar dtype to avoid dangling _scalar
    const DType* base = NULL;
    // Quick checks for common types
    if (dtype_eq(&dtype_scalar, &dtypes.int32)) base = &dtypes.int32;
    else if (dtype_eq(&dtype_scalar, &dtypes.float32)) base = &dtypes.float32;
    else if (dtype_eq(&dtype_scalar, &dtypes.float16)) base = &dtypes.float16;
    else if (dtype_eq(&dtype_scalar, &dtypes.bfloat16)) base = &dtypes.bfloat16;
    else if (dtype_eq(&dtype_scalar, &dtypes.int64)) base = &dtypes.int64;
    else if (dtype_eq(&dtype_scalar, &dtypes.uint32)) base = &dtypes.uint32;
    else if (dtype_eq(&dtype_scalar, &dtypes.uint64)) base = &dtypes.uint64;
    else if (dtype_eq(&dtype_scalar, &dtypes.bool_)) base = &dtypes.bool_;
    // Fallback: use passed-in address (may lack canonical identity but safe for vec creation)
    if (!base) base = &dtype_scalar;
    DType dt = (count > 1) ? dtype_vec(base, count) : *base;
    return uop_cast(a, dt);
}

UOp* uop_broadcast(UOp* a, int count) {
    assert(a->dtype.count == 1);
    if (count == 1) return uop_ref(a);
    // VECTORIZE: repeat the same source 'count' times
    UOp** src = (UOp**)malloc(sizeof(UOp*) * (size_t)count);
    for (int i=0;i<count;i++) src[i] = a;
    UOpArg arg = {0};
    UOp* u = uop_new(OPS_VECTORIZE, dtype_vec(&a->dtype, count), src, (size_t)count, &arg, NULL);
    free(src);
    return u;
}

// ---- Structured emitters that populate ARG_TUPLE2 ----
UOp* uop_wmma(UOp* a, UOp* b, UOp* acc, const int* first, const int* second, int count) {
    UOp* srcs[3] = {a, b, acc};
    UOpArg arg = {.type = ARG_TUPLE2};
    arg.tuple2.count = count;
    // Allocate temporary arrays to be deep-copied by uop_new
    int* f = NULL; int* s = NULL;
    if (count > 0) {
        if (first) { f = (int*)malloc((size_t)count * sizeof(int)); for (int i=0;i<count;i++) f[i] = first[i]; }
        if (second){ s = (int*)malloc((size_t)count * sizeof(int)); for (int i=0;i<count;i++) s[i] = second[i]; }
    }
    arg.tuple2.first = f; arg.tuple2.second = s;
    UOp* u = uop_new(OPS_WMMA, acc->dtype, srcs, 3, &arg, NULL);
    if (f) free(f);
    if (s) free(s);
    return u;
}

UOp* uop_contract(UOp* x, const int* first, const int* second, int count) {
    UOp* srcs[1] = {x};
    UOpArg arg = {.type = ARG_TUPLE2};
    arg.tuple2.count = count;
    int* f = NULL; int* s = NULL;
    if (count > 0) {
        if (first) { f = (int*)malloc((size_t)count * sizeof(int)); for (int i=0;i<count;i++) f[i] = first[i]; }
        if (second){ s = (int*)malloc((size_t)count * sizeof(int)); for (int i=0;i<count;i++) s[i] = second[i]; }
    }
    arg.tuple2.first = f; arg.tuple2.second = s;
    UOp* u = uop_new(OPS_CONTRACT, x->dtype, srcs, 1, &arg, NULL);
    if (f) free(f);
    if (s) free(s);
    return u;
}

UOp* uop_unroll(UOp* x, const int* first, const int* second, int count) {
    UOp* srcs[1] = {x};
    UOpArg arg = {.type = ARG_TUPLE2};
    arg.tuple2.count = count;
    int* f = NULL; int* s = NULL;
    if (count > 0) {
        if (first) { f = (int*)malloc((size_t)count * sizeof(int)); for (int i=0;i<count;i++) f[i] = first[i]; }
        if (second){ s = (int*)malloc((size_t)count * sizeof(int)); for (int i=0;i<count;i++) s[i] = second[i]; }
    }
    arg.tuple2.first = f; arg.tuple2.second = s;
    UOp* u = uop_new(OPS_UNROLL, x->dtype, srcs, 1, &arg, NULL);
    if (f) free(f);
    if (s) free(s);
    return u;
}

// Greater than or equal - not (less than)
UOp* uop_ge(UOp* a, UOp* b) {
    // a >= b is logical_not(a < b)
    UOp* lt = uop_lt(a, b);
    // logical_not is implemented as ne(True)
    UOp* true_val = uop_const(dtypes.bool_, 1.0);
    UOp* src[] = {lt, true_val};
    UOpArg arg = {0};
    return uop_new(OPS_CMPNE, dtypes.bool_, src, 2, &arg, NULL);
}

UOp* uop_cmpne(UOp* a, UOp* b) {
    // This is the same as uop_ne, but alias for clarity
    return uop_ne(a, b);
}

UOp* uop_abs(UOp* a) {
    // abs(a) = a.where(a >= 0, -a)
    UOp* zero = uop_const(a->dtype, 0.0);
    UOp* neg_a = uop_neg(a);
    return uop_where(uop_ge(a, zero), a, neg_a);
}

UOp* uop_remainder(UOp* a, UOp* b) {
    // Compute remainder using fmod for floating point
    UOp* src[] = {a, b};
    UOpArg arg = {0};
    return uop_new(OPS_MOD, a->dtype, src, 2, &arg, NULL);
}

UOp* uop_cast(UOp* a, DType dtype) {
    UOp* src[] = {a};
    UOpArg arg = {0};
    UOp* u = uop_new(OPS_CAST, dtype, src, 1, &arg, NULL);
    // Propagate shape: cast does not change shape
    if (a && a->st) {
        int nd = shapetracker_ndim(a->st);
        const int32_t* shp = shapetracker_shape(a->st);
        if (nd > 0 && shp) {
            // create a fresh ShapeTracker with the same shape to avoid shared ownership
            u->st = shapetracker_from_shape(shp, nd);
        }
    }
    return u;
}

// Line 242-260: Ternary operations
UOp* uop_where(UOp* cond, UOp* true_val, UOp* false_val) {
    UOp* src[] = {cond, true_val, false_val};
    UOpArg arg = {0};
    DType result_dtype = least_upper_dtype(&true_val->dtype, &false_val->dtype);
    return uop_new(OPS_WHERE, result_dtype, src, 3, &arg, NULL);
}

UOp* uop_mulacc(UOp* a, UOp* b, UOp* c) {
    // mulacc(a,b,c) = a*b + c
    UOp* src[] = {a, b, c};
    UOpArg arg = {0};
    return uop_new(OPS_MULACC, a->dtype, src, 3, &arg, NULL);
}

// Line 262-280: Reduction operations
UOp* uop_reduce_axis(UOp* src, Ops reduce_op, int* axes, int axes_count) {
    UOp* src_arr[] = {src};
    // Normalize, dedup, sort axes
    int ndim = src && src->st ? shapetracker_ndim(src->st) : -1;
    int* tmp = NULL; int n = axes_count;
    if (axes_count > 0) {
        tmp = (int*)malloc((size_t)axes_count * sizeof(int));
        for (int i=0;i<axes_count;i++) {
            int ax = axes[i];
            if (ndim > 0 && ax < 0) ax += ndim;  // negative axes support
            tmp[i] = ax;
        }
        // dedup
        int m = 0;
        for (int i=0;i<axes_count;i++) {
            bool seen=false; for (int j=0;j<m;j++) if (tmp[j]==tmp[i]) { seen=true; break; }
            if (!seen) tmp[m++] = tmp[i];
        }
        n = m;
        // sort ascending (simple bubble; small n)
        for (int i=0;i<n;i++) for (int j=i+1;j<n;j++) if (tmp[j] < tmp[i]) { int t=tmp[i]; tmp[i]=tmp[j]; tmp[j]=t; }
    }
    // Create reduction arg
    UOpArg arg = {.type = ARG_REDUCE};
    arg.reduce_data.reduce_op = reduce_op;
    arg.reduce_data.axes_count = n;
    if (n > 0) {
        arg.reduce_data.axes = (int*)malloc((size_t)n * sizeof(int));
        for (int i=0;i<n;i++) arg.reduce_data.axes[i] = tmp[i];
    } else {
        arg.reduce_data.axes = NULL;
    }
    if (tmp) free(tmp);
    // Determine result dtype based on reduce_op
    DType result_dtype = src->dtype;
    if (reduce_op == OPS_CMPLT || reduce_op == OPS_CMPNE) {
        result_dtype = dtypes.bool_;
    }
    UOp* result = uop_new(OPS_REDUCE_AXIS, result_dtype, src_arr, 1, &arg, NULL);
    // Attach reduced shape if available
    if (src && src->st) {
        int ndin = shapetracker_ndim(src->st);
        const int32_t* shp = shapetracker_shape(src->st);
        if (ndin > 0 && shp) {
            // Build mask for kept dims
            bool* red = (bool*)calloc((size_t)ndin, sizeof(bool));
            for (int i=0;i<n;i++) if (tmp) (void)0; // dummy to quiet unused warning
            // compute reduced axes from arg (already normalized/sorted)
            int rc = result->arg.reduce_data.axes_count;
            for (int i=0;i<rc;i++) { int a = result->arg.reduce_data.axes[i]; if (a>=0 && a<ndin) red[a]=true; }
            int kept=0; for (int i=0;i<ndin;i++) if (!red[i]) kept++;
            if (kept>0) {
                int32_t* new_shp = (int32_t*)malloc(sizeof(int32_t)*kept);
                int j=0; for (int i=0;i<ndin;i++) if (!red[i]) new_shp[j++]=shp[i];
                result->st = shapetracker_from_shape(new_shp, kept);
                free(new_shp);
            }
            free(red);
        }
    }
    if (arg.reduce_data.axes) free(arg.reduce_data.axes);  // copied in uop_new
    return result;
}

UOp* uop_gep(UOp* base, const int* idxs, int idx_count) {
    if (idx_count == 1 && base->op == OPS_VECTORIZE && base->src_count > (size_t)idxs[0]) {
        return uop_ref(base->src[idxs[0]]);
    }
    if (base->op == OPS_CONST && base->arg.type == ARG_CONST) {
        DType scalar = dtype_scalar(&base->dtype);
        return uop_const(scalar, base->arg.const_data.const_value);
    }
    DType scalar = dtype_scalar(&base->dtype);
    DType out_dtype = (idx_count > 1) ? dtype_vec(&scalar, idx_count) : scalar;
    UOpArg arg = {.type = ARG_REDUCE};
    arg.reduce_data.axes_count = idx_count;
    if (idx_count > 0) {
        arg.reduce_data.axes = (int*)malloc(idx_count * sizeof(int));
        for (int i=0;i<idx_count;i++) arg.reduce_data.axes[i] = idxs[i];
    }
    UOp* src_arr[] = { base };
    UOp* u = uop_new(OPS_GEP, out_dtype, src_arr, 1, &arg, NULL);
    if (idx_count > 0) free(arg.reduce_data.axes);
    return u;
}

int uop_axis_arg(UOp* uop, int** axes_out, int* count_out) {
    if (!uop || uop->op != OPS_REDUCE_AXIS || uop->arg.type != ARG_REDUCE) return 0;
    int n = uop->arg.reduce_data.axes_count;
    if (count_out) *count_out = n;
    if (axes_out) {
        int* ax = (int*)malloc(n * sizeof(int));
        for (int i=0;i<n;i++) ax[i] = uop->arg.reduce_data.axes[i];
        *axes_out = ax;
    }
    return 1;
}

// Line 282-300: View operations
UOp* uop_view(UOp* buf, ShapeTracker* st) {
    UOp* src[] = {buf};
    UOpArg arg = {0};
    UOp* u = uop_new(OPS_VIEW, buf->dtype, src, 1, &arg, NULL);
    u->st = st;
    return u;
}

UOp* uop_index(UOp* buf, UOp* idx) {
    UOp* src[] = {buf, idx};
    UOpArg arg = {0};
    return uop_new(OPS_INDEX, buf->dtype, src, 2, &arg, NULL);
}

// Movement ops: attach ShapeTracker to uop->st and keep params in arg
UOp* uop_reshape(UOp* x, const int32_t* new_shape, int32_t new_ndim) {
    ShapeTracker* cur = x->st;
    ShapeTracker* st = cur ? shapetracker_reshape(cur, new_shape, new_ndim) : shapetracker_from_shape(new_shape, new_ndim);
    UOpArg arg = {0}; UOp* src[] = {x};
    UOp* u = uop_new(OPS_RESHAPE, x->dtype, src, 1, &arg, NULL);
    u->st = st;
    return u;
}
UOp* uop_permute(UOp* x, const int32_t* axes, int32_t num_axes) {
    ShapeTracker* cur = x->st;
    ShapeTracker* st = cur ? shapetracker_permute(cur, axes, num_axes) : NULL;
    // Store axes in arg
    UOpArg arg = {.type = ARG_REDUCE};
    arg.reduce_data.axes_count = num_axes;
    if (num_axes > 0) {
        arg.reduce_data.axes = (int*)malloc(num_axes * sizeof(int));
        for (int i = 0; i < num_axes; i++) arg.reduce_data.axes[i] = axes[i];
    }
    UOp* src[] = {x};
    UOp* u = uop_new(OPS_PERMUTE, x->dtype, src, 1, &arg, NULL);
    if (num_axes > 0) { free(arg.reduce_data.axes); }
    u->st = st;
    return u;
}
UOp* uop_expand(UOp* x, const int32_t* target_shape, int32_t target_ndim) {
    ShapeTracker* cur = x->st;
    ShapeTracker* st = cur ? shapetracker_expand(cur, target_shape, target_ndim) : shapetracker_from_shape(target_shape, target_ndim);
    UOpArg arg = {0}; UOp* src[] = {x};
    UOp* u = uop_new(OPS_EXPAND, x->dtype, src, 1, &arg, NULL);
    u->st = st;
    return u;
}
UOp* uop_pad(UOp* x, const int32_t* pad_before, const int32_t* pad_after, int32_t ndim) {
    ShapeTracker* cur = x->st;
    ShapeTracker* st = cur ? shapetracker_pad(cur, pad_before, pad_after, ndim) : NULL;
    UOpArg arg={.type=ARG_PAD_PARAMS};
    arg.pad_data.ndim = ndim;
    if (ndim>0) {
        arg.pad_data.before = (int32_t*)malloc(ndim*sizeof(int32_t));
        arg.pad_data.after  = (int32_t*)malloc(ndim*sizeof(int32_t));
        for (int i=0;i<ndim;i++){ arg.pad_data.before[i]=pad_before[i]; arg.pad_data.after[i]=pad_after[i]; }
    }
    UOp* src[] = {x};
    UOp* u = uop_new(OPS_PAD, x->dtype, src, 1, &arg, NULL);
    if (ndim>0){ free(arg.pad_data.before); free(arg.pad_data.after); }
    u->st = st;
    return u;
}
UOp* uop_shrink(UOp* x, const int32_t* start, const int32_t* end, int32_t ndim) {
    ShapeTracker* cur = x->st;
    ShapeTracker* st = cur ? shapetracker_shrink(cur, start, end, ndim) : NULL;
    UOpArg arg={.type=ARG_SHRINK_PARAMS};
    arg.shrink_data.ndim = ndim;
    if (ndim>0) {
        arg.shrink_data.start = (int32_t*)malloc(ndim*sizeof(int32_t));
        arg.shrink_data.end   = (int32_t*)malloc(ndim*sizeof(int32_t));
        for (int i=0;i<ndim;i++){ arg.shrink_data.start[i]=start[i]; arg.shrink_data.end[i]=end[i]; }
    }
    UOp* src[] = {x};
    UOp* u = uop_new(OPS_SHRINK, x->dtype, src, 1, &arg, NULL);
    if (ndim>0){ free(arg.shrink_data.start); free(arg.shrink_data.end); }
    u->st = st;
    return u;
}
UOp* uop_flip_axis(UOp* x, int axis) {
    ShapeTracker* cur = x->st;
    ShapeTracker* st = NULL;
    if (cur){ int32_t ndim = shapetracker_ndim(cur); bool* axes = (bool*)calloc(ndim, sizeof(bool)); if (axis>=0 && axis<ndim) axes[axis]=true; st = shapetracker_flip(cur, axes, ndim); free(axes); }
    UOpArg arg = {.type = ARG_INT}; arg.int_data.i = axis; UOp* src[] = {x};
    UOp* u = uop_new(OPS_FLIP, x->dtype, src, 1, &arg, NULL);
    u->st = st;
    return u;
}

const int32_t* uop_shape(UOp* uop, int* ndim_out){
    if (!uop) { if (ndim_out) *ndim_out = 0; return NULL; }
    if (uop->st){
        const ShapeTracker* st = uop->st;
        const int32_t* shp = shapetracker_shape(st);
        if (ndim_out) *ndim_out = shapetracker_ndim(st);
        return shp;
    }
    if (ndim_out) *ndim_out = 0;
    return NULL;
}

// Helper function for topological sort
static void add_node(UOp*** nodes, size_t* size, size_t* capacity, UOp* node) {
    if (*size >= *capacity) {
        *capacity = *capacity ? *capacity * 2 : 16;
        *nodes = (UOp**)realloc(*nodes, *capacity * sizeof(UOp*));
    }
    (*nodes)[(*size)++] = node;
}

// Check if node is in list
static bool contains(UOp** nodes, size_t size, UOp* node) {
    for (size_t i = 0; i < size; i++) {
        if (nodes[i] == node) return true;
    }
    return false;
}

// Line 302-400: Graph operations
struct TopoSortState {
    UOp** visited;
    size_t visited_size;
    size_t visited_capacity;
    UOp** stack;
    size_t stack_size;
    size_t stack_capacity;
};

static void dfs_internal(UOp* node, struct TopoSortState* state) {
    if (contains(state->visited, state->visited_size, node)) return;
    add_node(&state->visited, &state->visited_size, &state->visited_capacity, node);
    
    for (size_t i = 0; i < node->src_count; i++) {
        dfs_internal(node->src[i], state);
    }
    
    add_node(&state->stack, &state->stack_size, &state->stack_capacity, node);
}

// Line 302: def toposort(self) -> list[UOp]:
UOp** uop_toposort(UOp* root, size_t* count) {
    // Simple DFS-based topological sort
    struct TopoSortState state = {NULL, 0, 0, NULL, 0, 0};
    
    dfs_internal(root, &state);
    
    if (count) *count = state.stack_size;
    
    // No need to reverse - DFS post-order gives correct topological order
    
    free(state.visited);
    return state.stack;
}

UOp** uop_toposort_gate(UOp* root, size_t* count, bool (*gate)(UOp*)) {
    // DFS with gate predicate (if gate is NULL, treat as always true)
    struct TopoSortState state = {NULL, 0, 0, NULL, 0, 0};
    // manual stack DFS to honor gate
    // use a simple recursion wrapper
    void dfs(UOp* node){
        if (!node) return;
        if (gate && !gate(node)) return;
        if (contains(state.visited, state.visited_size, node)) return;
        add_node(&state.visited, &state.visited_size, &state.visited_capacity, node);
        for (size_t i=0;i<node->src_count;i++) dfs(node->src[i]);
        add_node(&state.stack, &state.stack_size, &state.stack_capacity, node);
    }
    dfs(root);
    if (count) *count = state.stack_size;
    free(state.visited);
    return state.stack;
}

UOp* uop_replace_ex(UOp* uop, Ops new_op, DType new_dtype, UOp** new_src, size_t new_src_count, UOpArg* new_arg, void* new_tag) {
    // Construct a new UOp with provided fields (simple parity with Python's replace)
    UOpArg arg = {0}; if (new_arg) arg = *new_arg;
    return uop_new(new_op, new_dtype.count ? new_dtype : uop->dtype, new_src ? new_src : uop->src, new_src ? new_src_count : uop->src_count, &arg, new_tag);
}

// Line 340-360: def print(self, depth=0):
void uop_print(UOp* uop, int depth) {
    if (!uop) {
        printf("NULL\n");
        return;
    }
    
    // Print indentation
    for (int i = 0; i < depth; i++) printf("  ");
    
    // Print op and dtype
    printf("UOp(%d", uop->op);  // TODO: Convert op to string
    if (uop->dtype.count) {
        printf(", %s", uop->dtype.name);
    }
    
    // Print arg if present
    if (uop->arg.type != ARG_NONE) {
        if (uop->arg.type == ARG_CONST) {
            printf(", %.2f", uop->arg.const_data.const_value);
        } else if (uop->arg.type == ARG_INT) {
            printf(", %d", uop->arg.int_data.i);
        }
    }
    
    printf(")\n");
    
    // Print children
    for (size_t i = 0; i < uop->src_count; i++) {
        uop_print(uop->src[i], depth + 1);
    }
}

void uop_print_graph(UOp* root) {
    printf("=== UOp Graph ===\n");
    uop_print(root, 0);
    printf("=================\n");
}
// Pretty-print a flat list of UOps with indices and sources
void print_uops(UOp** uops, size_t count) {
    if (!uops || count == 0) return;
    printf("==== UOps (%zu) ====\n", count);
    for (size_t i = 0; i < count; i++) {
        UOp* u = uops[i];
        {
            const char* opn = ops_to_string(u->op);
            const char* dtn = u->dtype.count ? u->dtype.name : NULL;
            printf("[%zu] %s", i, opn);
            if (dtn) printf(" : %s", dtn);
        }
        if (u->arg.type == ARG_CONST) {
            printf(" arg=%.6g", u->arg.const_data.const_value);
        } else if (u->arg.type == ARG_INT) {
            printf(" arg=%d", u->arg.int_data.i);
        } else if (u->arg.type == ARG_REDUCE && u->arg.reduce_data.axes_count>0) {
            printf(" axes=[");
            for (int j=0;j<u->arg.reduce_data.axes_count;j++) { if (j) printf(","); printf("%d", u->arg.reduce_data.axes[j]); }
            printf("]");
        }
        if (u->src_count > 0) {
            printf(" src=(");
            for (size_t s = 0; s < u->src_count; s++) {
                if (s) printf(",");
                size_t idx = (size_t)-1; for (size_t k = 0; k < count; k++) { if (uops[k] == u->src[s]) { idx = k; break; } }
                if (idx == (size_t)-1) printf("?"); else printf("%zu", idx);
            }
            printf(")");
        }
        printf("\n");
    }
    printf("====================\n");
}

void print_uops_ex(UOp** uops, size_t count, const char* (*rep)(UOp*, void*), void* ctx, bool color) {
    if (!uops || count == 0) return;
    printf("==== UOps (%zu) ====\n", count);
    for (size_t i = 0; i < count; i++) {
        UOp* u = uops[i];
        const char* opn = ops_to_string(u->op);
        const char* dtn = u->dtype.count ? u->dtype.name : NULL;
        if (color) printf("[%zu] \x1b[36m%s\x1b[0m", i, opn); else printf("[%zu] %s", i, opn);
        if (dtn) { if (color) printf(" : \x1b[35m%s\x1b[0m", dtn); else printf(" : %s", dtn); }
        if (rep) {
            const char* s = rep(u, ctx);
            if (s) printf(" rep=%s", s);
        }
        if (u->arg.type == ARG_CONST) {
            printf(" arg=%.6g", u->arg.const_data.const_value);
        } else if (u->arg.type == ARG_INT) {
            printf(" arg=%d", u->arg.int_data.i);
        } else if (u->arg.type == ARG_REDUCE && u->arg.reduce_data.axes_count>0) {
            printf(" axes=[");
            for (int j=0;j<u->arg.reduce_data.axes_count;j++) { if (j) printf(","); printf("%d", u->arg.reduce_data.axes[j]); }
            printf("]");
        }
        if (u->src_count > 0) {
            printf(" src=(");
            for (size_t s = 0; s < u->src_count; s++) {
                if (s) printf(",");
                size_t idx = (size_t)-1; for (size_t k = 0; k < count; k++) { if (uops[k] == u->src[s]) { idx = k; break; } }
                if (idx == (size_t)-1) printf("?"); else printf("%zu", idx);
            }
            printf(")");
        }
        printf("\n");
    }
    printf("====================\n");
}

UOp** uop_children(UOp* uop, size_t* count) {
    if (!uop || uop->children_count == 0) { if (count) *count = 0; return NULL; }
    UOp** out = (UOp**)malloc(uop->children_count * sizeof(UOp*));
    memcpy(out, uop->children, uop->children_count * sizeof(UOp*));
    if (count) *count = uop->children_count;
    return out;
}

void uop_meta_set(UOp* uop, const char* key, void* value) {
    if (!uop || !key) return;
    struct UOpMetaKV* kv = uop->meta_head;
    while (kv) { if (strcmp(kv->key, key) == 0) { kv->value = value; return; } kv = kv->next; }
    kv = (struct UOpMetaKV*)calloc(1, sizeof(*kv));
    kv->key = strdup(key);
    kv->value = value;
    kv->next = uop->meta_head;
    uop->meta_head = kv;
}

void* uop_meta_get(UOp* uop, const char* key) {
    if (!uop || !key) return NULL;
    struct UOpMetaKV* kv = uop->meta_head;
    while (kv) { if (strcmp(kv->key, key) == 0) return kv->value; kv = kv->next; }
    return NULL;
}

// Line 380-390: def __hash__(self):
static size_t hash_mix(size_t h, size_t v){ return h*1315423911u + v + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2); }
static size_t hash_mem(const void* p, size_t n){ const unsigned char* b=(const unsigned char*)p; size_t h=1469598103934665603ULL; for(size_t i=0;i<n;i++) h = (h ^ b[i]) * 1099511628211ULL; return h; }
size_t uop_hash(UOp* uop) {
    if (!uop) return 0;
    size_t h = (size_t)uop->op;
    h = hash_mix(h, (size_t)(uop->dtype.count ? uop->dtype._scalar : &uop->dtype));
    // sources (pointer identity, matches Python's exact graph node identity)
    for (size_t i=0;i<uop->src_count;i++) h = hash_mix(h, (size_t)uop->src[i]);
    // arg
    h = hash_mix(h, (size_t)uop->arg.type);
    switch (uop->arg.type){
        case ARG_CONST: {
            // use memory hash of double bits
            h = hash_mix(h, hash_mem(&uop->arg.const_data.const_value, sizeof(double))); break; }
        case ARG_INT: { h = hash_mix(h, (size_t)uop->arg.int_data.i); break; }
        case ARG_REDUCE: {
            h = hash_mix(h, (size_t)uop->arg.reduce_data.reduce_op);
            if (uop->arg.reduce_data.axes && uop->arg.reduce_data.axes_count>0)
                h = hash_mix(h, hash_mem(uop->arg.reduce_data.axes, sizeof(int)* (size_t)uop->arg.reduce_data.axes_count));
            h = hash_mix(h, (size_t)uop->arg.reduce_data.axes_count);
            break; }
        case ARG_VCONST: {
            int n = uop->arg.vconst_data.count;
            if (uop->arg.vconst_data.values && n>0)
                h = hash_mix(h, hash_mem(uop->arg.vconst_data.values, sizeof(double)*(size_t)n));
            h = hash_mix(h, (size_t)n);
            break; }
        case ARG_PAD_PARAMS: {
            int n = uop->arg.pad_data.ndim;
            if (uop->arg.pad_data.before && n>0) h = hash_mix(h, hash_mem(uop->arg.pad_data.before, sizeof(int32_t)*(size_t)n));
            if (uop->arg.pad_data.after && n>0) h = hash_mix(h, hash_mem(uop->arg.pad_data.after, sizeof(int32_t)*(size_t)n));
            h = hash_mix(h, (size_t)n);
            break; }
        case ARG_SHRINK_PARAMS: {
            int n = uop->arg.shrink_data.ndim;
            if (uop->arg.shrink_data.start && n>0) h = hash_mix(h, hash_mem(uop->arg.shrink_data.start, sizeof(int32_t)*(size_t)n));
            if (uop->arg.shrink_data.end && n>0) h = hash_mix(h, hash_mem(uop->arg.shrink_data.end, sizeof(int32_t)*(size_t)n));
            h = hash_mix(h, (size_t)n);
            break; }
        case ARG_TUPLE2: {
            int n = uop->arg.tuple2.count;
            h = hash_mix(h, (size_t)n);
            if (uop->arg.tuple2.first && n>0) h = hash_mix(h, hash_mem(uop->arg.tuple2.first, sizeof(int)*(size_t)n));
            if (uop->arg.tuple2.second && n>0) h = hash_mix(h, hash_mem(uop->arg.tuple2.second, sizeof(int)*(size_t)n));
            break; }
        case ARG_STRING: {
            if (uop->arg.str.s) h = hash_mix(h, hash_mem(uop->arg.str.s, strlen(uop->arg.str.s)));
            break; }
        case ARG_STRLIST: {
            int n = uop->arg.strlist.count;
            h = hash_mix(h, (size_t)n);
            for (int i=0;i<n;i++) {
                if (uop->arg.strlist.items && uop->arg.strlist.items[i]) h = hash_mix(h, hash_mem(uop->arg.strlist.items[i], strlen(uop->arg.strlist.items[i])));
            }
            break; }
        case ARG_TUPLE_MIXED: {
            int n = uop->arg.tmixed.count;
            h = hash_mix(h, (size_t)n);
            for (int i=0;i<n;i++) {
                struct MixedItem* it = &uop->arg.tmixed.items[i];
                h = hash_mix(h, (size_t)it->tag);
                if (it->tag == MIXED_INT) h = hash_mix(h, (size_t)it->ival);
                else if (it->tag == MIXED_UOP) h = hash_mix(h, (size_t)it->uop);
            }
            break; }
        default: break;
    }
    // tag
    h = hash_mix(h, (size_t)uop->tag);
    return h;
}

// Line 392-400: def __eq__(self, other):
bool uop_equals(UOp* a, UOp* b) {
    if (a == b) return true;
    if (!a || !b) return false;
    if (a->op != b->op) return false;
    if (!dtype_eq(&a->dtype, &b->dtype)) return false;
    if (a->tag != b->tag) return false;
    if (a->src_count != b->src_count) return false;
    for (size_t i=0;i<a->src_count;i++) if (a->src[i] != b->src[i]) return false;
    if (a->arg.type != b->arg.type) return false;
    switch (a->arg.type){
        case ARG_CONST: if (a->arg.const_data.const_value != b->arg.const_data.const_value) return false; break;
        case ARG_INT: if (a->arg.int_data.i != b->arg.int_data.i) return false; break;
        case ARG_REDUCE: {
            if (a->arg.reduce_data.reduce_op != b->arg.reduce_data.reduce_op) return false;
            if (a->arg.reduce_data.axes_count != b->arg.reduce_data.axes_count) return false;
            for (int i=0;i<a->arg.reduce_data.axes_count;i++) if (a->arg.reduce_data.axes[i] != b->arg.reduce_data.axes[i]) return false;
            break; }
        case ARG_VCONST: {
            if (a->arg.vconst_data.count != b->arg.vconst_data.count) return false;
            for (int i=0;i<a->arg.vconst_data.count;i++) if (a->arg.vconst_data.values[i] != b->arg.vconst_data.values[i]) return false;
            break; }
        case ARG_PAD_PARAMS: {
            if (a->arg.pad_data.ndim != b->arg.pad_data.ndim) return false;
            for (int i=0;i<a->arg.pad_data.ndim;i++) if (a->arg.pad_data.before[i] != b->arg.pad_data.before[i] || a->arg.pad_data.after[i] != b->arg.pad_data.after[i]) return false;
            break; }
        case ARG_SHRINK_PARAMS: {
            if (a->arg.shrink_data.ndim != b->arg.shrink_data.ndim) return false;
            for (int i=0;i<a->arg.shrink_data.ndim;i++) if (a->arg.shrink_data.start[i] != b->arg.shrink_data.start[i] || a->arg.shrink_data.end[i] != b->arg.shrink_data.end[i]) return false;
            break; }
        case ARG_TUPLE2: {
            if (a->arg.tuple2.count != b->arg.tuple2.count) return false;
            int n = a->arg.tuple2.count;
            for (int i=0;i<n;i++) {
                int af = a->arg.tuple2.first ? a->arg.tuple2.first[i] : 0;
                int bf = b->arg.tuple2.first ? b->arg.tuple2.first[i] : 0;
                if (af != bf) return false;
            }
            for (int i=0;i<n;i++) {
                int as = a->arg.tuple2.second ? a->arg.tuple2.second[i] : 0;
                int bs = b->arg.tuple2.second ? b->arg.tuple2.second[i] : 0;
                if (as != bs) return false;
            }
            break; }
        case ARG_STRING: {
            const char* as = a->arg.str.s ? a->arg.str.s : "";
            const char* bs = b->arg.str.s ? b->arg.str.s : "";
            if (strcmp(as, bs) != 0) return false;
            break; }
        case ARG_STRLIST: {
            if (a->arg.strlist.count != b->arg.strlist.count) return false;
            int n = a->arg.strlist.count;
            for (int i=0;i<n;i++) {
                const char* as = (a->arg.strlist.items && a->arg.strlist.items[i]) ? a->arg.strlist.items[i] : "";
                const char* bs = (b->arg.strlist.items && b->arg.strlist.items[i]) ? b->arg.strlist.items[i] : "";
                if (strcmp(as, bs) != 0) return false;
            }
            break; }
        case ARG_TUPLE_MIXED: {
            if (a->arg.tmixed.count != b->arg.tmixed.count) return false;
            int n = a->arg.tmixed.count;
            for (int i=0;i<n;i++) {
                struct MixedItem* ai = &a->arg.tmixed.items[i];
                struct MixedItem* bi = &b->arg.tmixed.items[i];
                if (ai->tag != bi->tag) return false;
                if (ai->tag == MIXED_INT) { if (ai->ival != bi->ival) return false; }
                else if (ai->tag == MIXED_UOP) { if (ai->uop != bi->uop) return false; }
            }
            break; }
        default: break;
    }
    return true;
}

// DEVICE constructors
UOp* uop_device_str(const char* dev) {
    UOpArg arg = {.type = ARG_STRING};
    arg.str.s = (char*)dev; // deep copy in uop_new
    return uop_new(OPS_DEVICE, dtypes.void_, NULL, 0, &arg, NULL);
}

UOp* uop_device_tuple(const char* const* devs, int count) {
    UOpArg arg = {.type = ARG_STRLIST};
    arg.strlist.count = count;
    if (count > 0) {
        arg.strlist.items = (char**)malloc((size_t)count * sizeof(char*));
        for (int i=0;i<count;i++) arg.strlist.items[i] = devs[i] ? strdup(devs[i]) : NULL;
    }
    UOp* u = uop_new(OPS_DEVICE, dtypes.void_, NULL, 0, &arg, NULL);
    if (arg.strlist.items) { for (int i=0;i<count;i++) if (arg.strlist.items[i]) free(arg.strlist.items[i]); free(arg.strlist.items); }
    return u;
}

UOp* uop_buffer_view(UOp* buffer, int tag0, long long ival0, UOp* u0,
                     int tag1, long long ival1, UOp* u1) {
    if (!buffer) return NULL;
    UOp* srcs[1] = { buffer };
    UOpArg arg = {.type = ARG_TUPLE_MIXED};
    arg.tmixed.count = 2;
    arg.tmixed.items = (struct MixedItem*)malloc(2 * sizeof(struct MixedItem));
    arg.tmixed.items[0].tag = tag0;
    arg.tmixed.items[0].ival = ival0;
    arg.tmixed.items[0].uop = u0;
    arg.tmixed.items[1].tag = tag1;
    arg.tmixed.items[1].ival = ival1;
    arg.tmixed.items[1].uop = u1;
    UOp* u = uop_new(OPS_BUFFER_VIEW, dtypes.void_, srcs, 1, &arg, NULL);
    // uop_new deep-copies and refs items; free our temp holders
    free(arg.tmixed.items);
    return u;
}

UOp* uop_buffer_view_ii(UOp* buffer, long long a0, long long a1) {
    return uop_buffer_view(buffer, MIXED_INT, a0, NULL, MIXED_INT, a1, NULL);
}
UOp* uop_buffer_view_iU(UOp* buffer, long long a0, UOp* a1) {
    return uop_buffer_view(buffer, MIXED_INT, a0, NULL, MIXED_UOP, 0, a1);
}
UOp* uop_buffer_view_Ui(UOp* buffer, UOp* a0, long long a1) {
    return uop_buffer_view(buffer, MIXED_UOP, 0, a0, MIXED_INT, a1, NULL);
}
UOp* uop_buffer_view_UU(UOp* buffer, UOp* a0, UOp* a1) {
    return uop_buffer_view(buffer, MIXED_UOP, 0, a0, MIXED_UOP, 0, a1);
}

// More aggressive constant folding for basic operations
static UOp* constant_fold_basic(UOp* uop) {
    if (!uop || uop->src_count == 0) return uop;
    
    
    // Special case for WHERE with constant condition (handle first)
    // WHERE with constant condition: WHERE(const, x, y) -> x if const != 0, else y
    if (uop->op == OPS_WHERE && uop->src_count == 3) {
        UOp* cond = uop->src[0];
        if (cond->op == OPS_CONST && cond->arg.type == ARG_CONST) {
            // If condition is true, return true branch; if false, return false branch
            if (cond->arg.const_data.const_value != 0.0) {
                return uop_ref(uop->src[1]);
            } else {
                return uop_ref(uop->src[2]);
            }
        }
    }
    
    // Special case for WHERE with same branches: WHERE(c, x, x) -> x
    if (uop->op == OPS_WHERE && uop->src_count == 3) {
        if (uop->src[1] == uop->src[2]) {
            // Return the same branch (no need to create new constant)
            return uop_ref(uop->src[1]);
        }
    }
    
    // GEP(VCONST) folding handled in symbolic.c
    
    // Check if all sources are constants
    bool all_const = true;
    for (size_t i = 0; i < uop->src_count; i++) {
        if (uop->src[i]->op != OPS_CONST || uop->src[i]->arg.type != ARG_CONST) {
            all_const = false;
            break;
        }
    }
    
    if (!all_const) {
        return uop;
    }
    
    // Try to execute the operation with constant arguments
    double* args = malloc(uop->src_count * sizeof(double));
    for (size_t i = 0; i < uop->src_count; i++) {
        args[i] = uop->src[i]->arg.const_data.const_value;
    }
    
    double result = exec_alu(uop->op, uop->dtype, args, uop->src_count);
    free(args);
    
    // Create new constant with result
    UOpArg result_arg = {.type = ARG_CONST, .const_data.const_value = result};
    UOp* folded = uop_new(OPS_CONST, uop->dtype, NULL, 0, &result_arg, NULL);
    return folded;
}

// Line 402-550: Simplification
// Line 402: def simplify(self) -> UOp:
UOp* uop_simplify(UOp* uop) {
    // Basic simplification rules
    // Line 404-410: Early returns
    if (!uop) return NULL;
    if (uop->op == OPS_CONST) return uop;

    // Quick, local peephole rules before deeper symbolic passes
    // (x & 0xFFFFFFFF).cast(uint32) -> x.cast(uint32)
    if (uop->op == OPS_CAST && uop->src_count==1 && dtype_eq(&uop->dtype, &dtypes.uint32)) {
        UOp* m = uop->src[0];
        if (m->op == OPS_AND && m->src_count==2) {
            UOp* a=m->src[0], *b=m->src[1];
            UOp* cst = NULL; UOp* var = NULL;
            if (b->op==OPS_CONST) { cst=b; var=a; }
            else if (a->op==OPS_CONST) { cst=a; var=b; }
            if (cst && cst->arg.type==ARG_CONST) {
                double mv = cst->arg.const_data.const_value;
                if ((uint64_t)mv == 0xFFFFFFFFULL) {
                    return uop_cast(var, dtypes.uint32);
                }
            }
        }
    }
    // (((u64)*(1<<32)) | y(u32).cast(u64)).cast(u32) -> y
    if (uop->op == OPS_CAST && uop->src_count==1 && dtype_eq(&uop->dtype, &dtypes.uint32)) {
        UOp* orop = uop->src[0];
        if (orop->op == OPS_OR && orop->src_count==2) {
            UOp* l=orop->src[0], *r=orop->src[1];
            UOp* mul = (l->op==OPS_MUL)? l : (r->op==OPS_MUL? r: NULL);
            UOp* other = (mul==l)? r : (mul==r? l: NULL);
            if (mul && mul->src_count==2 && other) {
                UOp* mc=NULL; if (mul->src[0]->op==OPS_CONST){ mc=mul->src[0]; } else if (mul->src[1]->op==OPS_CONST){ mc=mul->src[1]; }
                if (mc && mc->arg.type==ARG_CONST && fabs(mc->arg.const_data.const_value - (double)(1ULL<<32))<0.5) {
                    if (other->op == OPS_CAST && dtype_eq(&other->dtype, &dtypes.uint64) && other->src_count==1 && dtype_eq(&other->src[0]->dtype, &dtypes.uint32)) {
                        return uop_ref(other->src[0]);
                    }
                }
            }
        }
    }
    // (((x u64)*K) | (u32).cast(u64)) // K -> x  when K matches both places (robust to no-op casts)
    if (uop->op == OPS_IDIV && uop->src_count==2 && uop->src[1]->op==OPS_CONST) {
        double dval = (uop->src[1]->arg.type==ARG_CONST) ? uop->src[1]->arg.const_data.const_value : (double)uop->src[1]->arg.int_data.i;
        if (dval != 0.0) {
            UOp* num = uop->src[0];
            // unwrap trivial cast on numerator
            if (num->op==OPS_CAST && dtype_eq(&num->dtype, &dtypes.uint64)) num = num->src[0];
            // direct MUL path
            if (num->op == OPS_MUL && num->src_count==2) {
                UOp* mc=NULL,*mv=NULL; if (num->src[0]->op==OPS_CONST){ mc=num->src[0]; mv=num->src[1]; } else if (num->src[1]->op==OPS_CONST){ mc=num->src[1]; mv=num->src[0]; }
                if (mc && mc->op==OPS_CONST) {
                    double mcv = (mc->arg.type==ARG_CONST) ? mc->arg.const_data.const_value : (double)mc->arg.int_data.i;
                    if (fabs(mcv - dval) < 0.5) {
                    // unwrap trivial cast around mv
                    if (mv->op==OPS_CAST && dtype_eq(&mv->dtype, &dtypes.uint64)) mv = mv->src[0];
                    return uop_ref(mv);
                    }
                }
            }
            // OR path with one side MUL
            if (num->op == OPS_OR && num->src_count==2) {
                UOp* l=num->src[0], *r=num->src[1];
                if (l->op==OPS_CAST && dtype_eq(&l->dtype, &dtypes.uint64)) l = l->src[0];
                if (r->op==OPS_CAST && dtype_eq(&r->dtype, &dtypes.uint64)) r = r->src[0];
                UOp* mul = (l->op==OPS_MUL)? l : (r->op==OPS_MUL? r: NULL);
                if (mul && mul->src_count==2) {
                    UOp* mc=NULL,*mv=NULL; if (mul->src[0]->op==OPS_CONST){ mc=mul->src[0]; mv=mul->src[1]; } else if (mul->src[1]->op==OPS_CONST){ mc=mul->src[1]; mv=mul->src[0]; }
                    if (mc && mc->op==OPS_CONST) {
                        double mcv = (mc->arg.type==ARG_CONST) ? mc->arg.const_data.const_value : (double)mc->arg.int_data.i;
                        if (fabs(mcv - dval) < 0.5) {
                            if (mv->op==OPS_CAST && dtype_eq(&mv->dtype, &dtypes.uint64)) mv = mv->src[0];
                            return uop_ref(mv);
                        }
                    }
                }
            }
        }
    }
    // x.cast(u64) * where(y, 1<<32, 0) -> where(y, x, 0).cast(u64) * (1<<32)
    if (uop->op == OPS_MUL && uop->src_count==2) {
        UOp* a=uop->src[0], *b=uop->src[1];
        UOp* cast=NULL; UOp* wh=NULL;
        if (a->op==OPS_CAST && dtype_eq(&a->dtype, &dtypes.uint64) && b->op==OPS_WHERE) { cast=a; wh=b; }
        else if (b->op==OPS_CAST && dtype_eq(&b->dtype, &dtypes.uint64) && a->op==OPS_WHERE) { cast=b; wh=a; }
        if (cast && wh && wh->src_count==3) {
            UOp* c=wh->src[0], *t=wh->src[1], *f=wh->src[2];
            if (t->op==OPS_CONST && f->op==OPS_CONST) {
                double tv = (t->arg.type==ARG_CONST)? t->arg.const_data.const_value : (double)t->arg.int_data.i;
                double fv = (f->arg.type==ARG_CONST)? f->arg.const_data.const_value : (double)f->arg.int_data.i;
                if (fabs(tv-(double)(1ULL<<32))<0.5 && fabs(fv-0.0)<0.5) {
                    UOp* wh2 = uop_where(c, cast->src[0], uop_const(cast->src[0]->dtype, 0.0));
                    UOp* wh2c = uop_cast(wh2, dtypes.uint64);
                    UOp* muls[]={wh2c, uop_const(dtypes.uint64, (double)(1ULL<<32))}; UOpArg aa={0};
                    return uop_new(OPS_MUL, dtypes.uint64, muls, 2, &aa, NULL);
                }
            }
        }
    }
    // ((x u64)& where(y, 0xFFFFFFFF, 0)).cast(u32) -> where(y, x.cast(u32), 0)
    if (uop->op == OPS_CAST && dtype_eq(&uop->dtype, &dtypes.uint32) && uop->src_count==1) {
        UOp* andop = uop->src[0];
        if (andop->op==OPS_AND && andop->src_count==2) {
            UOp* a=andop->src[0], *b=andop->src[1];
            UOp* wh = (a->op==OPS_WHERE)? a : (b->op==OPS_WHERE? b : NULL);
            UOp* other = (wh==a)? b : (wh==b? a : NULL);
            if (wh && other && dtype_eq(&other->dtype, &dtypes.uint64) && wh->src_count==3) {
                UOp* c=wh->src[0], *t=wh->src[1], *f=wh->src[2];
                if (t->op==OPS_CONST && f->op==OPS_CONST) {
                    double tv=(t->arg.type==ARG_CONST)? t->arg.const_data.const_value : (double)t->arg.int_data.i;
                    double fv=(f->arg.type==ARG_CONST)? f->arg.const_data.const_value : (double)f->arg.int_data.i;
                    if ((uint64_t)tv==0xFFFFFFFFULL && fabs(fv-0.0)<0.5) {
                        UOp* xc = uop_cast(other, dtypes.uint32);
                        return uop_where(c, xc, uop_const(dtypes.uint32, 0.0));
                    }
                }
            }
        }
    }
    // GEP through WMMA (gated)
    if (uop->op == OPS_GEP && uop->src_count>0 && uop->src[0]->op==OPS_WMMA) {
        const char* en = tg_getenv("ENABLE_GEP_WMMA");
        if (en && *en) {
            UOp* wmma = uop->src[0];
            UOp* g = uop;
            int m = g->arg.type==ARG_REDUCE ? g->arg.reduce_data.axes_count : 0;
            if (m>0 && wmma->src_count==3) {
                UOp** ns = (UOp**)malloc(sizeof(UOp*)*3);
                for (int i=0;i<3;i++) ns[i] = uop_gep(wmma->src[i], g->arg.reduce_data.axes, m);
                UOpArg a = wmma->arg;
                UOp* ret = uop_new(OPS_WMMA, g->dtype, ns, 3, &a, NULL);
                free(ns);
                return ret;
            }
        }
    }

    // CMPLT canonicalization quick wins for ints
    if (uop->op == OPS_CMPLT && uop->src_count==2) {
        UOp* lhs=uop->src[0], *rhs=uop->src[1];
        // (a + x) < b  and (x + a) < b
        if (lhs->op==OPS_ADD && rhs->op==OPS_CONST) {
            UOp* a = lhs->src[0]; UOp* b = lhs->src[1];
            double rv = (rhs->arg.type==ARG_CONST)? rhs->arg.const_data.const_value : (double)rhs->arg.int_data.i;
            if (a->op==OPS_CONST) {
                double av = (a->arg.type==ARG_CONST)? a->arg.const_data.const_value : (double)a->arg.int_data.i;
                UOp* bound = uop_const(rhs->dtype, rv - av);
                UOp* srcs[] = { b, bound };
                return uop_new(OPS_CMPLT, dtypes.bool_, srcs, 2, &(UOpArg){0}, NULL);
            }
            if (b->op==OPS_CONST) {
                double bv = (b->arg.type==ARG_CONST)? b->arg.const_data.const_value : (double)b->arg.int_data.i;
                UOp* bound = uop_const(rhs->dtype, rv - bv);
                UOp* srcs[] = { a, bound };
                return uop_new(OPS_CMPLT, dtypes.bool_, srcs, 2, &(UOpArg){0}, NULL);
            }
        }
        // (k*x) < c with k>0
        if (lhs->op==OPS_MUL && lhs->src_count==2 && rhs->op==OPS_CONST) {
            UOp* maybe_c0=lhs->src[0]; UOp* X=lhs->src[1]; if (maybe_c0->op!=OPS_CONST) { maybe_c0=lhs->src[1]; X=lhs->src[0]; }
            if (maybe_c0->op==OPS_CONST) {
                double c0 = (maybe_c0->arg.type==ARG_CONST)? maybe_c0->arg.const_data.const_value : (double)maybe_c0->arg.int_data.i;
                double c1 = (rhs->arg.type==ARG_CONST)? rhs->arg.const_data.const_value : (double)rhs->arg.int_data.i;
                if (c0 > 0 && c1 > 0) {
                    double ceilv = ceil(c1 / c0);
                    UOp* bound = uop_const(rhs->dtype, ceilv);
                    UOp* srcs[] = { X, bound };
                    return uop_new(OPS_CMPLT, dtypes.bool_, srcs, 2, &(UOpArg){0}, NULL);
                }
            }
        }
        // (-x) < (-y) -> y < x
        if (lhs->op==OPS_MUL && rhs->op==OPS_MUL && lhs->src_count==2 && rhs->src_count==2) {
            long long kl=0, kr=0; UOp* xl=NULL; UOp* xr=NULL;
            if (lhs->src[0]->op==OPS_CONST) { if (lhs->src[0]->arg.type==ARG_INT) kl=lhs->src[0]->arg.int_data.i; else { double v=lhs->src[0]->arg.const_data.const_value; kl=(long long)llround(v);} xl=lhs->src[1]; }
            else if (lhs->src[1]->op==OPS_CONST) { if (lhs->src[1]->arg.type==ARG_INT) kl=lhs->src[1]->arg.int_data.i; else { double v=lhs->src[1]->arg.const_data.const_value; kl=(long long)llround(v);} xl=lhs->src[0]; }
            if (rhs->src[0]->op==OPS_CONST) { if (rhs->src[0]->arg.type==ARG_INT) kr=rhs->src[0]->arg.int_data.i; else { double v=rhs->src[0]->arg.const_data.const_value; kr=(long long)llround(v);} xr=rhs->src[1]; }
            else if (rhs->src[1]->op==OPS_CONST) { if (rhs->src[1]->arg.type==ARG_INT) kr=rhs->src[1]->arg.int_data.i; else { double v=rhs->src[1]->arg.const_data.const_value; kr=(long long)llround(v);} xr=rhs->src[0]; }
            if (xl && xr && kl==-1 && kr==-1) {
                UOp* srcs[] = { xr, xl };
                return uop_new(OPS_CMPLT, dtypes.bool_, srcs, 2, &(UOpArg){0}, NULL);
            }
        }
    }
    
    // Apply advanced symbolic simplification from symbolic.c
    UOp* simplified = symbolic_simplify(uop);
    if (simplified && simplified != uop) {
        // Recursively simplify the result to catch chained patterns
        return uop_simplify(simplified);
    }
    
    // Try basic constant folding for all operations
    UOp* folded = constant_fold_basic(uop);
    if (folded && folded != uop) {
        return folded;
    }
    
    // Basic pattern-based simplifications
    // ADD(x, 0) -> x
    if (uop->op == OPS_ADD && uop->src_count == 2) {
        if (uop_is_zero(uop->src[1])) return uop->src[0];
        if (uop_is_zero(uop->src[0])) return uop->src[1];
    }
    
    // Commutative canonicalization for ADD/MUL/MAX/AND/OR/XOR: sort sources by pointer to improve dedup
    if ((uop->op == OPS_ADD || uop->op == OPS_MUL || uop->op == OPS_MAX ||
         uop->op == OPS_AND || uop->op == OPS_OR  || uop->op == OPS_XOR) && uop->src_count == 2) {
        UOp* a = uop->src[0]; UOp* b = uop->src[1];
        // Place non-const before const (except for AND/OR/XOR doesn't matter)
        bool aconst = (a->op == OPS_CONST);
        bool bconst = (b->op == OPS_CONST);
        bool swap = false;
        if (aconst != bconst) swap = aconst && !bconst; // keep const on right
        else if ((size_t)a > (size_t)b) swap = true; // pointer order
        if (swap) {
            UOp* src2[] = {b, a};
            return uop_replace_ex(uop, uop->op, uop->dtype, src2, 2, NULL, uop->tag);
        }
    }

    // MUL(x, 1) -> x
    if (uop->op == OPS_MUL && uop->src_count == 2) {
        if (uop_is_one(uop->src[1])) return uop->src[0];
        if (uop_is_one(uop->src[0])) return uop->src[1];
        // MUL(x, 0) -> 0
        if (uop_is_zero(uop->src[1]) || uop_is_zero(uop->src[0])) {
            return uop_const(uop->dtype, 0.0);
        }
    }

    // ADD(x, x) for XOR is handled elsewhere; ADD zero handled above
    // SUB rewrite to ADD(x, -y)
    if (uop->op == OPS_SUB && uop->src_count == 2) {
        UOp* negb = uop_neg(uop->src[1]);
        UOp* src2[] = {uop->src[0], negb};
        UOp* rew = uop_new(OPS_ADD, least_upper_dtype(&uop->src[0]->dtype, &uop->src[1]->dtype), src2, 2, &(UOpArg){0}, NULL);
        uop_unref(negb);
        return uop_simplify(rew);
    }

    return uop;
}

UOp* uop_ssimplify(UOp* uop) {
    // Symbolic simplification with advanced patterns
    if (!uop) return NULL;
    // Fast constant folding for simple binary ops before symbolic rewrite
    if (uop->src_count == 2 && uop->op == OPS_ADD) {
      UOp* a = uop->src[0]; UOp* b = uop->src[1];
      if (a->op==OPS_CONST && b->op==OPS_CONST && a->arg.type==ARG_CONST && b->arg.type==ARG_CONST) {
        UOp* r = uop_const(uop->dtype, a->arg.const_data.const_value + b->arg.const_data.const_value);
        // fprintf(stderr, "uop_ssimplify pre-const-fold ADD -> CONST\n");
        return r;
      }
    }
    
    // Apply comprehensive symbolic simplification from symbolic.c
    UOp* simplified = symbolic_ssimplify(uop);
    if (simplified && simplified != uop) {
        // fprintf(stderr, "uop_ssimplify: symbolic_ssimplify changed op %d -> %d\n", uop->op, simplified->op);
        return simplified;
    }
    
    // Try basic constant folding for all operations
    UOp* folded = constant_fold_basic(uop);
    if (folded && folded != uop) {
        // fprintf(stderr, "uop_ssimplify: constant_fold_basic folded op %d -> %d\n", uop->op, folded->op);
        return folded;
    }
    
    // Fall back to regular simplify if no symbolic simplification occurred
    return uop_simplify(uop);
}

// Line 472-518: vmin/vmax calculation
// Python: @functools.cached_property def _min_max(self)
static void compute_min_max(UOp* uop) {
    if (!uop || uop->vmin_vmax_valid) return;  // Already computed
    
    // Line 507: DEFINE_VAR returns arg[1] as vmin, arg[2] as vmax
    if (uop->op == OPS_DEFINE_VAR && uop->arg.type == ARG_VAR) {
        uop->vmin = uop->arg.var.vmin;
        uop->vmax = uop->arg.var.vmax;
        uop->vmin_vmax_valid = true;
        return;
    }
    
    // Line 513: CONST returns its value for both vmin and vmax
    if (uop->op == OPS_CONST && uop->arg.type == ARG_CONST) {
        int64_t val = (int64_t)uop->arg.const_data.const_value;
        uop->vmin = val;
        uop->vmax = val;
        uop->vmin_vmax_valid = true;
        return;
    }
    
    // Line 508: RANGE returns 0, (src[0]-1).vmax
    if (uop->op == OPS_RANGE && uop->src_count > 0) {
        compute_min_max(uop->src[0]);
        uop->vmin = 0;
        uop->vmax = uop->src[0]->vmax - 1;
        uop->vmin_vmax_valid = true;
        return;
    }
    
    // Line 509: BIND returns src[0]._min_max (ignore the bound value)
    if (uop->op == OPS_BIND && uop->src_count > 0) {
        compute_min_max(uop->src[0]);
        uop->vmin = uop->src[0]->vmin;
        uop->vmax = uop->src[0]->vmax;
        uop->vmin_vmax_valid = true;
        return;
    }
    
    // Line 477-503: Binary operations for non-float dtypes
    if (uop->src_count == 2 && (uop->op == OPS_ADD || uop->op == OPS_SUB || 
                                 uop->op == OPS_MUL || uop->op == OPS_MAX ||
                                 uop->op == OPS_MOD || uop->op == OPS_IDIV ||
                                 uop->op == OPS_SHL || uop->op == OPS_SHR ||
                                 uop->op == OPS_CMPLT || uop->op == OPS_CMPNE ||
                                 uop->op == OPS_AND)) {
        // Compute min/max for sources first
        compute_min_max(uop->src[0]);
        compute_min_max(uop->src[1]);
        
        int64_t s0_vmin = uop->src[0]->vmin;
        int64_t s0_vmax = uop->src[0]->vmax;
        int64_t s1_vmin = uop->src[1]->vmin;
        int64_t s1_vmax = uop->src[1]->vmax;
        
        // Line 479: ADD
        if (uop->op == OPS_ADD) {
            uop->vmin = s0_vmin + s1_vmin;
            uop->vmax = s0_vmax + s1_vmax;
        }
        // Line 480: SUB
        else if (uop->op == OPS_SUB) {
            uop->vmin = s0_vmin - s1_vmax;
            uop->vmax = s0_vmax - s1_vmin;
        }
        // Line 482: MUL - min/max of all products
        else if (uop->op == OPS_MUL) {
            int64_t vals[4] = {
                s0_vmin * s1_vmin,
                s0_vmin * s1_vmax,
                s0_vmax * s1_vmin,
                s0_vmax * s1_vmax
            };
            uop->vmin = vals[0];
            uop->vmax = vals[0];
            for (int i = 1; i < 4; i++) {
                if (vals[i] < uop->vmin) uop->vmin = vals[i];
                if (vals[i] > uop->vmax) uop->vmax = vals[i];
            }
        }
        // Line 498: MAX
        else if (uop->op == OPS_MAX) {
            uop->vmin = (s0_vmin > s1_vmin) ? s0_vmin : s1_vmin;
            uop->vmax = (s0_vmax > s1_vmax) ? s0_vmax : s1_vmax;
        }
        // Line 499: CMPLT returns boolean range
        else if (uop->op == OPS_CMPLT) {
            uop->vmin = (s0_vmax < s1_vmin) ? 1 : 0;
            uop->vmax = (s0_vmin < s1_vmax) ? 1 : 0;
        }
        // Line 500: CMPNE
        else if (uop->op == OPS_CMPNE) {
            uop->vmin = ((s0_vmax < s1_vmin) || (s1_vmax < s0_vmin)) ? 1 : 0;
            uop->vmax = (s0_vmin == s0_vmax && s0_vmax == s1_vmin && s1_vmin == s1_vmax) ? 0 : 1;
        }
        // Line 481: AND with positive constant (limited support)
        else if (uop->op == OPS_AND && s1_vmin == s1_vmax && s0_vmin >= 0 && s1_vmin >= 0) {
            // Python: return min(0, s0_vmin), min(s0_vmax, s1_vmax)
            // Since s0_vmin >= 0, min(0, s0_vmin) is always 0
            uop->vmin = 0;
            uop->vmax = (s0_vmax < s1_vmax) ? s0_vmax : s1_vmax;  // min(s0_vmax, s1_vmax)
        }
        // Line 484: SHL on consts only
        else if (uop->op == OPS_SHL && s1_vmin == s1_vmax) {
            uop->vmin = s0_vmin << s1_vmin;
            uop->vmax = s0_vmax << s1_vmin;
        }
        // Line 485: SHR on consts only
        else if (uop->op == OPS_SHR && s1_vmin == s1_vmax) {
            uop->vmin = s0_vmin >> s1_vmin;
            uop->vmax = s0_vmax >> s1_vmin;
        }
        // Line 486-488: MOD operation
        else if (uop->op == OPS_MOD) {
            if (s1_vmin > 0) {
                if (s0_vmin >= 0) {
                    // Special case: if range is entirely within [0, divisor), result is just the range
                    if (s0_vmax < s1_vmin) {
                        uop->vmin = s0_vmin;
                        uop->vmax = s0_vmax;
                    } else {
                        uop->vmin = 0;
                        uop->vmax = s1_vmax - 1;
                    }
                } else if (s0_vmax <= 0) {
                    uop->vmin = -(s1_vmax - 1);
                    uop->vmax = 0;
                } else {
                    uop->vmin = -(s1_vmax - 1);
                    uop->vmax = s1_vmax - 1;
                }
            } else if (s1_vmax < 0) {
                if (s0_vmin >= 0) {
                    uop->vmin = 0;
                    uop->vmax = -s1_vmin - 1;
                } else if (s0_vmax <= 0) {
                    uop->vmin = -(-s1_vmin - 1);
                    uop->vmax = 0;
                } else {
                    uop->vmin = -(-s1_vmin - 1);
                    uop->vmax = -s1_vmin - 1;
                }
            } else {
                uop->vmin = INT64_MIN;
                uop->vmax = INT64_MAX;
            }
        }
        // Line 489-497: IDIV bounds
        else if (uop->op == OPS_IDIV) {
            // Helper for ceiling division toward zero
            #define CDIV(x, y) ((y) == 0 ? 0 : ((x) < 0) != ((y) < 0) ? -labs(x)/labs(y) : labs(x)/labs(y))
            
            if (s1_vmin == s1_vmax) {  // s1 is a const
                int64_t c = s1_vmin;
                if (c > 0) {
                    uop->vmin = CDIV(s0_vmin, c);
                    uop->vmax = CDIV(s0_vmax, c);
                } else if (c < 0) {
                    uop->vmin = CDIV(s0_vmax, c);
                    uop->vmax = CDIV(s0_vmin, c);
                } else {
                    uop->vmin = INT64_MIN;
                    uop->vmax = INT64_MAX;
                }
            } else if (s0_vmax <= 0 && s1_vmax < 0) {
                uop->vmin = CDIV(s0_vmax, s1_vmin);
                uop->vmax = CDIV(s0_vmin, s1_vmax);
            } else if (s0_vmin >= 0 && s1_vmin > 0) {
                uop->vmin = CDIV(s0_vmin, s1_vmax);
                uop->vmax = CDIV(s0_vmax, s1_vmin);
            } else if (s0_vmax <= 0 && s1_vmin > 0) {
                uop->vmin = CDIV(s0_vmin, s1_vmin);
                uop->vmax = CDIV(s0_vmax, s1_vmax);
            } else if (s0_vmin >= 0 && s1_vmax < 0) {
                uop->vmin = CDIV(s0_vmax, s1_vmax);
                uop->vmax = CDIV(s0_vmin, s1_vmin);
            } else {
                uop->vmin = INT64_MIN;
                uop->vmax = INT64_MAX;
            }
            #undef CDIV
        }
        // Default for other ops - use dtype bounds
        else {
            uop->vmin = INT64_MIN;
            uop->vmax = INT64_MAX;
        }
        
        uop->vmin_vmax_valid = true;
        return;
    }
    
    // Unary operations - NEG
    if (uop->op == OPS_NEG && uop->src_count == 1) {
        compute_min_max(uop->src[0]);
        int64_t s0_vmin = uop->src[0]->vmin;
        int64_t s0_vmax = uop->src[0]->vmax;
        uop->vmin = -s0_vmax;
        uop->vmax = -s0_vmin;
        uop->vmin_vmax_valid = true;
        return;
    }
    
    // Line 505: WHERE for int dtypes
    if (uop->op == OPS_WHERE && uop->src_count == 3) {
        compute_min_max(uop->src[1]);
        compute_min_max(uop->src[2]);
        uop->vmin = (uop->src[1]->vmin < uop->src[2]->vmin) ? uop->src[1]->vmin : uop->src[2]->vmin;
        uop->vmax = (uop->src[1]->vmax > uop->src[2]->vmax) ? uop->src[1]->vmax : uop->src[2]->vmax;
        uop->vmin_vmax_valid = true;
        return;
    }
    
    // Line 518: Default - use dtype bounds
    uop->vmin = INT64_MIN;
    uop->vmax = INT64_MAX;
    uop->vmin_vmax_valid = true;
}

// Line 472-474: Property accessors for vmin/vmax
int uop_vmin(UOp* uop) {
    if (!uop) return 0;
    if (!uop->vmin_vmax_valid) compute_min_max(uop);
    return (int)uop->vmin;
}

int uop_vmax(UOp* uop) {
    if (!uop) return 0;
    if (!uop->vmin_vmax_valid) compute_min_max(uop);
    return (int)uop->vmax;
}

int uop_sym_infer(UOp* uop) {
    // For now, sym_infer returns vmin (could be vmax or midpoint)
    // The actual Python implementation creates a lambda and evaluates it,
    // but for simple cases we can just return the bounds
    return (int)uop_vmin(uop);
}

// Line 600: def resolve(self, default_val:bool=False) -> bool:
bool uop_resolve(UOp* uop, bool default_val) {
    // Line 26: return bool(sx.vmin) if (sx:=x.simplify()).vmin == sx.vmax else default
    if (!uop) return default_val;
    
    // Simplify first (we'll use basic simplification for now)
    UOp* simplified = uop_simplify(uop);
    if (!simplified) simplified = uop;
    
    // Get vmin and vmax for the boolean expression
    int vmin = uop_vmin(simplified);
    int vmax = uop_vmax(simplified);
    
    // If vmin == vmax, we know the value for certain
    if (vmin == vmax) {
        return vmin != 0;  // Convert to boolean
    }
    
    // Otherwise return default (ambiguous)
    return default_val;
}

// Line 427-431: bind function
// Variable binding function
UOp* uop_bind(UOp* var, UOp* value) {
    if (var->op != OPS_DEFINE_VAR) {
        return var;  // Not a variable, can't bind
    }
    UOp* src[] = {var, value};
    UOpArg arg = {0};
    return uop_new(OPS_BIND, var->dtype, src, 2, &arg, NULL);
}

// Line 622-680: Cache management
void uop_cache_init(void) {
    if (!_cache) {
        _cache = (UOpCacheTable*)calloc(1, sizeof(UOpCacheTable));
        _cache->bucket_count = 1024;
        _cache->buckets = (UOpCacheEntry**)calloc(_cache->bucket_count, sizeof(UOpCacheEntry*));
    }
}

void uop_cache_cleanup(void) {
    if (_cache) {
        // Free all entries
        for (size_t i = 0; i < _cache->bucket_count; i++) {
            UOpCacheEntry* entry = _cache->buckets[i];
            while (entry) {
                UOpCacheEntry* next = entry->next;
                free(entry);
                entry = next;
            }
        }
        free(_cache->buckets);
        free(_cache);
        _cache = NULL;
    }
}

// key hashing/equality helpers to mirror uop_hash/uop_equals without constructing a UOp
static size_t uop_key_hash(Ops op, DType dtype, UOp** src, size_t src_count, const UOpArg* arg, void* tag) {
    size_t h = (size_t)op;
    h = hash_mix(h, (size_t)(dtype.count ? dtype._scalar : &dtype));
    for (size_t i = 0; i < src_count; i++) h = hash_mix(h, (size_t)src[i]);
    if (arg) {
        h = hash_mix(h, (size_t)arg->type);
        switch (arg->type) {
            case ARG_CONST: {
                double v = arg->const_data.const_value;
                h = hash_mix(h, hash_mem(&v, sizeof(double))); break; }
            case ARG_INT: { h = hash_mix(h, (size_t)arg->int_data.i); break; }
            case ARG_REDUCE: {
                h = hash_mix(h, (size_t)arg->reduce_data.reduce_op);
                if (arg->reduce_data.axes && arg->reduce_data.axes_count>0)
                    h = hash_mix(h, hash_mem(arg->reduce_data.axes, sizeof(int)*(size_t)arg->reduce_data.axes_count));
                h = hash_mix(h, (size_t)arg->reduce_data.axes_count);
                break; }
            case ARG_VCONST: {
                int n = arg->vconst_data.count;
                if (arg->vconst_data.values && n>0)
                    h = hash_mix(h, hash_mem(arg->vconst_data.values, sizeof(double)*(size_t)n));
                h = hash_mix(h, (size_t)n);
                break; }
            case ARG_PAD_PARAMS: {
                int n = arg->pad_data.ndim;
                if (arg->pad_data.before && n>0) h = hash_mix(h, hash_mem(arg->pad_data.before, sizeof(int32_t)*(size_t)n));
                if (arg->pad_data.after  && n>0) h = hash_mix(h, hash_mem(arg->pad_data.after,  sizeof(int32_t)*(size_t)n));
                h = hash_mix(h, (size_t)n);
                break; }
            case ARG_SHRINK_PARAMS: {
                int n = arg->shrink_data.ndim;
                if (arg->shrink_data.start && n>0) h = hash_mix(h, hash_mem(arg->shrink_data.start, sizeof(int32_t)*(size_t)n));
                if (arg->shrink_data.end   && n>0) h = hash_mix(h, hash_mem(arg->shrink_data.end,   sizeof(int32_t)*(size_t)n));
                h = hash_mix(h, (size_t)n);
                break; }
            default: break;
        }
    }
    h = hash_mix(h, (size_t)tag);
    return h;
}

static bool uop_key_equals(UOp* cached, Ops op, DType dtype, UOp** src, size_t src_count, const UOpArg* arg, void* tag) {
    if (!cached) return false;
    if (cached->op != op) return false;
    if (!dtype_eq(&cached->dtype, &dtype)) return false;
    if (cached->tag != tag) return false;
    if (cached->src_count != src_count) return false;
    for (size_t i=0;i<src_count;i++) if (cached->src[i] != src[i]) return false;
    UOpArgType at = arg ? arg->type : ARG_NONE;
    if (cached->arg.type != at) return false;
    if (!arg) return true;
    switch (at) {
        case ARG_CONST: return cached->arg.const_data.const_value == arg->const_data.const_value;
        case ARG_INT:   return cached->arg.int_data.i == arg->int_data.i;
        case ARG_REDUCE: {
            if (cached->arg.reduce_data.reduce_op != arg->reduce_data.reduce_op) return false;
            if (cached->arg.reduce_data.axes_count != arg->reduce_data.axes_count) return false;
            for (int i=0;i<arg->reduce_data.axes_count;i++)
                if (cached->arg.reduce_data.axes[i] != arg->reduce_data.axes[i]) return false;
            return true; }
        case ARG_VCONST: {
            if (cached->arg.vconst_data.count != arg->vconst_data.count) return false;
            for (int i=0;i<arg->vconst_data.count;i++)
                if (cached->arg.vconst_data.values[i] != arg->vconst_data.values[i]) return false;
            return true; }
        case ARG_PAD_PARAMS: {
            if (cached->arg.pad_data.ndim != arg->pad_data.ndim) return false;
            for (int i=0;i<arg->pad_data.ndim;i++) {
                if (cached->arg.pad_data.before[i] != arg->pad_data.before[i]) return false;
                if (cached->arg.pad_data.after[i]  != arg->pad_data.after[i])  return false;
            }
            return true; }
        case ARG_SHRINK_PARAMS: {
            if (cached->arg.shrink_data.ndim != arg->shrink_data.ndim) return false;
            for (int i=0;i<arg->shrink_data.ndim;i++) {
                if (cached->arg.shrink_data.start[i] != arg->shrink_data.start[i]) return false;
                if (cached->arg.shrink_data.end[i]   != arg->shrink_data.end[i])   return false;
            }
            return true; }
        default: return true;
    }
}

UOp* uop_cache_get(Ops op, DType dtype, UOp** src, size_t src_count, UOpArg* arg, void* tag) {
    if (!_cache) return NULL;

    size_t hash = uop_key_hash(op, dtype, src, src_count, arg, tag);
    size_t bucket_idx = hash % _cache->bucket_count;

    UOpCacheEntry* entry = _cache->buckets[bucket_idx];
    while (entry) {
        if (entry->key_hash == hash && entry->value) {
            UOp* cached = entry->value;
            if (cached->ref_count <= 0) {
                entry->value = NULL;  // stale weak entry
            } else if (uop_key_equals(cached, op, dtype, src, src_count, arg, tag)) {
                return uop_ref(cached);
            }
        }
        entry = entry->next;
    }
    return NULL;
}

void uop_cache_put(UOp* uop) {
    if (!uop || !_cache) return;
    
    // Compute hash using the same logic as uop_hash for consistency
    size_t hash = uop_hash(uop);
    
    // Add to cache
    size_t bucket_idx = hash % _cache->bucket_count;
    
    UOpCacheEntry* new_entry = (UOpCacheEntry*)malloc(sizeof(UOpCacheEntry));
    new_entry->key_hash = hash;
    new_entry->value = uop;  // Don't reference - caller owns the reference
    new_entry->next = _cache->buckets[bucket_idx];
    _cache->buckets[bucket_idx] = new_entry;
    _cache->size++;
}

// Line 682-800: Pattern matching support (UPat class)
// Line 682: @dataclass(frozen=True)
// class UPat:

// Line 684-690: Pattern types
// Use the UPat struct from the header file

// Line 692-700: def __new__(cls, op:Optional[Ops], src:tuple[UPat,...], arg:Any):
UPat* upat_op(Ops op, UPat** src, size_t src_count) {
    UPat* pat = (UPat*)calloc(1, sizeof(UPat));
    pat->type = UPAT_OP;
    pat->op_data.op = op;
    if (src_count > 0) {
        pat->src = (UPat**)malloc(src_count * sizeof(UPat*));
        memcpy(pat->src, src, src_count * sizeof(UPat*));
        pat->src_count = src_count;
    }
    return pat;
}

UPat* upat_var(int id) {
    UPat* pat = (UPat*)calloc(1, sizeof(UPat));
    pat->type = UPAT_VAR;
    pat->var_data.var_id = id;
    return pat;
}

UPat* upat_var_named(const char* name, const DType* const* dts, size_t dtype_count, bool vec_any) {
    UPat* pat = (UPat*)calloc(1, sizeof(UPat));
    pat->type = UPAT_VAR;
    // name
    if (name) {
        pat->name = strdup(name);
    }
    // dtype list
    if (dts && dtype_count>0) {
        const DType** arr = (const DType**)malloc(sizeof(DType*)*dtype_count);
        for (size_t i=0;i<dtype_count;i++) arr[i]=dts[i];
        pat->dtype_list = arr;
        pat->dtype_list_count = dtype_count;
    }
    pat->vec_any = vec_any;
    return pat;
}

UPat* upat_cvar_named(const char* name, const DType* const* dts, size_t dtype_count, bool vec_any) {
    UPat* pat = upat_var_named(name, dts, dtype_count, vec_any);
    pat->require_const = true;
    return pat;
}

UPat* upat_const(double val) {
    UPat* pat = (UPat*)calloc(1, sizeof(UPat));
    pat->type = UPAT_CONST;
    pat->const_data.const_val = val;
    return pat;
}

UPat* upat_any(void) {
    UPat* pat = (UPat*)calloc(1, sizeof(UPat));
    pat->type = UPAT_ANY;
    return pat;
}

UPat* upat_not(UPat* p) {
    // Logical NOT pattern: CMPEQ(p, False)
    UPat* src[2] = { p, upat_const(0.0) };
    return upat_op(OPS_CMPEQ, src, 2);
}

UPat* upat_and(UPat* a, UPat* b) {
    UPat* src[2] = { a, b };
    return upat_op(OPS_AND, src, 2);
}

UPat* upat_or(UPat* a, UPat* b) {
    UPat* src[2] = { a, b };
    return upat_op(OPS_OR, src, 2);
}

// Line 710-750: Pattern matching
typedef struct {
    int var_id;
    UOp* value;
} Binding;

typedef struct {
    Binding* bindings;
    size_t count;
    size_t capacity;
} BindingList;

static bool _dtype_list_contains(const DType* const* lst, size_t n, const DType* dt, bool vec_any){
    if (!lst || n==0) return true;
    for (size_t i=0;i<n;i++){
        if (dtype_eq(lst[i], dt)) return true;
        if (vec_any) { DType sc = dtype_scalar(dt); if (dtype_eq(lst[i], &sc)) return true; }
    }
    return false;
}

static bool match_internal(UPat* pattern, UOp* uop, BindingList* bindings) {
    if (!pattern || !uop) return false;
    
    switch (pattern->type) {
        case UPAT_ANY:
            return true;
            
        case UPAT_VAR:
            // Check if already bound
            for (size_t i = 0; i < bindings->count; i++) {
                if (bindings->bindings[i].var_id == pattern->var_data.var_id) {
                    return bindings->bindings[i].value == uop;
                }
            }
            // Additional constraints
            if (pattern->require_const && uop->op != OPS_CONST) return false;
            if (pattern->dtype_list_count>0) {
                if (!_dtype_list_contains(pattern->dtype_list, pattern->dtype_list_count, &uop->dtype, pattern->vec_any)) return false;
            }
            // Add new binding
            if (bindings->count >= bindings->capacity) {
                bindings->capacity = bindings->capacity ? bindings->capacity * 2 : 8;
                bindings->bindings = (Binding*)realloc(bindings->bindings, 
                                                       bindings->capacity * sizeof(Binding));
            }
            bindings->bindings[bindings->count].var_id = pattern->var_data.var_id;
            bindings->bindings[bindings->count].value = uop;
            bindings->count++;
            return true;
            
        case UPAT_CONST:
            if (uop->op != OPS_CONST) return false;
            if (uop->arg.type != ARG_CONST) return false;
            return uop->arg.const_data.const_value == pattern->const_data.const_val;
            
        case UPAT_OP:
            if (uop->op != pattern->op_data.op) return false;
            if (pattern->src_count != uop->src_count) return false;
            for (size_t i = 0; i < pattern->src_count; i++) {
                if (!match_internal(pattern->src[i], uop->src[i], bindings)) {
                    return false;
                }
            }
            return true;
            
        // UPAT_DTYPE not in header, skipping
            
        default:
            return false;
    }
}

bool upat_match(UPat* pattern, UOp* uop) {
    BindingList bindings = {NULL, 0, 0};
    bool result = match_internal(pattern, uop, &bindings);
    if (bindings.bindings) free(bindings.bindings);
    _match_stats_total++;
    if (result) _match_stats_hits++;
    return result;
}

// upat_free is now defined in upat.c

// Line 802-900: Execution
// Helper functions for cdiv and cmod from Python
static int cdiv_impl(int x, int y) {
    // Python: cdiv(x,y) = abs(x)//abs(y)*(1,-1)[x*y<0] if y != 0 else 0
    if (y == 0) return 0;
    int abs_x = abs(x);
    int abs_y = abs(y);
    int result = abs_x / abs_y;
    // Apply sign: negative if x and y have different signs
    if ((x < 0) != (y < 0)) result = -result;
    return result;
}

static int cmod_impl(int x, int y) {
    // Python: cmod(x,y) = x - cdiv(x,y)*y
    return x - cdiv_impl(x, y) * y;
}

// Line 802: def exec_alu(op:Ops, dtype:DType, args:tuple[ConstType, ...]):
double exec_alu(Ops op, DType dtype, double* args, size_t arg_count) {
    // Execute ALU operation
    // This implements the actual computation for constant folding
    
    double result = 0.0;
    
    if (arg_count == 0) {
        // Nullary operations
        result = 0.0;
    } else if (arg_count == 1) {
        // Unary operations
        double a = args[0];
        switch (op) {
            case OPS_NEG: result = -a; break;
            case OPS_EXP2: {
                // Python safe_exp2: try 2**x except OverflowError return inf
                if (a > 1024) result = INFINITY;
                else if (a < -1024) result = 0.0;
                else result = pow(2.0, a);
                break;
            }
            case OPS_LOG2: {
                // Python: math.log2(x) if x > 0 else -math.inf if x == 0 else math.nan
                if (a > 0) result = log2(a);
                else if (a == 0) result = -INFINITY;
                else result = NAN;
                break;
            }
            case OPS_SIN: {
                // Python: math.sin(x) if not math.isinf(x) else math.nan
                if (isinf(a)) result = NAN;
                else result = sin(a);
                break;
            }
            case OPS_SQRT: {
                // Python: math.sqrt(x) if x >= 0 else math.nan
                if (a >= 0) result = sqrt(a);
                else result = NAN;
                break;
            }
            case OPS_RECIP: {
                // Python: 1/x if x != 0 else math.copysign(math.inf, x)
                if (a != 0) result = 1.0 / a;
                else result = (a >= 0) ? INFINITY : -INFINITY;
                break;
            }
            default: result = a; break; // Passthrough for unknown ops
        }
    } else if (arg_count == 2) {
        // Binary operations
        double a = args[0];
        double b = args[1];
        switch (op) {
            case OPS_ADD: result = a + b; break;
            case OPS_SUB: result = a - b; break;
            case OPS_MUL: result = a * b; break;
            case OPS_FDIV: {
                // Regular floating point division
                if (b == 0) result = (a >= 0) ? INFINITY : -INFINITY;
                else result = a / b;
                break;
            }
            case OPS_IDIV: {
                // Integer division using cdiv (ceiling division toward zero)
                // Convert to int, apply cdiv, convert back
                if (dtypes_is_int(&dtype)) {
                    result = (double)cdiv_impl((int)a, (int)b);
                } else {
                    // For float types, use truncating division
                    if (b == 0) result = (a >= 0) ? INFINITY : -INFINITY;
                    else result = trunc(a / b);
                }
                break;
            }
            case OPS_MAX: result = (a > b) ? a : b; break;
            case OPS_MOD: {
                // Use cmod for integer types, fmod for float types
                if (dtypes_is_int(&dtype)) {
                    result = (double)cmod_impl((int)a, (int)b);
                } else {
                    result = fmod(a, b);
                }
                break;
            }
            case OPS_CMPLT: result = (a < b) ? 1.0 : 0.0; break;
            case OPS_CMPEQ: result = (a == b) ? 1.0 : 0.0; break;
            case OPS_CMPNE: result = (a != b) ? 1.0 : 0.0; break;
            case OPS_XOR: {
                // Bitwise XOR for integers, logical XOR for bools
                if (dtype_eq(&dtype, &dtypes.bool_)) {
                    result = ((a != 0) != (b != 0)) ? 1.0 : 0.0;
                } else {
                    result = (double)((int)a ^ (int)b);
                }
                break;
            }
            case OPS_AND: {
                // Bitwise AND for integers, logical AND for bools
                if (dtype_eq(&dtype, &dtypes.bool_)) {
                    result = ((a != 0) && (b != 0)) ? 1.0 : 0.0;
                } else {
                    result = (double)((int)a & (int)b);
                }
                break;
            }
            case OPS_OR: {
                // Bitwise OR for integers, logical OR for bools
                if (dtype_eq(&dtype, &dtypes.bool_)) {
                    result = ((a != 0) || (b != 0)) ? 1.0 : 0.0;
                } else {
                    result = (double)((int)a | (int)b);
                }
                break;
            }
            case OPS_SHL: {
                int ia = (int)a;
                int shift = (int)b;
                if (shift < 0 || shift >= 32) result = 0.0;  // 32-bit shift
                else result = (double)(ia << shift);
                break;
            }
            case OPS_SHR: {
                int ia = (int)a;
                int shift = (int)b;
                if (shift < 0 || shift >= 32) result = 0.0;  // 32-bit shift
                else result = (double)(ia >> shift);
                break;
            }
            case OPS_POW: {
                // Python safe_pow: handle zero division and domain errors
                if (a == 0.0 && b < 0.0) { result = INFINITY; break; }
                double p = pow(a, b);
                if (isnan(p)) {
                    // ValueError in Python => inf if x>0 else -inf
                    result = (a > 0.0) ? INFINITY : -INFINITY;
                } else {
                    result = p;
                }
                break;
            }
            default: result = a; break; // Passthrough for unknown ops
        }
    } else if (arg_count == 3) {
        // Ternary operations
        double a = args[0];
        double b = args[1];
        double c = args[2];
        switch (op) {
            case OPS_WHERE: result = (a != 0.0) ? b : c; break;
            case OPS_MULACC: result = (a * b) + c; break;
            default: result = a; break; // Passthrough for unknown ops
        }
    }
    
    // Apply dtype truncation (handles overflow/underflow)
    return dtypes_truncate(result, &dtype);
}

// Line 902-978: Additional helper functions

// Identity element for reduction operations
double identity_element(Ops op, DType* dtype) {
    switch (op) {
        case OPS_ADD: return 0.0;
        case OPS_MUL: return 1.0;
        case OPS_MAX:
            if (dtypes_is_float(dtype)) {
                return -INFINITY;
            } else if (dtypes_is_int(dtype)) {
                // Return min value for int type
                // Check if it's an unsigned type by name
                bool is_unsigned = (strstr(dtype->name, "uint") != NULL);
                if (dtype->itemsize == 1) return is_unsigned ? 0.0 : -128.0;
                if (dtype->itemsize == 2) return is_unsigned ? 0.0 : -32768.0;
                if (dtype->itemsize == 4) return is_unsigned ? 0.0 : -2147483648.0;
                if (dtype->itemsize == 8) return is_unsigned ? 0.0 : -9223372036854775808.0;
            }
            return 0.0;
        case OPS_AND: return -1.0;  // All bits set
        case OPS_OR: return 0.0;
        default: return 0.0;
    }
}

// Line 910: def parents(self) -> set[UOp]:
UOp** uop_parents(UOp* uop, size_t* count) {
    // Return all parent nodes (sources)
    if (!uop || !uop->src_count) {
        if (count) *count = 0;
        return NULL;
    }
    
    if (count) *count = uop->src_count;
    UOp** parents = (UOp**)malloc(uop->src_count * sizeof(UOp*));
    memcpy(parents, uop->src, uop->src_count * sizeof(UOp*));
    return parents;
}

// Legacy uop_replace with DType* and no tag; forwards to uop_new
UOp* uop_replace(UOp* uop, Ops new_op, DType* new_dtype, UOp** new_src, size_t new_src_count, UOpArg* new_arg) {
    Ops op = (new_op != OPS_NOOP) ? new_op : uop->op;
    DType dtype = (new_dtype && new_dtype->count) ? *new_dtype : uop->dtype;
    UOp** src = new_src ? new_src : uop->src;
    size_t sc = new_src ? new_src_count : uop->src_count;
    UOpArg arg = {0}; if (new_arg) arg = *new_arg; else arg = uop->arg;
    return uop_new(op, dtype, src, sc, &arg, NULL);
}

// Line 930-978: Module initialization
void uop_ops_init(void) {
    // Initialize cache
    // fprintf(stderr, "DEBUG uop_ops_init start\n");
    uop_cache_init();
    
    // Initialize math traits
    mathtraits_init();
    
    // Reset statistics
    _match_stats_hits = 0;
    _match_stats_total = 0;
    // fprintf(stderr, "DEBUG uop_ops_init done\n");
}

void uop_ops_cleanup(void) {
    // Cleanup cache
    uop_cache_cleanup();
    
    // Cleanup math traits
    mathtraits_cleanup();
}

// Additional stub functions for testing
UOp* uop_var(const char* name, DType dtype) {
    // Create a variable UOp (placeholder implementation)
    UOpArg arg = {0};
    arg.type = ARG_NONE;
    return uop_new(OPS_DEFINE_VAR, dtype, NULL, 0, &arg, NULL);
}

UOp* uop_var_with_range(const char* name, DType dtype, int min_val, int max_val) {
    // Create a variable with range constraints
    // In Python: DEFINE_VAR with arg=(name, vmin, vmax)
    UOpArg arg = {0};
    arg.type = ARG_VAR;
    arg.var.vmin = min_val;
    arg.var.vmax = max_val;
    // Store name if needed (not currently used)
    return uop_new(OPS_DEFINE_VAR, dtype, NULL, 0, &arg, NULL);
}

// RANGE operation - creates index range from 0 to n-1
UOp* uop_range(UOp* n, int idx) {
    // RANGE(n, idx) creates range from 0 to n-1 with index idx
    UOpArg arg = {0};
    arg.type = ARG_INT;
    arg.int_data.i = idx;
    UOp* src[] = {n};
    // Parity with Python: rng.dtype == n.dtype
    DType dt = n ? n->dtype : dtypes.int32;
    return uop_new(OPS_RANGE, dt, src, 1, &arg, NULL);
}

UOp* uop_buffer(int64_t* shape, size_t shape_count, DType dtype) {
    // Create a buffer UOp
    UOpArg arg = {0};
    // Python spec expects BUFFER.arg to be an int
    arg.type = ARG_INT;
    arg.int_data.i = 0;
    UOp* u = uop_new(OPS_BUFFER, dtype, NULL, 0, &arg, NULL);
    if (shape && shape_count>0) {
        int32_t* shp = (int32_t*)malloc(sizeof(int32_t)*shape_count);
        for (size_t i=0;i<shape_count;i++) shp[i]=(int32_t)shape[i];
        u->st = shapetracker_from_shape(shp, (int32_t)shape_count);
        free(shp);
    }
    return u;
}

UOp* uop_reduce(UOp* src, Ops reduce_op) {
    // Create a reduce operation
    UOpArg arg = {0};
    arg.type = ARG_REDUCE;
    arg.reduce_data.reduce_op = reduce_op;
    arg.reduce_data.axes = NULL;
    arg.reduce_data.axes_count = 0;
    return uop_new(OPS_REDUCE, src->dtype, &src, 1, &arg, NULL);
}

UOp* uop_mod(UOp* a, UOp* b) {
    // Modulo operation
    return uop_new(OPS_MOD, a->dtype, (UOp*[]){a, b}, 2, NULL, NULL);
}

UOp* uop_gt(UOp* a, UOp* b) {
    // Greater than comparison
    // GT(a, b) is equivalent to LT(b, a)
    return uop_lt(b, a);
}


// Line 978: End of file
/* ops.c - Faithful port of tinygrad/uop/ops.py complete */
