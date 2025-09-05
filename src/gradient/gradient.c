/* gradient.c - Faithful port of tinygrad/gradient.py (symbolic only) */

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "uop/uop.h"
#include "gradient/gradient.h"
#include <stdio.h>

// Gradient result structure
struct gradient_entry { tg_uop_t* variable; tg_uop_t* gradient; };
struct tg_gradient_result { struct gradient_entry* entries; int entry_count; };

// Simple dict for gradients
typedef struct gradient_dict_entry { tg_uop_t* key; tg_uop_t* value; struct gradient_dict_entry* next; } gradient_dict_entry_t;
typedef struct { gradient_dict_entry_t** buckets; size_t bucket_count; } gradient_dict_t;
static gradient_dict_t* gradient_dict_create(size_t n){ gradient_dict_t* d=calloc(1,sizeof(*d)); d->bucket_count=n; d->buckets=calloc(n,sizeof(void*)); return d; }
static size_t gradient_dict_hash(tg_uop_t* k, size_t n){ return ((size_t)k>>3)%n; }
static void gradient_dict_set(gradient_dict_t* d, tg_uop_t* k, tg_uop_t* v){ size_t i=gradient_dict_hash(k,d->bucket_count); for(gradient_dict_entry_t*e=d->buckets[i];e;e=e->next) if(e->key==k){ e->value=v; return;} gradient_dict_entry_t* ne=calloc(1,sizeof(*ne)); ne->key=k; ne->value=v; ne->next=d->buckets[i]; d->buckets[i]=ne; }
static tg_uop_t* gradient_dict_get(gradient_dict_t* d, tg_uop_t* k){ size_t i=gradient_dict_hash(k,d->bucket_count); for(gradient_dict_entry_t*e=d->buckets[i];e;e=e->next) if(e->key==k) return e->value; return NULL; }
static bool gradient_dict_contains(gradient_dict_t* d, tg_uop_t* k){ return gradient_dict_get(d,k)!=NULL; }
static void gradient_dict_free(gradient_dict_t* d){ for(size_t i=0;i<d->bucket_count;i++){ gradient_dict_entry_t*e=d->buckets[i]; while(e){gradient_dict_entry_t*n=e->next; free(e); e=n;} } free(d->buckets); free(d); }

// Helpers for movement-aware gradients
static tg_uop_t* grad_reshape(tg_uop_t* ctx, tg_uop_t* target) {
  int nd=0; const int32_t* shp = uop_shape(target, &nd); if (!shp || nd<=0) return ctx;
  int32_t* s = (int32_t*)malloc(sizeof(int32_t)*nd); for (int i=0;i<nd;i++) s[i]=shp[i];
  tg_uop_t* r = tg_uop_reshape(ctx, s, nd); free(s); return r;
}
static tg_uop_t* grad_permute(tg_uop_t* ctx, tg_uop_t* t0) {
  // invert axes
  int n = t0->arg.reduce_data.axes_count; if (n<=0 || !t0->arg.reduce_data.axes) return ctx;
  int* inv = (int*)malloc(sizeof(int)*n);
  for (int i=0;i<n;i++) inv[i]=0; for (int i=0;i<n;i++) { int a=t0->arg.reduce_data.axes[i]; if (a<0) a+=n; inv[a]=i; }
  tg_uop_t* r = tg_uop_permute(ctx, inv, n); free(inv); return r;
}
static tg_uop_t* grad_expand(tg_uop_t* ctx, tg_uop_t* src, tg_uop_t* out) {
  int nds=0, ndo=0; const int32_t* ss = uop_shape(src, &nds); const int32_t* so = uop_shape(out, &ndo);
  if (!ss || !so || nds!=ndo) return ctx;
  int* axes = (int*)malloc(sizeof(int)*nds); int axc=0;
  for (int i=0;i<nds;i++) if (ss[i]==1 && so[i]>1) axes[axc++]=i;
  tg_uop_t* r = ctx;
  if (axc>0) { r = tg_uop_reduce_axis(ctx, OPS_ADD, axes, axc); r = tg_uop_reshape(r, ss, nds); }
  if (getenv("DEBUG_GRAD")) {
    fprintf(stderr, "grad_expand: ss=["); for(int i=0;i<nds;i++) fprintf(stderr, "%d%s", ss[i], i==nds-1?"]":" ");
    fprintf(stderr, ", so=["); for(int i=0;i<ndo;i++) fprintf(stderr, "%d%s", so[i], i==ndo-1?"]":" ");
    fprintf(stderr, ", axc=%d\n", axc);
  }
  free(axes); return r;
}
static tg_uop_t* grad_pad(tg_uop_t* ctx, tg_uop_t* t0) {
  int nd=0; const int32_t* oshp = uop_shape(t0, &nd); if (!oshp || t0->arg.type!=ARG_PAD_PARAMS) return ctx;
  int32_t* start = (int32_t*)malloc(sizeof(int32_t)*nd);
  int32_t* end   = (int32_t*)malloc(sizeof(int32_t)*nd);
  for (int i=0;i<nd;i++) { start[i]=t0->arg.pad_data.before[i]; end[i]=(int32_t)oshp[i]-t0->arg.pad_data.after[i]; }
  tg_uop_t* r = tg_uop_shrink(ctx, start, end, nd); free(start); free(end); return r;
}
static tg_uop_t* grad_shrink(tg_uop_t* ctx, tg_uop_t* t0) {
  int nd=0; const int32_t* ishp = uop_shape(t0->src[0], &nd); if (!ishp || t0->arg.type!=ARG_SHRINK_PARAMS) return ctx;
  int32_t* before = (int32_t*)malloc(sizeof(int32_t)*nd);
  int32_t* after  = (int32_t*)malloc(sizeof(int32_t)*nd);
  for (int i=0;i<nd;i++) { int32_t st=t0->arg.shrink_data.start[i]; int32_t ed=t0->arg.shrink_data.end[i]; if (ed<0) ed=ishp[i]; before[i]=st; after[i]=ishp[i]-ed; }
  tg_uop_t* r = tg_uop_pad(ctx, before, after, nd); free(before); free(after); return r;
}
static tg_uop_t* reduce_to_operand(tg_uop_t* ctx, tg_uop_t* out_op, tg_uop_t* src) {
  // derive broadcasted output shape from both operands
  int nd0=0, nd1=0, nds=0; const int32_t* s0=NULL; const int32_t* s1=NULL; const int32_t* ss=NULL;
  s0 = uop_shape(out_op->src[0], &nd0); s1 = uop_shape(out_op->src[1], &nd1); ss = uop_shape(src, &nds);
  if (!s0 || !s1 || !ss) return ctx;
  int nd = nd0>nd1? nd0: nd1;
  int32_t* so = (int32_t*)malloc(sizeof(int32_t)*nd);
  for (int i=0;i<nd;i++){
    int idx0 = i - (nd - nd0);
    int idx1 = i - (nd - nd1);
    int d0 = idx0<0 ? 1 : s0[idx0];
    int d1 = idx1<0 ? 1 : s1[idx1];
    so[i] = d0==1 ? d1 : d0;
  }
  // map src dims to broadcast dims aligned right
  int* axes = (int*)malloc(sizeof(int)*nd); int axc=0;
  for (int i=0;i<nd;i++){
    int idxs = i - (nd - nds);
    int ds = idxs<0 ? 1 : ss[idxs];
    if (ds==1 && so[i]>1) axes[axc++]=i;
  }
  if (getenv("DEBUG_GRAD")) {
    fprintf(stderr, "reduce_to_operand: so=["); for(int i=0;i<nd;i++) fprintf(stderr, "%d%s", so[i], i==nd-1?"]":" ");
    fprintf(stderr, ", ss=["); for(int i=0;i<nds;i++) fprintf(stderr, "%d%s", ss[i], i==nds-1?"]":" ");
    fprintf(stderr, ", axes="); for(int i=0;i<axc;i++) fprintf(stderr, "%d%s", axes[i], i==axc-1?"":" "); fprintf(stderr, "\n");
  }
  if (getenv("DEBUG_GRAD")) {
    fprintf(stderr, "reduce_to_operand: nd=%d nd0=%d nd1=%d nds=%d axc=%d\n", nd, nd0, nd1, nds, axc);
  }
  tg_uop_t* r = ctx; if (axc>0) {
    int nctx=0; const int32_t* sctx = uop_shape(ctx, &nctx);
    if (sctx && nctx==nd) {
      r = tg_uop_reduce_axis(ctx, OPS_ADD, axes, axc);
    } else {
      // Broadcast scalar ctx to output shape then reduce on axes
      int32_t* ones = (int32_t*)malloc(sizeof(int32_t)*nd); for(int i=0;i<nd;i++) ones[i]=1;
      tg_uop_t* bex = tg_uop_expand(tg_uop_reshape(ctx, ones, nd), so, nd);
      free(ones);
      r = tg_uop_reduce_axis(bex, OPS_ADD, axes, axc);
    }
    r = tg_uop_reshape(r, ss, nds);
  }
  free(axes); free(so); return r;
}

// reduce_gradient(ctx, ret) — movement and simple reduce handling
static tg_uop_t** reduce_gradient(tg_uop_t* ctx, tg_uop_t* ret, int* out_count){
  tg_uop_t** out=calloc(1,sizeof(tg_uop_t*)); *out_count=1; if(!ret){out[0]=ctx;return out;}
  if(ret->op==OPS_RESHAPE){ out[0]=grad_reshape(ctx, ret->src[0]); return out; }
  if(ret->op==OPS_REDUCE_AXIS && ret->arg.type==ARG_REDUCE && ret->arg.reduce_data.reduce_op==OPS_ADD){
    // broadcast ctx back to input shape
    int nd=0; const int32_t* shp = uop_shape(ret->src[0], &nd);
    int32_t* tmp_shape = NULL;
    if (!shp || nd<=0){
      // try to infer broadcasted shape for common binary ops
      UOp* a = ret->src[0];
      if (a && (a->op==OPS_ADD || a->op==OPS_SUB || a->op==OPS_MUL || a->op==OPS_MAX || a->op==OPS_FDIV || a->op==OPS_IDIV || a->op==OPS_POW)){
        int n0=0,n1=0; const int32_t* s0=uop_shape(a->src[0], &n0); const int32_t* s1=uop_shape(a->src[1], &n1);
        if (s0 && s1){ int ndm = n0>n1? n0:n1; tmp_shape=(int32_t*)malloc(sizeof(int32_t)*ndm); for(int i=0;i<ndm;i++){ int i0=i-(ndm-n0); int i1=i-(ndm-n1); int d0 = i0<0?1:s0[i0]; int d1=i1<0?1:s1[i1]; tmp_shape[i]= d0==1? d1: d0; } shp=tmp_shape; nd=ndm; }
      }
    }
    if (!shp || nd<=0){ out[0]=ctx; return out; }
    int32_t* ones = (int32_t*)malloc(sizeof(int32_t)*nd); for(int i=0;i<nd;i++) ones[i]=1;
    tg_uop_t* r = tg_uop_reshape(ctx, ones, nd); free(ones);
    if (getenv("DEBUG_GRAD")) { fprintf(stderr, "expand reduce_grad to nd=%d firstdim=%d\n", nd, shp[0]); }
    out[0] = tg_uop_expand(r, shp, nd);
    if (tmp_shape) free(tmp_shape);
    return out;
  }
  if(ret->op==OPS_PERMUTE){ out[0]=grad_permute(ctx, ret); return out; }
  if(ret->op==OPS_EXPAND){ out[0]=grad_expand(ctx, ret->src[0], ret); return out; }
  if(ret->op==OPS_PAD){ out[0]=grad_pad(ctx, ret); return out; }
  if(ret->op==OPS_SHRINK){ out[0]=grad_shrink(ctx, ret); return out; }
  if(ret->op==OPS_FLIP){ out[0]=tg_uop_flip(ctx, ret->arg.int_data.i); return out; }
  if(ret->arg.type==ARG_REDUCE && ret->arg.reduce_data.reduce_op==OPS_ADD){out[0]=ctx;return out;}
  if(ret->arg.type==ARG_REDUCE && ret->arg.reduce_data.reduce_op==OPS_MAX){out[0]=ctx;return out;}
  if(ret->arg.type==ARG_REDUCE && ret->arg.reduce_data.reduce_op==OPS_MUL){ out[0]=tg_uop_div(tg_uop_mul(ctx, ret), ret->src[0]); return out;}
  out[0]=ctx; return out; }

// pm_gradient rules (subset; mirrors python structure)
static tg_uop_t** pm_rules(tg_uop_t* t0, tg_uop_t* ctx, int* out_count){
  tg_uop_t** lgrads=NULL; switch(t0->op){
    case OPS_CAST: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=tg_uop_cast(ctx, &t0->src[0]->dtype); break;
    case OPS_RECIP: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=tg_uop_neg(tg_uop_mul(ctx, tg_uop_mul(t0,t0))); break;
    case OPS_SIN: { lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; tg_uop_t* pi2=tg_uop_const(TG_F32, (float)M_PI/2.0f); tg_uop_t* cosx=tg_uop_sin(tg_uop_sub(pi2, t0->src[0])); lgrads[0]=tg_uop_mul(cosx, ctx); break; }
    case OPS_LOG2: { lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; tg_uop_t* ln2=tg_uop_const(TG_F32, logf(2.0f)); lgrads[0]=tg_uop_div(ctx, tg_uop_mul(t0->src[0], ln2)); break; }
    case OPS_EXP2: { lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; tg_uop_t* ln2=tg_uop_const(TG_F32, logf(2.0f)); lgrads[0]=tg_uop_mul(tg_uop_mul(t0, ctx), ln2); break; }
    case OPS_SQRT: { lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; tg_uop_t* two=tg_uop_const(TG_F32, 2.0f); lgrads[0]=tg_uop_div(ctx, tg_uop_mul(t0, two)); break; }
    case OPS_CMPLT: case OPS_CMPNE: lgrads=calloc(2,sizeof(*lgrads)); *out_count=2; lgrads[0]=NULL; lgrads[1]=NULL; break;
    case OPS_ADD: lgrads=calloc(2,sizeof(*lgrads)); *out_count=2; lgrads[0]=ctx; lgrads[1]=ctx; ctx->ref_count+=2; break;
    case OPS_SUB: lgrads=calloc(2,sizeof(*lgrads)); *out_count=2; lgrads[0]=ctx; lgrads[1]=tg_uop_neg(ctx); break;
    case OPS_POW: {
      lgrads=calloc(2,sizeof(*lgrads)); *out_count=2; tg_uop_t* one=tg_uop_const(TG_F32,1.0f);
      lgrads[0]=tg_uop_mul(ctx, tg_uop_mul(t0->src[1], tg_uop_pow(t0->src[0], tg_uop_sub(t0->src[1], one))));
      tg_uop_t* ln = tg_uop_mul(tg_uop_log2(t0->src[0]), tg_uop_const(TG_F32, logf(2.0f)));
      lgrads[1]=tg_uop_mul(ctx, tg_uop_mul(t0, ln)); break; }
    case OPS_MAX: {
      lgrads=calloc(2,sizeof(*lgrads)); *out_count=2; tg_uop_t* cmp_gt=tg_uop_cmpgt(t0->src[0], t0->src[1]); tg_uop_t* cmp_lt=tg_uop_cmplt(t0->src[0], t0->src[1]); tg_uop_t* cmp_ne=tg_uop_cmpne(t0->src[0], t0->src[1]); tg_uop_t* zero=tg_uop_const_like(ctx, 0.0f); tg_uop_t* half=tg_uop_mul(ctx, tg_uop_const(TG_F32, 0.5f)); lgrads[0]=tg_uop_where(cmp_gt, ctx, tg_uop_where(cmp_ne, zero, half)); lgrads[1]=tg_uop_where(cmp_lt, ctx, tg_uop_where(cmp_ne, zero, half)); break; }
    case OPS_MUL: lgrads=calloc(2,sizeof(*lgrads)); *out_count=2; lgrads[0]=tg_uop_mul(t0->src[1], ctx); lgrads[1]=tg_uop_mul(t0->src[0], ctx); break;
    case OPS_RESHAPE: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=grad_reshape(ctx, t0->src[0]); break;
    case OPS_PERMUTE: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=grad_permute(ctx, t0); break;
    case OPS_EXPAND: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=grad_expand(ctx, t0->src[0], t0); break;
    case OPS_PAD: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=grad_pad(ctx, t0); break;
    case OPS_SHRINK: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=grad_shrink(ctx, t0); break;
    case OPS_FLIP: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=tg_uop_flip(ctx, t0->arg.int_data.i); break;
    case OPS_FDIV: case OPS_IDIV: {
      lgrads=calloc(2,sizeof(*lgrads)); *out_count=2;
      // wrt a: ctx / b
      lgrads[0]=tg_uop_div(ctx, t0->src[1]);
      // wrt b: -ctx * a / (b*b)
      tg_uop_t* b2 = tg_uop_mul(t0->src[1], t0->src[1]);
      tg_uop_t* num = tg_uop_mul(ctx, t0->src[0]);
      lgrads[1]=tg_uop_neg(tg_uop_div(num, b2));
      break; }
    case OPS_WHERE: lgrads=calloc(3,sizeof(*lgrads)); *out_count=3; lgrads[0]=NULL; lgrads[1]=tg_uop_where(t0->src[0], ctx, tg_uop_const_like(ctx,0.0f)); lgrads[2]=tg_uop_where(t0->src[0], tg_uop_const_like(ctx,0.0f), ctx); break;
    case OPS_REDUCE_AXIS: lgrads=reduce_gradient(ctx, t0, out_count); break;
    case OPS_CONTIGUOUS: case OPS_FUSE: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=ctx; break;
    case OPS_CONTIGUOUS_BACKWARD: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=tg_uop_contiguous(ctx); break;
    case OPS_MULTI: lgrads=calloc(t0->src_count,sizeof(*lgrads)); *out_count=t0->src_count; for(int i=0;i<t0->src_count;i++) lgrads[i]=ctx; break; // shard TODO
    case OPS_BITCAST: lgrads=calloc(1,sizeof(*lgrads)); *out_count=1; lgrads[0]=NULL; break;
    default: return NULL;
  }
  return lgrads;
}

// _deepwalk(root, targets)
static tg_uop_t** _deepwalk(tg_uop_t* root, tg_uop_t** targets, int target_count, int* out_count){
  int n=0; tg_uop_t** topo = tg_uop_toposort(root, &n); bool* in_path = calloc(n,sizeof(bool));
  for(int i=0;i<n;i++){
    tg_uop_t* u = topo[i]; bool ip=false;
    for(int j=0;j<u->src_count && !ip;j++){
      tg_uop_t* s = u->src[j];
      for(int t=0;t<target_count && !ip;t++) if (s==targets[t]) ip=true;
      if(!ip){ for(int k=0;k<i;k++) if (topo[k]==s && in_path[k]) { ip=true; break; } }
    }
    in_path[i]=ip;
  }
  tg_uop_t** out = calloc(n,sizeof(tg_uop_t*)); int m=0;
  for(int i=0;i<n;i++){ tg_uop_t* node=topo[i]; if (node->op!=OPS_DETACH && node->op!=OPS_ASSIGN && in_path[i]) out[m++]=node; }
  *out_count=m; free(in_path); free(topo); return out;
}

// compute_gradient(root, root_grad, targets)
static gradient_dict_t* compute_gradient_full(tg_uop_t* root, tg_uop_t* root_grad, tg_uop_t** targets, int target_count){
  gradient_dict_t* grads = gradient_dict_create(256); gradient_dict_set(grads, root, root_grad);
  int wc=0; tg_uop_t** walk = _deepwalk(root, targets, target_count, &wc);
  for(int i=wc-1;i>=0;i--){ tg_uop_t* t0=walk[i]; if(!gradient_dict_contains(grads,t0)) continue; tg_uop_t* grad_t0=gradient_dict_get(grads,t0);
    int lgc=0; tg_uop_t** lgrads = pm_rules(t0, grad_t0, &lgc); if(!lgrads){ free(walk); gradient_dict_free(grads); return NULL; }
    for(int j=0;j<t0->src_count;j++){ tg_uop_t* k=t0->src[j]; tg_uop_t* v=lgrads[j]; if(!v) continue; tg_uop_t* ex=gradient_dict_get(grads,k); gradient_dict_set(grads,k, ex? tg_uop_add(ex,v) : v); }
    free(lgrads);
  }
  free(walk); return grads;
}

// Public API
tg_gradient_result_t* tg_compute_gradient(tg_uop_t* expression, tg_uop_t* grad_seed, tg_uop_t** variables, int var_count){
  gradient_dict_t* dict = compute_gradient_full(expression, grad_seed, variables, var_count); if (!dict) return NULL;
  tg_gradient_result_t* res = calloc(1,sizeof(*res)); res->entries = calloc(var_count,sizeof(*res->entries)); res->entry_count=var_count;
  for(int i=0;i<var_count;i++){ res->entries[i].variable=variables[i]; res->entries[i].gradient=gradient_dict_get(dict, variables[i]); }
  gradient_dict_free(dict); return res;
}

tg_uop_t* tg_gradient_result_get(tg_gradient_result_t* r, tg_uop_t* v){ if(!r||!v) return NULL; for(int i=0;i<r->entry_count;i++){ if(r->entries[i].variable==v){ if(r->entries[i].gradient) r->entries[i].gradient->ref_count++; return r->entries[i].gradient; } } return NULL; }
void tg_gradient_result_free(tg_gradient_result_t* r){ if(!r) return; /* Gradients may be shared; let caller free returned UOps */ free(r->entries); free(r);} 
