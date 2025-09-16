#include "renderer/cstyle_base.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void cstyle_ctx_init(CStyleCtx* ctx){ if (!ctx) return; memset(ctx, 0, sizeof(*ctx)); }
void cstyle_ctx_free(CStyleCtx* ctx){
  if (!ctx) return;
  if (ctx->nm_vals){ for (int i=0;i<ctx->nm_count;i++) if (ctx->nm_vals[i]) free(ctx->nm_vals[i]); free(ctx->nm_vals); }
  if (ctx->nm_keys) free(ctx->nm_keys);
  if (ctx->cc_keys) free(ctx->cc_keys);
  if (ctx->cc_vals) free(ctx->cc_vals);
  memset(ctx, 0, sizeof(*ctx));
}

static int seen_id(int* ids, int count, int id){ for (int i=0;i<count;i++) if (ids[i]==id) return 1; return 0; }

int cstyle_ctx_collect_bufs(CStyleCtx* ctx, UOp** uops, int n, int* ids, const DType** dtypes, int max){
  int pc=0; if (!uops || !ids || !dtypes || max<=0) return 0;
  for (int i=0;i<n && pc<max;i++){
    UOp* u = uops[i]; if (!u) continue;
    if (u->op == OPS_DEFINE_GLOBAL){
      int id = (u->arg.type==ARG_INT)? u->arg.int_data.i : 0;
      if (!seen_id(ids, pc, id)) {
        ids[pc]=id; dtypes[pc] = &u->dtype; pc++;
        if (ctx && ctx->buf_count < 256){ int bi=ctx->buf_count++; ctx->buf_ids[bi]=id; ctx->buf_dtypes[bi]=&u->dtype; ctx->buf_writable[bi]=0; }
      }
    }
  }
  return pc;
}

static int buf_index_by_id(CStyleCtx* ctx, int id){ if (!ctx) return -1; for (int i=0;i<ctx->buf_count;i++) if (ctx->buf_ids[i]==id) return i; return -1; }

void cstyle_ctx_mark_writes(CStyleCtx* ctx, UOp** uops, int n){
  if (!ctx || !uops) return;
  for (int i=0;i<n;i++){
    UOp* u = uops[i]; if (!u) continue;
    if (u->op == OPS_STORE && u->src_count>=1){
      // find DEFINE_GLOBAL id from store dst buffer base
      int gid=-1; size_t tn=0; UOp** bt = uop_toposort(u->src[0], &tn);
      if (bt){ for (size_t k=0;k<tn;k++){ if (bt[k]->op==OPS_DEFINE_GLOBAL){ gid=(bt[k]->arg.type==ARG_INT)? bt[k]->arg.int_data.i : 0; break; } } free(bt);}      
      if (gid>=0){ int bi = buf_index_by_id(ctx, gid); if (bi>=0) ctx->buf_writable[bi] = 1; }
    }
  }
}

int cstyle_ctx_buf_writable(CStyleCtx* ctx, int id){ int bi = buf_index_by_id(ctx, id); return (bi>=0) ? ctx->buf_writable[bi] : 0; }

static void nm_put(CStyleCtx* ctx, UOp* u, const char* name){ if (!ctx||!u||!name) return; if (ctx->nm_count+1>ctx->nm_cap){ int nc = ctx->nm_cap?ctx->nm_cap*2:64; ctx->nm_cap=nc; ctx->nm_keys=(UOp**)realloc(ctx->nm_keys, nc*sizeof(UOp*)); ctx->nm_vals=(char**)realloc(ctx->nm_vals, nc*sizeof(char*)); }
  ctx->nm_keys[ctx->nm_count]=u; ctx->nm_vals[ctx->nm_count]=strdup(name); ctx->nm_count++; }
static const char* nm_get(CStyleCtx* ctx, UOp* u){ if (!ctx||!u) return NULL; for (int i=0;i<ctx->nm_count;i++) if (ctx->nm_keys[i]==u) return ctx->nm_vals[i]; return NULL; }

void cstyle_ctx_compute_child_counts(CStyleCtx* ctx, UOp** uops, int n){
  if (!ctx || !uops) return;
  for (int i=0;i<n;i++){
    UOp* u = uops[i]; if (!u) continue;
    for (size_t j=0;j<u->src_count;j++){
      UOp* v = u->src[j]; if (!v) continue;
      // accumulate count
      int found=-1; for (int k=0;k<ctx->cc_count;k++) if (ctx->cc_keys[k]==v){ found=k; break; }
      if (found<0){ if (ctx->cc_count+1>ctx->cc_cap){ int nc=ctx->cc_cap?ctx->cc_cap*2:64; ctx->cc_cap=nc; ctx->cc_keys=(UOp**)realloc(ctx->cc_keys, nc*sizeof(UOp*)); ctx->cc_vals=(int*)realloc(ctx->cc_vals, nc*sizeof(int)); }
        ctx->cc_keys[ctx->cc_count]=v; ctx->cc_vals[ctx->cc_count]=1; ctx->cc_count++; }
      else ctx->cc_vals[found]++;
    }
  }
}

static const char* prefix_for_op(Ops op){
  switch(op){
    case OPS_WMMA: return "wmma";
    case OPS_DEFINE_LOCAL: return "temp";
    case OPS_CONST: return "const";
    case OPS_CAST: case OPS_BITCAST: return "cast";
    case OPS_GEP: return "gep";
    case OPS_VECTORIZE: return "cast";
    case OPS_PRECAST: return "precast";
    case OPS_INDEX: return "bidx";
    case OPS_DEFINE_REG: return "acc";
    case OPS_LOAD: return "val";
    default: return "alu";
  }
}

void cstyle_ctx_assign_names(CStyleCtx* ctx, UOp** uops, int n){
  if (!ctx || !uops) return;
  int c_wmma=0,c_temp=0,c_const=0,c_cast=0,c_gep=0,c_precast=0,c_bidx=0,c_acc=0,c_val=0,c_alu=0,c_ridx=0;
  for (int i=0;i<n;i++){
    UOp* u = uops[i]; if (!u) continue;
    if (u->op==OPS_NOOP) continue;
    if (u->op==OPS_SINK){ continue; }
    if (u->op==OPS_DEFINE_GLOBAL){ char nm[32]; snprintf(nm,sizeof(nm),"data%d", (u->arg.type==ARG_INT)?u->arg.int_data.i:0); nm_put(ctx, u, nm); continue; }
    if (u->op==OPS_DEFINE_VAR && u->arg.type==ARG_VAR && u->arg.var.name){ nm_put(ctx, u, u->arg.var.name); continue; }
    if (u->op==OPS_SPECIAL && u->tag){ nm_put(ctx, u, (const char*)u->tag); continue; }
    if (u->op==OPS_RANGE){ char nm[32]; snprintf(nm,sizeof(nm),"ridx%d", c_ridx++); nm_put(ctx, u, nm); continue; }
    const char* pref = prefix_for_op(u->op); char nm[32];
    if (strcmp(pref,"wmma")==0) snprintf(nm,sizeof(nm),"wmma%d", c_wmma++);
    else if (strcmp(pref,"temp")==0) snprintf(nm,sizeof(nm),"temp%d", c_temp++);
    else if (strcmp(pref,"const")==0) snprintf(nm,sizeof(nm),"const%d", c_const++);
    else if (strcmp(pref,"cast")==0) snprintf(nm,sizeof(nm),"cast%d", c_cast++);
    else if (strcmp(pref,"gep")==0) snprintf(nm,sizeof(nm),"gep%d", c_gep++);
    else if (strcmp(pref,"precast")==0) snprintf(nm,sizeof(nm),"precast%d", c_precast++);
    else if (strcmp(pref,"bidx")==0) snprintf(nm,sizeof(nm),"bidx%d", c_bidx++);
    else if (strcmp(pref,"acc")==0) snprintf(nm,sizeof(nm),"acc%d", c_acc++);
    else if (strcmp(pref,"val")==0) snprintf(nm,sizeof(nm),"val%d", c_val++);
    else snprintf(nm,sizeof(nm),"alu%d", c_alu++);
    nm_put(ctx, u, nm);
  }
}

const char* cstyle_ctx_name_for(CStyleCtx* ctx, UOp* u){ return nm_get(ctx, u); }
