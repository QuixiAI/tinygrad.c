// Minimal C-style (Clang/CPU) renderer (M2 bootstrap)
#include "renderer/cstyle.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "dtype/dtype.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static char* sb_new_with(const char* a, const char* b) {
  size_t la=strlen(a), lb=strlen(b); char* s=(char*)malloc(la+lb+1); memcpy(s,a,la); memcpy(s+la,b,lb); s[la+lb]='\0'; return s;
}
static char* sb_append(char* base, const char* add) {
  if (!base) return strdup(add);
  size_t la=strlen(base), lb=strlen(add); base=(char*)realloc(base, la+lb+1); memcpy(base+la, add, lb); base[la+lb]='\0'; return base;
}

typedef struct { UOp** keys; char** names; int count; int cap; } name_map_t;
static const char* c_type_for(const DType* dt){
  const DType* s = dt;
  if (dt->count>1 && dt->_scalar) s = dt->_scalar;
  if (dtype_eq(s, &dtypes.float32)) return "float";
  if (dtype_eq(s, &dtypes.float64)) return "double";
  if (dtype_eq(s, &dtypes.int32)) return "int";
  if (dtype_eq(s, &dtypes.uint32)) return "unsigned int";
  if (dtype_eq(s, &dtypes.int8)) return "signed char";
  if (dtype_eq(s, &dtypes.uint8)) return "unsigned char";
  if (dtype_eq(s, &dtypes.bool_)) return "_Bool";
  return "int";
}
static char* render_const(UOp* u){ char buf[64]; if (dtypes_is_float(&u->dtype)) { double v=u->arg.const_data.const_value;
    if (dtype_eq(&u->dtype, &dtypes.float32)) snprintf(buf,sizeof(buf),"%gf",(float)v); else snprintf(buf,sizeof(buf),"%g",v);
  } else if (dtype_eq(&u->dtype,&dtypes.int64)) snprintf(buf,sizeof(buf),"%lldll",(long long)u->arg.const_data.const_value);
    else if (dtype_eq(&u->dtype,&dtypes.uint64)) snprintf(buf,sizeof(buf),"%llullu",(unsigned long long)u->arg.const_data.const_value);
    else if (dtype_eq(&u->dtype,&dtypes.uint32)) snprintf(buf,sizeof(buf),"%uu",(unsigned)u->arg.const_data.const_value);
    else if (dtype_eq(&u->dtype,&dtypes.bool_)) snprintf(buf,sizeof(buf),"%s",u->arg.const_data.const_value?"1":"0");
    else snprintf(buf,sizeof(buf),"%lld",(long long)u->arg.const_data.const_value);
  return strdup(buf);} 
static const char* ssa_get(name_map_t* m, UOp* u){
  for (int i=0;i<m->count;i++) if (m->keys[i]==u) return m->names[i];
  return NULL;
}
static const char* ssa_add(name_map_t* m, UOp* u){
  char tmp[32]; snprintf(tmp,sizeof(tmp),"v%d",m->count);
  if (m->count+1>m->cap){ int nc=m->cap?m->cap*2:32; m->names=(char**)realloc(m->names,nc*sizeof(char*)); m->keys=(UOp**)realloc(m->keys,nc*sizeof(UOp*)); m->cap=nc; }
  m->keys[m->count]=u; m->names[m->count]=strdup(tmp); m->count++;
  return m->names[m->count-1];
}
static char* render_alu(UOp* u, const char* a, const char* b){ const char* op=NULL;
  switch(u->op){ case OPS_ADD: op="+"; break; case OPS_SUB: op="-"; break; case OPS_MUL: op="*"; break;
    case OPS_OR: op="|"; break; case OPS_AND: op="&"; break; case OPS_XOR: op="^"; break; case OPS_MAX: op=NULL; break; default: op=NULL; }
  if (op){ size_t la=strlen(a),lb=strlen(b); char* s=(char*)malloc(la+lb+4); snprintf(s,la+lb+4,"(%s%s%s)",a,op,b); return s; }
  if (u->op==OPS_MAX){ size_t la=strlen(a),lb=strlen(b); char* s=(char*)malloc(la+lb+16); snprintf(s,la+lb+16,"((%s)>=(%s)?%s:%s)",a,b,a,b); return s; }
  return strdup(a); }
static char* render_cmp(UOp* u, const char* a, const char* b){
  const char* op = NULL;
  switch (u->op) {
    case OPS_CMPLT: op = "<";  break;
    case OPS_CMPEQ: op = "=="; break;
    case OPS_CMPNE: op = "!="; break;
    default: break;
  }
  if (!op) return strdup("0");
  size_t la=strlen(a), lb=strlen(b);
  char* s=(char*)malloc(la+lb+6);
  snprintf(s, la+lb+6, "(%s%s%s)", a, op, b);
  return s;
}
static char* clang_render(Renderer* self, UOp** uops, int uops_count) {
  (void)self;
  const char* header = "typedef float float4 __attribute__((aligned(4),vector_size(16)));\n";
  // collect DEFINE_GLOBAL ids to form function parameters
  int gids[128]; int gidc=0;
  for (int i=0;i<uops_count && gidc<128;i++) if (uops[i] && uops[i]->op==OPS_DEFINE_GLOBAL) gids[gidc++] = (uops[i]->arg.type==ARG_INT)? uops[i]->arg.int_data.i : 0;
  char fnhead[1024]; strcpy(fnhead, "void kernel_main(");
  for (int i=0;i<gidc;i++){ char seg[64]; snprintf(seg,sizeof(seg),"float* data%d%s",gids[i], (i<gidc-1)?", ":""); strcat(fnhead, seg);} strcat(fnhead, ") {\n");
  const char* end = "}\n";
  char* src = sb_new_with(header, fnhead);
  name_map_t map={0};
  int indent = 1;
  #define INDENT() for(int _i=0;_i<indent;_i++) src = sb_append(src, "  ")
  for (int i=0;i<uops_count;i++){
    UOp* u = uops[i]; if (!u || u->op==OPS_SINK || u->op==OPS_NOOP) continue;
    if (u->op==OPS_DEFINE_REG){ const PtrDType* pd=(const PtrDType*)&u->dtype; char line[256]; INDENT();
      snprintf(line,sizeof(line),"%s reg_%d[%d];\n", c_type_for(pd->base_dtype), 0, pd->size); src = sb_append(src, line); continue; }
    if (u->op==OPS_CONST){ char* val=render_const(u); const char* nm=ssa_add(&map, u); char line[256]; INDENT();
      snprintf(line,sizeof(line),"%s %s = %s;\n", c_type_for(&u->dtype), nm, val); src=sb_append(src,line); free(val); continue; }
    if (u->op==OPS_RANGE && u->src_count>=1){ int ax = (u->arg.type==ARG_INT)? u->arg.int_data.i : 0; char ridx[32]; snprintf(ridx,sizeof(ridx),"ridx%d",ax);
      char bound[32]; if (u->src[0]->arg.type==ARG_CONST) snprintf(bound,sizeof(bound),"%d", (int)u->src[0]->arg.const_data.const_value); else strcpy(bound,"1");
      char line[256]; INDENT(); snprintf(line,sizeof(line),"for (int %s = 0; %s < %s; %s++) {\n", ridx, ridx, bound, ridx); src = sb_append(src, line); indent++; continue; }
    if (u->op==OPS_ENDRANGE){ indent--; INDENT(); src = sb_append(src, "}\n"); continue; }
    if (u->op==OPS_IF && u->src_count>=1){
      // render condition if it's a cmp or non-zero expr fallback
      char* cond=NULL; UOp* c=u->src[0];
      if (c->op==OPS_CMPLT||c->op==OPS_CMPEQ||c->op==OPS_CMPNE){
        const char* an = ssa_get(&map, c->src[0]); const char* bn = ssa_get(&map, c->src[1]);
        char* a = an?strdup(an): (c->src[0]->op==OPS_CONST?render_const(c->src[0]):strdup("0"));
        char* b = bn?strdup(bn): (c->src[1]->op==OPS_CONST?render_const(c->src[1]):strdup("0"));
        cond = render_cmp(c, a, b); free(a); free(b);
      } else {
        const char* cn = ssa_get(&map, c); cond = strdup(cn?cn:"1");
      }
      INDENT(); char line[512]; snprintf(line,sizeof(line),"if %s {\n", cond); src = sb_append(src, line); free(cond); indent++; continue;
    }
    if (u->op==OPS_ENDIF){ indent--; INDENT(); src = sb_append(src, "}\n"); continue; }
    if (group_op.is_alu[u->op] && u->src_count==2){
      const char* an = ssa_get(&map, u->src[0]); const char* bn = ssa_get(&map, u->src[1]);
      char* a = an?strdup(an):(u->src[0]->op==OPS_CONST?render_const(u->src[0]):strdup("0"));
      char* b = bn?strdup(bn):(u->src[1]->op==OPS_CONST?render_const(u->src[1]):strdup("0"));
      char* expr=render_alu(u,a,b); const char* nm=ssa_add(&map, u);
      char line[512]; INDENT(); snprintf(line,sizeof(line),"%s %s = %s;\n", c_type_for(&u->dtype), nm, expr); src=sb_append(src,line); free(a); free(b); free(expr); continue; }
    if (u->op==OPS_INDEX && u->src_count>=2){
      // index(buffer, idx) → name representing pointer
      // try to find dataN param from DEFINE_GLOBAL id in src[0]
      int gid=-1; size_t tn=0; UOp** bt = uop_toposort(u->src[0], &tn);
      if (bt){ for(size_t k=0;k<tn;k++) if (bt[k]->op==OPS_DEFINE_GLOBAL){ gid=(bt[k]->arg.type==ARG_INT)? bt[k]->arg.int_data.i : 0; break; } free(bt);} 
      const char* in = ssa_get(&map, u->src[1]); char* idxs = in?strdup(in):(u->src[1]->op==OPS_CONST?render_const(u->src[1]):strdup("0"));
      const char* nm = ssa_add(&map, u); char line[256]; INDENT();
      if (gid>=0) snprintf(line,sizeof(line),"float* %s = data%d + %s;\n", nm, gid, idxs); else snprintf(line,sizeof(line),"float* %s = %s;\n", nm, idxs);
      src = sb_append(src, line); free(idxs); continue;
    }
    if (u->op==OPS_LOAD && u->src_count>=1){
      const char* ptr = ssa_get(&map, u->src[0]); if (!ptr) ptr = "ptr";
      const char* nm = ssa_add(&map, u); char line[256]; INDENT();
      snprintf(line,sizeof(line),"%s %s = *%s;\n", c_type_for(&u->dtype), nm, ptr); src = sb_append(src, line); continue;
    }
    if (u->op==OPS_CAST && u->src_count>=1){
      const char* srcn = ssa_get(&map, u->src[0]); char* srcv = srcn?strdup(srcn):(u->src[0]->op==OPS_CONST?render_const(u->src[0]):strdup("0"));
      const char* nm = ssa_add(&map, u);
      char line[512]; INDENT();
      if (u->dtype.count > 1) {
        // vector cast
        snprintf(line,sizeof(line),"%s %s = __builtin_convertvector(%s, %s);\n", c_type_for(&u->dtype), nm, srcv, c_type_for(&u->dtype));
      } else {
        snprintf(line,sizeof(line),"%s %s = (%s)(%s);\n", c_type_for(&u->dtype), nm, c_type_for(&u->dtype), srcv);
      }
      src = sb_append(src, line); free(srcv); continue;
    }
    if (u->op==OPS_BITCAST && u->src_count>=1){
      const char* srcn = ssa_get(&map, u->src[0]); char* srcv = srcn?strdup(srcn):(u->src[0]->op==OPS_CONST?render_const(u->src[0]):strdup("0"));
      const char* nm = ssa_add(&map, u);
      char line[512]; INDENT();
      snprintf(line,sizeof(line),"%s %s = *((%s*)&%s);\n", c_type_for(&u->dtype), nm, c_type_for(&u->dtype), srcv);
      src = sb_append(src, line); free(srcv); continue;
    }
    if (u->op==OPS_STORE && u->src_count>=2){
      const char* ptr = ssa_get(&map, u->src[0]); if (!ptr) ptr = "ptr";
      const char* valn = ssa_get(&map, u->src[1]); char* val = valn?strdup(valn):(u->src[1]->op==OPS_CONST?render_const(u->src[1]):strdup("0"));
      char line[256]; INDENT(); snprintf(line,sizeof(line),"*%s = %s;\n", ptr, val); src = sb_append(src, line); free(val); continue;
    }
  }
  src = sb_append(src, end); return src; }

Renderer* renderer_cstyle_clang(void) {
  Renderer* r = (Renderer*)calloc(1, sizeof(Renderer));
  r->device = "CPU";
  r->suffix = "";
  r->supports_float4 = true;
  r->has_local = false;
  r->has_shared = false;
  r->shared_max = 0;
  r->global_max[0]=r->global_max[1]=r->global_max[2]=0;
  r->local_max[0]=r->local_max[1]=r->local_max[2]=0;
  r->pre_matcher = NULL;
  r->extra_matcher = NULL;
  r->render = clang_render;
  return r;
}
