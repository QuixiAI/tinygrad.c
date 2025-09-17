// Minimal C-style (Clang/CPU) renderer (M2 bootstrap)
#include "renderer/cstyle.h"
#include "renderer/cstyle_base.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "dtype/dtype.h"
#include "helpers/string_builder.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

typedef enum {
  CSTYLE_FLAVOR_CLANG,
  CSTYLE_FLAVOR_OPENCL,
  CSTYLE_FLAVOR_CUDA,
  CSTYLE_FLAVOR_AMD,
  CSTYLE_FLAVOR_HIP,
  CSTYLE_FLAVOR_METAL,
  CSTYLE_FLAVOR_QCOM,
  CSTYLE_FLAVOR_NV,
  CSTYLE_FLAVOR_INTEL
} CStyleFlavor;

typedef struct {
  Renderer base;
  CStyleFlavor flavor;
  const char* kernel_typedef;
  const char* buffer_prefix;
  const char* buffer_suffix;
  const char* smem_align;
  const char* smem_prefix;
  bool smem_prefix_for_cast;
  const char* arg_int_prefix;
  const char* barrier;
  const char* float4;
  const char* float4_style_open;
  const char* float4_style_close;
  int gep_arr_threshold;
  const char* infinity;
  const char* nan;
  const char* const* extra_args;
  int extra_arg_count;
} CStyleRenderer;

static CStyleRenderer* cstyle_from_renderer(Renderer* r) {
  return (CStyleRenderer*)r;
}

static char* cstyle_render(Renderer* base, UOp** uops, int uops_count);

static CStyleRenderer* cstyle_renderer_alloc(void) {
  CStyleRenderer* self = (CStyleRenderer*)calloc(1, sizeof(CStyleRenderer));
  if (!self) return NULL;
  self->smem_prefix_for_cast = true;
  self->float4_style_open = "(";
  self->float4_style_close = ")";
  self->gep_arr_threshold = 4;
  self->base.render = cstyle_render;
  self->base.supports_float4 = true;
  self->base.has_local = true;
  self->base.has_shared = false;
  self->base.shared_max = 0;
  self->base.global_max[0] = self->base.global_max[1] = self->base.global_max[2] = 0;
  self->base.local_max[0] = self->base.local_max[1] = self->base.local_max[2] = 0;
  self->kernel_typedef = NULL;
  self->buffer_prefix = "";
  self->buffer_suffix = "";
  self->smem_align = "";
  self->smem_prefix = "";
  self->arg_int_prefix = "const int";
  self->barrier = "";
  self->float4 = NULL;
  self->infinity = NULL;
  self->nan = NULL;
  return self;
}

typedef struct { UOp** keys; char** names; int count; int cap; } name_map_t;
static int str_eq(const char* a, const char* b){ return a && b && strcmp(a,b)==0; }
static int is_opencl(const CStyleRenderer* self){
  return self && (self->flavor == CSTYLE_FLAVOR_OPENCL || self->flavor == CSTYLE_FLAVOR_QCOM || self->flavor == CSTYLE_FLAVOR_INTEL);
}
static int is_metal(const CStyleRenderer* self){ return self && self->flavor == CSTYLE_FLAVOR_METAL; }
static int is_cuda(const CStyleRenderer* self){ return self && (self->flavor == CSTYLE_FLAVOR_CUDA || self->flavor == CSTYLE_FLAVOR_NV); }
static int is_hip_or_amd(const CStyleRenderer* self){ return self && (self->flavor == CSTYLE_FLAVOR_HIP || self->flavor == CSTYLE_FLAVOR_AMD); }
static int gep_threshold_for(const CStyleRenderer* self){ return self ? self->gep_arr_threshold : 0; }
static const char* gep_components = "xyzwabcdefghijkl";
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
static const char* c_vec_type_for(const DType* dt){
  if (dt->count == 4 && dt->_scalar && dtype_eq(dt->_scalar, &dtypes.float32)) return "float4";
  // fallback
  return c_type_for(dt);
}
static const char* c_type_for_opencl(const DType* dt){
  const DType* s = dt;
  if (dt->count>1 && dt->_scalar) s = dt->_scalar;
  if (dtype_eq(s, &dtypes.float32)) return "float";
  if (dtype_eq(s, &dtypes.float64)) return "double";
  if (dtype_eq(s, &dtypes.int32)) return "int";
  if (dtype_eq(s, &dtypes.uint32)) return "uint";
  if (dtype_eq(s, &dtypes.int8)) return "char";
  if (dtype_eq(s, &dtypes.uint8)) return "uchar";
  if (dtype_eq(s, &dtypes.bool_)) return "bool";
  if (dtype_eq(s, &dtypes.uint64)) return "ulong";
  if (dtype_eq(s, &dtypes.int64)) return "long";
  return "int";
}
static const char* c_type_for_b(const CStyleRenderer* self, const DType* dt){
  return is_opencl(self) ? c_type_for_opencl(dt) : c_type_for(dt);
}
static const char* c_vec_type_for_b(const CStyleRenderer* self, const DType* dt){ (void)self; return c_vec_type_for(dt); }

static char* render_vec_literal(const CStyleRenderer* self, const DType* dt, const char* elems) {
  const char* t = c_vec_type_for_b(self, dt);
  if (is_opencl(self)) {
    size_t L = strlen(t) + strlen(elems) + 6;
    char* s = (char*)malloc(L);
    snprintf(s, L, "(%s)(%s)", t, elems);
    return s;
  } else if (is_metal(self)) {
    size_t L = strlen(t) + strlen(elems) + 4;
    char* s = (char*)malloc(L);
    snprintf(s, L, "%s(%s)", t, elems);
    return s;
  } else if (is_hip_or_amd(self)) {
    size_t L = strlen(t) + strlen(elems) + 12;
    char* s = (char*)malloc(L);
    snprintf(s, L, "make_%s(%s)", t, elems);
    return s;
  } else {
    size_t L = strlen(t) + strlen(elems) + 6;
    char* s = (char*)malloc(L);
    snprintf(s, L, "(%s){%s}", t, elems);
    return s;
  }
}
static char* render_const(const CStyleRenderer* self, UOp* u){
  char buf[128];
  if (dtypes_is_float(&u->dtype)) {
    double v = u->arg.const_data.const_value;
    if (dtype_eq(&u->dtype, &dtypes.float32)) {
      if (isnan(v)) return strdup(self->nan ? self->nan : "__builtin_nanf(\"\")");
      if (isinf(v)) return strdup((v < 0) ? "-__builtin_inff()" : (self->infinity ? self->infinity : "__builtin_inff()"));
      snprintf(buf, sizeof(buf), "%gf", (float)v);
    } else if (dtype_eq(&u->dtype, &dtypes.float64)) {
      if (isnan(v)) return strdup(self->nan ? self->nan : "__builtin_nan(\"\")");
      if (isinf(v)) return strdup((v < 0) ? "-__builtin_inf()" : (self->infinity ? self->infinity : "__builtin_inf()"));
      snprintf(buf, sizeof(buf), "%g", v);
    } else {
      snprintf(buf, sizeof(buf), "%g", v);
    }
  } else if (dtype_eq(&u->dtype,&dtypes.int64)) snprintf(buf,sizeof(buf),"%lldll",(long long)u->arg.const_data.const_value);
    else if (dtype_eq(&u->dtype,&dtypes.uint64)) snprintf(buf,sizeof(buf),"%llullu",(unsigned long long)u->arg.const_data.const_value);
    else if (dtype_eq(&u->dtype,&dtypes.uint32)) snprintf(buf,sizeof(buf),"%uu",(unsigned)u->arg.const_data.const_value);
    else if (dtype_eq(&u->dtype,&dtypes.bool_)) snprintf(buf,sizeof(buf),"%s",u->arg.const_data.const_value?"1":"0");
    else snprintf(buf,sizeof(buf),"%lld",(long long)u->arg.const_data.const_value);
  return strdup(buf);
} 
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
static const char* ssa_alias(name_map_t* m, UOp* u, const char* existing){
  if (!existing) return ssa_add(m, u);
  if (m->count+1>m->cap){ int nc=m->cap?m->cap*2:32; m->names=(char**)realloc(m->names,nc*sizeof(char*)); m->keys=(UOp**)realloc(m->keys,nc*sizeof(UOp*)); m->cap=nc; }
  m->keys[m->count]=u; m->names[m->count]=strdup(existing); m->count++;
  return m->names[m->count-1];
}
static char* render_alu(UOp* u, const char* a, const char* b){ const char* op=NULL;
  switch(u->op){ case OPS_ADD: op="+"; break; case OPS_SUB: op="-"; break; case OPS_MUL: op="*"; break;
    case OPS_OR: op="|"; break; case OPS_AND: op="&"; break; case OPS_XOR: op="^"; break; case OPS_SHL: op="<<"; break; case OPS_SHR: op= ">>"; break; case OPS_MOD: op="%"; break; case OPS_MAX: op=NULL; break; default: op=NULL; }
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
static char* cstyle_render(Renderer* base, UOp** uops, int uops_count) {
  CStyleRenderer* self = cstyle_from_renderer(base);
  const char* header = "";
  if (is_metal(self)) {
    header = "#include <metal_stdlib>\nusing namespace metal;\n";
  } else if (!is_opencl(self)) {
    header = "typedef float float4 __attribute__((aligned(4),vector_size(16)));\n";
  }
  // collect DEFINE_GLOBAL ids to form function parameters (with proper dtype)
  int param_ids[128]; const DType* param_dtypes[128]; bool param_is_image[128]; int param_count=0;
  {
    const DType* tmp_dt[128];
    CStyleCtx ctx; cstyle_ctx_init(&ctx);
    param_count = cstyle_ctx_collect_bufs(&ctx, uops, uops_count, param_ids, tmp_dt, 128);
    for (int i=0;i<param_count;i++){ param_dtypes[i]=tmp_dt[i]; param_is_image[i]=dtypes_is_image(param_dtypes[i]); }
    cstyle_ctx_free(&ctx);
  }
  // OpenCL image mutability detection: mark write_only for images used in STORE
  bool image_written_by_id[128]; memset(image_written_by_id, 0, sizeof(image_written_by_id));
  if (is_opencl(self)) {
    CStyleCtx wctx; cstyle_ctx_init(&wctx); (void)wctx; // separate ctx for writes
    // reuse earlier collected ids in param_ids
    // mark writes in a local ctx mirroring ids
    for (int i=0;i<param_count && i<256;i++){ wctx.buf_ids[i]=param_ids[i]; wctx.buf_dtypes[i]=param_dtypes[i]; wctx.buf_writable[i]=0; wctx.buf_count=param_count; }
    cstyle_ctx_mark_writes(&wctx, uops, uops_count);
    for (int j=0;j<param_count;j++){ image_written_by_id[j] = cstyle_ctx_buf_writable(&wctx, param_ids[j]) ? true : false; }
    cstyle_ctx_free(&wctx);
  }
  char fnhead[1024];
  if (self->kernel_typedef && self->kernel_typedef[0]) {
    snprintf(fnhead, sizeof(fnhead), "%s kernel_main(", self->kernel_typedef);
  } else if (is_cuda(self) || is_hip_or_amd(self)) {
    strcpy(fnhead, "extern \"C\" __global__ void kernel_main(");
  } else if (is_metal(self)) {
    strcpy(fnhead, "kernel void kernel_main(");
  } else if (is_opencl(self)) {
    strcpy(fnhead, "__kernel void kernel_main(");
  } else {
    strcpy(fnhead, "void kernel_main(");
  }
  for (int i=0;i<param_count;i++){ char seg[256];
    if (is_opencl(self) && param_is_image[i]) {
      const char* q = image_written_by_id[i]?"write_only":"read_only";
      snprintf(seg,sizeof(seg),"%s image2d_t data%d%s", q, param_ids[i], (i<param_count-1)?", ":"");
    } else {
      const char* pref = self->buffer_prefix ? self->buffer_prefix : "";
      const char* suf = self->buffer_suffix ? self->buffer_suffix : "";
      snprintf(seg,sizeof(seg),"%s%s*%s data%d%s", pref, c_type_for_b(self, param_dtypes[i]), suf, param_ids[i], (i<param_count-1)?", ":"");
    }
    strcat(fnhead, seg);
  }
  if (self->extra_arg_count > 0 && self->extra_args) {
    if (param_count > 0) strcat(fnhead, ", ");
    for (int i=0;i<self->extra_arg_count;i++){
      strcat(fnhead, self->extra_args[i]);
      if (i < self->extra_arg_count-1) strcat(fnhead, ", ");
    }
  }
  strcat(fnhead, ") {\n");
  const char* end = "}\n";
  char* src = tg_sb_new_with(header, fnhead);
  name_map_t map={0};
  int indent = 1;
  #define INDENT() for(int _i=0;_i<indent;_i++) src = tg_sb_append_owned(src, "  ")
  // OpenCL sampler helper when images are present
  bool any_image=false; for(int ii=0; ii<param_count; ii++){ if (param_is_image[ii]) { any_image=true; break; } }
  if (is_opencl(self) && any_image) { INDENT(); src = tg_sb_append_owned(src, "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"); }
  for (int i=0;i<uops_count;i++){
    UOp* u = uops[i]; if (!u || u->op==OPS_SINK || u->op==OPS_NOOP) continue;
    if (u->op==OPS_DEFINE_REG){ const PtrDType* pd=(const PtrDType*)&u->dtype; char line[256]; INDENT();
      snprintf(line,sizeof(line),"%s reg_%d[%d];\n", c_type_for_b(self, pd->base_dtype), 0, pd->size); src = tg_sb_append_owned(src, line); continue; }
    if (u->op==OPS_CONST){ char* val=render_const(self, u); const char* nm=ssa_add(&map, u); char line[256]; INDENT();
      snprintf(line,sizeof(line),"%s %s = %s;\n", c_type_for_b(self, &u->dtype), nm, val); src=tg_sb_append_owned(src,line); free(val); continue; }
    if (u->op==OPS_PRECAST && u->src_count>=1){
      const char* srcn = ssa_get(&map, u->src[0]);
      if (!srcn) srcn = ssa_add(&map, u->src[0]);
      (void)ssa_alias(&map, u, srcn);
      continue;
    }
    if (u->op==OPS_RANGE && u->src_count>=1){ int ax = (u->arg.type==ARG_INT)? u->arg.int_data.i : 0; char ridx[32]; snprintf(ridx,sizeof(ridx),"ridx%d",ax);
      char bound[32]; if (u->src[0]->arg.type==ARG_CONST) snprintf(bound,sizeof(bound),"%d", (int)u->src[0]->arg.const_data.const_value); else strcpy(bound,"1");
      char line[256]; INDENT(); snprintf(line,sizeof(line),"for (int %s = 0; %s < %s; %s++) {\n", ridx, ridx, bound, ridx); src = tg_sb_append_owned(src, line); indent++; continue; }
    if (u->op==OPS_ENDRANGE){ indent--; INDENT(); src = tg_sb_append_owned(src, "}\n"); continue; }
    if (u->op==OPS_IF && u->src_count>=1){
      // render condition if it's a cmp or non-zero expr fallback
      char* cond=NULL; UOp* c=u->src[0];
      if (c->op==OPS_CMPLT||c->op==OPS_CMPEQ||c->op==OPS_CMPNE){
        const char* an = ssa_get(&map, c->src[0]); const char* bn = ssa_get(&map, c->src[1]);
        char* a = an?strdup(an): (c->src[0]->op==OPS_CONST?render_const(self, c->src[0]):strdup("0"));
        char* b = bn?strdup(bn): (c->src[1]->op==OPS_CONST?render_const(self, c->src[1]):strdup("0"));
        cond = render_cmp(c, a, b); free(a); free(b);
      } else {
        const char* cn = ssa_get(&map, c); cond = strdup(cn?cn:"1");
      }
      INDENT(); char line[512]; snprintf(line,sizeof(line),"if %s {\n", cond); src = tg_sb_append_owned(src, line); free(cond); indent++; continue;
    }
    if (u->op==OPS_ENDIF){ indent--; INDENT(); src = tg_sb_append_owned(src, "}\n"); continue; }
    if (u->op==OPS_SPECIAL){
      const char* nm = ssa_add(&map, u);
      int axis = 0; char kind = 'i';
      if (u->tag){ const char* tg = (const char*)u->tag; size_t L=strlen(tg); if (L){ char last=tg[L-1]; if (last>='0'&&last<='9') axis=last-'0'; }
        if (strncmp((const char*)u->tag, "gidx", 4)==0) kind='g';
        else if (strncmp((const char*)u->tag, "lidx", 4)==0) kind='l';
        else if (strncmp((const char*)u->tag, "idx", 3)==0) kind='i'; }
      char expr[128];
      if (is_opencl(self)) snprintf(expr,sizeof(expr), (kind=='g')?"get_group_id(%d)": (kind=='l')?"get_local_id(%d)":"get_global_id(%d)", axis);
      else if (is_cuda(self) || is_hip_or_amd(self)) {
        const char axisc = (axis==0)?'x':((axis==1)?'y':'z');
        if (kind=='g') snprintf(expr,sizeof(expr),"blockIdx.%c", axisc);
        else if (kind=='l') snprintf(expr,sizeof(expr),"threadIdx.%c", axisc);
        else snprintf(expr,sizeof(expr),"(blockIdx.%c*blockDim.%c + threadIdx.%c)", axisc, axisc, axisc);
      } else if (is_metal(self)) {
        const char axisc = (axis==0)?'x':((axis==1)?'y':'z');
        if (kind=='g') snprintf(expr,sizeof(expr),"gid.%c", axisc);
        else if (kind=='l') snprintf(expr,sizeof(expr),"lid.%c", axisc);
        else snprintf(expr,sizeof(expr),"gid.%c", axisc);
      } else snprintf(expr,sizeof(expr),"0");
      char line[256]; INDENT(); snprintf(line,sizeof(line),"%s %s = %s;\n", c_type_for_b(self, &u->dtype), nm, expr); src = tg_sb_append_owned(src, line); continue;
    }
    // explicit comparisons first (so they don't get treated as generic ALU)
    if ((u->op==OPS_CMPLT || u->op==OPS_CMPEQ || u->op==OPS_CMPNE) && u->src_count==2){
      const char* an = ssa_get(&map, u->src[0]); const char* bn = ssa_get(&map, u->src[1]);
      char* a = an?strdup(an):(u->src[0]->op==OPS_CONST?render_const(self, u->src[0]):strdup("0"));
      char* b = bn?strdup(bn):(u->src[1]->op==OPS_CONST?render_const(self, u->src[1]):strdup("0"));
      char* expr = render_cmp(u, a, b);
      const char* nm = ssa_add(&map, u); char line[256]; INDENT();
      snprintf(line,sizeof(line),"%s %s = %s;\n", c_type_for_b(self, &u->dtype), nm, expr);
      src = tg_sb_append_owned(src, line); free(a); free(b); free(expr); continue;
    }
    if (group_op.is_alu[u->op] && u->src_count==2){
      const char* an = ssa_get(&map, u->src[0]); const char* bn = ssa_get(&map, u->src[1]);
      char* a = an?strdup(an):(u->src[0]->op==OPS_CONST?render_const(self, u->src[0]):strdup("0"));
      char* b = bn?strdup(bn):(u->src[1]->op==OPS_CONST?render_const(self, u->src[1]):strdup("0"));
      char* expr=render_alu(u,a,b); const char* nm=ssa_add(&map, u);
      char line[512]; INDENT(); snprintf(line,sizeof(line),"%s %s = %s;\n", c_type_for_b(self, &u->dtype), nm, expr); src=tg_sb_append_owned(src,line); free(a); free(b); free(expr); continue; }
    // unary builtins
    if (u->op==OPS_SQRT && u->src_count==1){
      const char* sn = ssa_get(&map, u->src[0]); char* sv = sn?strdup(sn):(u->src[0]->op==OPS_CONST?render_const(self, u->src[0]):strdup("0"));
      const char* nm = ssa_add(&map, u); char line[256]; INDENT();
      if (is_hip_or_amd(self)) {
        const char* suf = dtype_eq(&u->dtype, &dtypes.float64) ? "64" : (dtype_eq(&u->dtype, &dtypes.float16) ? "16" : "32");
        snprintf(line,sizeof(line),"%s %s = __ocml_sqrt_f%s(%s);\n", c_type_for_b(self, &u->dtype), nm, suf, sv);
      } else if (is_opencl(self) || is_cuda(self)) {
        snprintf(line,sizeof(line),"%s %s = sqrt(%s);\n", c_type_for_b(self, &u->dtype), nm, sv);
      } else {
        if (dtype_eq(&u->dtype, &dtypes.float32)) snprintf(line,sizeof(line),"%s %s = __builtin_sqrtf(%s);\n", c_type_for_b(self, &u->dtype), nm, sv);
        else snprintf(line,sizeof(line),"%s %s = __builtin_sqrt(%s);\n", c_type_for_b(self, &u->dtype), nm, sv);
      }
      src = tg_sb_append_owned(src, line); free(sv); continue;
    }
    if ((u->op==OPS_EXP2 || u->op==OPS_LOG2 || u->op==OPS_SIN) && u->src_count==1){
      const char* sn = ssa_get(&map, u->src[0]); char* sv = sn?strdup(sn):(u->src[0]->op==OPS_CONST?render_const(self, u->src[0]):strdup("0"));
      const char* nm = ssa_add(&map, u); char line[256]; INDENT();
      const char* fn = (u->op==OPS_EXP2) ? "exp2" : (u->op==OPS_LOG2 ? "log2" : "sin");
      if (is_hip_or_amd(self)) {
        const char* intr = (u->op==OPS_EXP2) ? "__ocml_exp2_f" : (u->op==OPS_LOG2 ? "__ocml_log2_f" : "__ocml_sin_f");
        const char* suf = dtype_eq(&u->dtype, &dtypes.float64) ? "64" : (dtype_eq(&u->dtype, &dtypes.float16) ? "16" : "32");
        snprintf(line,sizeof(line),"%s %s = %s%s(%s);\n", c_type_for_b(self, &u->dtype), nm, intr, suf, sv);
      } else if (is_metal(self) && u->op==OPS_SIN) {
        snprintf(line,sizeof(line),"%s %s = precise::sin(%s);\n", c_type_for_b(self, &u->dtype), nm, sv);
      } else {
        snprintf(line,sizeof(line),"%s %s = %s(%s);\n", c_type_for_b(self, &u->dtype), nm, fn, sv);
      }
      src = tg_sb_append_owned(src, line); free(sv); continue;
    }
    if (u->op==OPS_INDEX && u->src_count>=2){
      // index(buffer, idx) → name representing pointer
      // try to find dataN param from DEFINE_GLOBAL id in src[0]
      int gid=-1; const DType* gdt=&dtypes.float32; size_t tn=0; UOp** bt = uop_toposort(u->src[0], &tn);
      if (bt){ for(size_t k=0;k<tn;k++) if (bt[k]->op==OPS_DEFINE_GLOBAL){ gid=(bt[k]->arg.type==ARG_INT)? bt[k]->arg.int_data.i : 0; gdt=&bt[k]->dtype; break; } free(bt);} 
      const char* in = ssa_get(&map, u->src[1]); char* idxs = in?strdup(in):(u->src[1]->op==OPS_CONST?render_const(self, u->src[1]):strdup("0"));
      const char* nm = ssa_add(&map, u); char line[256]; INDENT();
      if (is_opencl(self) && dtypes_is_image(gdt)) {
        if (gid>=0) snprintf(line,sizeof(line),"image2d_t %s = data%d;\\n", nm, gid);
        else snprintf(line,sizeof(line),"image2d_t %s = %s;\\n", nm, idxs);
      } else {
        const char* pref = is_opencl(self)?"__global ": (is_metal(self)?"device ":"");
        if (gid>=0) snprintf(line,sizeof(line),"%s%s* restrict %s = data%d + %s;\\n", pref, c_type_for_b(self, gdt), nm, gid, idxs);
        else snprintf(line,sizeof(line),"%s%s* restrict %s = %s;\\n", pref, c_type_for_b(self, gdt), nm, idxs);
      }
      src = tg_sb_append_owned(src, line); free(idxs); continue;
    }
    if (u->op==OPS_GEP && u->src_count>=1){
      const char* base = ssa_get(&map, u->src[0]); if (!base) base = "v";
      int idx = 0; if (u->arg.type==ARG_REDUCE && u->arg.reduce_data.axes_count>0) idx = u->arg.reduce_data.axes[0];
      int threshold = gep_threshold_for(self);
      int count = u->src[0] ? u->src[0]->dtype.count : 1;
      char access[64];
      if (count > threshold && threshold >= 0) snprintf(access, sizeof(access), "%s[%d]", base, idx);
      else if (idx >= 0 && idx < (int)strlen(gep_components)) snprintf(access, sizeof(access), "%s.%c", base, gep_components[idx]);
      else snprintf(access, sizeof(access), "%s[%d]", base, idx);
      const char* nm = ssa_add(&map, u); char line[256]; INDENT();
      snprintf(line,sizeof(line),"%s %s = %s;\n", c_type_for_b(self, &u->dtype), nm, access);
      src = tg_sb_append_owned(src, line); continue;
    }
    if (u->op==OPS_DEFINE_LOCAL){
      // local/workgroup memory declaration
      int sz = (u->arg.type==ARG_INT) ? u->arg.int_data.i : 0;
      const char* nm = ssa_add(&map, u);
      char line[256]; INDENT();
      if (is_opencl(self)) snprintf(line,sizeof(line),"__local %s %s[%d];\n", c_type_for_b(self, &u->dtype), nm, sz);
      else if (is_metal(self)) snprintf(line,sizeof(line),"threadgroup __attribute__((aligned(16))) %s %s[%d];\n", c_type_for_b(self, &u->dtype), nm, sz);
      else if (is_cuda(self) || is_hip_or_amd(self)) snprintf(line,sizeof(line),"__shared__ %s %s[%d];\n", c_type_for_b(self, &u->dtype), nm, sz);
      else snprintf(line,sizeof(line),"%s %s[%d];\n", c_type_for_b(self, &u->dtype), nm, sz);
      src = tg_sb_append_owned(src, line); continue;
    }
    if (u->op==OPS_BARRIER){
      char line[128]; INDENT();
      if (is_opencl(self)) strcpy(line, "barrier(CLK_LOCAL_MEM_FENCE);\n");
      else if (is_metal(self)) strcpy(line, "threadgroup_barrier(mem_flags::mem_threadgroup);\n");
      else if (is_cuda(self) || is_hip_or_amd(self)) strcpy(line, "__syncthreads();\n");
      else strcpy(line, ";/* barrier */\n");
      src = tg_sb_append_owned(src, line); continue;
    }
    if (u->op==OPS_MULACC && u->src_count==3){
      const char* aN = ssa_get(&map, u->src[0]); const char* bN = ssa_get(&map, u->src[1]); const char* cN = ssa_get(&map, u->src[2]);
      char* a = aN?strdup(aN):(u->src[0]->op==OPS_CONST?render_const(self, u->src[0]):strdup("0"));
      char* b = bN?strdup(bN):(u->src[1]->op==OPS_CONST?render_const(self, u->src[1]):strdup("0"));
      char* c = cN?strdup(cN):(u->src[2]->op==OPS_CONST?render_const(self, u->src[2]):strdup("0"));
      const char* nm = ssa_add(&map, u);
      char* expr = tg_sb_append_ownedf(NULL, "((%s*%s)+(%s))", a, b, c);
      INDENT();
      src = tg_sb_append_ownedf(src, "%s %s = %s;\n", c_type_for(&u->dtype), nm, expr ? expr : "0");
      if (expr) free(expr);
      free(a); free(b); free(c); continue;
    }
    if (u->op==OPS_LOAD && u->src_count>=1){
      const char* ptr = ssa_get(&map, u->src[0]); if (!ptr) ptr = "ptr";
      // optional gate: third src of INDEX
      UOp* idxu = u->src[0]; UOp* gateu = NULL;
      if (idxu && idxu->op==OPS_INDEX && idxu->src_count>=3) gateu = idxu->src[2];
      char* alt = NULL; if (u->src_count>=2){ const char* vn = ssa_get(&map, u->src[1]); alt = vn?strdup(vn):(u->src[1]->op==OPS_CONST?render_const(self, u->src[1]):strdup("0")); }
      char* cond = NULL; if (gateu){ const char* cn = ssa_get(&map, gateu); cond = cn?strdup(cn):(gateu->op==OPS_CONST?render_const(self, gateu):strdup("0")); }
      const char* nm = ssa_add(&map, u); char line[768]; INDENT();
      // OpenCL image read path (read_imagef)
      if (is_opencl(self) && idxu && idxu->op==OPS_INDEX && idxu->src_count>=2){
        size_t tn=0; UOp** bt = uop_toposort(idxu->src[0], &tn);
        int gid=-1; const DType* gdt=&dtypes.float32; if (bt){ for(size_t k=0;k<tn;k++) if (bt[k]->op==OPS_DEFINE_GLOBAL){ gid=(bt[k]->arg.type==ARG_INT)? bt[k]->arg.int_data.i:0; gdt=&bt[k]->dtype; break; } free(bt);}        
        if (gid>=0 && dtypes_is_image(gdt)){
          const char* idxn = ssa_get(&map, idxu->src[1]); char* idxv = idxn?strdup(idxn):(idxu->src[1]->op==OPS_CONST?render_const(self, idxu->src[1]):strdup("0"));
          const char* vtype = (u->dtype.count>1) ? c_vec_type_for_b(self, &u->dtype) : c_type_for_b(self, &u->dtype);
          if (cond && alt) snprintf(line,sizeof(line),"%s %s = ((%s)?read_imagef(data%d, smp, %s):%s);\n", vtype, nm, cond, gid, idxv, alt);
          else snprintf(line,sizeof(line),"%s %s = read_imagef(data%d, smp, %s);\n", vtype, nm, gid, idxv);
          src = tg_sb_append_owned(src, line);
          free(idxv);
          if (alt) free(alt);
          if (cond) free(cond);
          continue;
        }
      }
      const char* vtype = (u->dtype.count>1) ? c_vec_type_for_b(self, &u->dtype) : c_type_for_b(self, &u->dtype);
      if (cond && alt) snprintf(line,sizeof(line),"%s %s = ((%s)?*%s:%s);\n", vtype, nm, cond, ptr, alt);
      else snprintf(line,sizeof(line),"%s %s = *%s;\n", vtype, nm, ptr);
      src = tg_sb_append_owned(src, line); if (alt) free(alt); if (cond) free(cond); continue;
    }
    if (u->op==OPS_CAST && u->src_count>=1){
      const char* srcn = ssa_get(&map, u->src[0]); char* srcv = srcn?strdup(srcn):(u->src[0]->op==OPS_CONST?render_const(self, u->src[0]):strdup("0"));
      const char* nm = ssa_add(&map, u);
      char line[512]; INDENT();
      if (u->dtype.count > 1 && !is_opencl(self) && !is_metal(self)) {
        // vector cast
        snprintf(line,sizeof(line),"%s %s = __builtin_convertvector(%s, %s);\n", c_vec_type_for_b(self, &u->dtype), nm, srcv, c_vec_type_for_b(self, &u->dtype));
      } else {
        snprintf(line,sizeof(line),"%s %s = (%s)(%s);\n", c_type_for_b(self, &u->dtype), nm, c_type_for_b(self, &u->dtype), srcv);
      }
      src = tg_sb_append_owned(src, line); free(srcv); continue;
    }
    if (u->op==OPS_BITCAST && u->src_count>=1){
      const char* srcn = ssa_get(&map, u->src[0]); char* srcv = srcn?strdup(srcn):(u->src[0]->op==OPS_CONST?render_const(self, u->src[0]):strdup("0"));
      const char* nm = ssa_add(&map, u);
      char line[512]; INDENT();
      if (is_opencl(self)) snprintf(line,sizeof(line),"%s %s = as_%s(%s);\n", c_type_for_b(self, &u->dtype), nm, c_type_for_b(self, &u->dtype), srcv);
      else if (is_metal(self)) snprintf(line,sizeof(line),"%s %s = as_type<%s>(%s);\n", c_type_for_b(self, &u->dtype), nm, c_type_for_b(self, &u->dtype), srcv);
      else snprintf(line,sizeof(line),"%s %s = *((%s*)&%s);\n", c_type_for_b(self, &u->dtype), nm, c_type_for_b(self, &u->dtype), srcv);
      src = tg_sb_append_owned(src, line); free(srcv); continue;
    }
    if (u->op==OPS_STORE && u->src_count>=2){
      const char* ptr = ssa_get(&map, u->src[0]); if (!ptr) ptr = "ptr";
      const char* valn = ssa_get(&map, u->src[1]); char* val = valn?strdup(valn):(u->src[1]->op==OPS_CONST?render_const(self, u->src[1]):strdup("0"));
      char* cond = NULL; if (u->src_count>=3){ const char* cn = ssa_get(&map, u->src[2]); cond = cn?strdup(cn):(u->src[2]->op==OPS_CONST?render_const(self, u->src[2]):strdup("0")); }
      char line[768];
      // OpenCL image write path
      bool did_img=false; if (is_opencl(self) && u->src[0]->op==OPS_INDEX){
        UOp* idxu2 = u->src[0];
        size_t tn=0; UOp** bt = uop_toposort(idxu2->src[0], &tn);
        int gid=-1; const DType* gdt=&dtypes.float32; if (bt){ for(size_t k=0;k<tn;k++) if (bt[k]->op==OPS_DEFINE_GLOBAL){ gid=(bt[k]->arg.type==ARG_INT)? bt[k]->arg.int_data.i:0; gdt=&bt[k]->dtype; break; } free(bt);}        
        if (gid>=0 && dtypes_is_image(gdt)){
          const char* idxn = ssa_get(&map, idxu2->src[1]); char* idxv = idxn?strdup(idxn):(idxu2->src[1]->op==OPS_CONST?render_const(self, idxu2->src[1]):strdup("0"));
          if (cond){ INDENT(); snprintf(line,sizeof(line),"if (%s) {\n", cond); src = tg_sb_append_owned(src, line); indent++; }
          INDENT(); snprintf(line,sizeof(line),"write_imagef(data%d, %s, %s);\n", gid, idxv, val); src = tg_sb_append_owned(src, line);
          if (cond){ indent--; INDENT(); src = tg_sb_append_owned(src, "}\n"); }
          free(idxv); did_img=true;
        }
      }
      if (!did_img){
        if (cond){ INDENT(); snprintf(line,sizeof(line),"if (%s) {\n", cond); src = tg_sb_append_owned(src, line); indent++; }
        INDENT(); snprintf(line,sizeof(line),"*%s = %s;\n", ptr, val); src = tg_sb_append_owned(src, line);
        if (cond){ indent--; INDENT(); src = tg_sb_append_owned(src, "}\n"); }
      }
      if (cond){ free(cond); }
      free(val); continue;
    }
    // VECTORIZE literal to C-style vector (backend-specific constructors)
    if (u->op==OPS_VECTORIZE && u->src_count>0){
      const char* nm = ssa_add(&map, u);
      char line[1024]; INDENT();
      char elems[768]; elems[0]='\0';
      for (size_t k=0;k<u->src_count;k++){
        const char* vn = ssa_get(&map, u->src[k]); char* vv = vn?strdup(vn):(u->src[k]->op==OPS_CONST?render_const(self, u->src[k]):strdup("0"));
        if (k>0) strcat(elems, ", ");
        strcat(elems, vv); free(vv);
      }
      char* ctor = render_vec_literal(self, &u->dtype, elems);
      snprintf(line, sizeof(line), "%s %s = %s;\n", c_vec_type_for_b(self, &u->dtype), nm, ctor);
      free(ctor);
      src = tg_sb_append_owned(src, line); continue;
    }
    if (u->op==OPS_WHERE && u->src_count==3){
      const char* cn = ssa_get(&map, u->src[0]); char* c = cn?strdup(cn):(u->src[0]->op==OPS_CONST?render_const(self, u->src[0]):strdup("0"));
      const char* tn = ssa_get(&map, u->src[1]); char* t = tn?strdup(tn):(u->src[1]->op==OPS_CONST?render_const(self, u->src[1]):strdup("0"));
      const char* fn = ssa_get(&map, u->src[2]); char* f = fn?strdup(fn):(u->src[2]->op==OPS_CONST?render_const(self, u->src[2]):strdup("0"));
      const char* nm = ssa_add(&map, u); char line[512]; INDENT();
      snprintf(line,sizeof(line),"%s %s = ((%s)?(%s):(%s));\n", c_type_for_b(self, &u->dtype), nm, c, t, f);
      src = tg_sb_append_owned(src, line); free(c); free(t); free(f); continue;
    }
    if ((u->op==OPS_CUSTOM || u->op==OPS_CUSTOMI) && u->arg.type==ARG_STRING){
      const char* fmt = u->arg.str.s ? u->arg.str.s : "";
      char* out = strdup("");
      int si = 0; size_t L = strlen(fmt);
      for (size_t p=0; p<L;){
        if (fmt[p]=='{' && p+1<L && fmt[p+1]=='}'){
          char* rep = strdup("0");
          if (si < (int)u->src_count){
            if (u->src[si]->op == OPS_CONST) { free(rep); rep = render_const(self, u->src[si]); }
            else { const char* sn = ssa_get(&map, u->src[si]); free(rep); rep = sn?strdup(sn):strdup("0"); }
          }
          out = tg_sb_append_owned(out, rep); free(rep); si++; p+=2; continue;
        }
        char tmp[2] = {fmt[p], 0}; out = tg_sb_append_owned(out, tmp); p++;
      }
      const char* nm = ssa_add(&map, u); char line[1024]; INDENT();
      snprintf(line, sizeof(line), "%s %s = %s;\n", c_type_for_b(self, &u->dtype), nm, out);
      src = tg_sb_append_owned(src, line); free(out); continue;
    }
  }
  src = tg_sb_append_owned(src, end);
  return src; }

Renderer* renderer_cstyle_clang(void) {
  CStyleRenderer* self = cstyle_renderer_alloc();
  if (!self) return NULL;
  self->flavor = CSTYLE_FLAVOR_CLANG;
  self->base.device = "CPU";
  self->base.suffix = "";
  self->base.has_local = false;
  self->kernel_typedef = "void";
  self->buffer_prefix = "";
  self->buffer_suffix = " restrict";
  self->smem_align = "";
  self->smem_prefix = "";
  self->arg_int_prefix = "const int";
  self->barrier = "";
  self->float4 = "(float4)";
  self->float4_style_open = "{";
  self->float4_style_close = "}";
  self->gep_arr_threshold = 0;
  self->infinity = "__builtin_inff()";
  self->nan = "__builtin_nanf(\"\")";
  return &self->base;
}

Renderer* renderer_cstyle_opencl(void) {
  CStyleRenderer* self = cstyle_renderer_alloc();
  if (!self) return NULL;
  self->flavor = CSTYLE_FLAVOR_OPENCL;
  self->base.device = "GPU";
  self->base.suffix = "OPENCL";
  self->kernel_typedef = "__kernel void";
  self->buffer_prefix = "__global ";
  self->buffer_suffix = "";
  self->smem_align = "__attribute__ ((aligned (16))) ";
  self->smem_prefix = "__local ";
  self->arg_int_prefix = "const int";
  self->barrier = "barrier(CLK_LOCAL_MEM_FENCE);";
  self->float4 = "(float4)";
  self->infinity = "INFINITY";
  self->nan = "NAN";
  return &self->base;
}

Renderer* renderer_cstyle_cuda(const char* arch) {
  CStyleRenderer* self = cstyle_renderer_alloc();
  if (!self) return NULL;
  self->flavor = CSTYLE_FLAVOR_CUDA;
  self->base.device = "CUDA";
  self->base.suffix = arch ? arch : "";
  self->base.has_local = true;
  self->base.has_shared = true;
  self->base.shared_max = 49152;
  self->kernel_typedef = "extern \"C\" __global__ void";
  self->float4 = "(float4)";
  self->infinity = "INFINITY";
  self->nan = "NAN";
  return &self->base;
}

Renderer* renderer_cstyle_nv(const char* arch) {
  Renderer* base = renderer_cstyle_cuda(arch);
  if (base) {
    CStyleRenderer* self = cstyle_from_renderer(base);
    self->flavor = CSTYLE_FLAVOR_NV;
    self->base.device = "NV";
  }
  return base;
}

Renderer* renderer_cstyle_amd(const char* arch) {
  CStyleRenderer* self = cstyle_renderer_alloc();
  if (!self) return NULL;
  self->flavor = CSTYLE_FLAVOR_AMD;
  self->base.device = "AMD";
  self->base.suffix = arch ? arch : "";
  self->base.has_local = true;
  self->base.has_shared = true;
  self->base.shared_max = 65536;
  self->kernel_typedef = "void";
  self->float4 = "(float4)";
  self->infinity = "INFINITY";
  self->nan = "NAN";
  return &self->base;
}

Renderer* renderer_cstyle_hip(const char* arch) {
  CStyleRenderer* self = cstyle_renderer_alloc();
  if (!self) return NULL;
  self->flavor = CSTYLE_FLAVOR_HIP;
  self->base.device = "HIP";
  self->base.suffix = arch ? arch : "";
  self->base.has_local = true;
  self->base.has_shared = true;
  self->base.shared_max = 65536;
  self->kernel_typedef = "extern \"C\" __global__ void";
  self->float4 = "(float4)";
  self->infinity = "INFINITY";
  self->nan = "NAN";
  return &self->base;
}

Renderer* renderer_cstyle_metal(void) {
  CStyleRenderer* self = cstyle_renderer_alloc();
  if (!self) return NULL;
  self->flavor = CSTYLE_FLAVOR_METAL;
  self->base.device = "METAL";
  self->base.suffix = "";
  self->base.has_local = true;
  self->base.has_shared = true;
  self->base.shared_max = 32*1024;
  self->kernel_typedef = "kernel void";
  self->buffer_prefix = "device ";
  self->buffer_suffix = "";
  self->smem_prefix = "threadgroup __attribute__((aligned(16))) ";
  self->arg_int_prefix = "constant int&";
  self->barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);";
  static const char* metal_extra_args[] = {
    "uint3 gid [[threadgroup_position_in_grid]]",
    "uint3 lid [[thread_position_in_threadgroup]]"
  };
  self->extra_args = metal_extra_args;
  self->extra_arg_count = 2;
  self->float4 = "float4";
  self->float4_style_open = "(";
  self->float4_style_close = ")";
  self->infinity = "INFINITY";
  self->nan = "NAN";
  return &self->base;
}

Renderer* renderer_cstyle_qcom(void) {
  CStyleRenderer* self = cstyle_renderer_alloc();
  if (!self) return NULL;
  self->flavor = CSTYLE_FLAVOR_QCOM;
  self->base.device = "QCOM";
  self->base.suffix = "OPENCL";
  self->kernel_typedef = "__kernel void";
  self->buffer_prefix = "__global ";
  self->buffer_suffix = "";
  self->smem_align = "__attribute__ ((aligned (16))) ";
  self->smem_prefix = "__local ";
  self->barrier = "barrier(CLK_LOCAL_MEM_FENCE);";
  self->float4 = "(float4)";
  self->infinity = "INFINITY";
  self->nan = "NAN";
  return &self->base;
}

Renderer* renderer_cstyle_intel(void) {
  CStyleRenderer* self = cstyle_renderer_alloc();
  if (!self) return NULL;
  self->flavor = CSTYLE_FLAVOR_INTEL;
  self->base.device = "GPU";
  self->base.suffix = "INTEL";
  self->kernel_typedef = "__attribute__((intel_reqd_sub_group_size(8)))\n__kernel void";
  self->buffer_prefix = "__global ";
  self->buffer_suffix = "";
  self->smem_align = "__attribute__ ((aligned (16))) ";
  self->smem_prefix = "__local ";
  self->barrier = "barrier(CLK_LOCAL_MEM_FENCE);";
  self->float4 = "(float4)";
  self->infinity = "INFINITY";
  self->nan = "NAN";
  return &self->base;
}
