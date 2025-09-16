#include "renderer/wgsl.h"
#include "uop/uop.h"
#include "dtype/dtype.h"
#include "renderer/cstyle_base.h"
#include "helpers/string_builder.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "shape/shapetracker.h"

static char* strip_parens_copy(const char* s){ if (!s) return NULL; size_t n=strlen(s); if (n<2) return strdup(s); const char* p=s; const char* q=s+n-1; while (n>=2 && *p=='(' && *q==')') { int bal=0; int ok=1; for (size_t i=0;i<n;i++){ char c=s[i]; if (c=='(') bal++; else if (c==')'){ bal--; if (bal==0 && i!=n-1){ ok=0; break; } } } if (!ok) break; p++; q--; n-=2; } char* out=(char*)malloc(n+1); memcpy(out,p,n); out[n]='\0'; return out; }

// ---- Pattern matcher callbacks (file-scope for portability) ----
static void* wgsl_cb_boolop_cmpxor(void* ctx, void* node, const char** names, UOp** values, size_t nbinds) {
  (void)ctx; (void)node; (void)names;
  UOp *a=NULL, *b=NULL, *c=NULL; for(size_t i=0;i<nbinds;i++){ if (!names[i]) continue; if (strcmp(names[i],"a")==0) a=values[i]; else if (strcmp(names[i],"b")==0) b=values[i]; else if (strcmp(names[i],"c")==0) c=values[i]; }
  if (!a||!b||!c) return NULL;
  UOp* ac = uop_cast(a, dtypes.int32);
  UOp* bc = uop_cast(b, dtypes.int32);
  if (c->op == OPS_CMPLT) return uop_lt(ac, bc);
  return uop_cast(uop_xor(ac, bc), dtypes.bool_);
}

static void* wgsl_cb_shl_fix(void* ctx, void* node, const char** names, UOp** values, size_t nbinds) {
  (void)ctx; (void)node; (void)names;
  UOp *a=NULL, *b=NULL; for(size_t i=0;i<nbinds;i++){ if (!names[i]) continue; if (strcmp(names[i],"a")==0) a=values[i]; else if (strcmp(names[i],"b")==0) b=values[i]; }
  if (!a||!b) return NULL;
  if (dtype_eq(&b->dtype, &dtypes.uint32)) return NULL;
  UOp* ab = uop_bitcast(a, dtypes.uint32);
  UOp* bc = uop_cast(b, dtypes.uint32);
  UOp* r = uop_shl(ab, bc);
  return uop_bitcast(r, a->dtype);
}

static void* wgsl_cb_shr_fix(void* ctx, void* node, const char** names, UOp** values, size_t nbinds) {
  (void)ctx; (void)node; (void)names;
  UOp *x=NULL, *y=NULL; for(size_t i=0;i<nbinds;i++){ if (!names[i]) continue; if (strcmp(names[i],"x")==0) x=values[i]; else if (strcmp(names[i],"y")==0) y=values[i]; }
  if (!x||!y) return NULL;
  if (dtype_eq(&y->dtype, &dtypes.uint)) return NULL;
  UOp* yc = uop_cast(y, dtypes.uint);
  return uop_shr(x, yc);
}

// Scalar WGSL type for a dtype's base scalar
static const char* wgsl_scalar_ty(const DType* s){
  if (dtype_eq(s,&dtypes.float32)) return "f32";
  if (dtype_eq(s,&dtypes.float16) || dtype_eq(s,&dtypes.half)) return "f16";
  if (dtype_eq(s,&dtypes.int32) || dtype_eq(s,&dtypes.int16) || dtype_eq(s,&dtypes.int8) || dtype_eq(s,&dtypes.short_) || dtype_eq(s,&dtypes.char_)) return "i32";
  if (dtype_eq(s,&dtypes.uint32) || dtype_eq(s,&dtypes.uint16) || dtype_eq(s,&dtypes.uint8) || dtype_eq(s,&dtypes.ushort) || dtype_eq(s,&dtypes.uchar)) return "u32";
  if (dtype_eq(s,&dtypes.bool_)) return "bool";
  // fallback
  return "f32";
}

// Full WGSL type including vectors: vecN<scalar>
static char* wgsl_dtype(const DType* dt){
  const DType* s = dt;
  if (dt->count>1 && dt->_scalar) s = dt->_scalar;
  const char* sc = wgsl_scalar_ty(s);
  if (dt->count>1){ size_t cap = strlen(sc) + 16; char* out=(char*)malloc(cap); snprintf(out,cap,"vec%d<%s>", dt->count, sc); return out; }
  return strdup(sc);
}

typedef struct { UOp** keys; char** names; int count; int cap; } name_map_t;
static const char* nm_get(name_map_t* m, UOp* u){ for(int i=0;i<m->count;i++) if(m->keys[i]==u) return m->names[i]; return NULL; }
static const char* nm_add(name_map_t* m, UOp* u, int* c){ char tmp[32]; snprintf(tmp,sizeof(tmp),"v%d", (*c)++); if (m->count+1>m->cap){ int nc=m->cap?m->cap*2:32; m->keys=(UOp**)realloc(m->keys,nc*sizeof(UOp*)); m->names=(char**)realloc(m->names,nc*sizeof(char*)); m->cap=nc; } m->keys[m->count]=u; m->names[m->count]=strdup(tmp); m->count++; return m->names[m->count-1]; }
static void nm_set(name_map_t* m, UOp* u, const char* name){
  for(int i=0;i<m->count;i++) if (m->keys[i]==u){ free(m->names[i]); m->names[i]=strdup(name); return; }
  // not found: add
  if (m->count+1>m->cap){ int nc=m->cap?m->cap*2:32; m->keys=(UOp**)realloc(m->keys,nc*sizeof(UOp*)); m->names=(char**)realloc(m->names,nc*sizeof(char*)); m->cap=nc; }
  m->keys[m->count]=u; m->names[m->count]=strdup(name); m->count++;
}
static char* wgsl_const(UOp* u){
  char buf[128];
  const DType* s = &u->dtype; if (u->dtype.count>1 && u->dtype._scalar) s = u->dtype._scalar;
  if (dtype_eq(s, &dtypes.bool_)) { snprintf(buf,sizeof(buf), "%s", u->arg.const_data.const_value ? "true" : "false"); return strdup(buf); }
  if (dtype_eq(s, &dtypes.uint32) || dtype_eq(s,&dtypes.uint16) || dtype_eq(s,&dtypes.ushort) || dtype_eq(s,&dtypes.uint8) || dtype_eq(s,&dtypes.uchar)) {
    long long iv = (long long)u->arg.const_data.const_value;
    if (iv < 0) { snprintf(buf,sizeof(buf), "bitcast<u32>(%lld)", iv); return strdup(buf); }
    else { snprintf(buf,sizeof(buf), "%lluu", iv); return strdup(buf); }
  }
  if (dtype_eq(s, &dtypes.int32)) { snprintf(buf,sizeof(buf), "%d", (int)u->arg.const_data.const_value); return strdup(buf); }
  if (dtype_eq(s, &dtypes.float16) || dtype_eq(s, &dtypes.half)) { snprintf(buf,sizeof(buf), "f16(%g)", (double)u->arg.const_data.const_value); return strdup(buf); }
  /* default f32 */ snprintf(buf,sizeof(buf), "%g", (double)u->arg.const_data.const_value); return strdup(buf);
}

static const DType* scalar_dt(const DType* dt){ return (dt->count>1 && dt->_scalar) ? dt->_scalar : dt; }
static int itemsize(const DType* dt){ const DType* s=scalar_dt(dt); return s->itemsize; }
static int is_packed_dt(const DType* dt){ const DType* s=scalar_dt(dt); return (s->itemsize < 4) && !dtype_eq(s, &dtypes.half); }
static char* str_or_const(name_map_t* map, UOp* u){
  const char* n = nm_get(map, u);
  if (n) return strdup(n);
  if (u->op==OPS_CONST) return wgsl_const(u);
  // fallback
  return strdup("0");
}

static char* wgsl_zero_expr(const DType* dt){
  const DType* s = scalar_dt(dt);
  if (dtype_eq(s, &dtypes.bool_)) return strdup("false");
  if (dtype_eq(s, &dtypes.uint32) || dtype_eq(s, &dtypes.uint16) || dtype_eq(s, &dtypes.uint8) || dtype_eq(s, &dtypes.ushort) || dtype_eq(s, &dtypes.uchar)) return strdup("0u");
  if (dtype_eq(s, &dtypes.float16) || dtype_eq(s, &dtypes.half)) return strdup("f16(0.0)");
  if (dtype_eq(s, &dtypes.float32)) return strdup("0.0");
  // default i32
  if (dt->count>1){
    char* sc = strdup(wgsl_scalar_ty(s));
    size_t cap = 32 + strlen(sc);
    char* out = (char*)malloc(cap);
    snprintf(out, cap, "vec%d<%s>(0)", dt->count, sc);
    free(sc);
    return out;
  }
  return strdup("0");
}

static char* wgsl_float_lit(const DType* dt, double v){
  const DType* s = scalar_dt(dt);
  char buf[64];
  if (dtype_eq(s, &dtypes.float16) || dtype_eq(s,&dtypes.half)) { snprintf(buf,sizeof(buf),"f16(%g)", v); return strdup(buf); }
  snprintf(buf,sizeof(buf),"%g", v); return strdup(buf);
}

static char* wgsl_render(Renderer* self, UOp** uops, int n){ (void)self;
  // Build and apply a small WGSL-specific pattern matcher on the fly
  // Patterns:
  // 1) (CMPLT/XOR) with a bool -> cast to i32 before op, cast result to bool for XOR
  // 2) SHL: cast rhs to u32 and bitcast lhs to u32, then bitcast back
  // 3) SHR: cast rhs to u32
  PatternMatcher* pm = NULL; do {
    PatternMatch entries[3]; int m=0;
    // Pattern 1: (CMPLT|XOR)(a<bool>, b) as c
    {
      UPat* a = upat_var_named("a", (const DType* const[]){ &dtypes.bool_ }, 1, false);
      UPat* b = upat_var_named("b", NULL, 0, false);
      UPat* srcs[2] = { a, b };
      UPat* p = upat_op(OPS_CMPLT, srcs, 2);
      Ops oplist[2] = { OPS_CMPLT, OPS_XOR };
      upat_set_op_list(p, oplist, 2);
      upat_set_name(p, "c");
      entries[m++] = (PatternMatch){ .pattern=p, .callback=NULL, .callback_ex=wgsl_cb_boolop_cmpxor, .user_data=NULL };
    }
    // Pattern 2: SHL(a, b) -> bitcast<u32>(a) << cast<u32>(b)
    {
      UPat* a = upat_var_named("a", NULL, 0, false);
      UPat* b = upat_var_named("b", NULL, 0, false);
      UPat* srcs[2] = { a, b };
      UPat* p = upat_op(OPS_SHL, srcs, 2);
      entries[m++] = (PatternMatch){ .pattern=p, .callback=NULL, .callback_ex=wgsl_cb_shl_fix, .user_data=NULL };
    }
    // Pattern 3: SHR(x, y) -> SHR(x, cast<uint>(y)) if rhs not uint
    {
      UPat* x = upat_var_named("x", NULL, 0, false);
      UPat* y = upat_var_named("y", NULL, 0, false);
      UPat* srcs[2] = { x, y };
      UPat* p = upat_op(OPS_SHR, srcs, 2);
      entries[m++] = (PatternMatch){ .pattern=p, .callback=NULL, .callback_ex=wgsl_cb_shr_fix, .user_data=NULL };
    }
    pm = pattern_matcher_new(entries, m, false);
  } while(0);
  if (pm){ for (int i=0;i<n;i++){ if (uops[i]) uops[i] = upat_graph_rewrite(uops[i], pm, "wgsl-pm"); } pattern_matcher_free(pm); }
  // collect buffers (shared pipeline)
  int ids[64]; const DType* dts[64]; CStyleCtx ctx; cstyle_ctx_init(&ctx);
  int pc = cstyle_ctx_collect_bufs(&ctx, uops, n, ids, dts, 64);
  char* s = tg_sb_new_with("", "");
  name_map_t map={0}; int vc=0;
  // header prelude: f16 enable, nan helper, INFINITY uniform
  int any_f16 = 0; for (int i=0;i<n;i++){ if (uops[i]){ const DType* b = scalar_dt(&uops[i]->dtype); if (dtype_eq(b, &dtypes.float16) || dtype_eq(b, &dtypes.half)) { any_f16=1; break; } } }
  if (any_f16) s = tg_sb_append_owned(s, "enable f16;\n");
  s = tg_sb_append_owned(s, "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n");
  s = tg_sb_append_owned(s, "@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n");
  int bind = 1;
  for (int i=0;i<pc;i++){
    // Storage buffers (read_write); packed types become atomic<u32>
    const DType* bdt = dtype_base(dts[i]); const DType* sdt = scalar_dt(bdt);
    const char* el = is_packed_dt(sdt) ? "atomic<u32>" : wgsl_scalar_ty(sdt);
    char line[256]; snprintf(line,sizeof(line), "@group(0) @binding(%d) var<storage, read_write> data%d: array<%s>;\n", bind++, ids[i], el);
    s = tg_sb_append_owned(s, line);
  }
  // Local/workgroup memory and registers
  for (int i=0;i<n;i++){
    UOp* u = uops[i]; if (!u) continue;
    if (u->op==OPS_DEFINE_LOCAL || u->op==OPS_DEFINE_REG){
      const char* nm = nm_add(&map, u, &vc);
      // size from shape or arg
      int sz = 1; if (u->st){ int nd = shapetracker_ndim(u->st); const int32_t* shp = shapetracker_shape(u->st); if (nd>0 && shp) sz = shp[0]; }
      if (sz <= 0 && u->arg.type==ARG_INT) sz = u->arg.int_data.i;
      int es = itemsize(&u->dtype); int units = es>0 ? (4/es) : 1; if (units<=0) units=1;
      int cnt = is_packed_dt(&u->dtype) ? (sz/units) : sz; if (cnt<=0) cnt=1;
      const char* el = is_packed_dt(&u->dtype) ? "atomic<u32>" : wgsl_scalar_ty(scalar_dt(&u->dtype));
      char line[256];
      if (u->op==OPS_DEFINE_LOCAL) snprintf(line,sizeof(line), "var<workgroup> %s: array<%s,%d>;\n", nm, el, cnt);
      else snprintf(line,sizeof(line), "var %s: array<%s,%d>;\n", nm, el, cnt);
      s = tg_sb_append_owned(s, line);
    }
  }
  // Compute entry with builtin indices and workgroup size from SPECIAL args
  int lsz[3] = {1,1,1};
  for (int i=0;i<n;i++){
    UOp* u = uops[i]; if (!u || u->op!=OPS_SPECIAL) continue;
    if (!u->tag) continue;
    const char* tg = (const char*)u->tag;
    if (strncmp(tg, "lidx", 4)==0) {
      int axis = 0; size_t L=strlen(tg); if (L){ char last=tg[L-1]; if (last>='0'&&last<='9') axis=last-'0'; }
      int bound = 1; if (u->arg.type==ARG_TUPLE2 && u->arg.tuple2.count>=1 && u->arg.tuple2.second) bound = u->arg.tuple2.second[0];
      if (axis>=0 && axis<3 && bound>0) lsz[axis] = bound;
    }
  }
  s = tg_sb_append_ownedf(s, "@compute @workgroup_size(%d,%d,%d) fn kernel_main(@builtin(workgroup_id) gindex: vec3<u32>, @builtin(local_invocation_id) lindex: vec3<u32>) {\n", lsz[0], lsz[1], lsz[2]);
  // minimal SSA: consts, barriers, SPECIAL indices, INDEX, LOAD/STORE, BITCAST, WHERE (with vector support)
  for (int i=0;i<n;i++){
    UOp* u = uops[i]; if (!u || u->op==OPS_NOOP || u->op==OPS_SINK) continue;
    if (u->op==OPS_CONST){ const char* nm = nm_add(&map, u, &vc); char* lit = wgsl_const(u); char* ty = wgsl_dtype(&u->dtype); char line[160]; snprintf(line,sizeof(line),"  let %s: %s = %s;\n", nm, ty, lit); s = tg_sb_append_owned(s, line); free(lit); free(ty); continue; }
    if (u->op==OPS_BARRIER){ s = tg_sb_append_owned(s, "  workgroupBarrier();\n"); continue; }
    if (u->op==OPS_SPECIAL){
      const char* nm = nm_add(&map, u, &vc);
      int axis = 0; char kind='g';
      if (u->tag){ const char* tg=(const char*)u->tag; size_t L=strlen(tg); if (L){ char last=tg[L-1]; if (last>='0'&&last<='9') axis=last-'0'; }
        if (strncmp((const char*)u->tag, "gidx", 4)==0) kind='g';
        else if (strncmp((const char*)u->tag, "lidx", 4)==0) kind='l'; }
      const char axisc = (axis==0)?'x':((axis==1)?'y':'z');
      char expr[128]; snprintf(expr,sizeof(expr), (kind=='g')?"i32(gindex.%c)":"i32(lindex.%c)", axisc);
      char* ty = wgsl_dtype(&u->dtype);
      char line[192]; snprintf(line,sizeof(line),"  let %s: %s = %s;\n", nm, ty, expr); s = tg_sb_append_owned(s, line); free(ty); continue;
    }
    // Explicit comparisons and bitwise boolean ops (limited)
    if ((u->op==OPS_CMPLT || u->op==OPS_CMPEQ || u->op==OPS_CMPNE) && u->src_count==2){
      char* a = str_or_const(&map, u->src[0]);
      char* b = str_or_const(&map, u->src[1]);
      const char* nm = nm_add(&map, u, &vc);
      if (u->op==OPS_CMPNE && strcmp(a,b)==0 && dtypes_is_float(&u->src[0]->dtype)){
        char* one = wgsl_float_lit(&u->src[0]->dtype, 1.0);
        char* n1 = wgsl_float_lit(&u->src[0]->dtype, -1.0);
        char line[512]; snprintf(line,sizeof(line),
          "  let %s: bool = ((min(%s, %s) == %s) && (max(%s, %s) == %s));\n",
          nm, a, one, one, a, n1, n1);
        s = tg_sb_append_owned(s, line); free(a); free(b); free(one); free(n1); continue;
      }
      const char* op = (u->op==OPS_CMPLT)? "<" : (u->op==OPS_CMPEQ? "==" : "!=");
      char line[256]; snprintf(line,sizeof(line), "  let %s: bool = (%s %s %s);\n", nm, a, op, b);
      s = tg_sb_append_owned(s, line); free(a); free(b); continue;
    }
    if (u->op==OPS_INDEX && u->src_count>=2){
      // map to data<gid>[idx]
      int gid=-1; size_t tn=0; UOp** bt=uop_toposort(u->src[0], &tn);
      if (bt){ for(size_t k=0;k<tn;k++) if (bt[k]->op==OPS_DEFINE_GLOBAL){ gid=(bt[k]->arg.type==ARG_INT)? bt[k]->arg.int_data.i:0; break; } free(bt); }
      char* idx = str_or_const(&map, u->src[1]);
      char* idx2 = strip_parens_copy(idx);
      char expr[256]; snprintf(expr,sizeof(expr), "data%d[%s]", gid>=0?gid:0, idx2); nm_set(&map, u, expr); free(idx2); free(idx); continue;
    }
    if (u->op==OPS_LOAD && u->src_count>=1){
      // LOAD from INDEX or raw name
      const char* addr = nm_get(&map, u->src[0]);
      UOp* idxu = u->src[0];
      UOp* gate = NULL; if (idxu && idxu->op==OPS_INDEX && idxu->src_count>=3) gate = idxu->src[2];
      char* gate_str = NULL; if (gate){ const char* gn = nm_get(&map, gate); if (gn) gate_str=strdup(gn); else if (gate->op==OPS_CONST) gate_str = wgsl_const(gate); }
      char* ty = wgsl_dtype(&u->dtype);
      const char* nm = nm_add(&map, u, &vc);
      if (addr && strstr(addr, "data")==addr){
        // attempt to detect idx inside brackets
        const char* lb = strchr(addr,'['); const char* rb = lb?strchr(lb,']'):NULL;
        if (lb && rb){
          // parse gid after "data"
          int gid = 0; const char* p = addr+4; while (*p && *p>='0' && *p<='9'){ gid = gid*10 + (*p - '0'); p++; }
          char idxbuf[128]; size_t L = (size_t)(rb-lb-1); if (L>=sizeof(idxbuf)) L=sizeof(idxbuf)-1; memcpy(idxbuf, lb+1, L); idxbuf[L]='\0';
          // decide packed
          if (is_packed_dt(&u->dtype)){
            int es = itemsize(&u->dtype); int units = 4/es; int bits = es*8; unsigned maskv = (es==1)?0xFFu:0xFFFFu;
            char line[768];
            // common helpers
            snprintf(line,sizeof(line),
              "  let %s_div: i32 = %s / %d;\n"
              "  let %s_shift: u32 = (u32(%s) %% %d) * %d;\n"
              "  let %s_load: u32 = atomicLoad(&data%d[%s_div]);\n"
              "  let %s_seg: u32 = (%s_load >> %s_shift) & %uu;\n",
              nm, idxbuf, units,
              nm, idxbuf, units, bits,
              nm, gid, nm,
              nm, nm, nm, maskv);
            s = tg_sb_append_owned(s, line);
            // final value
            if (dtype_eq(scalar_dt(&u->dtype), &dtypes.int8) || dtype_eq(scalar_dt(&u->dtype), &dtypes.char_) || dtype_eq(scalar_dt(&u->dtype), &dtypes.int16) || dtype_eq(scalar_dt(&u->dtype), &dtypes.short_)) {
              char fin[256];
              snprintf(fin,sizeof(fin),
                "  let %s_tmp: i32 = bitcast<i32>(%s_seg << (32u - %d));\n  let %s: %s = %s_tmp >> %d;\n",
                nm, nm, bits, nm, ty, nm, (32-bits));
              if (gate_str){
                char zero_sel[64]; strcpy(zero_sel, "0");
                char* z = wgsl_zero_expr(&u->dtype);
                char sel[512]; snprintf(sel,sizeof(sel), "  let %s: %s = select(%s, %s, %s);\n", nm, ty, z, nm, gate_str);
                s = tg_sb_append_owned(s, fin); s = tg_sb_append_owned(s, sel); free(z);
              } else {
                s = tg_sb_append_owned(s, fin);
              }
            } else {
              if (gate_str){
                char* z = wgsl_zero_expr(&u->dtype);
                char fin[512]; snprintf(fin,sizeof(fin), "  let %s: %s = select(%s, %s_seg, %s);\n", nm, ty, z, nm, gate_str);
                s = tg_sb_append_owned(s, fin); free(z);
              } else {
                char fin[256]; snprintf(fin,sizeof(fin), "  let %s: %s = %s_seg;\n", nm, ty, nm); s = tg_sb_append_owned(s, fin);
              }
            }
          } else {
            if (gate_str){
              char* z = wgsl_zero_expr(&u->dtype);
              char line[512]; snprintf(line,sizeof(line), "  let %s: %s = select(%s, %s, %s);\n", nm, ty, z, addr, gate_str);
              s = tg_sb_append_owned(s, line); free(z);
            } else {
              char line[256]; snprintf(line,sizeof(line), "  let %s: %s = %s;\n", nm, ty, addr); s = tg_sb_append_owned(s, line);
            }
          }
          free(ty); if (gate_str) free(gate_str); continue;
        }
      }
      // fallback
      if (gate_str){ char* z = wgsl_zero_expr(&u->dtype); char line[512]; snprintf(line,sizeof(line), "  let %s: %s = select(%s, %s, %s);\n", nm, ty, z, addr?addr:"0", gate_str); s = tg_sb_append_owned(s, line); free(z); free(ty); free(gate_str); continue; }
      char line[256]; snprintf(line,sizeof(line), "  let %s: %s = %s;\n", nm, ty, addr?addr:"0"); s = tg_sb_append_owned(s, line); free(ty); if (gate_str) free(gate_str); continue;
    }
    if (u->op==OPS_STORE && u->src_count>=2){
      const char* addr = nm_get(&map, u->src[0]); char* val = str_or_const(&map, u->src[1]);
      if (addr && strstr(addr, "data")==addr){
        const char* lb = strchr(addr,'['); const char* rb = lb?strchr(lb,']'):NULL;
        if (lb && rb){
          int gid = 0; const char* p = addr+4; while (*p && *p>='0' && *p<='9'){ gid = gid*10 + (*p - '0'); p++; }
          char idxbuf[128]; size_t L = (size_t)(rb-lb-1); if (L>=sizeof(idxbuf)) L=sizeof(idxbuf)-1; memcpy(idxbuf, lb+1, L); idxbuf[L]='\0';
          // find gating condition if present on INDEX
          UOp* idxu = u->src[0]; UOp* gate = NULL; if (idxu && idxu->op==OPS_INDEX && idxu->src_count>=3) gate = idxu->src[2];
          char* gate_str = NULL; if (gate){ const char* gn = nm_get(&map, gate); if (gn) gate_str=strdup(gn); else if (gate->op==OPS_CONST) gate_str = wgsl_const(gate); }
          if (is_packed_dt(&u->src[1]->dtype)){
            int es = itemsize(&u->src[1]->dtype); int units = 4/es; int bits = es*8; unsigned maskv = (es==1)?0xFFu:0xFFFFu;
            char line[512]; snprintf(line,sizeof(line),
              "  let _idx_div: i32 = %s / %d;\n  let _shift: u32 = (u32(%s) %% %d) * %d;\n  let _mask: u32 = ~((%uu << _shift));\n  atomicAnd(&data%d[_idx_div], _mask);\n  atomicAdd(&data%d[_idx_div], (u32(%s) & %uu) << _shift);\n",
              idxbuf, units,
              idxbuf, units, bits,
              maskv,
              gid,
              gid,
              val, maskv);
            if (gate_str){ char hdr[256]; snprintf(hdr,sizeof(hdr), "  if (%s) {\n", gate_str); s = tg_sb_append_owned(s, hdr); }
            s = tg_sb_append_owned(s, line);
            if (gate_str){ s = tg_sb_append_owned(s, "  }\n"); free(gate_str); }
            free(val); continue;
          } else {
            if (gate_str){ char hdr[256]; snprintf(hdr,sizeof(hdr), "  if (%s) { %s = %s; }\n", gate_str, addr, val); s = tg_sb_append_owned(s, hdr); free(gate_str); free(val); continue; }
            char line[256]; snprintf(line,sizeof(line), "  %s = %s;\n", addr, val); s = tg_sb_append_owned(s, line); free(val); continue;
          }
        }
      }
      // fallback
      char line[256]; snprintf(line,sizeof(line), "  /* store */ /* %s */ = %s;\n", addr?addr:"addr", val); s = tg_sb_append_owned(s, line); free(val); continue;
    }
    if (u->op==OPS_BITCAST && u->src_count>=1){
      char* src = str_or_const(&map, u->src[0]);
      char* ty = wgsl_dtype(&u->dtype);
      const char* nm = nm_add(&map, u, &vc);
      const DType* sd = scalar_dt(&u->src[0]->dtype);
      const DType* dd = scalar_dt(&u->dtype);
      char expr[256];
      if (dtype_eq(dd,&dtypes.float16)||dtype_eq(dd,&dtypes.half)){
        snprintf(expr,sizeof(expr), "bitcast<vec2<f16>>(%s)[0]", src);
      } else if (dtype_eq(dd,&dtypes.char_)||dtype_eq(dd,&dtypes.uint8)){
        snprintf(expr,sizeof(expr), "bitcast<%s>(%s & 0xFF)", wgsl_scalar_ty(dd), src);
      } else if (dtype_eq(dd,&dtypes.short_)||dtype_eq(dd,&dtypes.uint16)){
        if (dtype_eq(sd,&dtypes.half)||dtype_eq(sd,&dtypes.float16)) snprintf(expr,sizeof(expr), "bitcast<%s>(vec2<f16>(%s,0))", wgsl_scalar_ty(dd), src);
        else snprintf(expr,sizeof(expr), "bitcast<%s>(%s & 0xFFFF)", wgsl_scalar_ty(dd), src);
      } else {
        snprintf(expr,sizeof(expr), "bitcast<%s>(%s)", wgsl_scalar_ty(dd), src);
      }
      char line[320]; snprintf(line,sizeof(line), "  let %s: %s = %s;\n", nm, ty, expr); s = tg_sb_append_owned(s, line); free(src); free(ty); continue;
    }
    if (u->op==OPS_WHERE && u->src_count==3){
      const char* cn = nm_get(&map, u->src[0]); if (!cn){ if (u->src[0]->op==OPS_CONST){ cn = wgsl_const(u->src[0]); } }
      const char* tn = nm_get(&map, u->src[1]); if (!tn){ if (u->src[1]->op==OPS_CONST){ tn = wgsl_const(u->src[1]); } }
      const char* fn = nm_get(&map, u->src[2]); if (!fn){ if (u->src[2]->op==OPS_CONST){ fn = wgsl_const(u->src[2]); } }
      const char* nm = nm_add(&map, u, &vc); char* ty = wgsl_dtype(&u->dtype);
      char line[320]; snprintf(line,sizeof(line),"  let %s: %s = select(%s, %s, %s);\n", nm, ty, fn?fn:"0", tn?tn:"0", cn?cn:"false"); s = tg_sb_append_owned(s, line); free(ty); continue; }
  }
  s = tg_sb_append_owned(s, "}\n");
  cstyle_ctx_free(&ctx);
  return s; }

Renderer* renderer_wgsl(void){ Renderer* r=(Renderer*)calloc(1,sizeof(Renderer)); r->device="WGSL"; r->suffix=""; r->render=wgsl_render; return r; }
