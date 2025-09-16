#include "renderer/llvmir.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "dtype/dtype.h"
#include "helpers/string_builder.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static const char* llvm_ty(const DType* dt){
  const DType* s = dt; if (dt->count>1 && dt->_scalar) s = dt->_scalar;
  if (dtype_eq(s, &dtypes.float16) || dtype_eq(s, &dtypes.half)) return "half";
  if (dtype_eq(s, &dtypes.bfloat16)) return "bfloat";
  if (dtype_eq(s, &dtypes.float32) || dtype_eq(s, &dtypes.float_)) return "float";
  if (dtype_eq(s, &dtypes.float64) || dtype_eq(s, &dtypes.double_)) return "double";
  if (dtype_eq(s, &dtypes.int8)) return "i8";
  if (dtype_eq(s, &dtypes.uint8) || dtype_eq(s, &dtypes.uchar)) return "i8";
  if (dtype_eq(s, &dtypes.int16) || dtype_eq(s, &dtypes.short_)) return "i16";
  if (dtype_eq(s, &dtypes.uint16) || dtype_eq(s, &dtypes.ushort)) return "i16";
  if (dtype_eq(s, &dtypes.int32) || dtype_eq(s, &dtypes.int_)) return "i32";
  if (dtype_eq(s, &dtypes.uint32) || dtype_eq(s, &dtypes.uint)) return "i32";
  if (dtype_eq(s, &dtypes.int64) || dtype_eq(s, &dtypes.long_)) return "i64";
  if (dtype_eq(s, &dtypes.uint64) || dtype_eq(s, &dtypes.ulong)) return "i64";
  if (dtype_eq(s, &dtypes.bool_)) return "i1";
  return "i32";
}
static char* llvm_vec_ty(const DType* dt){
  const DType* s = dt; if (dt->count>1 && dt->_scalar) s = dt->_scalar;
  char buf[64];
  if (dt->count>1) { snprintf(buf,sizeof(buf),"<%d x %s>", dt->count, llvm_ty(s)); return strdup(buf); }
  return strdup(llvm_ty(s));
}
static char* llvm_const_str(const DType* dt, UOp* u){
  // Only scalar literals handled here
  char buf[64];
  if (dtypes_is_float(dt)) {
    double v = (u && u->arg.type==ARG_CONST) ? u->arg.const_data.const_value : 0.0;
    if (isnan(v) || isinf(v)) {
      union { double d; uint64_t u; } cv; cv.d = v;
      // print as 0xHH.. (big endian order in textual)
      uint8_t b[8]; memcpy(b, &cv.u, 8);
      snprintf(buf, sizeof(buf), "0x%02X%02X%02X%02X%02X%02X%02X%02X", b[7], b[6], b[5], b[4], b[3], b[2], b[1], b[0]);
    } else {
      // precise decimal via dtype truncate
      double tv = dtypes_truncate(v, dt);
      snprintf(buf,sizeof(buf),"%.*e", 6, tv);
    }
  } else {
    long long v = (u && u->arg.type==ARG_CONST) ? (long long)u->arg.const_data.const_value : 0;
    snprintf(buf,sizeof(buf),"%lld", v);
  }
  return strdup(buf);
}

typedef struct { UOp** keys; char** names; int count; int cap; } name_map_t;
static const char* nm_get(name_map_t* m, UOp* u){ for(int i=0;i<m->count;i++) if(m->keys[i]==u) return m->names[i]; return NULL; }
static const char* nm_add(name_map_t* m, UOp* u){
  char tmp[32]; snprintf(tmp,sizeof(tmp),"%%v%d", m->count);
  if (m->count+1>m->cap){ int nc=m->cap?m->cap*2:32; m->keys=(UOp**)realloc(m->keys, nc*sizeof(UOp*)); m->names=(char**)realloc(m->names, nc*sizeof(char*)); m->cap=nc; }
  m->keys[m->count]=u; m->names[m->count]=strdup(tmp); m->count++; return m->names[m->count-1];
}
static int align_of(const DType* dt){ const DType* s=dt; if (dt->count>1 && dt->_scalar) s=dt->_scalar; if (dtype_eq(s,&dtypes.float64)) return 8; if (dtype_eq(s,&dtypes.int64)||dtype_eq(s,&dtypes.uint64)) return 8; if (dtype_eq(s,&dtypes.float32)) return 4; if (dtype_eq(s,&dtypes.int32)||dtype_eq(s,&dtypes.uint32)) return 4; if (dtype_eq(s,&dtypes.int8)||dtype_eq(s,&dtypes.uint8)||dtype_eq(s,&dtypes.bool_)) return 1; return 4; }

static char* llvm_render(Renderer* self, UOp** uops, int n){ (void)self;
  int is_amd = (self && self->device && (strcmp(self->device, "LLVM-AMD")==0 || strcmp(self->device, "AMD")==0));
  // Gather parameters from DEFINE_GLOBAL
  int ids[64]; const DType* dts[64]; int pc=0;
  for (int i=0;i<n && pc<64;i++) if (uops[i] && uops[i]->op==OPS_DEFINE_GLOBAL){
    int id = (uops[i]->arg.type==ARG_INT)? uops[i]->arg.int_data.i : 0;
    int seen=0; for (int k=0;k<pc;k++) if (ids[k]==id){ seen=1; break; }
    if (!seen){ ids[pc]=id; dts[pc]=&uops[i]->dtype; pc++; }
  }
  // compute AMD workgroup attribute if needed
  int required_wg = 1;
  if (is_amd) {
    for (int i=0;i<n;i++){
      UOp* u = uops[i]; if (!u || u->op!=OPS_SPECIAL) continue;
      const char* tag = (const char*)u->tag; if (!tag) continue;
      if (strncmp(tag, "lidx", 4)==0) {
        int v = 1; if (u->arg.type==ARG_INT) v = u->arg.int_data.i; if (v<1) v=1; required_wg *= v;
      }
    }
    if (required_wg < 1) required_wg = 1;
  }

  // Pre-scan for AMD DEFINE_LOCAL globals
  name_map_t loc_syms = {0}; int loc_counter = 0;
  char* globs = strdup("");
  if (is_amd) {
    for (int i=0;i<n;i++){
      UOp* u=uops[i]; if (!u || u->op!=OPS_DEFINE_LOCAL) continue;
      int sz = (u->arg.type==ARG_INT)? u->arg.int_data.i : 0;
      char* elty = llvm_vec_ty(&u->dtype);
      char gname[32]; snprintf(gname,sizeof(gname),"l%d", loc_counter++);
      // map u -> gname
      // reuse name_map_t to store symbol name
      if (loc_syms.count+1>loc_syms.cap){ int nc=loc_syms.cap?loc_syms.cap*2:32; loc_syms.keys=(UOp**)realloc(loc_syms.keys,nc*sizeof(UOp*)); loc_syms.names=(char**)realloc(loc_syms.names,nc*sizeof(char*)); loc_syms.cap=nc; }
      loc_syms.keys[loc_syms.count]=u; loc_syms.names[loc_syms.count]=strdup(gname); loc_syms.count++;
      char line[256]; snprintf(line,sizeof(line),"@%s = internal unnamed_addr addrspace(3) global [%d x %s] undef, align 16\n", gname, sz, elty);
      globs = tg_sb_append_owned(globs, line); free(elty);
    }
  }

  char* s = strdup("; ModuleID = 'tinygradc'\n");
  if (is_amd && globs && globs[0]) s = tg_sb_append_owned(s, globs);
  s = tg_sb_append_owned(s, "define void @kernel_main(");
  for (int i=0;i<pc;i++){
    char seg[64];
    snprintf(seg,sizeof(seg),"ptr %%%s%d%s", "data", ids[i], (i<pc-1)?", ":""); s=tg_sb_append_owned(s, seg);
  }
  if (is_amd) {
    s = tg_sb_append_owned(s, ") #0 amdgpu_kernel {\n");
  } else {
    s = tg_sb_append_owned(s, ") {\n");
  }
  s = tg_sb_append_owned(s, "entry:\n");
  name_map_t map={0};
  int lbl_counter = 0;
  // simple stacks for control flow labels
  const char* if_end[64]; int if_sp=0;
  typedef struct { char pre[64], head[64], body[64], latch[64], end[64]; const char* iv; const char* bound; } LoopCtx;
  LoopCtx loops[64]; int loop_sp=0;
  int need_footer_attrs = 0;
  for (int i=0;i<n;i++){
    UOp* u = uops[i]; if (!u || u->op==OPS_NOOP || u->op==OPS_SINK) continue;
    // AMD specials and barriers
    if (is_amd && u->op==OPS_SPECIAL){
      // derive axis/kind from tag as in cstyle
      int axis = 0; char kind='i'; const char* tag = (const char*)u->tag; if (tag){ size_t L=strlen(tag); if (L){ char last=tag[L-1]; if (last>='0'&&last<='9') axis=last-'0'; }
        if (strncmp(tag, "gidx", 4)==0) kind='g'; else if (strncmp(tag, "lidx", 4)==0) kind='l'; else kind='i'; }
      char line[128]; const char axisc = axis==0?'x':(axis==1?'y':'z'); const char* nmv = nm_add(&map, u);
      if (kind=='g') snprintf(line,sizeof(line),"  %s = tail call i32 @llvm.amdgcn.workgroup.id.%c()\n", nmv, axisc);
      else if (kind=='l') snprintf(line,sizeof(line),"  %s = tail call i32 @llvm.amdgcn.workitem.id.%c()\n", nmv, axisc);
      else snprintf(line,sizeof(line),"  %s = add i32 0, 0\n", nmv);
      s = tg_sb_append_owned(s, line); need_footer_attrs=1; continue;
    }
    if (is_amd && u->op==OPS_BARRIER){
      s = tg_sb_append_owned(s, "  fence syncscope(\"workgroup\") release\n");
      s = tg_sb_append_owned(s, "  tail call void @llvm.amdgcn.s.barrier()\n");
      s = tg_sb_append_owned(s, "  fence syncscope(\"workgroup\") acquire\n");
      need_footer_attrs=1; continue;
    }
    // AMD WMMA intrinsics (stubbed based on dtype)
    if (is_amd && u->op==OPS_WMMA){
      const char* nmv = nm_add(&map, u);
      const char* arch = self && self->suffix ? self->suffix : "";
      // operand types
      char* ta = llvm_vec_ty(&u->src[0]->dtype);
      char* tb = llvm_vec_ty(&u->src[1]->dtype);
      char* tc = llvm_vec_ty(&u->src[2]->dtype);
      char* td = llvm_vec_ty(&u->dtype);
      // scalar dtype suffix mapping
      const char* sfx_in = NULL; const DType* sa = &u->src[0]->dtype; const DType* ss = sa; if (sa->count>1 && sa->_scalar) ss=sa->_scalar;
      if (dtype_eq(ss, &dtypes.float16) || dtype_eq(ss, &dtypes.half)) sfx_in = "f16";
      else if (dtype_eq(ss, &dtypes.bfloat16)) sfx_in = "bf16";
      else sfx_in = "f32";
      const char* a = nm_get(&map, u->src[0]); if (!a) a="%a";
      const char* b = nm_get(&map, u->src[1]); if (!b) b="%b";
      const char* c = nm_get(&map, u->src[2]); if (!c) c="%c";
      char line[512];
      if (strncmp(arch, "gfx942", 6)==0 || strncmp(arch, "gfx950", 6)==0) {
        snprintf(line,sizeof(line),"  %s = call %s @llvm.amdgcn.mfma.f32.16x16x16.%s(%s %s, %s %s, %s %s, i32 0, i32 0, i32 0)\n", nmv, td, sfx_in, ta, a, tb, b, tc, c);
      } else if (strncmp(arch, "gfx12", 5)==0) {
        snprintf(line,sizeof(line),"  %s = call %s @llvm.amdgcn.wmma.f32.16x16x16.%s(%s %s, %s %s, %s %s)\n", nmv, td, sfx_in, ta, a, tb, b, tc, c);
      } else {
        snprintf(line,sizeof(line),"  %s = call %s @llvm.amdgcn.wmma.f32.16x16x16.%s(%s %s, %s %s, %s %s)\n", nmv, td, sfx_in, ta, a, tb, b, tc, c);
      }
      s = tg_sb_append_owned(s, line); free(ta); free(tb); free(tc); free(td); need_footer_attrs=1; continue;
    }
    // ALU ops (binary)
    if (group_op.is_alu[u->op] && u->src_count==2){
      const DType* ty_dt = &u->dtype;
      char* ty = llvm_vec_ty(ty_dt);
      const char* a = nm_get(&map, u->src[0]); char* av = NULL;
      const char* b = nm_get(&map, u->src[1]); char* bv = NULL;
      if (!a) av = llvm_const_str(ty_dt, u->src[0]);
      if (!b) bv = llvm_const_str(ty_dt, u->src[1]);
      const char* nmv = nm_add(&map, u);
      const char* op = NULL; int isfloat = dtypes_is_float(ty_dt);
      switch (u->op){
        case OPS_ADD: op = isfloat?"fadd nsz arcp contract afn":"add"; break;
        case OPS_SUB: op = isfloat?"fsub nsz arcp contract afn":"sub"; break;
        case OPS_MUL: op = isfloat?"fmul nsz arcp contract afn":"mul"; break;
        case OPS_FDIV: op = "fdiv nsz arcp contract afn"; break;
        case OPS_IDIV: op = dtypes_is_unsigned(ty_dt)?"udiv":"sdiv"; break;
        case OPS_SHL:  op = "shl"; break;
        case OPS_SHR:  op = dtypes_is_unsigned(ty_dt)?"lshr":"ashr"; break;
        case OPS_AND: op = "and"; break;
        case OPS_OR:  op = "or"; break;
        case OPS_XOR: op = "xor"; break;
        default: op = NULL; break;
      }
      if (op){
      char line[256]; snprintf(line,sizeof(line),"  %s = %s %s %s, %s\n", nmv, op, ty, a?a:av, b?b:bv);
        s = tg_sb_append_owned(s, line);
        if (av) free(av); if (bv) free(bv); free(ty); continue;
      }
      if (av) free(av); if (bv) free(bv); free(ty);
    }
    // Comparisons
    if ((u->op==OPS_CMPLT || u->op==OPS_CMPEQ || u->op==OPS_CMPNE) && u->src_count==2){
      const DType* opd = &u->src[0]->dtype; // assume same type
      int isfloat = dtypes_is_float(opd);
      char* ty = llvm_vec_ty(opd);
      const char* a = nm_get(&map, u->src[0]); char* av = NULL;
      const char* b = nm_get(&map, u->src[1]); char* bv = NULL;
      if (!a) av = llvm_const_str(opd, u->src[0]);
      if (!b) bv = llvm_const_str(opd, u->src[1]);
      const char* nmv = nm_add(&map, u);
      const char* cc = NULL;
      if (isfloat) {
        switch (u->op){ case OPS_CMPLT: cc="olt"; break; case OPS_CMPEQ: cc="oeq"; break; case OPS_CMPNE: cc="one"; break; default: cc="oeq"; }
        char line[256]; snprintf(line,sizeof(line),"  %s = fcmp %s %s %s, %s\n", nmv, cc, ty, a?a:av, b?b:bv); s = tg_sb_append_owned(s, line);
      } else {
        switch (u->op){ case OPS_CMPLT: cc="slt"; break; case OPS_CMPEQ: cc="eq"; break; case OPS_CMPNE: cc="ne"; break; default: cc="eq"; }
        char line[256]; snprintf(line,sizeof(line),"  %s = icmp %s %s %s, %s\n", nmv, cc, ty, a?a:av, b?b:bv); s = tg_sb_append_owned(s, line);
      }
      if (av) free(av); if (bv) free(bv); free(ty); continue;
    }
    // Casts
    if (u->op==OPS_CAST && u->src_count>=1){
      const DType* dst = &u->dtype; const DType* srcd = &u->src[0]->dtype;
      char* dty = llvm_vec_ty(dst); char* sty = llvm_vec_ty(srcd);
      const char* ssa = nm_get(&map, u->src[0]); if (!ssa) ssa = "%src";
      const char* nmv = nm_add(&map, u);
      int s_is_float = dtypes_is_float(srcd), d_is_float = dtypes_is_float(dst);
      int s_bytes = srcd->itemsize, d_bytes = dst->itemsize;
      const char* op = NULL;
      if (s_is_float && d_is_float) op = (d_bytes > s_bytes)?"fpext":"fptrunc";
      else if (!s_is_float && !d_is_float) {
        if (d_bytes > s_bytes) op = dtypes_is_unsigned(srcd)?"zext":"sext"; else if (d_bytes < s_bytes) op = "trunc"; else op = "bitcast";
      } else if (!s_is_float && d_is_float) op = dtypes_is_unsigned(srcd)?"uitofp":"sitofp";
      else if (s_is_float && !d_is_float) op = dtypes_is_unsigned(dst)?"fptoui":"fptosi";
      if (!op) op = "bitcast";
      char line[256]; snprintf(line,sizeof(line),"  %s = %s %s %s to %s\n", nmv, op, sty, ssa, dty);
      s = tg_sb_append_owned(s, line); free(dty); free(sty); continue;
    }
    if (u->op==OPS_BITCAST && u->src_count>=1){
      char* dty = llvm_vec_ty(&u->dtype); char* sty = llvm_vec_ty(&u->src[0]->dtype);
      const char* ssa = nm_get(&map, u->src[0]); if (!ssa) ssa = "%src";
      const char* nmv = nm_add(&map, u);
      char line[256]; snprintf(line,sizeof(line),"  %s = bitcast %s %s to %s\n", nmv, sty, ssa, dty); s = tg_sb_append_owned(s, line);
      free(dty); free(sty); continue;
    }
    // Control flow: IF/ENDIF
    if (u->op==OPS_IF && u->src_count>=1){
      const char* cond = nm_get(&map, u->src[0]); if (!cond) cond = "%cond";
      char then_lbl[64], end_lbl[64]; snprintf(then_lbl,sizeof(then_lbl),"if.then%d", lbl_counter); snprintf(end_lbl,sizeof(end_lbl),"if.end%d", lbl_counter); lbl_counter++;
      char line[256]; snprintf(line,sizeof(line),"  br i1 %s, label %%%s, label %%%s\n%s:\n", cond, then_lbl, end_lbl, then_lbl); s = tg_sb_append_owned(s, line);
      if_end[if_sp++] = strdup(end_lbl);
      continue;
    }
    if (u->op==OPS_ENDIF){
      const char* end = if_sp>0 ? if_end[--if_sp] : "if.end";
      char line[128]; snprintf(line,sizeof(line),"  br label %%%s\n%s:\n", end, end); s = tg_sb_append_owned(s, line);
      continue;
    }
    // Control flow: RANGE/ENDRANGE with preheader, IV phi and latch (nested safe)
    if (u->op==OPS_RANGE && u->src_count>=1){
      int id = lbl_counter++;
      LoopCtx *lc = &loops[loop_sp++];
      snprintf(lc->pre, sizeof(lc->pre),   "loop%d.preheader", id);
      snprintf(lc->head, sizeof(lc->head), "loop%d.head", id);
      snprintf(lc->body, sizeof(lc->body), "loop%d.body", id);
      snprintf(lc->latch,sizeof(lc->latch),"loop%d.latch", id);
      snprintf(lc->end, sizeof(lc->end),   "loop%d.end", id);
      // branch to preheader from current block
      char line[256]; snprintf(line,sizeof(line),"  br label %%%s\n%s:\n", lc->pre, lc->pre); s = tg_sb_append_owned(s, line);
      // go to head
      snprintf(line,sizeof(line),"  br label %%%s\n%s:\n", lc->head, lc->head); s = tg_sb_append_owned(s, line);
      // IV phi
      lc->iv = nm_add(&map, u);
      snprintf(line,sizeof(line),"  %s = phi i32 [ 0, %%%s ], [ %s.next, %%%s ]\n", lc->iv, lc->pre, lc->iv, lc->latch); s = tg_sb_append_owned(s, line);
      // bound (if const or named)
      lc->bound = "0";
      if (u->src_count>=1){ const char* bnm=nm_get(&map, u->src[0]); if (bnm) lc->bound=bnm; else if (u->src[0]->op==OPS_CONST){ static char bb[64]; snprintf(bb,sizeof(bb),"%lld", (long long)u->src[0]->arg.const_data.const_value); lc->bound=strdup(bb); } }
      // branch to body
      snprintf(line,sizeof(line),"  br label %%%s\n%s:\n", lc->body, lc->body); s = tg_sb_append_owned(s, line);
      continue;
    }
    if (u->op==OPS_ENDRANGE){
      if (loop_sp<=0) continue; LoopCtx *lc = &loops[loop_sp-1];
      // go to latch
      char line[256]; snprintf(line,sizeof(line),"  br label %%%s\n%s:\n", lc->latch, lc->latch); s = tg_sb_append_owned(s, line);
      // iv.next and cmp
      snprintf(line,sizeof(line),"  %s.next = add i32 %s, 1\n", lc->iv, lc->iv); s = tg_sb_append_owned(s, line);
      snprintf(line,sizeof(line),"  %%cmp%d = icmp slt i32 %s.next, %s\n", lbl_counter, lc->iv, lc->bound); s = tg_sb_append_owned(s, line);
      // backedge to head or exit to end
      snprintf(line,sizeof(line),"  br i1 %%cmp%d, label %%%s, label %%%s\n%s:\n", lbl_counter, lc->head, lc->end, lc->end); s = tg_sb_append_owned(s, line);
      lbl_counter++;
      loop_sp--;
      continue;
    }
    if (u->op==OPS_DEFINE_LOCAL){
      if (is_amd){
        int sz = (u->arg.type==ARG_INT)? u->arg.int_data.i : 0; char* elty = llvm_vec_ty(&u->dtype);
        // find symbol
        const char* sym=NULL; for (int k=0;k<loc_syms.count;k++) if (loc_syms.keys[k]==u){ sym=loc_syms.names[k]; break; }
        const char* nmv = nm_add(&map, u);
        char line[256]; snprintf(line,sizeof(line),"  %s = addrspacecast [%d x %s] addrspace(3)* @%s to [%d x %s]*\n", nmv, sz, elty, sym?sym:"l0", sz, elty);
        s = tg_sb_append_owned(s, line); free(elty); continue;
      } else {
        // generic: alloca array
        int sz = (u->arg.type==ARG_INT)? u->arg.int_data.i : 0; char* elty = llvm_vec_ty(&u->dtype);
        const char* nmv = nm_add(&map, u);
        char line[256]; snprintf(line,sizeof(line),"  %s = alloca [%d x %s]\n", nmv, sz, elty); s = tg_sb_append_owned(s, line); free(elty); continue;
      }
    }
    if (u->op==OPS_INDEX && u->src_count>=2){
      // find base param id and dtype
      int gid=-1; const DType* gdt=&dtypes.float32; size_t tn=0; UOp** bt=uop_toposort(u->src[0], &tn);
      if (bt){ for(size_t k=0;k<tn;k++){ if (bt[k]->op==OPS_DEFINE_GLOBAL){ gid=(bt[k]->arg.type==ARG_INT)? bt[k]->arg.int_data.i:0; gdt=&bt[k]->dtype; break; } } free(bt); }
      const char* idxnm = nm_get(&map, u->src[1]); char ibuf[64]; const char* idxstr=idxnm; if (!idxstr){ if (u->src[1]->op==OPS_CONST) { long long v=(long long)u->src[1]->arg.const_data.const_value; snprintf(ibuf,sizeof(ibuf),"%lld", v); idxstr=ibuf; } else idxstr="0"; }
      char* elty = llvm_vec_ty(dtype_base(gdt)); // scalar in practice
      const char* nmv = nm_add(&map, u);
      char line[256]; snprintf(line,sizeof(line),"  %s = getelementptr inbounds %s, ptr %%data%d, i32 %s\n", nmv, elty, gid>=0?gid:0, idxstr);
      s = tg_sb_append_owned(s, line); free(elty); continue;
    }
    if (u->op==OPS_LOAD && u->src_count>=1){
      const char* ptr = nm_get(&map, u->src[0]); if (!ptr) ptr = "%ptr";
      const char* nmv = nm_add(&map, u);
      char* ty = llvm_vec_ty(&u->dtype); int al = align_of(&u->dtype);
      char line[256]; snprintf(line,sizeof(line),"  %s = load %s, ptr %s, align %d\n", nmv, ty, ptr, al);
      s = tg_sb_append_owned(s, line); free(ty); continue;
    }
    if (u->op==OPS_STORE && u->src_count>=2){
      const char* ptr = nm_get(&map, u->src[0]); if (!ptr) ptr = "%ptr";
      const char* valnm = nm_get(&map, u->src[1]); char vbuf[64]; const char* vstr = valnm;
      if (!vstr){ if (u->src[1]->op==OPS_CONST){ if (dtypes_is_float(&u->src[1]->dtype)) snprintf(vbuf,sizeof(vbuf),"%gf", (float)u->src[1]->arg.const_data.const_value); else snprintf(vbuf,sizeof(vbuf),"%lld", (long long)u->src[1]->arg.const_data.const_value); vstr=vbuf; }
        else vstr = "%val"; }
      char* ty = llvm_vec_ty(&u->src[1]->dtype); int al=align_of(&u->src[1]->dtype);
      char line[256]; snprintf(line,sizeof(line),"  store %s %s, ptr %s, align %d\n", ty, vstr, ptr, al);
      s = tg_sb_append_owned(s, line); free(ty); continue;
    }
    // simple integer const to bind name for indexing
    if (u->op==OPS_CONST && dtypes_is_int(&u->dtype)) { const char* nmv=nm_add(&map,u); char line[128]; snprintf(line,sizeof(line),"  %s = add i32 0, %lld\n", nmv, (long long)u->arg.const_data.const_value); s = tg_sb_append_owned(s, line); continue; }
  }
  s = tg_sb_append_owned(s, "  ret void\n}\n");
  if (is_amd && need_footer_attrs){
    // minimal attribute footer akin to Python's AMDLLVMRenderer, include flat workgroup size
    char attr[256];
    snprintf(attr, sizeof(attr), "attributes #0 = { alwaysinline nounwind \"no-builtins\" \"amdgpu-flat-work-group-size\"=\"1,%d\" \"no-trapping-math\"=\"true\" }\n", required_wg);
    s = tg_sb_append_owned(s, attr);
  }
  return s; }

Renderer* renderer_llvm_generic(void){
  Renderer* r=(Renderer*)calloc(1,sizeof(Renderer));
  r->device = "LLVM";
  r->suffix = "GEN";
  r->render = llvm_render;
  return r;
}
Renderer* renderer_llvm_amd(const char* arch){ (void)arch; Renderer* r=renderer_llvm_generic(); r->device="LLVM-AMD"; r->suffix = arch?arch:""; return r; }
