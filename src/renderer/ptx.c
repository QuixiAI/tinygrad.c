#include "renderer/ptx.h"
#include "uop/uop.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "helpers/string_builder.h"
typedef struct { UOp** keys; char** names; int count; int cap; } name_map_t;
static const char* nm_get(name_map_t* m, UOp* u){ for(int i=0;i<m->count;i++) if(m->keys[i]==u) return m->names[i]; return NULL; }
static const char* nm_add(name_map_t* m, UOp* u, const char* prefix, int* counter){ char tmp[32]; snprintf(tmp,sizeof(tmp),"%s%d", prefix, (*counter)++); if (m->count+1>m->cap){ int nc=m->cap?m->cap*2:32; m->keys=(UOp**)realloc(m->keys,nc*sizeof(UOp*)); m->names=(char**)realloc(m->names,nc*sizeof(char*)); m->cap=nc; } m->keys[m->count]=u; m->names[m->count]=strdup(tmp); m->count++; return m->names[m->count-1]; }
static int dtype_itemsize(const DType* dt){ const DType* s=dt; if (dt->count>1 && dt->_scalar) s=dt->_scalar; return s->itemsize; }
static int is_float(const DType* dt){ const DType* s=dt; if (dt->count>1 && dt->_scalar) s=dt->_scalar; return dtype_eq(s, &dtypes.float32) || dtype_eq(s, &dtypes.float64) || dtype_eq(s, &dtypes.float16) || dtype_eq(s, &dtypes.bfloat16); }
static char* ptx_render(Renderer* self, UOp** uops, int n){ (void)self;
  const char* header = ".version 7.8\n.target sm_80\n.address_size 64\n";
  int ids[64]; int pc=0;
  for (int i=0;i<n && pc<64;i++) if (uops && uops[i] && uops[i]->op==OPS_DEFINE_GLOBAL){
    int id = (uops[i]->arg.type==ARG_INT)? uops[i]->arg.int_data.i : 0;
    int seen=0; for (int k=0;k<pc;k++) if (ids[k]==id){ seen=1; break; }
    if(!seen) ids[pc++]=id;
  }
  char sig[1024]; strcpy(sig, ".entry kernel_main(");
  for (int i=0;i<pc;i++){ char seg[64]; snprintf(seg,sizeof(seg),".param .u64 data%d%s", ids[i], (i<pc-1)?", ":"" ); strcat(sig, seg);} strcat(sig, ") {\n");
  char* s = tg_sb_new_with(header, sig);
  // declare some registers
  s = tg_sb_append_owned(s, "  .reg .pred %p1;\n  .reg .b32 %r1, %r2, %r3;\n  .reg .b64 %rd1, %rd2, %rd3;\n  .reg .f32 %f1, %f2, %f3;\n");
  // load param pointers
  for (int i=0;i<pc;i++){ char line[128]; snprintf(line,sizeof(line),"  ld.param.u64 %%rd_data%d, [data%d];\n", ids[i], ids[i]); s = tg_sb_append_owned(s, line); }

  name_map_t map={0}; int rc=1, rdc=1, fc=1, pcnt=1;
  const char* if_end[64]; int if_sp=0; int loop_id=0; const char* loop_end[64]; int loop_sp=0; char tmpbuf[64];
  for (int i=0;i<n;i++){
    UOp* u = uops[i]; if (!u || u->op==OPS_NOOP || u->op==OPS_SINK) continue;
    if (u->op==OPS_CONST){
      if (is_float(&u->dtype)){
        const char* nm = nm_add(&map, u, "%f", &fc); char line[128]; snprintf(line,sizeof(line),"  mov.f32 %s, %gf;\n", nm, (float)u->arg.const_data.const_value); s = tg_sb_append_owned(s, line); continue;
      } else {
        const char* nm = nm_add(&map, u, "%r", &rc); char line[128]; snprintf(line,sizeof(line),"  mov.s32 %s, %d;\n", nm, (int)u->arg.const_data.const_value); s = tg_sb_append_owned(s, line); continue;
      }
    }
    if (u->op==OPS_INDEX && u->src_count>=2){
      int gid=-1; size_t tn=0; UOp** bt=uop_toposort(u->src[0], &tn);
      if (bt){ for(size_t k=0;k<tn;k++) if (bt[k]->op==OPS_DEFINE_GLOBAL){ gid=(bt[k]->arg.type==ARG_INT)? bt[k]->arg.int_data.i:0; break; } free(bt); }
      const char* idx = nm_get(&map, u->src[1]); char idxbuf[32]; if (!idx){ if (u->src[1]->op==OPS_CONST){ snprintf(idxbuf,sizeof(idxbuf),"%d", (int)u->src[1]->arg.const_data.const_value); idx=idxbuf; } else idx="%r1"; }
      int sz = dtype_itemsize(&u->dtype); const char* nmaddr = nm_add(&map, u, "%rd", &rdc);
      char line[256]; snprintf(line,sizeof(line),"  mul.wide.s32 %%rd%dc, %s, %d;\n  add.s64 %s, %%rd_data%d, %%rd%dc;\n", rdc, idx, sz, nmaddr, gid>=0?gid:0, rdc); s = tg_sb_append_owned(s, line); continue;
    }
    if (u->op==OPS_LOAD && u->src_count>=1){ const char* addr = nm_get(&map, u->src[0]); if (!addr) addr="%rd1"; const char* nmv = nm_add(&map, u, is_float(&u->dtype)?"%f":"%r", is_float(&u->dtype)?&fc:&rc); char line[128]; if (is_float(&u->dtype)) snprintf(line,sizeof(line),"  ld.global.f32 %s, [%s];\n", nmv, addr); else snprintf(line,sizeof(line),"  ld.global.s32 %s, [%s];\n", nmv, addr); s = tg_sb_append_owned(s, line); continue; }
    if (u->op==OPS_STORE && u->src_count>=2){ const char* addr = nm_get(&map, u->src[0]); if (!addr) addr="%rd1"; const char* val = nm_get(&map, u->src[1]); if (!val) val = is_float(&u->src[1]->dtype)?"%f1":"%r1"; char line[128]; if (is_float(&u->src[1]->dtype)) snprintf(line,sizeof(line),"  st.global.f32 [%s], %s;\n", addr, val); else snprintf(line,sizeof(line),"  st.global.s32 [%s], %s;\n", addr, val); s = tg_sb_append_owned(s, line); continue; }
    if (u->op==OPS_ADD && u->src_count==2){ const char* a = nm_get(&map, u->src[0]); const char* b = nm_get(&map, u->src[1]); if (!a) a="%r1"; if (!b) b="%r2"; const char* nmv = nm_add(&map, u, "%r", &rc); char line[128]; snprintf(line,sizeof(line),"  add.s32 %s, %s, %s;\n", nmv, a, b); s = tg_sb_append_owned(s, line); continue; }
    if (u->op==OPS_AND && u->src_count==2){ const char* a2 = nm_get(&map, u->src[0]); const char* b2 = nm_get(&map, u->src[1]); const char* nmv3 = nm_add(&map, u, "%r", &rc); char line3[128]; snprintf(line3,sizeof(line3),"  and.b32 %s, %s, %s;\n", nmv3, a2?a2:"%r1", b2?b2:"%r2"); s = tg_sb_append_owned(s, line3); continue; }
    if (u->op==OPS_OR && u->src_count==2){ const char* a3 = nm_get(&map, u->src[0]); const char* b3 = nm_get(&map, u->src[1]); const char* nmv4 = nm_add(&map, u, "%r", &rc); char line4[128]; snprintf(line4,sizeof(line4),"  or.b32 %s, %s, %s;\n", nmv4, a3?a3:"%r1", b3?b3:"%r2"); s = tg_sb_append_owned(s, line4); continue; }
    if (u->op==OPS_XOR && u->src_count==2){ const char* a4 = nm_get(&map, u->src[0]); const char* b4 = nm_get(&map, u->src[1]); const char* nmv5 = nm_add(&map, u, "%r", &rc); char line5[128]; snprintf(line5,sizeof(line5),"  xor.b32 %s, %s, %s;\n", nmv5, a4?a4:"%r1", b4?b4:"%r2"); s = tg_sb_append_owned(s, line5); continue; }
    // comparisons to predicate
    if ((u->op==OPS_CMPLT || u->op==OPS_CMPEQ || u->op==OPS_CMPNE) && u->src_count==2){
      const char* a = nm_get(&map, u->src[0]); const char* b = nm_get(&map, u->src[1]);
      if (!a) a = is_float(&u->src[0]->dtype)?"%f1":"%r1";
      if (!b) b = is_float(&u->src[1]->dtype)?"%f2":"%r2";
      const char* pred = nm_add(&map, u, "%p", &pcnt);
      const char* op = (u->op==OPS_CMPLT)?"lt":(u->op==OPS_CMPEQ?"eq":"ne");
      char line[128]; snprintf(line,sizeof(line),"  setp.%s.%s %s, %s, %s;\n", op, is_float(&u->src[0]->dtype)?"f32":"s32", pred, a, b);
      s = tg_sb_append_owned(s, line); continue;
    }
    // IF/ENDIF control flow
    if (u->op==OPS_IF && u->src_count>=1){ const char* pred = nm_get(&map, u->src[0]); if (!pred) pred="%p1"; snprintf(tmpbuf,sizeof(tmpbuf),"IF_END_%d", if_sp); if_end[if_sp++] = strdup(tmpbuf); char line[128]; snprintf(line,sizeof(line),"  @!%s bra %s;\n", pred, tmpbuf); s = tg_sb_append_owned(s, line); continue; }
    if (u->op==OPS_ENDIF){ if (if_sp>0){ const char* lbl = if_end[--if_sp]; char line[64]; snprintf(line,sizeof(line),"%s:\n", lbl); s = tg_sb_append_owned(s, line); free((void*)lbl);} continue; }
    // RANGE/ENDRANGE loop with compare+branch
    if (u->op==OPS_RANGE && u->src_count>=1){ int id = loop_id++; char head[64], end[64]; snprintf(head,sizeof(head),"LOOP_%d", id); snprintf(end,sizeof(end),"LOOP_END_%d", id); loop_end[loop_sp++] = strdup(end); // iv
      const char* iv = nm_add(&map, u, "%r", &rc); char line[256]; snprintf(line,sizeof(line),"  mov.s32 %s, 0;\n%s:\n", iv, head); s = tg_sb_append_owned(s, line);
      const char* bound = nm_get(&map, u->src[0]); char bbuf[32]; if (!bound){ if (u->src[0]->op==OPS_CONST){ snprintf(bbuf,sizeof(bbuf),"%d", (int)u->src[0]->arg.const_data.const_value); bound=bbuf; } else bound="%r3"; }
      snprintf(line,sizeof(line),"  setp.lt.s32 %%p1, %s, %s;\n  @!%%p1 bra %s;\n", iv, bound, end); s = tg_sb_append_owned(s, line);
      continue; }
    if (u->op==OPS_ENDRANGE){ if (loop_sp>0){ const char* end = loop_end[--loop_sp]; // assume last iv is current
        // increment last iv (simplistic: %r(rc-1))
        char line[128]; snprintf(line,sizeof(line),"  add.s32 %%r%d, %%r%d, 1;\n  bra LOOP_%d\n%s:\n", rc-1, rc-1, loop_id-1, end); s = tg_sb_append_owned(s, line); free((void*)end);} continue; }
    if (u->op==OPS_MUL && u->src_count==2){ const char* a = nm_get(&map, u->src[0]); const char* b = nm_get(&map, u->src[1]); if (!a) a="%r1"; if (!b) b="%r2"; const char* nmv = nm_add(&map, u, "%r", &rc); char line[128]; snprintf(line,sizeof(line),"  mul.lo.s32 %s, %s, %s;\n", nmv, a, b); s = tg_sb_append_owned(s, line); continue; }
    if (u->op==OPS_IDIV && u->src_count==2){ const DType* dt=&u->dtype; const char* a = nm_get(&map, u->src[0]); const char* b = nm_get(&map, u->src[1]); if (!a) a=is_float(dt)?"%f1":"%r1"; if (!b) b=is_float(dt)?"%f2":"%r2"; const char* nmv2 = nm_add(&map, u, is_float(dt)?"%f":"%r", is_float(dt)?&fc:&rc); char line2[128]; if (is_float(dt)) snprintf(line2,sizeof(line2),"  div.rn.f32 %s, %s, %s;\n", nmv2, a, b); else snprintf(line2,sizeof(line2),"  div.s32 %s, %s, %s;\n", nmv2, a, b); s = tg_sb_append_owned(s, line2); continue; }
    if (u->op==OPS_SHL && u->src_count==2){ const char* a = nm_get(&map, u->src[0]); const char* b = nm_get(&map, u->src[1]); if (!a) a="%r1"; if (!b) b="%r2"; const char* nmv = nm_add(&map, u, "%r", &rc); char line[128]; snprintf(line,sizeof(line),"  shl.b32 %s, %s, %s;\n", nmv, a, b); s = tg_sb_append_owned(s, line); continue; }
    if (u->op==OPS_SHR && u->src_count==2){ const char* a = nm_get(&map, u->src[0]); const char* b = nm_get(&map, u->src[1]); if (!a) a="%r1"; if (!b) b="%r2"; const char* nmv = nm_add(&map, u, "%r", &rc); char line[128]; snprintf(line,sizeof(line),"  shr.s32 %s, %s, %s;\n", nmv, a, b); s = tg_sb_append_owned(s, line); continue; }
  }
  s = tg_sb_append_owned(s, "  ret;\n}\n");
  return s;
}

Renderer* renderer_ptx(const char* arch, const char* device){ (void)device; Renderer* r=(Renderer*)calloc(1,sizeof(Renderer)); r->device="PTX"; r->suffix = arch?arch:""; r->render = ptx_render; return r; }
