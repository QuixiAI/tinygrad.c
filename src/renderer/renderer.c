#include "renderer/renderer.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "helpers/helpers.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Base file intentionally minimal: the concrete backends implement .render.
// This source exists to ensure the symbol table is consistent and the
// interface compiles/links cleanly.

// -- to_function_name --
char* renderer_to_function_name(const char* s) {
  if (!s) return NULL;
  char* clean = tg_ansistrip(s);
  size_t n = strlen(clean);
  // worst case each char becomes 2 hex chars
  char* out = (char*)malloc(n*2 + 1);
  size_t j=0;
  for (size_t i=0;i<n;i++) {
    unsigned char c = (unsigned char)clean[i];
    if ((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_') {
      out[j++] = (char)c;
    } else {
      static const char* hex = "0123456789ABCDEF";
      out[j++] = hex[(c>>4)&0xF];
      out[j++] = hex[c&0xF];
    }
  }
  out[j] = '\0';
  free(clean);
  return out;
}

// -- Estimates.from_uops (simplified but parity-minded) --
Estimates renderer_estimates_from_uops(UOp** uops, int count, int ignore_indexing) {
  Estimates e = {0,0,0};
  if (!uops || count<=0) return e;
  // Build dont_count set for indexing when requested
  UOp** dont=NULL; int dcnt=0, dcap=32;
  if (ignore_indexing) { dont = (UOp**)malloc(sizeof(UOp*)*dcap); }
  if (ignore_indexing) {
    for (int i=0;i<count;i++) {
      UOp* u=uops[i]; if(!u) continue;
      if (u->op == OPS_LOAD || u->op == OPS_STORE) {
        // skip REG addrspace
        if (u->src_count>=1) {
          const PtrDType* pd = (const PtrDType*)&u->src[0]->dtype;
          if (pd->addrspace != ADDRSPACE_REG) {
            size_t n=0; UOp** topo = uop_toposort(u->src[0], &n);
            if (topo){ for (size_t k=0;k<n;k++){ if (dcnt>=dcap){ dcap*=2; dont=(UOp**)realloc(dont,sizeof(UOp*)*dcap);} dont[dcnt++]=topo[k]; } free(topo); }
            if (u->src_count>2) {
              size_t n2=0; UOp** topo2 = uop_toposort(u->src[2], &n2);
              if (topo2){ for (size_t k=0;k<n2;k++){ if (dcnt>=dcap){ dcap*=2; dont=(UOp**)realloc(dont,sizeof(UOp*)*dcap);} dont[dcnt++]=topo2[k]; } free(topo2);}            
            }
          }
        }
      } else if (u->op == OPS_IF && u->src_count>=1) {
        size_t n=0; UOp** topo = uop_toposort(u->src[0], &n);
        if (topo){ for (size_t k=0;k<n;k++){ if (dcnt>=dcap){ dcap*=2; dont=(UOp**)realloc(dont,sizeof(UOp*)*dcap);} dont[dcnt++]=topo[k]; } free(topo);}      
      }
    }
  }
  // helper
  #define IN_DONT(x) ({ int _f=0; if (dont){ for(int _i=0;_i<dcnt;_i++){ if (dont[_i]==(x)){ _f=1; break; } } } _f; })
  long long mults = 1; long long* stack=NULL; int sp=0, sc=8; stack=(long long*)malloc(sizeof(long long)*sc);
  for (int i=0;i<count;i++) {
    UOp* u = uops[i]; if (!u) continue;
    if (u->op == OPS_RANGE && u->src_count>=1) {
      if (sp>=sc){ sc*=2; stack=(long long*)realloc(stack,sizeof(long long)*sc);} stack[sp++]=mults;
      int r = 1; /* TODO: resolve bound exactly; placeholder to avoid hard dep */ if (r<=0) r=1; mults *= r;
    } else if (u->op == OPS_ENDRANGE) {
      if (sp>0) mults = stack[--sp];
    } else if (u->op == OPS_SPECIAL) {
      // No reliable access to arg; skip multiplying
    } else if (u->op == OPS_LOAD) {
      const PtrDType* pd = (const PtrDType*)&u->src[0]->dtype;
      if (pd->addrspace != ADDRSPACE_REG) e.lds += (long long)u->dtype.itemsize * mults;
    } else if (u->op == OPS_STORE && u->src_count>=2) {
      const PtrDType* pd = (const PtrDType*)&u->src[0]->dtype;
      if (pd->addrspace != ADDRSPACE_REG) e.lds += (long long)u->src[1]->dtype.itemsize * mults;
    } else if (group_op.is_alu[u->op]) {
      if (!IN_DONT(u)) {
        int per = (u->op == OPS_MULACC) ? 2 : 1;
        e.ops += (long long)per * mults * u->dtype.count;
      }
    } else if (u->op == OPS_WMMA) {
      if (!IN_DONT(u)) e.ops += 1024 * mults; // placeholder without full arg parsing
    }
  }
  if (dont) free(dont);
  if (stack) free(stack);
  e.mem = e.lds; // high estimate fallback
  return e;
}
