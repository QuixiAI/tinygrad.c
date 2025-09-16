#include "renderer/renderer.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "dtype/dtype.h"
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

static void add_unique_uop(UOp*** arr, int* count, int* cap, UOp* u) {
  if (!u) return;
  for (int i = 0; i < *count; i++) {
    if ((*arr)[i] == u) return;
  }
  int newcap = *cap;
  if (*count >= *cap) {
    newcap = (*cap > 0) ? (*cap * 2) : 32;
    *arr = (UOp**)realloc(*arr, sizeof(UOp*) * newcap);
    *cap = newcap;
  }
  (*arr)[(*count)++] = u;
}

static AddrSpace detect_addrspace(UOp* ptr) {
  if (!ptr) return ADDRSPACE_GLOBAL;
  size_t n = 0; UOp** topo = uop_toposort(ptr, &n);
  AddrSpace as = ADDRSPACE_GLOBAL;
  if (topo) {
    for (size_t i = 0; i < n; i++) {
      UOp* t = topo[i]; if (!t) continue;
      if (t->op == OPS_DEFINE_REG) { as = ADDRSPACE_REG; break; }
      if (t->op == OPS_DEFINE_LOCAL) { as = ADDRSPACE_LOCAL; break; }
      if (t->op == OPS_DEFINE_GLOBAL) { as = ADDRSPACE_GLOBAL; break; }
    }
    free(topo);
  }
  return as;
}

static long long const_from_uop(UOp* u) {
  if (!u) return 1;
  if (u->op == OPS_CONST && u->arg.type == ARG_CONST) {
    double v = u->arg.const_data.const_value;
    if (v <= 0) return 0;
    return (long long)v;
  }
  return 1;
}

// -- Estimates.from_uops (simplified but parity-minded) --
Estimates renderer_estimates_from_uops(UOp** uops, int count, int ignore_indexing) {
  Estimates e = (Estimates){0,0,0};
  if (!uops || count <= 0) return e;

  UOp** dont = NULL; int dcnt = 0, dcap = 0;
  if (ignore_indexing) {
    for (int i = 0; i < count; i++) {
      UOp* u = uops[i]; if (!u) continue;
      if ((u->op == OPS_LOAD || u->op == OPS_STORE) && u->src_count >= 1) {
        AddrSpace as = detect_addrspace(u->src[0]);
        if (as != ADDRSPACE_REG) {
          size_t n = 0; UOp** topo = uop_toposort(u->src[0], &n);
          if (topo) {
            for (size_t k = 0; k < n; k++) add_unique_uop(&dont, &dcnt, &dcap, topo[k]);
            free(topo);
          }
          if (u->src_count > 2) {
            size_t n2 = 0; UOp** topo2 = uop_toposort(u->src[2], &n2);
            if (topo2) {
              for (size_t k = 0; k < n2; k++) add_unique_uop(&dont, &dcnt, &dcap, topo2[k]);
              free(topo2);
            }
          }
        }
      } else if (u->op == OPS_IF && u->src_count >= 1) {
        size_t n = 0; UOp** topo = uop_toposort(u->src[0], &n);
        if (topo) {
          for (size_t k = 0; k < n; k++) add_unique_uop(&dont, &dcnt, &dcap, topo[k]);
          free(topo);
        }
      }
    }
  }

  #define IN_DONT(x) ({ int _f=0; if (dont){ for(int _i=0;_i<dcnt;_i++){ if (dont[_i]==(x)){ _f=1; break; } } } _f; })
  long long mults = 1;
  long long* stack = NULL; int sp = 0, scap = 0;

  for (int i = 0; i < count; i++) {
    UOp* u = uops[i]; if (!u) continue;
    if (u->op == OPS_RANGE && u->src_count >= 1) {
      if (sp >= scap) {
        scap = scap ? scap * 2 : 8;
        stack = (long long*)realloc(stack, sizeof(long long) * scap);
      }
      stack[sp++] = mults;
      long long bound = const_from_uop(u->src[0]);
      if (bound <= 0) bound = 1;
      mults *= bound;
      continue;
    }
    if (u->op == OPS_ENDRANGE) {
      if (sp > 0) mults = stack[--sp];
      else mults = 1;
      continue;
    }
    if (u->op == OPS_SPECIAL && u->arg.type == ARG_TUPLE2 && u->arg.tuple2.second && u->arg.tuple2.count > 0) {
      long long bound = u->arg.tuple2.second[0];
      if (bound > 0) mults *= bound;
      continue;
    }
    if (u->op == OPS_LOAD && u->src_count >= 1) {
      AddrSpace as = detect_addrspace(u->src[0]);
      if (as != ADDRSPACE_REG) e.lds += (long long)u->dtype.itemsize * mults;
      continue;
    }
    if (u->op == OPS_STORE && u->src_count >= 2) {
      AddrSpace as = detect_addrspace(u->src[0]);
      if (as != ADDRSPACE_REG) e.lds += (long long)u->src[1]->dtype.itemsize * mults;
      continue;
    }
    if (group_op.is_alu[u->op]) {
      if (!IN_DONT(u)) {
        int per = (u->op == OPS_MULACC) ? 2 : 1;
        long long vec = u->dtype.count > 0 ? u->dtype.count : 1;
        e.ops += (long long)per * mults * vec;
      }
      continue;
    }
    if (u->op == OPS_WMMA) {
      if (!IN_DONT(u)) e.ops += 1024 * mults;
      continue;
    }
  }

  if (dont) free(dont);
  if (stack) free(stack);
  e.mem = e.lds;
  return e;
}
