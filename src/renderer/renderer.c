#include "renderer/renderer.h"
#include "uop/uop.h"
#include "uop/ops.h"
#include "dtype/dtype.h"
#include "helpers/helpers.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

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
    return (long long)llround(v);
  }
  UOp* simplified = uop_ssimplify(u);
  if (simplified) {
    long long val = 1;
    if (simplified->op == OPS_CONST && simplified->arg.type == ARG_CONST) {
      double v = simplified->arg.const_data.const_value;
      if (v <= 0) val = 0;
      else val = (long long)llround(v);
    }
    uop_unref(simplified);
    return val;
  }
  return 1;
}

typedef struct {
  int op;
  int gid;
  long long maxbytes;
} MemGroup;

static int cmp_variable_by_name(const void* a, const void* b) {
  const Variable* va = *(const Variable* const*)a;
  const Variable* vb = *(const Variable* const*)b;
  const UOp* ua = (const UOp*)va;
  const UOp* ub = (const UOp*)vb;
  const char* na = (ua && ua->arg.type == ARG_VAR && ua->arg.var.name) ? ua->arg.var.name : "";
  const char* nb = (ub && ub->arg.type == ARG_VAR && ub->arg.var.name) ? ub->arg.var.name : "";
  return strcmp(na, nb);
}

static int cmp_int(const void* a, const void* b) {
  const int ia = *(const int*)a;
  const int ib = *(const int*)b;
  if (ia < ib) return -1;
  if (ia > ib) return 1;
  return 0;
}

static void sort_and_dedup_ints(int* arr, int* count) {
  if (!arr || !count || *count <= 0) return;
  qsort(arr, (size_t)*count, sizeof(int), cmp_int);
  int write = 1;
  for (int i = 1; i < *count; i++) {
    if (arr[i] != arr[write - 1]) {
      arr[write++] = arr[i];
    }
  }
  *count = write;
}

static int find_global_id(UOp* ptr) {
  if (!ptr) return -1;
  size_t n = 0;
  UOp** topo = uop_toposort(ptr, &n);
  if (!topo) return -1;
  int gid = -1;
  for (size_t i = 0; i < n; i++) {
    UOp* node = topo[i];
    if (node && node->op == OPS_DEFINE_GLOBAL) {
      gid = (node->arg.type == ARG_INT) ? node->arg.int_data.i : 0;
      break;
    }
  }
  free(topo);
  return gid;
}

static int dtype_value_nbytes(const DType* dt) {
  if (!dt) return 0;
  int count = dt->count > 0 ? dt->count : 1;
  return dt->itemsize * count;
}

static int parse_special_axis(const char* tag) {
  if (!tag) return 0;
  int axis = 0;
  for (const char* c = tag; *c; c++) {
    if (isdigit((unsigned char)*c)) axis = *c - '0';
  }
  if (axis < 0 || axis > 2) axis = 0;
  return axis;
}

// -- Estimates.from_uops (simplified but parity-minded) --
Estimates renderer_estimates_from_uops(UOp** uops, int count, int ignore_indexing) {
  Estimates e = (Estimates){0,0,0};
  if (!uops || count <= 0) return e;

  UOp** dont = NULL; int dcnt = 0, dcap = 0;
  MemGroup* lds_groups = NULL; int lds_group_count = 0, lds_group_cap = 0;
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
      if (as != ADDRSPACE_REG) {
        int gid = find_global_id(u->src[0]);
        if (gid >= 0) {
          long long nbytes = (long long)dtype_value_nbytes(&u->dtype) * mults;
          int existing = -1;
          for (int g = 0; g < lds_group_count; g++) {
            if (lds_groups[g].op == (int)u->op && lds_groups[g].gid == gid) { existing = g; break; }
          }
          if (existing < 0) {
            if (lds_group_count >= lds_group_cap) {
              lds_group_cap = lds_group_cap ? lds_group_cap * 2 : 8;
              lds_groups = (MemGroup*)realloc(lds_groups, sizeof(MemGroup) * (size_t)lds_group_cap);
            }
            lds_groups[lds_group_count++] = (MemGroup){ .op = u->op, .gid = gid, .maxbytes = nbytes };
          } else if (nbytes > lds_groups[existing].maxbytes) {
            lds_groups[existing].maxbytes = nbytes;
          }
        } else {
          e.lds += (long long)dtype_value_nbytes(&u->dtype) * mults;
        }
      }
      continue;
    }
    if (u->op == OPS_STORE && u->src_count >= 2) {
      AddrSpace as = detect_addrspace(u->src[0]);
      if (as != ADDRSPACE_REG) {
        int gid = find_global_id(u->src[0]);
        if (gid >= 0) {
          long long nbytes = (long long)dtype_value_nbytes(&u->src[1]->dtype) * mults;
          int existing = -1;
          for (int g = 0; g < lds_group_count; g++) {
            if (lds_groups[g].op == (int)u->op && lds_groups[g].gid == gid) { existing = g; break; }
          }
          if (existing < 0) {
            if (lds_group_count >= lds_group_cap) {
              lds_group_cap = lds_group_cap ? lds_group_cap * 2 : 8;
              lds_groups = (MemGroup*)realloc(lds_groups, sizeof(MemGroup) * (size_t)lds_group_cap);
            }
            lds_groups[lds_group_count++] = (MemGroup){ .op = u->op, .gid = gid, .maxbytes = nbytes };
          } else if (nbytes > lds_groups[existing].maxbytes) {
            lds_groups[existing].maxbytes = nbytes;
          }
        } else {
          e.lds += (long long)dtype_value_nbytes(&u->src[1]->dtype) * mults;
        }
      }
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
  for (int g = 0; g < lds_group_count; g++) {
    e.lds += lds_groups[g].maxbytes;
  }
  free(lds_groups);
  e.mem = e.lds;
  return e;
}

void programspec_finalize(ProgramSpec* spec) {
  if (!spec || !spec->uops || spec->uops_count <= 0) return;

  int vars_cap = spec->vars ? spec->vars_count : 8;
  if (!spec->vars) spec->vars = (Variable**)malloc(sizeof(Variable*) * vars_cap);
  spec->vars_count = 0;

  int globals_cap = spec->globals ? spec->globals_count : 8;
  if (!spec->globals) spec->globals = (int*)malloc(sizeof(int) * globals_cap);
  spec->globals_count = 0;

  int ins_cap = spec->ins ? spec->ins_count : 16;
  if (!spec->ins) spec->ins = (int*)malloc(sizeof(int) * ins_cap);
  spec->ins_count = 0;

  int outs_cap = spec->outs ? spec->outs_count : 16;
  if (!spec->outs) spec->outs = (int*)malloc(sizeof(int) * outs_cap);
  spec->outs_count = 0;

  for (int i = 0; i < spec->uops_count; i++) {
    UOp* u = spec->uops[i];
    if (!u) continue;

    if (u->op == OPS_DEFINE_VAR) {
      if (spec->vars_count >= vars_cap) {
        vars_cap = vars_cap ? vars_cap * 2 : 8;
        spec->vars = (Variable**)realloc(spec->vars, sizeof(Variable*) * vars_cap);
      }
      spec->vars[spec->vars_count++] = (Variable*)u;
    }

    if (u->op == OPS_DEFINE_GLOBAL) {
      int id = (u->arg.type == ARG_INT) ? u->arg.int_data.i : 0;
      if (spec->globals_count >= globals_cap) {
        globals_cap = globals_cap ? globals_cap * 2 : 8;
        spec->globals = (int*)realloc(spec->globals, sizeof(int) * globals_cap);
      }
      spec->globals[spec->globals_count++] = id;
    }

    if (u->op == OPS_STORE && u->src_count >= 1) {
      int gid = find_global_id(u->src[0]);
      if (gid >= 0) {
        if (spec->outs_count >= outs_cap) {
          outs_cap = outs_cap ? outs_cap * 2 : 16;
          spec->outs = (int*)realloc(spec->outs, sizeof(int) * outs_cap);
        }
        spec->outs[spec->outs_count++] = gid;
      }
    }

    if (u->op == OPS_LOAD && u->src_count >= 1) {
      int gid = find_global_id(u->src[0]);
      if (gid >= 0) {
        if (spec->ins_count >= ins_cap) {
          ins_cap = ins_cap ? ins_cap * 2 : 16;
          spec->ins = (int*)realloc(spec->ins, sizeof(int) * ins_cap);
        }
        spec->ins[spec->ins_count++] = gid;
      }
    }

    if (u->op == OPS_SPECIAL) {
      const char* tag = (const char*)u->tag;
      int axis = parse_special_axis(tag);
      int bound = 0;
      if (u->arg.type == ARG_TUPLE2 && u->arg.tuple2.second && u->arg.tuple2.count > 0) {
        bound = u->arg.tuple2.second[0];
      }
      if (tag && tag[0] == 'i') {
        spec->local_size_valid = false;
        spec->local_size[0] = spec->local_size[1] = spec->local_size[2] = 0;
      }
      int* target = spec->global_size;
      if (tag && tag[0] == 'l') {
        if (spec->has_local && spec->local_size_valid) target = spec->local_size;
        else target = NULL;
      } else {
        spec->global_size_valid = true;
      }
      if (target && axis >= 0 && axis < 3) {
        target[axis] = bound;
      }
    }
  }

  if (spec->vars && spec->vars_count > 1) {
    qsort(spec->vars, (size_t)spec->vars_count, sizeof(Variable*), cmp_variable_by_name);
  }

  sort_and_dedup_ints(spec->globals, &spec->globals_count);
  sort_and_dedup_ints(spec->ins, &spec->ins_count);
  sort_and_dedup_ints(spec->outs, &spec->outs_count);

  long long mem_est = 0;
  size_t topo_count = 0;
  UOp** topo = uop_toposort(spec->ast, &topo_count);
  if (topo) {
    MemGroup* groups = NULL;
    int group_count = 0;
    int group_cap = 8;
    if (group_cap > 0) {
      groups = (MemGroup*)malloc(sizeof(MemGroup) * (size_t)group_cap);
    }

    for (size_t i = 0; i < topo_count; i++) {
      UOp* node = topo[i];
      if (!node) continue;
      if (node->op != OPS_LOAD && node->op != OPS_STORE) continue;

      if (node->src_count < 1) continue;
      int gid = find_global_id(node->src[0]);
      if (gid < 0) continue;

      long long nbytes = 0;
      if (node->op == OPS_LOAD) {
        nbytes = dtype_value_nbytes(&node->dtype);
      } else if (node->op == OPS_STORE && node->src_count >= 2) {
        nbytes = dtype_value_nbytes(&node->src[1]->dtype);
      }

      int existing = -1;
      for (int g = 0; g < group_count; g++) {
        if (groups[g].op == (int)node->op && groups[g].gid == gid) {
          existing = g;
          break;
        }
      }

      if (existing < 0) {
        if (group_count >= group_cap) {
          group_cap = group_cap ? group_cap * 2 : 8;
          groups = (MemGroup*)realloc(groups, sizeof(MemGroup) * (size_t)group_cap);
        }
        groups[group_count++] = (MemGroup){ .op = node->op, .gid = gid, .maxbytes = nbytes };
      } else if (nbytes > groups[existing].maxbytes) {
        groups[existing].maxbytes = nbytes;
      }
    }

    for (int g = 0; g < group_count; g++) {
      mem_est += groups[g].maxbytes;
    }
    free(groups);
    free(topo);
  }

  spec->estimates.mem = mem_est;
}

char* ps_function_name(const ProgramSpec* spec) {
  if (!spec) return NULL;
  if (spec->function_name) return strdup(spec->function_name);
  if (!spec->name) return NULL;
  return renderer_to_function_name(spec->name);
}

void ps_launch_dims(const ProgramSpec* spec, UOp** vars, const int* vals, int n,
                    int out_global[3], int out_local[3]) {
  (void)vars; (void)vals; (void)n;
  if (out_global) {
    if (spec && spec->global_size_valid) {
      memcpy(out_global, spec->global_size, sizeof(int) * 3);
    } else {
      out_global[0] = out_global[1] = out_global[2] = 0;
    }
  }
  if (out_local) {
    if (spec && spec->local_size_valid) {
      memcpy(out_local, spec->local_size, sizeof(int) * 3);
    } else {
      out_local[0] = out_local[1] = out_local[2] = 0;
    }
  }
}
