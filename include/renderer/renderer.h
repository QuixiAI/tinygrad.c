#ifndef TG_RENDERER_RENDERER_H
#define TG_RENDERER_RENDERER_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward decls
typedef struct UOp UOp;
typedef struct Variable Variable;
typedef struct PatternMatcher PatternMatcher;

// Base Renderer interface (parity with Python Renderer fields where applicable)
typedef struct Renderer {
  const char* device;         // e.g., "CPU", "GPU", "CUDA", "AMD", "WEBGPU"
  const char* suffix;         // backend-specific suffix (optional)
  bool supports_float4;       // language supports vector literal constructors
  bool has_local;             // backend supports local/workgroup memory
  bool has_shared;            // backend supports shared memory model
  int shared_max;             // bytes of shared memory available
  int global_max[3];          // max global dims (x,y,z); 0 or negative => unset
  int local_max[3];           // max local dims (x,y,z); 0 or negative => unset
  // Optional pattern matchers applied by renderer
  PatternMatcher* pre_matcher;
  PatternMatcher* extra_matcher;
  // Render function. Returns heap-allocated source string; caller frees.
  char* (*render)(struct Renderer* self, UOp** uops, int uops_count);
} Renderer;

// Helper to call the renderer vtable (returns NULL on error)
static inline char* renderer_render(Renderer* r, UOp** uops, int count) {
  if (!r || !r->render) return NULL;
  return r->render(r, uops, count);
}

// -------- Program helpers (parity with Python renderer/__init__.py) --------

// to_function_name: sanitize a display name to a valid function symbol
// Caller owns returned string (heap-allocated)
char* renderer_to_function_name(const char* s);

// Estimates: rough FLOPs / bytes counters
typedef struct Estimates {
  long long ops;  // FLOPs (and ALU ops)
  long long lds;  // bytes loaded/stored (sum of loads + stores)
  long long mem;  // total bytes accessed (high estimate: use lds)
} Estimates;

// Compute estimates from a uops list. If ignore_indexing!=0, attempts to
// ignore indexing-cost expressions (best-effort parity with Python).
Estimates renderer_estimates_from_uops(UOp** uops, int count, int ignore_indexing);

// -------- ProgramSpec helpers (parity with Python ProgramSpec) --------

typedef struct ProgramSpec {
  char* name;
  char* src;
  char* device;
  UOp* ast;
  UOp** uops;
  int uops_count;
  int global_size[3];
  int local_size[3];
  bool global_size_valid;
  bool local_size_valid;
  bool has_local;
  char* function_name;
  Variable** vars;
  int vars_count;
  int* globals;
  int globals_count;
  int* outs;
  int outs_count;
  int* ins;
  int ins_count;
  Estimates estimates;
} ProgramSpec;

// Populate ProgramSpec derived fields (vars/globals/ins/outs/mem).
void programspec_finalize(ProgramSpec* spec);

// Resolve launch dimensions given optional variable bindings. Any unset axis is left as 0.
void ps_launch_dims(const ProgramSpec* spec, UOp** vars, const int* vals, int n,
                    int out_global[3], int out_local[3]);

// Convenience accessor mirroring Python's ProgramSpec.function_name cached property.
// Returns owned string (caller frees) if spec/function present, otherwise NULL.
char* ps_function_name(const ProgramSpec* spec);

#ifdef __cplusplus
}
#endif

#endif // TG_RENDERER_RENDERER_H
