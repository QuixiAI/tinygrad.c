Deep Port Plan: `reference/tinygrad/renderer/*` → `src/renderer/*`

Objective

- Port TinyGrad’s renderer system line-for-line in behavior, covering: base `Renderer` APIs, C-style backends (CPU/CL/CUDA/Metal/AMD/NV/QCOM), LLVM IR (generic + AMD), PTX, and WGSL.
- Keep codegen output parity for a curated corpus of UOp graphs, including tricky cases (packed types, half upcasts, pointer arithmetic, WMMA intrinsics).
- Provide ProgramSpec and Estimates utilities matching Python semantics for device integration.

Deliverables

- Base: `include/renderer/renderer.h`, `src/renderer/renderer.c` (ProgramSpec, Estimates, core Renderer vtable, helpers)
- C-style: `include/renderer/cstyle.h`, `src/renderer/cstyle.c`
- LLVM IR: `include/renderer/llvmir.h`, `src/renderer/llvmir.c`
- PTX: `include/renderer/ptx.h`, `src/renderer/ptx.c`
- WGSL: `include/renderer/wgsl.h`, `src/renderer/wgsl.c`
- Tests: `tests/renderer/test_*.c` with snapshot diffs or structural asserts against Python outputs

Parity Requirements

- ProgramSpec fields derived from uops (vars/globals/ins/outs/local/global sizes) identical to Python ordering and set semantics (dedup / sorted where applicable).
- Estimates.from_uops counts flops/lds/mem with the same rules (ignore_indexing option, mults accumulation, dont_count sets, WMMA flops calc).
- Renderer outputs match string-wise (normalized whitespace) for: ALU, CAST/BITCAST, RANGE/ENDRANGE, IF/ENDIF, LOAD/STORE/INDEX, VECTORIZE, SPECIAL, CUSTOM, GEP, WHERE/MULACC, WMMA, images (OpenCL), packed WGSL paths.
- Device caps (supports_float4, global/local/shared max, has_local/shared) reflect Python defaults per backend.

Milestones & TODOs

- [x] M1: Base Renderer layer (DONE)
  - [x] Add `renderer.h/.c` with:
    - [x] Renderer struct + vtable (`render` function pointer) and capability fields
    - [x] to_function_name (C port using helpers `tg_ansistrip`) → `renderer_to_function_name`
    - [x] Estimates struct + `renderer_estimates_from_uops` (ALU/LOAD/STORE, MULACC=2, RANGE multipliers, dont_count when ignore_indexing). TODO: WMMA arg-based flops; SPECIAL multiplier.
    - [x] ProgramSpec fields derived from uops in `get_program`: vars (sorted by name), globals, outs/ins (dedup+sorted), function_name, base sizes, mem_estimate grouping (LOAD/STORE by (op, gid) max nbytes).
    - [ ] Small string builder utility for kernel text (defer to M2 when emitting code).
  - [ ] Unit tests for to_function_name, ProgramSpec extraction, Estimates on mini-uops (covered indirectly; add focused tests later).

- [ ] M2: CStyleLanguage (Clang/CPU) core
  - [x] Bootstrapped renderer with minimal kernel skeleton and basic SSA + ALU/CONST rendering (placeholder semantics; expands next).
  - [ ] Implement base_rewrite subset line-for-line:
    - [ ] CONST (nan/inf and dtype suffixes)
    - [ ] ALU ops (ADD/MUL/SUB/DIV/MAX/OR/AND/XOR) with correct parentheses
    - [ ] RANGE/ENDRANGE → for-loops; IF/ENDIF blocks
    - [ ] INDEX addressing; LOAD/STORE (non-image), with gating
    - [ ] CAST/BITCAST (including vector casts)
  - [ ] SSA map keyed by UOp identity + buffer bindings; function signature and args
  - [ ] Tests: small graphs vs Python ClangRenderer (normalized whitespace)

- [ ] M3: Full C-style backends
  - [ ] Complete base_rewrite parity: VECTORIZE, PRECAST, SPECIAL, WMMA stubs, CUSTOM/CUSTOMI, GEP formatting, devectorize rules for bools
  - [ ] Per-backend adapters:
    - [ ] ClangRenderer: type_map, float4 typedef, sqrt builtin, AMX, entry/footer
    - [ ] OpenCLRenderer: buffer/smem qualifiers, barrier, workitem code, image load/store patterns
    - [ ] IntelRenderer, MetalRenderer: specific overrides (inherit from CStyleLanguage/OpenCL)
    - [ ] CUDARenderer/AMDRenderer/NV/HIP/QCOM: device names, memory caps, string rewrites
  - [ ] Tests: each backend snapshot vs Python for representative graphs

- [ ] M4: PTXRenderer
  - [ ] `asm_for_op` map and dtype name table
  - [ ] PTX matcher parity: bool lowering; half-op upcast to f32; CAST elision for pointers; INDEX pointer arithmetic; SHIFT operand casting; mask shuttle to LOAD/STORE
  - [ ] SSA naming + .reg declarations; .entry header and parameter layout; mem_types; barrier
  - [ ] WMMA pack/unpack registers (sm75/sm80 differences) and arch selection
  - [ ] Tests: PTX output for ALU, memory, shifts, bool, WMMA (tensor-core smoke + matrix shapes)

- [ ] M5: LLVMRenderer + AMDLLVMRenderer
  - [ ] Type lowering helpers: `ldt`, `lconst`, `lcast`
  - [ ] base_rewrite for LLVM textual IR (icmp/fcmp forms, flags, gep, loads/stores, casts)
  - [ ] AMD-specific: MFMA/WMMA intrinsics, attributes footer, ABI, workitem code
  - [ ] Tests: IR output snapshots for patterns including WMMA (tensor cores)

- [ ] M6: WGSLRenderer
  - [ ] Packed load/store emulation (atomics) + sign_extend helpers
  - [ ] wgsl_matcher parity (bool expressions, shifts, bitcasts, const formatting, index, NaN checks)
  - [ ] Type maps, storage classes, barrier/select, @group/@binding layout, f16 enable
  - [ ] Tests: packed paths, select/where, atomics, bindings

- [ ] M7: PatternMatcher completeness
  - [ ] Ensure C UPat engine supports: named vars, dtype lists, allow_any_len srcs, fork/repeat groups as used in Python
  - [ ] Port remaining minor patterns from CStyle extra_pm, PTX, LLVM, WGSL

- [ ] M8: Integration & Device hooks
  - [ ] Public constructor functions per backend in headers, with capability fields
  - [ ] Ensure device code paths can call into renderer uniformly
  - [ ] Round-trip ProgramSpec.launch_dims with variable maps (add sym_infer-based resolution once SPECIAL sizes are symbolic)
  - [ ] Expose default device renderer to tests (to let tests call get_program with Device.default.renderer analog)

- [ ] M9: Test suite and polish
  - [ ] Snapshot normalizer for whitespace to compare strings across languages
  - [ ] Expand corpus: scalar/vector dtypes, pointer addrspaces, PRECAST/BITCAST, MULACC/WHERE, images
  - [ ] Warnings cleanup; coverage report for pattern hits

API Sketches (C)

```c
// include/renderer/renderer.h
typedef struct Renderer Renderer;
struct Renderer {
  const char *device, *suffix;
  int supports_float4, has_local, has_shared;
  int shared_max;
  int global_max[3], local_max[3];
  struct PatternMatcher *pre_matcher, *extra_matcher;
  char* (*render)(Renderer *self, struct UOp **uops, int count);
};

typedef struct Estimates { long long flops, lds, mem; } Estimates;
Estimates estimates_from_uops(struct UOp **uops, int count, int ignore_indexing);
Estimates estimates_add(Estimates a, Estimates b);
Estimates estimates_simplify(Estimates e);

typedef struct ProgramSpec {
  char *name, *src, *device;
  struct UOp *ast;
  struct UOp **uops; int uops_count;
  int global_size[3], local_size[3];
  struct UOp **vars; int vars_count;
  int *globals; int globals_count;
  int *outs; int outs_count;
  int *ins; int ins_count;
} ProgramSpec;

ProgramSpec ps_from_uops(const char *name, const char *device, struct UOp *ast, struct UOp **uops, int count);
char* ps_function_name(const ProgramSpec*);
void ps_launch_dims(const ProgramSpec*, struct UOp** vars, const int* vals, int n, int out_global[3], int out_local[3]);
char* renderer_to_function_name(const char*);
```

Backend Construction

- C-style: `Renderer* renderer_cstyle_clang(void);`, `renderer_cstyle_opencl`, `renderer_cstyle_cuda`, `renderer_cstyle_amd`, `renderer_cstyle_nv`, `renderer_cstyle_hip`, `renderer_cstyle_metal`, `renderer_cstyle_qcom`.
- LLVM: `Renderer* renderer_llvm_amd(const char* arch);` (and optional generic LLVM renderer)
- PTX: `Renderer* renderer_ptx(const char* arch, const char* device);`
- WGSL: `Renderer* renderer_wgsl(void);`

Tests (incremental)

- Unit tests for ProgramSpec/Estimates (M1)
- Renderer string snapshots:
  - CStyle/Clang: minimal ALU + memory, RANGE/IF, vectorize, casts, where/mulacc
  - OpenCL images + locals/shared (barriers, smem), Metal specifics
  - PTX ALU/memory/shift/where/bool, WMMA (tensor cores)
  - LLVM IR ALU/memory/gep/casts/icmp/fcmp, AMD intrinsics (tensor cores)
  - WGSL atomics packed paths, select/where, bitcasts
- Normalization: trim whitespace differences while comparing

Risks & Mitigations

- Pattern parity: ensure C UPat engine supports all constructs used (we already rely on named captures, dtype filters, allow_any_len in other modules). Add missing features before porting a backend relying on them.
- Float/half handling: faithfully replicate upcasts and literal formatting; add focused tests for these.
- Packed types (WGSL/OpenCL): atomics and bitcasts can be subtle; lock parity with small targeted tests.
- String diffs: Employ normalization and ensure predictable formatting (SSA naming order stable; deterministic buffer ordering).

Acceptance Criteria

- All planned backends render identical code (modulo whitespace) to the Python reference for the test corpus.
- ProgramSpec.estimates/launch_dims match Python.
- Device-integration tests continue to pass; no regressions in existing suites.
