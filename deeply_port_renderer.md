Deep Port Plan: `reference/tinygrad/renderer/*` → `src/renderer/*`

Objective

- Port TinyGrad’s renderer system line-for-line in behavior, covering: base `Renderer` APIs, C-style backends (CPU/CL/CUDA/Metal/AMD/NV/QCOM), LLVM IR (generic + AMD), PTX, and WGSL.
- Keep codegen output parity for a curated corpus of UOp graphs, including tricky cases (packed types, half upcasts, pointer arithmetic, WMMA intrinsics).
- Provide ProgramSpec and Estimates utilities matching Python semantics for device integration.

Status Update

- Base (M1): Complete. Renderer API, `renderer_to_function_name`, `renderer_estimates_from_uops` (with ignore_indexing + RANGE multipliers), and ProgramSpec post-init (vars sorted, globals/ins/outs dedup+sorted, mem_estimate grouping) are implemented. get_program wires these in and fills `function_name`; mem_estimate mirrors Python grouping semantics. Added shared string builder helper used by all renderers plus `programspec_finalize` with unit coverage mirroring Python `ProgramSpec.__post_init__` behavior.
- CStyle/Clang (M2): Complete for current test corpus (412/412 passing).
  - Implemented: type mapping, dtype-aware consts (NaN/Inf builtins), SSA naming, comparisons (CMPLT/CMPEQ/CMPNE), RANGE/ENDRANGE and IF/ENDIF blocks, INDEX pointer aliasing to `dataN + offset` with dtype-based pointer type, LOAD/STORE with gating (ternary guarded loads, if-guarded stores), CAST/BITCAST (scalar + vector), and VECTORIZE literal construction. Function signature includes typed params per DEFINE_GLOBAL (deduped).
  - ALU coverage includes ADD/SUB/MUL/OR/AND/XOR/SHL/SHR/MOD/MAX; WHERE and MULACC rendered with correct parentheses.
  - Unary builtins: exp2/log2/sin/sqrt mapped appropriately for Clang CPU.
  - Tests: renderer suites for ALU, vectorize, casts, bitcasts, gating, params, precedence are green.
- CStyle backends: initial backend-specific string differences implemented and tested.
  - OpenCL/QCOM: `__kernel void` signature; params use `__global TYPE*`; bitcasts via `as_TYPE(x)`; vector literals `(float4)(...)`; local memory `__local TYPE name[N]`; barrier `barrier(CLK_LOCAL_MEM_FENCE);`. Image read/write implemented (read_only/write_only image2d_t params, sampler, `read_imagef`/`write_imagef`).
  - Metal: `kernel void` signature with Metal headers; params use `device TYPE*`; bitcasts via `as_type<TYPE>(x)`; vector literals `float4(...)`; local memory `threadgroup __attribute__((aligned(16)))`; barrier `threadgroup_barrier(mem_flags::mem_threadgroup);`; sin uses `precise::sin`.
  - CUDA/HIP/AMD: `extern "C" __global__ void` signature; locals via `__shared__`; barrier `__syncthreads();`; vector literal style `make_float4(...)` for AMD/HIP; OCML intrinsics for exp2/log2/sin/sqrt on AMD/HIP with 16/32/64 suffix selection.
  - Added constructors for all C-style backends and routed codegen through the backend-aware emitter.
- CStyle backends: extended features (now implemented)
  - OpenCL/QCOM: image read/write implemented (read_only/write_only image2d_t params, sampler, `read_imagef`/`write_imagef`), vectorized loads, and special handling for INDEX on images.
- WGSL backend: expanded and integrated with shared base
  - Packed paths: i8/i16 load/store via `atomicLoad` + `atomicAnd/atomicAdd` with lane masks and shifts; signed extension for signed 8/16.
  - Gating: gated LOAD lowers to `select(zero, value, gate)`; gated STORE guarded with `if (gate) { ... }`.
  - Bitcasts and const typing: half via `bitcast<vec2<f16>>(...)[0]`; 8/16-bit masked `bitcast`s; unsigned consts use `u` suffix or `bitcast<u32>(-1)` when negative.
  - Header prelude and bindings: emits `enable f16;`, `fn nan() -> f32 { ... }`, and `@group(0) @binding(0) var<uniform> INFINITY : f32;`; storage buffers start at binding 1.
  - Workgroup size: derives `@workgroup_size(x,y,z)` from SPECIAL nodes (e.g., `lidx0/1/2` bounds).
  - INDEX formatting polish: trims redundant outer parentheses in subscripts.
- Shared C-style base (in use by WGSL and CStyle)
  - New `include/renderer/cstyle_base.h`, `src/renderer/cstyle_base.c` provide a reusable rendering context: buffer collection, write detection, preliminary SSA naming, and child-count scaffolding.
  - CStyle param collection and OpenCL image write detection refactored to use the shared context. GEP formatting now mirrors Python (`v[idx]` vs `.xyzw`) with per-backend thresholds and regression tests covering both Clang and OpenCL paths.
- Tests: Added `tests/renderer/test_cstyle_backends.c` covering signatures, parameter qualifiers/types, bitcast forms, vector literal styles, AMD OCML intrinsics, local memory/barriers, and OpenCL image I/O. Added LLVM IR tests (`tests/renderer/test_llvmir_min.c`, `tests/renderer/test_llvmir_ops.c`, `tests/renderer/test_llvmir_wmma.c`, `tests/renderer/test_llvmir_local_amd.c`). Added ProgramSpec unit to confirm globals/ins/outs/mem-estimate parity and reused renderer estimate smoke tests for builder regression detection.

Deliverables

- Base: `include/renderer/renderer.h`, `src/renderer/renderer.c` (ProgramSpec, Estimates, core Renderer vtable, helpers)
- C-style: `include/renderer/cstyle.h`, `src/renderer/cstyle.c`
- LLVM IR: `include/renderer/llvmir.h`, `src/renderer/llvmir.c`
- PTX: `include/renderer/ptx.h`, `src/renderer/ptx.c`
- WGSL: `include/renderer/wgsl.h`, `src/renderer/wgsl.c`
- CStyle base: `include/renderer/cstyle_base.h`, `src/renderer/cstyle_base.c`
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
    - [x] Small string builder utility for kernel text (defer to M2 when emitting code).
  - [x] Unit tests for to_function_name, ProgramSpec extraction, Estimates on mini-uops (covered indirectly; add focused tests later).

- [x] M2: CStyleLanguage (Clang/CPU) core
  - [x] SSA + ALU/CONST rendering
  - [x] CONST (dtype-aware suffixes; NaN/Inf handling)
  - [x] ALU ops (ADD/SUB/MUL/OR/AND/XOR/SHL/SHR/MOD/MAX, WHERE, MULACC)
  - [x] RANGE/ENDRANGE; IF/ENDIF
  - [x] INDEX addressing; LOAD/STORE with gating
  - [x] CAST/BITCAST (scalar + vector); VECTORIZE literal
  - [x] Function params from DEFINE_GLOBAL (deduped, typed)
  - [x] Backend-specific string differences (OpenCL/Metal/CUDA/HIP/AMD/QCOM): signatures, qualifiers, bitcasts, vector literal style, local/barrier, AMD OCML, OPS_SPECIAL
  - [x] Tests added for backend coverage

M3: Backends and LLVM IR (in progress)

- [x] OpenCL images: image param typing and read/write forms with sampler.
- [x] OPS_SPECIAL per backend (OpenCL get_*_id, CUDA/HIP blockIdx/threadIdx, Metal gid/lid).
- [x] LLVM IR generic backend: signature + getelementptr, load/store, ALU (fadd/fsub/fmul/fdiv, add/sub/mul, shl/lshr/ashr, and/or/xor), comparisons (fcmp/icmp), casts (sitofp/fptosi/zext/sext/trunc/fpext/fptrunc), bitcast, IF/ENDIF and RANGE loops with PHI and latch.
- [x] LLVM IR MFMA/WMMA: arch-based selection (gfx942/950 mfma; gfx12* wmma), typed operands/results, dtype suffixes f16/bf16/f32.
- [x] LLVM IR AMD local memory: addrspace(3) globals for DEFINE_LOCAL + addrspacecast inside function.
- [ ] LLVM IR improvements: fully mirror WMMA/MFMA combos, fcmp flags parity, addrspaces in more cases.
- [x] PTX backend: kernel signature, ld.param, INDEX addr calc (mul.wide/add.s64), ld.global/st.global, basic ALU (add/mul/shl/shr/div), bitwise (and/or/xor), IF/ENDIF via setp + predicated bra, RANGE loops.
- [x] WGSL backend: storage bindings, compute entry, WHERE via select, workgroupBarrier.
- [x] WGSL backend: vector lowering for WHERE, special indices, more type coverage.
- [x] WGSL backend: packed i8/i16 load/store via atomics; gated load/store; bitcasts for half/8/16; unsigned const typing; header prelude (f16, nan, INFINITY); bindings start at 1; workgroup_size from SPECIAL.
- [x] Shared CStyle base: context for buffers, writes, preliminary names; integrated in WGSL and CStyle param collection.

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
  - CStyle/Clang: ALU + memory, RANGE/IF, vectorize, casts, where/mulacc
  - OpenCL locals/barriers/bitcasts/vector style; images (read/write + sampler)
  - Metal signature/bitcasts/vector style/locals/barriers and precise::sin
  - CUDA/HIP/AMD: signature, __shared__/__syncthreads, AMD OCML intrinsics
  - PTX ALU/memory/shift/where/bool, WMMA (tensor cores)
  - LLVM IR ALU/memory/gep/casts/icmp/fcmp/control flow, AMD intrinsics (tensor cores), WMMA/MFMA, addrspace(3) locals
  - WGSL header prelude (f16/nan/INFINITY), atomics packed paths, select/where, bitcasts, workgroup_size from SPECIAL
- Normalization: trim whitespace differences while comparing

Additional WGSL tests added
- `tests/renderer/test_wgsl_packed.c` — packed i8/i16 load/store (atomic paths), masked bitcasts
- `tests/renderer/test_wgsl_gating.c` — gated LOAD/STORE (select/if)
- `tests/renderer/test_wgsl_locals.c` — var<workgroup>/var declarations (packed atomic<u32>), negative u32 const bitcast
- `tests/renderer/test_wgsl_nancheck.c` — NaN check lowering for `x != x`
- `tests/renderer/test_wgsl_workgroup_size.c` — `@workgroup_size` from SPECIAL
- `tests/renderer/test_wgsl_f16_header.c` — `enable f16;` header emission

Risks & Mitigations

- Pattern parity: ensure C UPat engine supports all constructs used (we already rely on named captures, dtype filters, allow_any_len in other modules). Add missing features before porting a backend relying on them.
- Float/half handling: faithfully replicate upcasts and literal formatting; add focused tests for these.
- Packed types (WGSL/OpenCL): atomics and bitcasts can be subtle; lock parity with small targeted tests.
- String diffs: Employ normalization and ensure predictable formatting (SSA naming order stable; deterministic buffer ordering).

Acceptance Criteria

- All planned backends render identical code (modulo whitespace) to the Python reference for the test corpus.
- ProgramSpec.estimates/launch_dims match Python.
- Device-integration tests continue to pass; no regressions in existing suites.
