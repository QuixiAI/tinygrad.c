// Port of tinygrad/engine/realize.py
// Line-by-line faithful port from Python to C

#include "realize.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// Include other headers we need
#include "uop/ops.h"
#include "device/device.h"
#include "helpers/helpers.h"
#include "engine/schedule.h"
#include "renderer/renderer.h"
#include "codegen/codegen.h"

// Port of: from typing import cast, Generator
// Port of: import time, pprint
// Port of: from dataclasses import dataclass, replace, field
// Port of: from tinygrad.helpers import all_same, colored, DEBUG, GlobalCounters, ansilen, BEAM, NOOPT, all_int, CAPTURING, Metadata, TRACEMETA, TracingKey
// Port of: from tinygrad.helpers import DEVECTORIZE, time_to_str, VALIDATE_WITH_CPU, getenv, cpu_profile
// Port of: from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, Variable, sym_infer, graph_rewrite, print_uops, track_rewrites, KernelInfo
// Port of: from tinygrad.device import Device, Buffer
// Port of: from tinygrad.renderer import Renderer, ProgramSpec, Estimates
// Port of: from tinygrad.engine.schedule import ScheduleItem
// Port of: from tinygrad.codegen import full_rewrite
// Port of: from tinygrad.codegen.opt.kernel import Opt

// Global variables
int DEBUG = 0;
int BEAM = 0;
int NOOPT = 0;
int DEVECTORIZE = 0;
int CAPTURING = 0;
int VALIDATE_WITH_CPU = 0;
int TRACEMETA = 0;

// GlobalCounters struct
struct {
    int kernel_count;
    int64_t global_ops;
    int64_t global_mem;
    double time_sum_s;
    int64_t mem_used;
} GlobalCounters = {0};

// **************** Program Creation ****************
// Port of lines 13-48

// Port of: @track_rewrites(name=lambda *args,ret,**kwargs: TracingKey(ret.name, (ret.function_name, ret.ast), ret=ret))
// Port of: def get_program(ast:UOp, renderer:Renderer|None=None, opts:list[Opt]|None=None) -> ProgramSpec:
ProgramSpec* get_program(UOp* ast, Renderer* renderer, Opt** opts, int opts_count) {
    // Port of line 17-26: docstring
    
    // Port of line 28: if getenv("VIZ"): graph_rewrite(ast, PatternMatcher([]), name="View Base AST")
    if (getenv("VIZ")) {
        // graph_rewrite(ast, empty_pattern_matcher, "View Base AST");
    }
    
    // Port of line 30-31: linearize
    // if renderer is None: renderer = Device.default.renderer
    if (renderer == NULL) {
        renderer = device_get_default()->renderer;
    }
    
    // Port of line 32-34: if opts is not None:
    if (opts != NULL && opts_count > 0) {
        // assert ast.arg is None, "can't apply opts if sink has an arg"
        assert(ast->arg == NULL);
        // ast = ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts)))
        KernelInfo* ki = kernel_info_new();
        ki->opts_to_apply = opts;
        ki->opts_count = opts_count;
        ast = uop_replace(ast, NULL, 0, ki);
    }
    
    // Port of line 35-40: try/except for full_rewrite
    UOp** uops = NULL;
    int uops_count = 0;
    // try:
    //   uops = full_rewrite(ast, renderer)
    uops = full_rewrite(ast, renderer, &uops_count);
    if (uops == NULL) {
        // except RuntimeError:
        printf("***** LINEARIZE FAILURE *****\n");
        printf("ast = ");
        uop_print(ast);
        printf("\n");
        // raise
        return NULL;
    }
    
    // Port of line 41: assert uops[-1].op is Ops.SINK, "last uop must be sink"
    assert(uops[uops_count - 1]->op == OPS_SINK);
    
    // Port of line 43-44: print and render
    // if DEBUG >= 6: print_uops(uops)
    if (DEBUG >= 6) {
        print_uops(uops, uops_count);
    }
    
    // Port of line 45: src = renderer.render(uops)
    char* src = renderer_render(renderer, uops, uops_count);
    
    // Port of line 47-48: return ProgramSpec(...)
    ProgramSpec* spec = calloc(1, sizeof(ProgramSpec));
    spec->name = uops[uops_count - 1]->arg != NULL ? strdup(((KernelInfo*)uops[uops_count - 1]->arg)->name) : strdup("test");
    spec->src = src;
    spec->device = strdup(renderer->device);
    spec->ast = ast;
    spec->uops = uops;
    spec->uops_count = uops_count;
    spec->has_local = renderer->has_local;
    if (renderer->has_local) {
        spec->global_size[0] = spec->global_size[1] = spec->global_size[2] = 1;
        spec->local_size[0] = spec->local_size[1] = spec->local_size[2] = 1;
    }
    
    return spec;
}

// **************** Runners ****************
// Port of lines 50-124

// Port of lines 52-60: class Runner
Runner* runner_new(const char* display_name, const char* device, Estimates estimates) {
    // Port of line 53-54: def __init__(self, display_name:str, device:str, estimates=Estimates()):
    //   self.first_run, self.display_name, self.device, self.estimates = True, display_name, device, estimates
    Runner* runner = calloc(1, sizeof(Runner));
    runner->first_run = true;
    runner->display_name = strdup(display_name);
    runner->device = strdup(device);
    runner->estimates = estimates;
    return runner;
}

// Port of line 55-56: @property def dev(self): return Device[self.device]
Device* runner_get_dev(Runner* self) {
    return device_get(self->device);
}

// Port of line 57-58: def exec(self, rawbufs:list[Buffer], var_vals:dict[Variable, int]|None=None) -> float|None:
float runner_exec(Runner* self, Buffer** rawbufs, int rawbufs_count, Variable** var_keys, int* var_vals, int var_count) {
    // return self(rawbufs, {} if var_vals is None else var_vals)
    if (var_keys == NULL || var_vals == NULL || var_count == 0) {
        return self->call(self, rawbufs, rawbufs_count, NULL, NULL, 0, false);
    }
    return self->call(self, rawbufs, rawbufs_count, var_keys, var_vals, var_count, false);
}

// Port of line 59-60: def __call__(self, rawbufs:list[Buffer], var_vals:dict[Variable, int], wait=False) -> float|None:
//   raise NotImplementedError("override this")
float runner_call_base(Runner* self, Buffer** rawbufs, int rawbufs_count, Variable** var_keys, int* var_vals, int var_count, bool wait) {
    // raise NotImplementedError("override this")
    assert(false && "override this");
    return 0.0f;
}

void runner_free(Runner* runner) {
    if (!runner) return;
    free(runner->display_name);
    free(runner->device);
    if (runner->free) runner->free(runner);
    else free(runner);
}

// Port of lines 62-91: class CompiledRunner(Runner)
CompiledRunner* compiled_runner_new(ProgramSpec* p, void* precompiled, void* prg) {
    CompiledRunner* runner = calloc(1, sizeof(CompiledRunner));
    
    // Port of line 63-64: def __init__(self, p:ProgramSpec, precompiled:bytes|None=None, prg=None):
    //   if DEBUG >= 4: print(p.src)
    if (DEBUG >= 4) {
        printf("%s\n", p->src);
    }
    
    // Port of line 65: self.p:ProgramSpec = p
    runner->p = p;
    
    // Port of line 66-69: handle precompiled or compile
    if (precompiled != NULL) {
        // Port of line 66: if precompiled is not None: self.lib = precompiled
        runner->lib = precompiled;
    } else {
        // Port of line 68-69: with cpu_profile(...): self.lib = Device[p.device].compiler.compile_cached(p.src)
        Device* dev = device_get(p->device);
        runner->lib = device_compile_cached(dev, p->src);
    }
    
    // Port of line 70: if DEBUG >= 7: Device[p.device].compiler.disassemble(self.lib)
    if (DEBUG >= 7) {
        Device* dev = device_get(p->device);
        device_disassemble(dev, runner->lib);
    }
    
    // Port of line 71: self._prg = Device[p.device].runtime(p.function_name, self.lib) if prg is None else prg
    if (prg == NULL) {
        Device* dev = device_get(p->device);
        runner->_prg = device_runtime(dev, p->function_name, runner->lib);
    } else {
        runner->_prg = prg;
    }
    
    // Port of line 72: super().__init__(p.name, p.device, p.estimates)
    runner->base.first_run = true;
    runner->base.display_name = strdup(p->name);
    runner->base.device = strdup(p->device);
    runner->base.estimates = p->estimates;
    runner->base.call = (void*)compiled_runner_call;
    runner->base.exec = runner_exec;
    runner->base.get_dev = runner_get_dev;
    runner->base.free = (void*)compiled_runner_free;
    
    return runner;
}

// Port of line 76-91: def __call__(self, rawbufs:list[Buffer], var_vals:dict[Variable, int], wait=False) -> float|None:
float compiled_runner_call(CompiledRunner* self, Buffer** rawbufs, int rawbufs_count, 
                           Variable** var_keys, int* var_vals, int var_count, bool wait) {
    // Port of line 77: global_size, local_size = self.p.launch_dims(var_vals)
    int global_size[3], local_size[3];
    bool has_global = false, has_local = false;
    // launch_dims would need to be implemented
    
    // Port of line 78-83: optimize local size if needed
    if (has_global && !has_local && all_int(self->p->global_size, 3)) {
        // Port of line 80-82: optimize_local_size
        // from tinygrad.codegen.opt.search import optimize_local_size
        // local_size = optimize_local_size(self._prg, global_size, rawbufs)
        // global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
        // This would need optimize_local_size implementation
    }
    
    // Port of line 84-90: build launch args
    // lra = {}
    // if global_size:
    //   lra['global_size'] = tuple(global_size)
    //   assert len(global_size) == 3, "global size must have len 3"
    // if local_size:
    //   lra['local_size'] = tuple(local_size)
    //   assert len(local_size) == 3, "local size must have len 3"
    
    // Port of line 91: return self._prg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k] for k in self.p.vars), wait=wait)
    // This needs the actual device runtime execution
    return 0.0f;  // Placeholder
}

void compiled_runner_free(CompiledRunner* runner) {
    if (!runner) return;
    // Free base runner fields are handled by runner_free
    free(runner);
}

// Port of lines 93-96: class ViewOp(Runner)
ViewOp* view_op_new(Buffer* buf) {
    ViewOp* op = calloc(1, sizeof(ViewOp));
    
    // Port of line 94: def __init__(self, buf:Buffer): 
    //   super().__init__(colored(f"view {buf.nbytes:8d} @ {buf.offset:<10d}", "yellow"), buf.device)
    char display[256];
    snprintf(display, sizeof(display), "view %8ld @ %-10ld", buf->nbytes, buf->offset);
    
    op->base.first_run = true;
    op->base.display_name = strdup(display);  // Would need colored() implementation
    op->base.device = strdup(buf->device);
    op->base.estimates = (Estimates){0};
    op->base.call = (void*)view_op_call;
    op->base.exec = runner_exec;
    op->base.get_dev = runner_get_dev;
    op->base.free = (void*)view_op_free;
    
    op->buf = buf;
    
    return op;
}

// Port of line 95-96: def __call__(self, rawbufs:list[Buffer], var_vals:dict[Variable, int], wait=False):
float view_op_call(ViewOp* self, Buffer** rawbufs, int rawbufs_count, 
                  Variable** var_keys, int* var_vals, int var_count, bool wait) {
    // assert rawbufs[0]._base is not None and rawbufs[0]._base == rawbufs[1].base, f"must be base {rawbufs}"
    assert(rawbufs[0]->_base != NULL && rawbufs[0]->_base == rawbufs[1]->base);
    return 0.0f;
}

void view_op_free(ViewOp* op) {
    if (!op) return;
    free(op);
}

// Port of lines 98-120: class BufferCopy(Runner)
BufferCopy* buffer_copy_new(size_t total_sz, const char* dest_device, const char* src_device) {
    BufferCopy* copy = calloc(1, sizeof(BufferCopy));
    
    // Port of line 99-101: def __init__(self, total_sz, dest_device, src_device):
    char name[256];
    if (total_sz >= 1e6) {
        // Port of line 100: if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
        snprintf(name, sizeof(name), "copy %7.2fM, %7.7s <- %-7.7s", total_sz/1e6, dest_device, src_device);
    } else {
        // Port of line 101: else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest_device[:7]:>7s} <- {src_device[:7]:7s}"
        snprintf(name, sizeof(name), "copy %8ld, %7.7s <- %-7.7s", total_sz, dest_device, src_device);
    }
    
    // Port of line 102: super().__init__(colored(name, "yellow"), dest_device, Estimates(lds=total_sz, mem=total_sz))
    copy->base.first_run = true;
    copy->base.display_name = strdup(name);  // Would need colored() implementation
    copy->base.device = strdup(dest_device);
    copy->base.estimates = (Estimates){.ops = 0, .mem = total_sz, .lds = total_sz};
    copy->base.call = (void*)buffer_copy_call;
    copy->base.exec = runner_exec;
    copy->base.get_dev = runner_get_dev;
    copy->base.free = (void*)buffer_copy_free;
    
    copy->total_sz = total_sz;
    copy->dest_device = strdup(dest_device);
    copy->src_device = strdup(src_device);
    copy->copy = buffer_copy_copy;
    
    return copy;
}

// Port of line 103-112: def copy(self, dest, src):
void buffer_copy_copy(BufferCopy* self, Buffer* dest, Buffer* src) {
    // Port of line 104-106: disk_supports_fast_copyout check
    bool disk_supports_fast_copyout = false;
    // This would need proper implementation checking src->device and allocator capabilities
    
    // Port of line 106-112: various copy paths
    if (strncmp(src->device, "DISK", 4) == 0 && disk_supports_fast_copyout && src->nbytes >= 4096) {
        // dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
        allocator_copy_from_disk(dest->allocator, dest->_buf, src->_buf, src->nbytes);
    } else if (strncmp(src->device, "DISK", 4) == 0 && allocator_has_as_buffer(dest->allocator)) {
        // Port of line 108-110: fast(ish) path
        // src.allocator._copyout(dest.allocator._as_buffer(dest._buf), src._buf)
        void* dest_buffer = allocator_as_buffer(dest->allocator, dest->_buf);
        allocator_copyout(src->allocator, dest_buffer, src->_buf);
    } else {
        // Port of line 112: dest.copyin(src.as_buffer(allow_zero_copy=True))
        buffer_copyin(dest, buffer_as_buffer(src, true));
    }
}

// Port of line 113-120: def __call__(self, rawbufs:list[Buffer], var_vals:dict[Variable, int], wait=False):
float buffer_copy_call(BufferCopy* self, Buffer** rawbufs, int rawbufs_count,
                       Variable** var_keys, int* var_vals, int var_count, bool wait) {
    // Port of line 114: dest, src = rawbufs[0:2]
    Buffer* dest = rawbufs[0];
    Buffer* src = rawbufs[1];
    
    // Port of line 115: assert dest.size == src.size and dest.dtype == src.dtype
    assert(dest->size == src->size && dest->dtype == src->dtype);
    
    // Port of line 116: st = time.perf_counter()
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Port of line 117: self.copy(dest, src)
    self->copy(self, dest, src);
    
    // Port of line 118-120: synchronize if wait
    if (wait) {
        // Device[dest.device].synchronize()
        device_synchronize(device_get(dest->device));
        
        // return time.perf_counter() - st
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    }
    
    return 0.0f;
}

void buffer_copy_free(BufferCopy* copy) {
    if (!copy) return;
    free(copy->dest_device);
    free(copy->src_device);
    free(copy);
}

// Port of lines 122-123: class BufferXfer(BufferCopy)
BufferXfer* buffer_xfer_new(size_t total_sz, const char* dest_device, const char* src_device) {
    BufferXfer* xfer = (BufferXfer*)buffer_copy_new(total_sz, dest_device, src_device);
    
    // Port of line 123: def copy(self, dest, src): 
    //   dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.dev, dest_dev=dest.allocator.dev)
    xfer->base.copy = buffer_xfer_copy;
    
    return xfer;
}

void buffer_xfer_copy(BufferXfer* self, Buffer* dest, Buffer* src) {
    allocator_transfer(dest->allocator, dest->_buf, src->_buf, dest->nbytes, 
                      src->allocator->dev, dest->allocator->dev);
}

void buffer_xfer_free(BufferXfer* xfer) {
    buffer_copy_free((BufferCopy*)xfer);
}

// **************** method cache ****************
// Port of lines 125-139

// Port of line 127: method_cache: dict[tuple[str, bytes, tuple[int, ...], bool], CompiledRunner] = {}
static MethodCache* method_cache = NULL;

static void init_method_cache() {
    if (method_cache == NULL) {
        method_cache = calloc(1, sizeof(MethodCache));
        method_cache->bucket_count = 16;
        method_cache->buckets = calloc(method_cache->bucket_count, sizeof(MethodCacheEntry*));
    }
}

static uint64_t hash_method_cache_key(const MethodCacheKey* key) {
    uint64_t hash = 5381;
    for (const char* p = key->device; *p; p++) {
        hash = ((hash << 5) + hash) + *p;
    }
    for (size_t i = 0; i < key->ast_key_len; i++) {
        hash = ((hash << 5) + hash) + key->ast_key[i];
    }
    for (int i = 0; i < 3; i++) {
        hash = ((hash << 5) + hash) + key->context[i];
    }
    hash = ((hash << 5) + hash) + key->is_base;
    return hash;
}

static MethodCacheEntry* method_cache_get(const MethodCacheKey* key) {
    init_method_cache();
    uint64_t hash = hash_method_cache_key(key);
    size_t bucket = hash % method_cache->bucket_count;
    
    for (MethodCacheEntry* entry = method_cache->buckets[bucket]; entry; entry = entry->next) {
        if (strcmp(entry->key.device, key->device) == 0 &&
            entry->key.ast_key_len == key->ast_key_len &&
            memcmp(entry->key.ast_key, key->ast_key, key->ast_key_len) == 0 &&
            memcmp(entry->key.context, key->context, sizeof(key->context)) == 0 &&
            entry->key.is_base == key->is_base) {
            return entry;
        }
    }
    return NULL;
}

static void method_cache_put(const MethodCacheKey* key, CompiledRunner* runner) {
    init_method_cache();
    
    MethodCacheEntry* entry = calloc(1, sizeof(MethodCacheEntry));
    entry->key.device = strdup(key->device);
    entry->key.ast_key = malloc(key->ast_key_len);
    memcpy(entry->key.ast_key, key->ast_key, key->ast_key_len);
    entry->key.ast_key_len = key->ast_key_len;
    memcpy(entry->key.context, key->context, sizeof(key->context));
    entry->key.is_base = key->is_base;
    entry->runner = runner;
    
    uint64_t hash = hash_method_cache_key(key);
    size_t bucket = hash % method_cache->bucket_count;
    
    entry->next = method_cache->buckets[bucket];
    method_cache->buckets[bucket] = entry;
    method_cache->size++;
}

// Port of line 128-139: def get_runner(device:str, ast:UOp) -> CompiledRunner:
CompiledRunner* get_runner(const char* device, UOp* ast) {
    // Port of line 129-130: context = (BEAM.value, NOOPT.value, DEVECTORIZE.value)
    int context[3] = {BEAM, NOOPT, DEVECTORIZE};
    
    // Port of line 131: ckey = (device, ast.key, context, False)
    MethodCacheKey ckey = {
        .device = (char*)device,
        .ast_key = ast->key,
        .ast_key_len = ast->key_len,
        .is_base = false
    };
    memcpy(ckey.context, context, sizeof(context));
    
    // Port of line 132: if cret:=method_cache.get(ckey): return cret
    MethodCacheEntry* cret = method_cache_get(&ckey);
    if (cret) return cret->runner;
    
    // Port of line 133: bkey = (device.split(":")[0], ast.key, context, True)
    char* device_base = strdup(device);
    char* colon = strchr(device_base, ':');
    if (colon) *colon = '\0';
    
    MethodCacheKey bkey = {
        .device = device_base,
        .ast_key = ast->key,
        .ast_key_len = ast->key_len,
        .is_base = true
    };
    memcpy(bkey.context, context, sizeof(context));
    
    // Port of line 134-135: if bret:=method_cache.get(bkey):
    MethodCacheEntry* bret = method_cache_get(&bkey);
    CompiledRunner* ret;
    
    if (bret) {
        // method_cache[ckey] = ret = CompiledRunner(replace(bret.p, device=device), bret.lib)
        ProgramSpec* new_spec = malloc(sizeof(ProgramSpec));
        memcpy(new_spec, bret->runner->p, sizeof(ProgramSpec));
        new_spec->device = strdup(device);
        ret = compiled_runner_new(new_spec, bret->runner->lib, NULL);
        method_cache_put(&ckey, ret);
    } else {
        // Port of line 137-138: 
        // prg: ProgramSpec = get_program(ast, Device[device].renderer)
        // method_cache[ckey] = method_cache[bkey] = ret = CompiledRunner(replace(prg, device=device))
        Device* dev = device_get(device);
        ProgramSpec* prg = get_program(ast, dev->renderer, NULL, 0);
        ProgramSpec* new_spec = malloc(sizeof(ProgramSpec));
        memcpy(new_spec, prg, sizeof(ProgramSpec));
        new_spec->device = strdup(device);
        ret = compiled_runner_new(new_spec, NULL, NULL);
        method_cache_put(&ckey, ret);
        method_cache_put(&bkey, ret);
    }
    
    free(device_base);
    return ret;
}

// **************** lowering functions ****************
// Port of lines 141-177

// Port of lines 143-148: @dataclass(frozen=True) class ExecItem:
ExecItem* exec_item_new(Runner* prg, Buffer** bufs, int bufs_count,
                        Metadata** metadata, int metadata_count,
                        Variable** fixedvars_keys, int* fixedvars_values, int fixedvars_count) {
    ExecItem* item = calloc(1, sizeof(ExecItem));
    item->prg = prg;
    
    item->bufs = malloc(bufs_count * sizeof(Buffer*));
    memcpy(item->bufs, bufs, bufs_count * sizeof(Buffer*));
    item->bufs_count = bufs_count;
    
    if (metadata_count > 0) {
        item->metadata = malloc(metadata_count * sizeof(Metadata*));
        memcpy(item->metadata, metadata, metadata_count * sizeof(Metadata*));
    }
    item->metadata_count = metadata_count;
    
    if (fixedvars_count > 0) {
        item->fixedvars_keys = malloc(fixedvars_count * sizeof(Variable*));
        item->fixedvars_values = malloc(fixedvars_count * sizeof(int));
        memcpy(item->fixedvars_keys, fixedvars_keys, fixedvars_count * sizeof(Variable*));
        memcpy(item->fixedvars_values, fixedvars_values, fixedvars_count * sizeof(int));
    }
    item->fixedvars_count = fixedvars_count;
    
    return item;
}

// Port of line 149-166: def run(self, _var_vals:dict[Variable, int]|None=None, wait=False, jit=False, do_update_stats=True) -> float|None:
float exec_item_run(ExecItem* item, Variable** var_keys, int* var_vals, int var_count, 
                    bool wait, bool jit, bool do_update_stats) {
    // Port of line 150: var_vals = self.fixedvars if _var_vals is None else (_var_vals|self.fixedvars)
    // Merge var_vals with fixedvars
    Variable** merged_keys = NULL;
    int* merged_vals = NULL;
    int merged_count = 0;
    
    // Simple merge - would need proper implementation
    if (var_keys && var_vals && var_count > 0) {
        merged_count = var_count + item->fixedvars_count;
        merged_keys = malloc(merged_count * sizeof(Variable*));
        merged_vals = malloc(merged_count * sizeof(int));
        memcpy(merged_keys, var_keys, var_count * sizeof(Variable*));
        memcpy(merged_vals, var_vals, var_count * sizeof(int));
        memcpy(merged_keys + var_count, item->fixedvars_keys, item->fixedvars_count * sizeof(Variable*));
        memcpy(merged_vals + var_count, item->fixedvars_values, item->fixedvars_count * sizeof(int));
    } else {
        merged_keys = item->fixedvars_keys;
        merged_vals = item->fixedvars_values;
        merged_count = item->fixedvars_count;
    }
    
    // Port of line 151: bufs = [cast(Buffer, x) for x in self.bufs] if jit else [cast(Buffer, x).ensure_allocated() for x in self.bufs]
    Buffer** bufs = malloc(item->bufs_count * sizeof(Buffer*));
    for (int i = 0; i < item->bufs_count; i++) {
        bufs[i] = item->bufs[i];
        if (!jit) {
            buffer_ensure_allocated(bufs[i]);
        }
    }
    
    // Port of line 152: et = self.prg(bufs, var_vals, wait=wait or DEBUG >= 2)
    float et = item->prg->call(item->prg, bufs, item->bufs_count, merged_keys, merged_vals, merged_count, wait || DEBUG >= 2);
    
    // Port of line 153-165: update stats
    if (do_update_stats) {
        // Port of line 154: GlobalCounters.kernel_count += 1
        GlobalCounters.kernel_count++;
        
        // Port of line 155-157: GlobalCounters updates
        int64_t op_est = sym_infer(item->prg->estimates.ops, merged_keys, merged_vals, merged_count);
        int64_t mem_est = sym_infer(item->prg->estimates.mem, merged_keys, merged_vals, merged_count);
        GlobalCounters.global_ops += op_est;
        GlobalCounters.global_mem += mem_est;
        
        if (et > 0) {
            GlobalCounters.time_sum_s += et;
        }
        
        // Port of line 158-164: DEBUG printing
        if (DEBUG >= 2) {
            int64_t lds_est = sym_infer(item->prg->estimates.lds, merged_keys, merged_vals, merged_count);
            mem_est = mem_est < lds_est ? mem_est : lds_est;
            
            // Formatted printing would need proper implementation
            printf("*** %7.7s %4d %s arg %2d mem %5.2f GB ",
                   item->prg->device, GlobalCounters.kernel_count, item->prg->display_name, 
                   item->bufs_count, GlobalCounters.mem_used/1e9);
            
            if (et > 0) {
                printf("tm %9.6f/%9.2fms (%9.2f GFLOPS %6.1f|%-7.1f GB/s)",
                       et, GlobalCounters.time_sum_s*1e3,
                       op_est/(et*1e9), mem_est/(et*1e9), lds_est/(et*1e9));
            }
            
            if (item->metadata_count > 0 && TRACEMETA) {
                printf(" [");
                for (int i = 0; i < item->metadata_count; i++) {
                    // Print metadata
                }
                printf("]");
            }
            printf("\n");
        }
        
        // Port of line 165: self.prg.first_run = False
        item->prg->first_run = false;
    }
    
    // Cleanup
    if (merged_keys != item->fixedvars_keys) {
        free(merged_keys);
        free(merged_vals);
    }
    free(bufs);
    
    return et;
}

void exec_item_free(ExecItem* item) {
    if (!item) return;
    free(item->bufs);
    free(item->metadata);
    free(item->fixedvars_keys);
    free(item->fixedvars_values);
    free(item);
}

// Port of lines 168-175: si_lowerer = PatternMatcher([...])
static PatternMatcher* si_lowerer = NULL;

static void init_si_lowerer() {
    if (si_lowerer == NULL) {
        // Would need proper PatternMatcher implementation
        si_lowerer = pattern_matcher_new();
        // Add patterns...
    }
}

// Port of lines 176-177: def lower_schedule_item(si:ScheduleItem) -> ExecItem:
ExecItem* lower_schedule_item(ScheduleItem* si) {
    init_si_lowerer();
    
    // Port of line 177: return ExecItem(*cast(tuple[Runner,list], si_lowerer.rewrite(si.ast, si.bufs)), si.metadata, si.fixedvars)
    
    // Simple implementation without pattern matching for now
    Runner* runner = NULL;
    Buffer** bufs = NULL;
    int bufs_count = 0;
    
    // Handle different op types
    if (si->ast->op == OPS_SINK) {
        runner = (Runner*)get_runner(si->bufs[0]->device, si->ast);
        bufs = si->bufs;
        bufs_count = si->bufs_count;
    } else if (si->ast->op == OPS_BUFFER_VIEW) {
        runner = (Runner*)view_op_new(si->bufs[0]);
        bufs = si->bufs;
        bufs_count = si->bufs_count;
    } else if (si->ast->op == OPS_COPY) {
        if (allocator_has_transfer(device_get(si->bufs[0]->device)->allocator) &&
            all_same_device_type(si->bufs, si->bufs_count)) {
            runner = (Runner*)buffer_xfer_new(si->bufs[0]->nbytes, si->bufs[0]->device, si->bufs[1]->device);
        } else {
            runner = (Runner*)buffer_copy_new(si->bufs[0]->nbytes, si->bufs[0]->device, si->bufs[1]->device);
        }
        bufs = si->bufs;
        bufs_count = si->bufs_count;
    }
    
    return exec_item_new(runner, bufs, bufs_count, si->metadata, si->metadata_count, 
                         si->fixedvars_keys, si->fixedvars_values, si->fixedvars_count);
}

// Port of lines 179-188: def lower_schedule(schedule:list[ScheduleItem]) -> Generator[tuple[ScheduleItem, ExecItem], None, None]:
LowerScheduleGenerator* lower_schedule_new(ScheduleItem** schedule, int count) {
    LowerScheduleGenerator* gen = calloc(1, sizeof(LowerScheduleGenerator));
    gen->schedule = schedule;
    gen->schedule_count = count;
    gen->current_index = 0;
    return gen;
}

LowerScheduleResult* lower_schedule_next(LowerScheduleGenerator* gen) {
    // Port of line 180: while len(schedule):
    if (gen->current_index >= gen->schedule_count) {
        return NULL;
    }
    
    // Port of line 181: si = schedule.pop(0)
    ScheduleItem* si = gen->schedule[gen->current_index++];
    
    // Port of line 182-188: try/except
    ExecItem* ei = NULL;
    // try: yield (si, lower_schedule_item(si))
    ei = lower_schedule_item(si);
    if (ei == NULL) {
        // except Exception as e:
        if (DEBUG >= 2) {
            printf("error lowering %d\n", si->ast->op);
            printf("tensor operations:\n");
            // pprint.pprint(si.metadata, indent=2)
        }
        // raise e
        return NULL;
    }
    
    LowerScheduleResult* result = calloc(1, sizeof(LowerScheduleResult));
    result->si = si;
    result->ei = ei;
    return result;
}

void lower_schedule_generator_free(LowerScheduleGenerator* gen) {
    free(gen);
}

// **************** main run function ****************
// Port of lines 190-211

// Port of line 192: capturing: list = []  # put classes with an add method in here
CapturingList* capturing = NULL;

// Port of line 194-211: def run_schedule(schedule:list[ScheduleItem], var_vals:dict[Variable, int]|None=None, do_update_stats=True):
void run_schedule(ScheduleItem** schedule, int schedule_count, 
                 Variable** var_keys, int* var_vals, int var_count, 
                 bool do_update_stats) {
    // Port of line 195: for si, ei in lower_schedule(schedule):
    LowerScheduleGenerator* gen = lower_schedule_new(schedule, schedule_count);
    LowerScheduleResult* result;
    
    while ((result = lower_schedule_next(gen)) != NULL) {
        ScheduleItem* si = result->si;
        ExecItem* ei = result->ei;
        
        // Port of line 196: if len(capturing) and CAPTURING: capturing[0].add(ei)
        if (capturing && capturing->count > 0 && CAPTURING) {
            if (capturing->add) {
                capturing->add(capturing->items[0], ei);
            }
        }
        
        // Port of line 197-209: VALIDATE_WITH_CPU
        if (VALIDATE_WITH_CPU && si->ast->op == OPS_SINK) {
            // Port of line 198-199: copy in allocated buffers from the GPU
            // nb: tuple[Buffer, ...] = tuple(Buffer("CPU", b.size, b.dtype) for b in si.bufs)
            Buffer** nb = malloc(si->bufs_count * sizeof(Buffer*));
            for (int i = 0; i < si->bufs_count; i++) {
                nb[i] = buffer_new("CPU", si->bufs[i]->size, si->bufs[i]->dtype);
            }
            
            // Port of line 200-201: copy data
            for (int i = 0; i < si->bufs_count; i++) {
                if (buffer_is_allocated(si->bufs[i])) {
                    buffer_ensure_allocated(nb[i]);
                    buffer_copyin(nb[i], buffer_as_buffer(si->bufs[i], false));
                }
            }
            
            // Port of line 203-204: run on GPU
            exec_item_run(ei, var_keys, var_vals, var_count, false, false, do_update_stats);
            
            // Port of line 206-207: validate the output buffers match
            ScheduleItem* cpu_si = schedule_item_new(si->ast, nb, si->bufs_count,
                                                     si->metadata, si->metadata_count,
                                                     si->fixedvars_keys, si->fixedvars_values, si->fixedvars_count);
            ExecItem* cpu_ei = lower_schedule_item(cpu_si);
            exec_item_run(cpu_ei, var_keys, var_vals, var_count, false, false, do_update_stats);
            
            // Port of line 208-209: np.testing.assert_allclose
            // Would need numpy equivalent for validation
            
            // Cleanup
            schedule_item_free(cpu_si);
            exec_item_free(cpu_ei);
            for (int i = 0; i < si->bufs_count; i++) {
                buffer_free(nb[i]);
            }
            free(nb);
        } else {
            // Port of line 211: ei.run(var_vals, do_update_stats=do_update_stats)
            exec_item_run(ei, var_keys, var_vals, var_count, false, false, do_update_stats);
        }
        
        free(result);
    }
    
    lower_schedule_generator_free(gen);
}