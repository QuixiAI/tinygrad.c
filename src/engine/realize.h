// Port of tinygrad/engine/realize.py
#ifndef TINYGRAD_ENGINE_REALIZE_H
#define TINYGRAD_ENGINE_REALIZE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "renderer/renderer.h"

// Forward declarations
typedef struct UOp UOp;
typedef struct Variable Variable;
typedef struct Buffer Buffer;
typedef struct Metadata Metadata;
typedef struct Device Device;
typedef struct ScheduleItem ScheduleItem;
typedef struct Opt Opt;
// Minimal KernelInfo used by renderer to name functions
typedef struct KernelInfo {
    // Display name (unsanitized). Used to derive function_name
    char* name;
    // Cached/sanitized function symbol (renderer_to_function_name(name))
    char* function_name;
    // Optional: placeholder for opts_to_apply parity (not used yet)
    void* opts_to_apply;
    int opts_count;
} KernelInfo;

// Helpers to create/destroy KernelInfo
KernelInfo* kernel_info_new(void);
void kernel_info_free(KernelInfo* ki);
typedef struct PatternMatcher PatternMatcher;

// **************** Program Creation ****************
// Port of lines 13-48

// Port of: @track_rewrites(name=lambda *args,ret,**kwargs: TracingKey(ret.name, (ret.function_name, ret.ast), ret=ret))
// Port of: def get_program(ast:UOp, renderer:Renderer|None=None, opts:list[Opt]|None=None) -> ProgramSpec:
ProgramSpec* get_program(UOp* ast, Renderer* renderer, Opt** opts, int opts_count);

// **************** Runners ****************
// Port of lines 50-124

// Port of: class Runner:
typedef struct Runner {
    bool first_run;
    char* display_name;
    char* device;
    Estimates estimates;
    
    // Virtual function pointers
    float (*exec)(struct Runner* self, Buffer** rawbufs, int rawbufs_count, Variable** var_keys, int* var_vals, int var_count);
    float (*call)(struct Runner* self, Buffer** rawbufs, int rawbufs_count, Variable** var_keys, int* var_vals, int var_count, bool wait);
    Device* (*get_dev)(struct Runner* self);
    void (*free)(struct Runner* self);
} Runner;

// Port of: class CompiledRunner(Runner):
typedef struct CompiledRunner {
    Runner base;  // Inheritance
    ProgramSpec* p;
    void* lib;  // Compiled library
    void* _prg;  // Program runtime
} CompiledRunner;

// Port of: class ViewOp(Runner):
typedef struct ViewOp {
    Runner base;  // Inheritance
    Buffer* buf;
} ViewOp;

// Port of: class BufferCopy(Runner):
typedef struct BufferCopy {
    Runner base;  // Inheritance
    size_t total_sz;
    char* dest_device;
    char* src_device;
    
    // Virtual function for copy
    void (*copy)(struct BufferCopy* self, Buffer* dest, Buffer* src);
} BufferCopy;

// Port of: class BufferXfer(BufferCopy):
typedef struct BufferXfer {
    BufferCopy base;  // Inheritance
} BufferXfer;

// Runner constructors
// Port of lines 52-60
Runner* runner_new(const char* display_name, const char* device, Estimates estimates);
void runner_free(Runner* runner);

// Port of lines 62-91
CompiledRunner* compiled_runner_new(ProgramSpec* p, void* precompiled, void* prg);
void compiled_runner_free(CompiledRunner* runner);

// Port of lines 93-96
ViewOp* view_op_new(Buffer* buf);
void view_op_free(ViewOp* op);

// Port of lines 98-120
BufferCopy* buffer_copy_new(size_t total_sz, const char* dest_device, const char* src_device);
void buffer_copy_free(BufferCopy* copy);

// Port of lines 122-123
BufferXfer* buffer_xfer_new(size_t total_sz, const char* dest_device, const char* src_device);
void buffer_xfer_free(BufferXfer* xfer);

// **************** method cache ****************
// Port of lines 125-139

// Port of: method_cache: dict[tuple[str, bytes, tuple[int, ...], bool], CompiledRunner] = {}
typedef struct MethodCacheKey {
    char* device;
    uint8_t* ast_key;
    size_t ast_key_len;
    int context[3];  // BEAM, NOOPT, DEVECTORIZE
    bool is_base;
} MethodCacheKey;

typedef struct MethodCacheEntry {
    MethodCacheKey key;
    CompiledRunner* runner;
    struct MethodCacheEntry* next;
} MethodCacheEntry;

typedef struct MethodCache {
    MethodCacheEntry** buckets;
    size_t bucket_count;
    size_t size;
} MethodCache;

// Port of: def get_runner(device:str, ast:UOp) -> CompiledRunner:
CompiledRunner* get_runner(const char* device, UOp* ast);

// **************** lowering functions ****************
// Port of lines 141-177

// Port of: @dataclass(frozen=True)
// Port of: class ExecItem:
typedef struct ExecItem {
    Runner* prg;
    Buffer** bufs;
    int bufs_count;
    Metadata** metadata;
    int metadata_count;
    // Port of: fixedvars: dict[Variable, int] = field(default_factory=dict)
    Variable** fixedvars_keys;
    int* fixedvars_values;
    int fixedvars_count;
} ExecItem;

// Port of line 149-166: def run(self, _var_vals:dict[Variable, int]|None=None, wait=False, jit=False, do_update_stats=True) -> float|None:
float exec_item_run(ExecItem* item, Variable** var_keys, int* var_vals, int var_count, 
                    bool wait, bool jit, bool do_update_stats);

// Port of lines 168-175: si_lowerer = PatternMatcher([...])
// Port of lines 176-177: def lower_schedule_item(si:ScheduleItem) -> ExecItem:
ExecItem* lower_schedule_item(ScheduleItem* si);

// Port of lines 179-188: def lower_schedule(schedule:list[ScheduleItem]) -> Generator[tuple[ScheduleItem, ExecItem], None, None]:
typedef struct LowerScheduleResult {
    ScheduleItem* si;
    ExecItem* ei;
} LowerScheduleResult;

typedef struct LowerScheduleGenerator {
    ScheduleItem** schedule;
    int schedule_count;
    int current_index;
} LowerScheduleGenerator;

LowerScheduleGenerator* lower_schedule_new(ScheduleItem** schedule, int count);
LowerScheduleResult* lower_schedule_next(LowerScheduleGenerator* gen);
void lower_schedule_generator_free(LowerScheduleGenerator* gen);

// **************** main run function ****************
// Port of lines 190-211

// Port of line 192: capturing: list = []  # put classes with an add method in here
typedef struct CapturingList {
    void** items;
    int count;
    int capacity;
    void (*add)(void* item, ExecItem* ei);
} CapturingList;

extern CapturingList* capturing;

// Port of line 194-211: def run_schedule(schedule:list[ScheduleItem], var_vals:dict[Variable, int]|None=None, do_update_stats=True):
void run_schedule(ScheduleItem** schedule, int schedule_count, 
                 Variable** var_keys, int* var_vals, int var_count, 
                 bool do_update_stats);

// Helper functions
ExecItem* exec_item_new(Runner* prg, Buffer** bufs, int bufs_count,
                        Metadata** metadata, int metadata_count,
                        Variable** fixedvars_keys, int* fixedvars_values, int fixedvars_count);
void exec_item_free(ExecItem* item);

#endif // TINYGRAD_ENGINE_REALIZE_H
