#ifndef UPAT_H
#define UPAT_H

#include "uop/ops.h"
#include <stdbool.h>

// Forward declarations
typedef struct CompiledMatch CompiledMatch;
typedef struct CodeGenContext CodeGenContext;

// Exception type for compile errors
typedef struct {
    char message[256];
} UPatCompileError;

// Code generation context
struct CodeGenContext {
    char** code_lines;
    int code_count;
    int code_capacity;
    void** dyn_lookup;  // Dynamic lookup table
    int dyn_count;
    bool has_ctx;
};

// Compiled match function type
typedef UOp* (*CompiledMatchFunc)(UOp* uop, void* ctx);

// Compiled match result
struct CompiledMatch {
    CompiledMatchFunc func;
    char* code_str;
    void** dyn_lookup;
    int dyn_count;
};

// **** UPat compilation functions ****

// Line 10-43: Get clause from UPat pattern
UOp* upat_get_clause(UPat* self, UOp* base, int depth);

// Line 47-92: Process AND operations
UOp* do_process_and(UOp* a);

// Line 98-101: Wrap function for renderer
UOp* wrap(CodeGenContext* ctx, UOp* x);

// Line 116-137: Final render of compiled pattern
char** upat_final_render(UOp* x, bool has_ctx, int depth, int* line_count);

// Line 139-151: Get compiled code for pattern
CompiledMatch* upat_get_code(UPat* self, bool has_ctx);

// Line 154-163: Main compile function
CompiledMatchFunc upat_compile(UPat* self, void* fxn);

// Pattern matcher instances (Line 95, 102-114)
extern PatternMatcher* pm_proc;
extern PatternMatcher* pm_renderer;

// Helper functions
void code_gen_init(CodeGenContext* ctx, bool has_ctx);
void code_gen_add_line(CodeGenContext* ctx, const char* format, ...);
void code_gen_cleanup(CodeGenContext* ctx);
char* code_gen_join(CodeGenContext* ctx);

// String helpers for code generation
char* format_custom_arg(const char* format, char** args, int arg_count);
bool is_and_op(UOp* x);
bool is_or_op(UOp* x);
bool is_store_op(UOp* x);
bool is_noop_op(UOp* x);

#endif // UPAT_H