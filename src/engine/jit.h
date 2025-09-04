// Port of tinygrad/engine/jit.py - stub for now
#ifndef TINYGRAD_ENGINE_JIT_H
#define TINYGRAD_ENGINE_JIT_H

#include <stdbool.h>

// JIT is complex and not needed for gradient tests
// This is a minimal stub

typedef struct TinyJit TinyJit;
typedef struct CapturedJit CapturedJit;
typedef struct GraphRunner GraphRunner;

// Global JIT flag
extern int JIT_ENABLED;

// Stub functions
bool jit_is_enabled(void);
void jit_set_enabled(bool enabled);

#endif // TINYGRAD_ENGINE_JIT_H