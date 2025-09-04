// Port of tinygrad/engine/jit.py - stub implementation
#include "jit.h"

// Global JIT flag
int JIT_ENABLED = 0;

bool jit_is_enabled(void) {
    return JIT_ENABLED != 0;
}

void jit_set_enabled(bool enabled) {
    JIT_ENABLED = enabled ? 1 : 0;
}