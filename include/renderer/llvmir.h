#ifndef SRC_RENDERER_LLVMIR_H
#define SRC_RENDERER_LLVMIR_H

#include "renderer/renderer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Minimal LLVM IR renderers
Renderer* renderer_llvm_generic(void);
Renderer* renderer_llvm_amd(const char* arch);

#ifdef __cplusplus
}
#endif
#endif /* SRC_RENDERER_LLVMIR_H */

