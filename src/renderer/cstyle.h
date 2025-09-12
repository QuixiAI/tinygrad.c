#ifndef SRC_RENDERER_CSTYLE_H
#define SRC_RENDERER_CSTYLE_H

#include "renderer/renderer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Minimal C-style (Clang/CPU) renderer constructor
Renderer* renderer_cstyle_clang(void);

#ifdef __cplusplus
}
#endif
#endif /* SRC_RENDERER_CSTYLE_H */
