#ifndef SRC_RENDERER_PTX_H
#define SRC_RENDERER_PTX_H

#include "renderer/renderer.h"

#ifdef __cplusplus
extern "C" {
#endif

Renderer* renderer_ptx(const char* arch, const char* device);

#ifdef __cplusplus
}
#endif
#endif /* SRC_RENDERER_PTX_H */

