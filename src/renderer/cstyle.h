#ifndef SRC_RENDERER_CSTYLE_H
#define SRC_RENDERER_CSTYLE_H

#include "renderer/renderer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Minimal C-style (Clang/CPU) renderer constructor
Renderer* renderer_cstyle_clang(void);

// Additional C-style backends (ported from reference Python)
Renderer* renderer_cstyle_opencl(void);
Renderer* renderer_cstyle_cuda(const char* arch);
Renderer* renderer_cstyle_amd(const char* arch);
Renderer* renderer_cstyle_hip(const char* arch);
Renderer* renderer_cstyle_metal(void);
Renderer* renderer_cstyle_qcom(void);

#ifdef __cplusplus
}
#endif
#endif /* SRC_RENDERER_CSTYLE_H */
