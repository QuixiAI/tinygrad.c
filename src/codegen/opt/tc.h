#ifndef TINYGRAD_CODEGEN_OPT_TC_H
#define TINYGRAD_CODEGEN_OPT_TC_H

#include <stddef.h>
#include "dtype/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int count;
  const char* const* items;
} TCStrList;

typedef struct {
  TCStrList parts[3];
} TCSwizzlePart;

typedef struct TensorCore {
  int dims[3];
  int threads;
  int elements_per_thread[3];
  const DType* dtype_in;
  const DType* dtype_out;
  TCStrList opts;
  TCSwizzlePart swizzle[2];
} TensorCore;

extern const TensorCore tc_cuda_sm80[];
size_t tc_cuda_sm80_count(void);

extern const TensorCore tc_cuda_sm75[];
size_t tc_cuda_sm75_count(void);

extern const TensorCore tc_amd_rdna3[];
size_t tc_amd_rdna3_count(void);

extern const TensorCore tc_amd_rdna4[];
size_t tc_amd_rdna4_count(void);

extern const TensorCore tc_amd_cdna[];
size_t tc_amd_cdna_count(void);

extern const TensorCore tc_metal[];
size_t tc_metal_count(void);

extern const TensorCore tc_amx[];
size_t tc_amx_count(void);

extern const TensorCore tc_intel[];
size_t tc_intel_count(void);

#ifdef __cplusplus
}
#endif

#endif /* TINYGRAD_CODEGEN_OPT_TC_H */
