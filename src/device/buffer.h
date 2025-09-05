// Buffer API - stub for memory planning
#ifndef TINYGRAD_DEVICE_BUFFER_H
#define TINYGRAD_DEVICE_BUFFER_H

#include <stdbool.h>
#include <stddef.h>

// Forward declarations
typedef struct Buffer Buffer;
typedef struct Device Device;
typedef struct BufferSpec BufferSpec;

// Buffer functions needed by memory planner
Buffer* buffer_new(const char* device, size_t size, void* dtype);
Buffer* buffer_new_with_base(const char* device, size_t size, void* dtype, Buffer* base, int offset);
bool buffer_is_allocated(Buffer* buf);
Buffer* buffer_get_base(Buffer* buf);
int buffer_get_refcount(Buffer* buf);
const char* buffer_get_device(Buffer* buf);
size_t buffer_get_size(Buffer* buf);
size_t buffer_get_nbytes(Buffer* buf);
void* buffer_get_dtype(Buffer* buf);
void* buffer_get_options(Buffer* buf);

// Device functions needed by memory planner
Device* device_get(const char* device);
bool device_has_offset_allocator(Device* dev);

// DType functions
void* dtype_int8(void);
bool dtype_is_image(void* dtype);

#endif // TINYGRAD_DEVICE_BUFFER_H

// New faithful Buffer API (tg_* namespace) mirroring tinygrad.device.Buffer
#ifndef TINYGRAD_TG_BUFFER_H
#define TINYGRAD_TG_BUFFER_H

#include "dtype/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tg_buffer tg_buffer_t;

// BufferSpec equivalent
typedef struct tg_bufferspec {
  const ImageDType* image;  // optional image dtype
  int uncached;
  int cpu_access;
  int host;
  int nolru;
  void* external_ptr;       // optional external pointer
} tg_bufferspec_t;

// Simple memory view for as_buffer
typedef struct tg_memview {
  void* data;
  size_t nbytes;
} tg_memview_t;

// Creation / destruction
tg_buffer_t* tg_buffer_create(const char* device, size_t size, const DType* dtype,
                              const tg_bufferspec_t* options,
                              const void* initial_value, size_t initial_nbytes,
                              int uop_refcount, tg_buffer_t* base, size_t offset);
void tg_buffer_destroy(tg_buffer_t* buf);

// Allocation
int tg_buffer_allocate(tg_buffer_t* buf);
int tg_buffer_deallocate(tg_buffer_t* buf);

// Info
const char* tg_buffer_device(const tg_buffer_t* buf);
size_t tg_buffer_size(const tg_buffer_t* buf);
size_t tg_buffer_nbytes(const tg_buffer_t* buf);
const DType* tg_buffer_dtype(const tg_buffer_t* buf);
int tg_buffer_is_allocated(const tg_buffer_t* buf);

// Views
tg_buffer_t* tg_buffer_view(tg_buffer_t* base, size_t size, const DType* dtype, size_t offset);

// Copy in/out
int tg_buffer_copyin(tg_buffer_t* buf, const void* src, size_t nbytes);
int tg_buffer_copyout(const tg_buffer_t* buf, void* dst, size_t nbytes);

// As buffer
tg_memview_t tg_buffer_as_buffer(tg_buffer_t* buf, int allow_zero_copy, int force_zero_copy);

// Numpy compatibility: build a 1D np_array view of the buffer data
struct np_array; // forward declare
struct np_array* tg_buffer_numpy(tg_buffer_t* buf);

// DMARef support (CPU/FD)
typedef struct tg_dma_cpu_ref {
  uintptr_t addr;
  size_t size;
} tg_dma_cpu_ref_t;
#include <stdint.h>
tg_dma_cpu_ref_t tg_buffer_as_dmaref_cpu(tg_buffer_t* buf);

// File-descriptor based DMA reference (placeholder for future backends)
typedef struct tg_dma_fd_ref {
  int fd;
  size_t size;
  size_t offset;
} tg_dma_fd_ref_t;
// Not yet implemented; returns zeroed struct for unsupported devices
tg_dma_fd_ref_t tg_buffer_as_dmaref_fd(tg_buffer_t* buf);

#ifdef __cplusplus
}
#endif

#endif // TINYGRAD_TG_BUFFER_H
