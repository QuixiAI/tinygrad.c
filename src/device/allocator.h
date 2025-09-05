#ifndef TINYGRAD_TG_ALLOCATOR_H
#define TINYGRAD_TG_ALLOCATOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tg_allocator tg_allocator_t;
typedef struct tg_bufferspec tg_bufferspec_t; // forward declare
typedef struct tg_dma_cpu_ref tg_dma_cpu_ref_t; // forward declare

struct tg_allocator {
  void* (*alloc)(size_t nbytes, const tg_bufferspec_t* opts);
  void  (*free)(void* opaque, const tg_bufferspec_t* opts);
  // Optional size-aware free for LRU
  void  (*free_sized)(void* opaque, size_t nbytes, const tg_bufferspec_t* opts);
  int   (*copyin)(void* dst_opaque, const void* src, size_t nbytes);
  int   (*copyout)(void* dst, const void* src_opaque, size_t nbytes);
  // Optional: direct device-to-device transfer (tinygrad allocator._transfer)
  int   (*transfer)(void* dst_opaque, const char* dst_device,
                    const void* src_opaque, const char* src_device,
                    size_t nbytes);
  // Optional: expose a CPU DMA reference
  int   (*as_dmaref_cpu)(const void* opaque, size_t nbytes, tg_dma_cpu_ref_t* out);
  /* reserved for future: as_buffer, offset */
};

// Returns a default allocator for a device (currently malloc-based CPU)
const tg_allocator_t* tg_get_default_allocator(const char* device);

// Convenience wrappers (align with engine expectations)
int allocator_has_transfer(const tg_allocator_t* a);
int allocator_transfer(const tg_allocator_t* a, void* dst_opaque, const void* src_opaque, size_t nbytes,
                       const char* src_dev, const char* dst_dev);
int allocator_has_as_buffer(const tg_allocator_t* a);
void* allocator_as_buffer(const tg_allocator_t* a, void* opaque);
int allocator_copyout(const tg_allocator_t* a, void* dst, const void* src_opaque);
int allocator_copyin(const tg_allocator_t* a, void* dst_opaque, const void* src);

// LRU cache controls (optional bounds)
void tg_allocator_set_cache_cap(size_t cap_bytes);
size_t tg_allocator_get_cache_cap(void);
size_t tg_allocator_get_cached_bytes(void);

#ifdef __cplusplus
}
#endif

#endif // TINYGRAD_TG_ALLOCATOR_H
