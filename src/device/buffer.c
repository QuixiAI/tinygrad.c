// Buffer API - stub implementation for memory planning
#include "buffer.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "device/allocator.h"
#include "device/device.h"
// Avoid heavy include; forward declare minimal numpy_compat API
typedef struct np_array np_array_t;
extern np_array_t* np_frombuffer(void* buffer, size_t size, const DType* dtype);

// Stub structures - these would be replaced with real implementations
struct Buffer {
    char* device;
    size_t size;
    void* dtype;
    void* options;
    Buffer* base;
    int offset;
    bool allocated;
    int uop_refcount;
};

struct Device {
    char* name;
    bool has_offset;
};

// Stub implementations
Buffer* buffer_new(const char* device, size_t size, void* dtype) {
    Buffer* buf = calloc(1, sizeof(Buffer));
    buf->device = strdup(device);
    buf->size = size;
    buf->dtype = dtype;
    buf->base = NULL;
    buf->offset = 0;
    buf->allocated = false;
    buf->uop_refcount = 0;
    return buf;
}

Buffer* buffer_new_with_base(const char* device, size_t size, void* dtype, Buffer* base, int offset) {
    Buffer* buf = buffer_new(device, size, dtype);
    buf->base = base;
    buf->offset = offset;
    return buf;
}

bool buffer_is_allocated(Buffer* buf) {
    return buf ? buf->allocated : false;
}

Buffer* buffer_get_base(Buffer* buf) {
    return buf ? buf->base : NULL;
}

int buffer_get_refcount(Buffer* buf) {
    return buf ? buf->uop_refcount : 0;
}

const char* buffer_get_device(Buffer* buf) {
    return buf ? buf->device : NULL;
}

size_t buffer_get_size(Buffer* buf) {
    return buf ? buf->size : 0;
}

size_t buffer_get_nbytes(Buffer* buf) {
    // Assuming itemsize of 1 for now
    return buf ? buf->size : 0;
}

void* buffer_get_dtype(Buffer* buf) {
    return buf ? buf->dtype : NULL;
}

void* buffer_get_options(Buffer* buf) {
    return buf ? buf->options : NULL;
}

Device* device_get(const char* device) {
    Device* dev = calloc(1, sizeof(Device));
    dev->name = strdup(device);
    // Stub: assume CPU doesn't have offset, GPU does
    dev->has_offset = (strstr(device, "GPU") != NULL);
    return dev;
}

bool device_has_offset_allocator(Device* dev) {
    return dev ? dev->has_offset : false;
}

// DType stubs
static struct { int dummy; } int8_dtype = {0};

void* dtype_int8(void) {
    return &int8_dtype;
}

bool dtype_is_image(void* dtype) {
    // Stub: no image dtypes for now
    return false;
}

// ===== New faithful Buffer implementation (minimal backend-agnostic) =====
struct tg_buffer {
  char* device;
  size_t size;
  const DType* dtype;
  tg_bufferspec_t opts;
  void* data;           // backing store (malloc) when allocated and not external
  tg_buffer_t* base;
  size_t offset;        // byte offset into base
  int uop_refcount;
  int allocated;
  int owns_data;        // whether this buffer owns data
  const tg_allocator_t* allocator;
  size_t alloc_nbytes;  // size of allocation for LRU return
};

static size_t _dtype_itemsize(const DType* dt){ return (size_t)(dt ? dt->itemsize : 1); }

tg_buffer_t* tg_buffer_create(const char* device, size_t size, const DType* dtype,
                              const tg_bufferspec_t* options,
                              const void* initial_value, size_t initial_nbytes,
                              int uop_refcount, tg_buffer_t* base, size_t offset){
  tg_buffer_t* b = (tg_buffer_t*)calloc(1, sizeof(tg_buffer_t));
  if (!b) return NULL;
  b->device = device? strdup(device) : strdup("CPU");
  b->size = size;
  b->dtype = dtype? dtype : &dtypes.uint8;
  if (options) b->opts = *options; else memset(&b->opts, 0, sizeof(b->opts));
  b->base = base; b->offset = offset; b->uop_refcount = uop_refcount;
  b->allocated = 0; b->data=NULL; b->owns_data=0;
  b->allocator = tg_device_get_allocator(b->device);
  b->alloc_nbytes = 0;
  if (initial_value && initial_nbytes>0){
    // allocate and initialize now
    if (tg_buffer_allocate(b)==0){
      size_t nb = tg_buffer_nbytes(b);
      memcpy(b->data, initial_value, initial_nbytes<nb? initial_nbytes: nb);
    }
  }
  return b;
}

void tg_buffer_destroy(tg_buffer_t* buf){ if(!buf) return; if(buf->allocated && buf->owns_data && buf->data){ if (buf->allocator && buf->allocator->free_sized) buf->allocator->free_sized(buf->data, buf->alloc_nbytes, &buf->opts); else if (buf->allocator && buf->allocator->free) buf->allocator->free(buf->data, &buf->opts); else free(buf->data);} free(buf->device); free(buf); }

int tg_buffer_allocate(tg_buffer_t* buf){
  if (!buf) return -1;
  if (buf->allocated) return 0;
  if (buf->opts.external_ptr){ buf->data = buf->opts.external_ptr; buf->owns_data=0; buf->allocated=1; return 0; }
  size_t nb = tg_buffer_nbytes(buf);
  if (buf->allocator && buf->allocator->alloc) buf->data = buf->allocator->alloc(nb, &buf->opts);
  else buf->data = malloc(nb);
  if (!buf->data) return -1;
  buf->owns_data=1; buf->allocated=1; buf->alloc_nbytes=nb; return 0;
}

int tg_buffer_deallocate(tg_buffer_t* buf){ if(!buf) return -1; if(!buf->allocated) return 0; if(buf->owns_data && buf->data){ if (buf->allocator && buf->allocator->free_sized) buf->allocator->free_sized(buf->data, buf->alloc_nbytes, &buf->opts); else if (buf->allocator && buf->allocator->free) buf->allocator->free(buf->data, &buf->opts); else free(buf->data);} buf->data=NULL; buf->allocated=0; buf->owns_data=0; buf->alloc_nbytes=0; return 0; }

const char* tg_buffer_device(const tg_buffer_t* buf){ return buf? buf->device: NULL; }
size_t tg_buffer_size(const tg_buffer_t* buf){ return buf? buf->size: 0; }
size_t tg_buffer_nbytes(const tg_buffer_t* buf){ if(!buf) return 0; return buf->size * _dtype_itemsize(buf->dtype); }
const DType* tg_buffer_dtype(const tg_buffer_t* buf){ return buf? buf->dtype: NULL; }
int tg_buffer_is_allocated(const tg_buffer_t* buf){ return buf? buf->allocated: 0; }

tg_buffer_t* tg_buffer_view(tg_buffer_t* base, size_t size, const DType* dtype, size_t offset){
  assert(base);
  assert(offset < (base->size * _dtype_itemsize(base->dtype)));
  tg_bufferspec_t opts = base->opts; // inherit options
  // Note: for views we don't allocate; point to base data with offset at copy time
  return tg_buffer_create(base->device, size, dtype? dtype: base->dtype, &opts, NULL, 0, base->uop_refcount, base, offset);
}

int tg_buffer_copyin(tg_buffer_t* buf, const void* src, size_t nbytes){
  if(!buf||!src) return -1;
  if(!buf->allocated) {
    if (tg_buffer_allocate(buf)!=0) return -1;
  }
  size_t nb = tg_buffer_nbytes(buf); if (nbytes != nb) return -1;
  if (buf->base){
    // write into base at offset
    assert(buf->base->allocated);
    if (buf->allocator && buf->allocator->copyin) return buf->allocator->copyin((uint8_t*)buf->base->data + buf->offset, src, nbytes);
    memcpy((uint8_t*)buf->base->data + buf->offset, src, nbytes);
  } else {
    if (buf->allocator && buf->allocator->copyin) return buf->allocator->copyin(buf->data, src, nbytes);
    memcpy(buf->data, src, nbytes);
  }
  return 0;
}

int tg_buffer_copyout(const tg_buffer_t* buf, void* dst, size_t nbytes){
  if(!buf||!dst) return -1;
  if(!buf->allocated) return -1;
  size_t nb = tg_buffer_nbytes(buf);
  if (nbytes != nb) return -1;
  if (buf->base){ 
    if (buf->allocator && buf->allocator->copyout) return buf->allocator->copyout(dst, (uint8_t*)buf->base->data + buf->offset, nbytes);
    memcpy(dst, (uint8_t*)buf->base->data + buf->offset, nbytes);
  } else {
    if (buf->allocator && buf->allocator->copyout) return buf->allocator->copyout(dst, buf->data, nbytes);
    memcpy(dst, buf->data, nbytes);
  }
  return 0;
}

tg_memview_t tg_buffer_as_buffer(tg_buffer_t* buf, int allow_zero_copy, int force_zero_copy){
  tg_memview_t mv = (tg_memview_t){0};
  if(!buf) return mv;
  if(!buf->allocated) tg_buffer_allocate(buf);
  size_t nbytes = tg_buffer_nbytes(buf);
  // zero-copy path only if requested; for CPU opaque is the pointer
  int can_zero = (buf->opts.image == NULL);
  if ((force_zero_copy || allow_zero_copy) && can_zero){
    mv.data = buf->base ? (void*)((uint8_t*)buf->base->data + buf->offset) : buf->data;
    mv.nbytes = nbytes;
    return mv;
  }
  // if force requested but can't do zero-copy, assert like Python
  assert(!force_zero_copy && "force zero copy was passed, but copy is required");
  // fallback: return internal pointer view
  mv.data = buf->base ? (void*)((uint8_t*)buf->base->data + buf->offset) : buf->data;
  mv.nbytes = nbytes;
  return mv;
}

struct np_array* tg_buffer_numpy(tg_buffer_t* buf){
  if (!buf) return NULL; tg_memview_t mv = tg_buffer_as_buffer(buf, 0, 0);
  return np_frombuffer(mv.data, mv.nbytes, buf->dtype);
}

tg_dma_cpu_ref_t tg_buffer_as_dmaref_cpu(tg_buffer_t* buf){
  tg_dma_cpu_ref_t r = (tg_dma_cpu_ref_t){0,0};
  if (!buf) return r;
  const char* canon = tg_device_canonicalize(buf->device);
  // Only CPU exposes a valid host pointer; others return a NOOP ref
  if (canon && strncmp(canon, "CPU", 3)==0) {
    if (!buf->allocated) tg_buffer_allocate(buf);
    size_t nbytes = tg_buffer_nbytes(buf);
    void* opaque = buf->base ? (void*)((uint8_t*)buf->base->data + buf->offset) : buf->data;
    if (buf->allocator && buf->allocator->as_dmaref_cpu) {
      if (buf->allocator->as_dmaref_cpu(opaque, nbytes, &r) == 0) return r;
    }
    // Fallback if allocator hook not present: construct CPURef directly
    r.addr = (uintptr_t)opaque;
    r.size = nbytes;
  }
  return r;
}

tg_dma_fd_ref_t tg_buffer_as_dmaref_fd(tg_buffer_t* buf){
  (void)buf;
  tg_dma_fd_ref_t r = (tg_dma_fd_ref_t){0,0,0};
  // No backends provide FD-based DMA yet; return zeroed struct
  return r;
}
