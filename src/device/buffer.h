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