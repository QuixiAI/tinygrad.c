// Buffer API - stub implementation for memory planning
#include "buffer.h"
#include <stdlib.h>
#include <string.h>

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