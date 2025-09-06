#ifndef SRC_DEVICE_DEVICE_H
#define SRC_DEVICE_DEVICE_H

#include "tg.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device API - faithful port of Python tinygrad.device functionality

// Device management
typedef struct tg_device tg_device_t;

// Device canonicalization
TG_API const char* tg_device_canonicalize(const char* device_str);
TG_API const char* tg_device_get_default(void);
TG_API int tg_device_set_default(const char* device_str);

// Device lookup and creation
TG_API tg_device_t* tg_device_get(const char* device_str);
TG_API int tg_device_exists(const char* device_str);

// Compiler functionality
typedef struct tg_compiler tg_compiler_t;

TG_API tg_compiler_t* tg_compiler_create(const char* key);
TG_API void tg_compiler_destroy(tg_compiler_t* compiler);
TG_API int tg_compiler_compile(tg_compiler_t* compiler, const char* src, char** output, size_t* output_size);
TG_API int tg_compiler_compile_cached(tg_compiler_t* compiler, const char* src, char** output, size_t* output_size);

// Context management for compiler cache
TG_API int tg_context_set_compiler_cache(int enabled);
TG_API int tg_context_get_compiler_cache(void);

// Device operations
TG_API int tg_device_compile_test(const char* device_str);

// Finalize (close) all opened devices (mirrors Python atexit finalize)
TG_API void tg_device_finalize_all(void);

// Return allocator for a device (backend-provided or default)
struct tg_allocator; // fwd decl
TG_API const struct tg_allocator* tg_device_get_allocator(const char* device_str);

// Allow or disallow device usage (mirrors Python Context/ALLOW_DEVICE_USAGE)
TG_API void tg_device_set_allow_usage(int allow);

// List available devices by attempting to open/close backends (no side effects)
// Returns the number of devices written to out_names (canonical names), up to max_names
TG_API int tg_device_get_available(const char** out_names, int max_names);

#ifdef __cplusplus
}
#endif

#endif /* SRC_DEVICE_DEVICE_H */
