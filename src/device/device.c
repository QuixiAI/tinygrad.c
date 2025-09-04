#include "device.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

// Global state for default device
static char* default_device = NULL;
static int compiler_cache_enabled = 1;  // Python default: DISABLE_COMPILER_CACHE=False means enabled

// Compiler structure
struct tg_compiler {
    char* cachekey;
};

// Device structure
struct tg_device {
    char name[64];
};

// Helper to convert string to uppercase
static void str_toupper(char* str) {
    while (*str) {
        *str = toupper(*str);
        str++;
    }
}

// Port of Python: @functools.cache  # this class is a singleton, pylint: disable=method-cache-max-size-none
// def _canonicalize(self, device:str) -> str: return re.sub(r":0$", "", (d:=device.split(":", 1)[0].upper()) + device[len(d):])
const char* tg_device_canonicalize(const char* device_str) {
    static char canonicalized[256];  // Static buffer for result
    
    // Handle NULL case - return default device
    if (device_str == NULL) {
        const char* default_dev = tg_device_get_default();
        if (default_dev == NULL) {
            // If no default is set, use "CPU" as fallback (common in Python tests)
            strcpy(canonicalized, "CPU");
            return canonicalized;
        }
        return default_dev;
    }
    
    // Copy input string to work with
    strcpy(canonicalized, device_str);
    
    // Find the colon position
    char* colon = strchr(canonicalized, ':');
    
    // Convert the device name part (before colon) to uppercase
    if (colon) {
        // Temporarily null-terminate at colon
        *colon = '\0';
        str_toupper(canonicalized);
        *colon = ':';  // Restore colon
        
        // Check if it ends with ":0" and remove it
        size_t len = strlen(canonicalized);
        if (len >= 2 && strcmp(canonicalized + len - 2, ":0") == 0) {
            canonicalized[len - 2] = '\0';
        }
    } else {
        // No colon, just uppercase the whole string
        str_toupper(canonicalized);
    }
    
    return canonicalized;
}

// Port of Python: @property def DEFAULT(self) -> str:
const char* tg_device_get_default(void) {
    if (default_device == NULL) {
        // Python default behavior: try to find available device
        // For now, return "CPU" as the basic default
        default_device = strdup("CPU");
    }
    return default_device;
}

int tg_device_set_default(const char* device_str) {
    if (default_device) {
        free(default_device);
    }
    default_device = strdup(device_str);
    return TG_SUCCESS;
}

tg_device_t* tg_device_get(const char* device_str) {
    // Port of Python: def __getitem__(self, ix:str) -> Compiled
    const char* canonical = tg_device_canonicalize(device_str);
    
    // Check if device exists
    if (!tg_device_exists(canonical)) {
        return NULL;  // Python raises ModuleNotFoundError, we return NULL
    }
    
    // For now, create a simple device struct
    tg_device_t* device = malloc(sizeof(tg_device_t));
    if (!device) return NULL;
    
    strncpy(device->name, canonical, sizeof(device->name) - 1);
    device->name[sizeof(device->name) - 1] = '\0';
    
    return device;
}

int tg_device_exists(const char* device_str) {
    // For basic implementation, accept CPU and GPU devices
    const char* canonical = tg_device_canonicalize(device_str);
    return (strcmp(canonical, "CPU") == 0 || 
            strcmp(canonical, "GPU") == 0 ||
            strcmp(canonical, "METAL") == 0 ||
            strcmp(canonical, "CUDA") == 0) ? 1 : 0;
}

// Port of Python Compiler class
tg_compiler_t* tg_compiler_create(const char* key) {
    tg_compiler_t* compiler = malloc(sizeof(tg_compiler_t));
    if (!compiler) return NULL;
    
    // Port of: def __init__(self, cachekey:str|None=None): self.cachekey = None if DISABLE_COMPILER_CACHE else cachekey
    if (compiler_cache_enabled && key) {
        compiler->cachekey = strdup(key);
    } else {
        compiler->cachekey = NULL;
    }
    
    return compiler;
}

void tg_compiler_destroy(tg_compiler_t* compiler) {
    if (compiler) {
        if (compiler->cachekey) free(compiler->cachekey);
        free(compiler);
    }
}

// Port of: def compile(self, src:str) -> bytes: return src.encode()   # NOTE: empty compiler is the default
int tg_compiler_compile(tg_compiler_t* compiler, const char* src, char** output, size_t* output_size) {
    (void)compiler;  // Unused in basic implementation
    
    if (!src || !output || !output_size) return TG_ERR_INVALID;
    
    // Python default: return src.encode()
    size_t len = strlen(src);
    *output = malloc(len + 1);
    if (!*output) return TG_ERR_NOMEM;
    
    strcpy(*output, src);
    *output_size = len;
    
    return TG_SUCCESS;
}

// Port of: def compile_cached(self, src:str) -> bytes:
int tg_compiler_compile_cached(tg_compiler_t* compiler, const char* src, char** output, size_t* output_size) {
    if (!compiler || !src || !output || !output_size) return TG_ERR_INVALID;
    
    // Check cache if cachekey exists
    if (compiler->cachekey != NULL) {
        // Try to get from cache
        int cache_result = tg_diskcache_get(compiler->cachekey, src, output, output_size);
        if (cache_result == TG_SUCCESS) {
            return TG_SUCCESS;  // Found in cache
        }
    }
    
    // Not in cache or no cachekey - compile normally
    int compile_result = tg_compiler_compile(compiler, src, output, output_size);
    if (compile_result != TG_SUCCESS) {
        return compile_result;
    }
    
    // Store in cache if cachekey exists
    if (compiler->cachekey != NULL) {
        tg_diskcache_put(compiler->cachekey, src, *output, *output_size);
    }
    
    return TG_SUCCESS;
}

// Simple in-memory cache for disk cache simulation
typedef struct cache_entry {
    char* key;
    char* src;
    char* data;
    size_t data_size;
    struct cache_entry* next;
} cache_entry_t;

static cache_entry_t* cache_head = NULL;

int tg_diskcache_get(const char* key, const char* src, char** output, size_t* output_size) {
    if (!key || !src || !output || !output_size) return TG_ERR_INVALID;
    
    // Search cache
    cache_entry_t* entry = cache_head;
    while (entry) {
        if (strcmp(entry->key, key) == 0 && strcmp(entry->src, src) == 0) {
            // Found in cache
            *output = malloc(entry->data_size + 1);
            if (!*output) return TG_ERR_NOMEM;
            memcpy(*output, entry->data, entry->data_size);
            (*output)[entry->data_size] = '\0';
            *output_size = entry->data_size;
            return TG_SUCCESS;
        }
        entry = entry->next;
    }
    
    return TG_ERR_RUNTIME;  // Not found
}

int tg_diskcache_put(const char* key, const char* src, const char* data, size_t data_size) {
    if (!key || !src || !data) return TG_ERR_INVALID;
    
    // Create new cache entry
    cache_entry_t* entry = malloc(sizeof(cache_entry_t));
    if (!entry) return TG_ERR_NOMEM;
    
    entry->key = strdup(key);
    entry->src = strdup(src);
    entry->data = malloc(data_size + 1);
    if (!entry->key || !entry->src || !entry->data) {
        if (entry->key) free(entry->key);
        if (entry->src) free(entry->src);
        if (entry->data) free(entry->data);
        free(entry);
        return TG_ERR_NOMEM;
    }
    
    memcpy(entry->data, data, data_size);
    entry->data[data_size] = '\0';
    entry->data_size = data_size;
    
    // Add to cache list
    entry->next = cache_head;
    cache_head = entry;
    
    return TG_SUCCESS;
}

int tg_context_set_compiler_cache(int enabled) {
    compiler_cache_enabled = enabled;
    return TG_SUCCESS;
}

int tg_context_get_compiler_cache(void) {
    return compiler_cache_enabled;
}

int tg_device_compile_test(const char* device_str) {
    // Simple test compile
    tg_compiler_t* compiler = tg_compiler_create(device_str);
    if (!compiler) return TG_ERR_NOMEM;
    
    char* output;
    size_t output_size;
    const char* test_src = "test";
    
    int result = tg_compiler_compile(compiler, test_src, &output, &output_size);
    if (result == TG_SUCCESS) {
        free(output);
    }
    
    tg_compiler_destroy(compiler);
    return result;
}

// Keep the old stub function for compatibility
int tg_unimpl_stub_device(void) {
    return TG_ERR_UNIMPL;
}
