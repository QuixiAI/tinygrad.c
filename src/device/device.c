#include "device.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include "device/allocator.h"
#include "helpers/helpers.h"

// Backend registry scaffold (to mirror Python importlib of runtime backends)
typedef const tg_allocator_t* (*get_allocator_fn)(const char*);
typedef struct tg_backend_vtable {
  const char* name;                 // canonical device name, e.g., "CPU"
  void*      (*open)(const char*);  // open/construct backend compiled device
  void       (*finalize)(void*);    // finalize/close
  get_allocator_fn get_allocator;   // return allocator for this backend
} tg_backend_vtable;

// Minimal CPU backend placeholders
static void* _cpu_open(const char* dev){ (void)dev; return (void*)0x1; }
static void  _cpu_finalize(void* h){ (void)h; }

static const tg_allocator_t* _cpu_get_allocator(const char* dev){ return tg_get_default_allocator(dev); }
// Minimal LLVM backend placeholders
static void* _llvm_open(const char* dev){ (void)dev; return (void*)0x1; }
static void  _llvm_finalize(void* h){ (void)h; }
static const tg_allocator_t* _llvm_get_allocator(const char* dev){ return tg_get_default_allocator(dev); }
static void* _disk_open(const char* dev){ (void)dev; return (void*)0x1; }
static void  _disk_finalize(void* h){ (void)h; }
static const tg_allocator_t* _disk_get_allocator(const char* dev){ return tg_get_default_allocator(dev); }
static void* _npy_open(const char* dev){ (void)dev; return (void*)0x1; }
static void  _npy_finalize(void* h){ (void)h; }
static const tg_allocator_t* _npy_get_allocator(const char* dev){ return tg_get_default_allocator(dev); }
static const tg_backend_vtable k_backends[] = {
  { "CPU", _cpu_open, _cpu_finalize, _cpu_get_allocator },
  { "LLVM", _llvm_open, _llvm_finalize, _llvm_get_allocator },
  { "DISK", _disk_open, _disk_finalize, _disk_get_allocator },
  { "NPY", _npy_open, _npy_finalize, _npy_get_allocator },
};
static const int k_backends_count = (int)(sizeof(k_backends)/sizeof(k_backends[0]));

// Track opened devices for atexit-like finalize
typedef struct {
  char** names; int count; int cap;
} opened_set_t;
static opened_set_t g_opened = {0};
static void opened_add(const char* name){
  if (g_opened.count+1 > g_opened.cap){ int nc = g_opened.cap? g_opened.cap*2 : 8; g_opened.names = (char**)realloc(g_opened.names, nc*sizeof(char*)); g_opened.cap = nc; }
  g_opened.names[g_opened.count++] = strdup(name);
}
static void opened_finalize_all(void){
  for (int i=0;i<g_opened.count;i++){
    // find backend by name and call finalize (opaque handle ignored in this scaffold)
    const char* nm = g_opened.names[i];
    for (int b=0;b<k_backends_count;b++) if (strcmp(k_backends[b].name, nm)==0) { k_backends[b].finalize((void*)0x1); break; }
    free(g_opened.names[i]);
  }
  free(g_opened.names); g_opened.names=NULL; g_opened.count=g_opened.cap=0;
}

void tg_device_finalize_all(void){ opened_finalize_all(); }
__attribute__((constructor)) static void _tg_device_register_atexit(void){ atexit(tg_device_finalize_all); }

// For now, return default allocator (LRU-wrapped CPU). This can be per-backend later.
const tg_allocator_t* tg_device_get_allocator(const char* device_str){
  // canonicalize and compare base name (before any colon)
  const char* canon = tg_device_canonicalize(device_str);
  char base[64]; strncpy(base, canon?canon:"CPU", sizeof(base)-1); base[sizeof(base)-1]='\0';
  char* colon = strchr(base, ':'); if (colon) *colon='\0';
  for (int i=0;i<k_backends_count;i++) if (strcmp(k_backends[i].name, base)==0){
    if (k_backends[i].get_allocator) return k_backends[i].get_allocator(canon);
    break;
  }
  return tg_get_default_allocator(canon);
}

// Global state for default device
static char* default_device = NULL;
static int compiler_cache_enabled = 1;  // Python default: DISABLE_COMPILER_CACHE=False means enabled
static int allow_device_usage = 1;      // mirrors Python ALLOW_DEVICE_USAGE
void tg_device_set_allow_usage(int allow){ allow_device_usage = (allow!=0); }

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
static int _is_true_env(const char* name){ const char* v=getenv(name); if(!v) return 0; return strcmp(v,"1")==0 || strcasecmp(v,"true")==0 || strlen(v)>0; }
const char* tg_device_get_default(void) {
    // Mirror Python DEFAULT selection, avoid freeing existing default unless it changes
    const char* dev = getenv("DEV");
    if (dev && *dev){
        // Use DEV override
        char buf[64]; strncpy(buf, dev, sizeof(buf)-1); buf[sizeof(buf)-1]='\0';
        const char* canon = tg_device_canonicalize(buf);
        if (!default_device || strcmp(default_device, canon) != 0) {
            if (default_device) free(default_device);
            default_device = strdup(canon);
        }
        return default_device;
    }
    // From environment flags matching backend names (exclude DISK, NPY like Python)
    int chosen = -1;
    for (int i=0;i<k_backends_count;i++){
        const char* nm = k_backends[i].name;
        if (strcmp(nm, "DISK")==0 || strcmp(nm, "NPY")==0) continue;
        if (_is_true_env(nm)) { if (chosen==-1) chosen = i; }
    }
    if (chosen!=-1) {
        if (!default_device || strcmp(default_device, k_backends[chosen].name) != 0) {
            if (default_device) free(default_device);
            default_device = strdup(k_backends[chosen].name);
        }
        return default_device;
    }
    // Fallback: pick the first available backend, set env for children
    if (k_backends_count > 0){
        if (!default_device || strcmp(default_device, k_backends[0].name) != 0) {
            if (default_device) free(default_device);
            default_device = strdup(k_backends[0].name);
        }
        // propagate to env (best-effort)
        setenv(default_device, "1", 1);
        return default_device;
    }
    // Absolute fallback if no backends: CPU
    if (!default_device || strcmp(default_device, "CPU") != 0) {
        if (default_device) free(default_device);
        default_device = strdup("CPU");
    }
    return default_device;
}

int tg_device_set_default(const char* device_str) {
    // Do not free previous default to avoid invalidating external pointers returned by tg_device_get_default
    default_device = strdup(device_str);
    return TG_SUCCESS;
}

tg_device_t* tg_device_get(const char* device_str) {
    // Port of Python: def __getitem__(self, ix:str) -> Compiled
    const char* canonical = tg_device_canonicalize(device_str);

    // resolve backend by base name
    const tg_backend_vtable* be = NULL;
    char base[64]; strncpy(base, canonical?canonical:"CPU", sizeof(base)-1); base[sizeof(base)-1]='\0';
    char* colon = strchr(base, ':'); if (colon) *colon='\0';
    for (int i=0;i<k_backends_count;i++) if (strcmp(k_backends[i].name, base)==0) { be = &k_backends[i]; break; }
    // allow device usage checks: permit DISK/NPY unconditionally (no PYTHON backend in C)
    if (!allow_device_usage && strcmp(base, "DISK")!=0 && strcmp(base, "NPY")!=0) {
        return NULL;
    }
    if (!be) {
      // fallback to previous behavior for names we accept elsewhere
      if (!tg_device_exists(canonical)) return NULL;
    } else {
      // open backend (opaque)
      (void)be->open(canonical);
      opened_add(canonical);
    }

    tg_device_t* device = malloc(sizeof(tg_device_t));
    if (!device) return NULL;
    strncpy(device->name, canonical, sizeof(device->name) - 1);
    device->name[sizeof(device->name) - 1] = '\0';
    return device;
}

int tg_device_exists(const char* device_str) {
    const char* canonical = tg_device_canonicalize(device_str);
    char base[64]; strncpy(base, canonical?canonical:"CPU", sizeof(base)-1); base[sizeof(base)-1]='\0';
    char* colon = strchr(base, ':'); if (colon) *colon='\0';
    for (int i=0;i<k_backends_count;i++) if (strcmp(k_backends[i].name, base)==0) return 1;
    // legacy support
    return (strcmp(base, "GPU") == 0 || strcmp(base, "METAL") == 0 || strcmp(base, "CUDA") == 0) ? 1 : 0;
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

// List available devices by iterating statically-registered backends
int tg_device_get_available(const char** out_names, int max_names) {
    int count = 0;
    for (int i = 0; i < k_backends_count && count < max_names; i++) {
        const char* nm = k_backends[i].name;
        if (!allow_device_usage && strcmp(nm, "DISK")!=0 && strcmp(nm, "NPY")!=0) continue;
        if (out_names) out_names[count] = nm;
        count++;
    }
    return count;
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

// diskcache_get/put moved to helpers/helpers.c

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
