#include "device/allocator.h"
#include "device/buffer.h"
#include "helpers/helpers.h"
#include <stdlib.h>
#include <string.h>

// ---- Base CPU allocator ----
static void* cpu_alloc(size_t nbytes, const tg_bufferspec_t* opts){ (void)opts; return malloc(nbytes); }
static void  cpu_free(void* opaque, const tg_bufferspec_t* opts){ (void)opts; if (opaque) free(opaque); }
static int   cpu_copyin(void* dst_opaque, const void* src, size_t nbytes){ if(!dst_opaque||!src) return -1; memcpy(dst_opaque, src, nbytes); return 0; }
static int   cpu_copyout(void* dst, const void* src_opaque, size_t nbytes){ if(!dst||!src_opaque) return -1; memcpy(dst, src_opaque, nbytes); return 0; }
/* as_buffer/offset reserved */

static int   cpu_transfer(void* dst_opaque, const char* dst_device, const void* src_opaque, const char* src_device, size_t nbytes){
  (void)dst_device; (void)src_device;
  if (!dst_opaque || !src_opaque) return -1;
  // CPU fallback: memcpy, ignore device strings for now
  memcpy(dst_opaque, src_opaque, nbytes);
  return 0;
}
static int   cpu_as_dmaref_cpu(const void* opaque, size_t nbytes, tg_dma_cpu_ref_t* out){
  if (!opaque || !out) return -1;
  out->addr = (uintptr_t)opaque;
  out->size = nbytes;
  return 0;
}

static __attribute__((unused)) tg_allocator_t k_cpu_allocator = {
  .alloc = cpu_alloc,
  .free = cpu_free,
  .copyin = cpu_copyin,
  .copyout = cpu_copyout,
  .transfer = cpu_transfer,
  .as_dmaref_cpu = cpu_as_dmaref_cpu,
};

// ---- LRU wrapper (faithful behavior) ----
typedef struct {
  size_t size;
  tg_bufferspec_t opts;
  void** opaques;
  int count, cap;
} lru_bucket_t;

static lru_bucket_t* g_buckets = NULL;
static int g_bucket_count = 0, g_bucket_cap = 0;
static size_t g_cached_bytes = 0;
static size_t g_cache_cap = 0; // 0 = unlimited

static int _opts_equal(const tg_bufferspec_t* a, const tg_bufferspec_t* b){
  return a->image==b->image && a->uncached==b->uncached && a->cpu_access==b->cpu_access &&
         a->host==b->host && a->nolru==b->nolru && a->external_ptr==b->external_ptr;
}

static lru_bucket_t* _find_bucket(size_t size, const tg_bufferspec_t* opts){
  for (int i=0;i<g_bucket_count;i++) if (g_buckets[i].size==size && _opts_equal(&g_buckets[i].opts, opts)) return &g_buckets[i];
  return NULL;
}

static lru_bucket_t* _get_bucket(size_t size, const tg_bufferspec_t* opts){
  lru_bucket_t* b = _find_bucket(size, opts);
  if (b) return b;
  if (g_bucket_count+1 > g_bucket_cap){ int nc = g_bucket_cap? g_bucket_cap*2 : 8; g_buckets = (lru_bucket_t*)realloc(g_buckets, nc*sizeof(lru_bucket_t)); g_bucket_cap = nc; }
  b = &g_buckets[g_bucket_count++]; memset(b,0,sizeof(*b)); b->size=size; b->opts=*opts; return b;
}

static void _bucket_push(lru_bucket_t* b, void* opaque){
  if (b->count+1 > b->cap){ int nc = b->cap? b->cap*2 : 8; b->opaques = (void**)realloc(b->opaques, nc*sizeof(void*)); b->cap = nc; }
  b->opaques[b->count++] = opaque;
}

static void* _bucket_pop(lru_bucket_t* b){ if (b->count==0) return NULL; return b->opaques[--b->count]; }

static void _free_cache_all(void){
  for (int i=0;i<g_bucket_count;i++){
    lru_bucket_t* b = &g_buckets[i];
    for (int j=0;j<b->count;j++) cpu_free(b->opaques[j], &b->opts);
    free(b->opaques); b->opaques=NULL; b->count=b->cap=0;
  }
  g_cached_bytes = 0;
}

static void _evict_until_cap(void){
  if (g_cache_cap == 0) return;
  for (int i = 0; i < g_bucket_count && g_cached_bytes > g_cache_cap; i++){
    lru_bucket_t* b = &g_buckets[i];
    while (b->count > 0 && g_cached_bytes > g_cache_cap){
      void* blk = _bucket_pop(b);
      cpu_free(blk, &b->opts);
      g_cached_bytes -= b->size;
    }
  }
}

static void* lru_alloc(size_t nbytes, const tg_bufferspec_t* opts){
  // use size in bytes as key; Python uses size (elements) but we only know bytes here
  lru_bucket_t* b = _get_bucket(nbytes, opts);
  void* blk = _bucket_pop(b);
  if (blk) return blk;
  blk = cpu_alloc(nbytes, opts);
  if (!blk){ _free_cache_all(); blk = cpu_alloc(nbytes, opts); }
  return blk;
}
static void  lru_free(void* opaque, const tg_bufferspec_t* opts){ if (!opaque) return; // push back into any matching bucket by size unknown — caller passes options; assume size tracked externally
  // Without size we cannot find the bucket; require caller to free with same options and size context.
  // For this shim, we cannot derive size; fallback to freeing.
  // NOTE: Faithful LRU requires integration with Buffer to free by size; will be wired when buffer calls allocator->free with bucket context.
  cpu_free(opaque, opts);
}
static int   lru_copyin(void* dst_opaque, const void* src, size_t nbytes){ return cpu_copyin(dst_opaque, src, nbytes); }
static int   lru_copyout(void* dst, const void* src_opaque, size_t nbytes){ return cpu_copyout(dst, src_opaque, nbytes); }

// forward declare free_sized used in initializer
static void lru_free_sized(void* opaque, size_t nbytes, const tg_bufferspec_t* opts);

static tg_allocator_t k_lru_allocator = {
  .alloc = lru_alloc,
  .free = lru_free,
  .free_sized = lru_free_sized,
  .copyin = lru_copyin,
  .copyout = lru_copyout,
  .transfer = cpu_transfer,      // delegate to CPU memcpy for now
  .as_dmaref_cpu = cpu_as_dmaref_cpu,
};

static void lru_free_sized(void* opaque, size_t nbytes, const tg_bufferspec_t* opts){
  if (!opaque) return;
  lru_bucket_t* b = _get_bucket(nbytes, opts);
  if (b) { _bucket_push(b, opaque); g_cached_bytes += nbytes; _evict_until_cap(); return; }
  cpu_free(opaque, opts);
}

const tg_allocator_t* tg_get_default_allocator(const char* device){
  (void)device;
  // debug: ensure function is entered and global is addressable
  // fprintf(stderr, "tg_get_default_allocator %p\n", (void*)&k_lru_allocator);
  return &k_lru_allocator;
}

// ---- Convenience wrappers ----
int allocator_has_transfer(const tg_allocator_t* a){ return a && a->transfer != NULL; }
int allocator_transfer(const tg_allocator_t* a, void* dst_opaque, const void* src_opaque, size_t nbytes,
                       const char* src_dev, const char* dst_dev){ return a && a->transfer ? a->transfer(dst_opaque, dst_dev, src_opaque, src_dev, nbytes) : -1; }
int allocator_has_as_buffer(const tg_allocator_t* a){ (void)a; return 1; }
void* allocator_as_buffer(const tg_allocator_t* a, void* opaque){ (void)a; return opaque; }
int allocator_copyout(const tg_allocator_t* a, void* dst, const void* src_opaque){ return a && a->copyout ? a->copyout(dst, src_opaque, 0 /* unknown size in wrapper */) : -1; }
int allocator_copyin(const tg_allocator_t* a, void* dst_opaque, const void* src){ return a && a->copyin ? a->copyin(dst_opaque, src, 0 /* unknown size in wrapper */) : -1; }

// ---- LRU bounds controls ----
void tg_allocator_set_cache_cap(size_t cap_bytes){ g_cache_cap = cap_bytes; _evict_until_cap(); }
size_t tg_allocator_get_cache_cap(void){ return g_cache_cap; }
size_t tg_allocator_get_cached_bytes(void){ return g_cached_bytes; }

// env: LRU_CACHE_CAP supports optional K/M/G suffix (base 1024)
static size_t _parse_cap_bytes(const char* s){
  if (!s || !*s) return 0;
  char *end = NULL;
  unsigned long long v = strtoull(s, &end, 10);
  if (end && *end){
    if (*end=='K' || *end=='k') v *= 1024ull;
    else if (*end=='M' || *end=='m') v *= 1024ull*1024ull;
    else if (*end=='G' || *end=='g') v *= 1024ull*1024ull*1024ull;
  }
  return (size_t)v;
}

__attribute__((constructor)) static void _tg_allocator_init_from_env(void){
  const char* cap = tg_getenv("LRU_CACHE_CAP");
  if (cap && *cap) tg_allocator_set_cache_cap(_parse_cap_bytes(cap));
}
