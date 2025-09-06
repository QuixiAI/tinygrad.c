#ifndef SRC_HELPERS_HELPERS_H
#define SRC_HELPERS_HELPERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

// Environment variable functions
const char* tg_getenv(const char* name);
const char* tg_getenv_default(const char* name, const char* default_val);
int tg_is_osx(void);
int tg_is_ci(void);
void tg_windows_ansi_enable(void);

// Product function for shapes
int tg_prod(const int* shape, int len);

// Global counters (moved from engine/realize)
typedef struct tg_global_counters {
    int kernel_count;
    long long global_ops;
    long long global_mem;
    double time_sum_s;
    long long mem_used;
} tg_global_counters_t;

extern tg_global_counters_t GlobalCounters;
void tg_global_counters_reset(void);

// Disk cache (moved from device.c; simple in-memory implementation for now)
int tg_diskcache_get(const char* key, const char* src, char** output, size_t* output_size);
int tg_diskcache_put(const char* key, const char* src, const char* data, size_t data_size);

// Text/ANSI helpers
char* tg_colored(const char* st, const char* color, int background);
char* tg_colorize_float(float x);
char* tg_ansistrip(const char* s);
int   tg_ansilen(const char* s);
char* tg_time_to_str(double t, int w);
char* tg_strip_parens(const char* s);
char* tg_word_wrap(const char* s, int wrap);

// Math/bit helpers (Python-faithful integer semantics)
#include <stdint.h>
int64_t tg_py_floor_div(int64_t a, int64_t b);
int64_t tg_ceildiv(int64_t num, int64_t amt);
int64_t tg_round_up(int64_t num, int64_t amt);
int64_t tg_round_down(int64_t num, int64_t amt);
int64_t tg_cdiv(int64_t x, int64_t y);
int64_t tg_cmod(int64_t x, int64_t y);
uint32_t tg_lo32(uint64_t x);
uint32_t tg_hi32(uint64_t x);
void     tg_data64(uint64_t data, uint32_t* hi, uint32_t* lo);
void     tg_data64_le(uint64_t data, uint32_t* lo, uint32_t* hi);
uint64_t tg_getbits(uint64_t value, int start, int end);
uint64_t tg_i2u(int bits, int64_t value);

// Wait until callback returns value or timeout (ms). Returns 1 if met, 0 on timeout.
int tg_wait_cond(int (*cb)(void*), void* ctx, int value, int timeout_ms, const char* msg);

// Simple sequence helpers
int  tg_all_same_int(const int* arr, int n);
void tg_argsort_int(const int* arr, int n, int* out_idx);
// Dedup strings (preserves order). Returns new count; out array allocated.
int  tg_dedup_str(char** in, int n, char*** out);

// Diskcache management
void tg_diskcache_clear(void);

// Profiling helpers (minimal)
typedef struct tg_profile_range_event {
  const char* device;
  const char* name;
  unsigned long long st_us;
  unsigned long long en_us;
  int is_copy;
} tg_profile_range_event_t;

void tg_profile_events_clear(void);
int  tg_profile_events_count(void);
const tg_profile_range_event_t* tg_profile_events_data(void);
// Begin/end range; returns index handle for end
int  tg_cpu_profile_begin(const char* name, const char* device, int is_copy);
void tg_cpu_profile_end(int handle, int display);
// Enable/disable profiling (mirrors Python PROFILE). Default from env PROFILE (non-empty => on).
void tg_profile_set_enabled(int enabled);
int  tg_profile_get_enabled(void);

// Download/fetch helper (faithful shape):
// If url starts with '/' or '.', returns that path. Otherwise, downloads to downloads dir under build/ (or tinybox path),
// using optional "name" and "subdir". If gunzip!=0, decompresses .gz into final file.
// If allow_caching==0, forces re-download. Writes the output path to out_path (size out_sz).
// Returns 0 on success, non-zero on error or unsupported.
int tg_fetch(const char* url, const char* name, const char* subdir, int gunzip, int allow_caching,
             char* out_path, size_t out_sz);

// ctypes-like helpers (minimal)
#include <stdint.h>
uintptr_t tg_mv_address(void* p);
char** tg_to_char_p_p(const char* const* arr, int n);

// Exec helpers (stubs; optional integrations may override)
void tg_cpu_objdump(const unsigned char* lib, size_t len, const char* tool);
int  tg_capstone_flatdump(const unsigned char* lib, size_t len);

#ifdef __cplusplus
}
#endif
#endif /* SRC_HELPERS_HELPERS_H */
