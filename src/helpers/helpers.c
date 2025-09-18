/* helpers.c
 * Minimal implementation of helper functions needed for dtype.c
 */
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <glib.h>
#include "helpers/helpers.h"
#include "compat/fetch.h"
#include "compat/tqdm.h"
#ifdef TG_HAVE_CAPSTONE
#include <capstone/capstone.h>
#endif

// Environment variable functions
const char* tg_getenv(const char* name) {
    return getenv(name);
}

const char* tg_getenv_default(const char* name, const char* default_val) {
    const char* val = getenv(name);
    return val ? val : default_val;
}

static int _amx_initialized = 0;
static int _amx_enabled = 0;

static void _amx_init(void) {
    if (_amx_initialized) return;
    const char* v = tg_getenv("AMX");
    _amx_enabled = (v && *v && strcmp(v, "0") != 0);
    _amx_initialized = 1;
}

// Env/platform flags
int tg_is_osx(void){
#ifdef __APPLE__
  return 1;
#else
  return 0;
#endif
}
int tg_is_ci(void){ const char* v = getenv("CI"); return v && *v; }

#ifdef _WIN32
#include <windows.h>
void tg_windows_ansi_enable(void){
  HANDLE hs[2] = { GetStdHandle(STD_OUTPUT_HANDLE), GetStdHandle(STD_ERROR_HANDLE) };
  for (int i=0;i<2;i++){
    if (hs[i] == INVALID_HANDLE_VALUE) continue;
    DWORD mode = 0; if (GetConsoleMode(hs[i], &mode)) {
      mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING; SetConsoleMode(hs[i], mode);
    }
  }
}
#else
void tg_windows_ansi_enable(void) { /* no-op */ }
#endif

// Product function for shapes
int tg_prod(const int* shape, int len) {
    int result = 1;
    for (int i = 0; i < len; i++) {
        result *= shape[i];
    }
    return result;
}

// Global counters
tg_global_counters_t GlobalCounters = {0};
void tg_global_counters_reset(void) {
    GlobalCounters.kernel_count = 0;
    GlobalCounters.global_ops = 0;
    GlobalCounters.global_mem = 0;
    GlobalCounters.time_sum_s = 0.0;
    // NOTE: mem_used is not reset in Python, but we won't modify it here
}

// ---- Disk cache: sqlite-backed when available, else in-memory fallback ----
#ifdef TG_HAVE_SQLITE3
#include <sqlite3.h>
#endif

typedef struct cache_entry {
    char* key;
    char* src;
    char* data;
    size_t data_size;
    struct cache_entry* next;
} cache_entry_t;

static cache_entry_t* cache_head = NULL; // fallback

static int _cachelevel(void){ const char* v=tg_getenv("CACHELEVEL"); return (v&&*v)? atoi(v) : 1; }

#ifdef TG_HAVE_SQLITE3
static sqlite3* _db = NULL;
static int _db_open(void){
  if (_db) return 0;
  const char* path = tg_getenv("CACHEDB");
  char defpath[256]; if (!path || !*path){
    // default under build/cache/cache.db
    snprintf(defpath, sizeof(defpath), "build/cache/cache.db"); path = defpath;
  }
  // ensure parent dirs exist
  char dir[256]; strncpy(dir, path, sizeof(dir)-1); dir[sizeof(dir)-1]='\0';
  char* slash = strrchr(dir, '/'); if (slash){ *slash='\0';
#ifdef _WIN32
    _mkdir(dir);
#else
    char tmp[256]; strncpy(tmp, dir, sizeof(tmp)-1); tmp[sizeof(tmp)-1]='\0';
    for(char* p=tmp+1; *p; p++){ if(*p=='/'){ *p='\0'; mkdir(tmp, 0755); *p='/'; } }
    mkdir(tmp,0755);
#endif
  }
  if (sqlite3_open(path, &_db) != SQLITE_OK) { _db=NULL; return -1; }
  sqlite3_busy_timeout(_db, 60000);
  sqlite3_exec(_db, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
  return 0;
}
static void _sanitize_table(const char* in, char* out, size_t max){ size_t j=0; for(size_t i=0; in[i] && j<max-1; i++){ char c=in[i]; out[j++] = (isalnum((unsigned char)c) || c=='_') ? c : '_'; } out[j]='\0'; }
static void _table_name(const char* key, char* out, size_t max){ char base[128]; _sanitize_table(key, base, sizeof(base)); snprintf(out, max, "%s_%d", base, 22); }
#endif

int tg_diskcache_get(const char* key, const char* src, char** output, size_t* output_size) {
    if (!key || !src || !output || !output_size) return -1;
    if (_cachelevel() < 1) return 1;
#ifdef TG_HAVE_SQLITE3
    if (_db_open()==0){
      char tbl[160]; _table_name(key, tbl, sizeof(tbl));
      char sql[256]; snprintf(sql, sizeof(sql), "SELECT val FROM '%s' WHERE key=?1", tbl);
      sqlite3_stmt* st=NULL; if (sqlite3_prepare_v2(_db, sql, -1, &st, NULL)!=SQLITE_OK) return 1;
      sqlite3_bind_text(st, 1, src, -1, SQLITE_TRANSIENT);
      int rc = sqlite3_step(st);
      if (rc == SQLITE_ROW){ const void* blob = sqlite3_column_blob(st,0); int blen = sqlite3_column_bytes(st,0);
        *output = (char*)malloc((size_t)blen+1); if(!*output){ sqlite3_finalize(st); return -2; }
        memcpy(*output, blob, (size_t)blen); (*output)[blen]='\0'; *output_size=(size_t)blen; sqlite3_finalize(st); return 0;
      }
      sqlite3_finalize(st); return 1;
    }
#endif
    // fallback in-memory
    cache_entry_t* entry = cache_head;
    while (entry) {
        if (strcmp(entry->key, key) == 0 && strcmp(entry->src, src) == 0) {
            *output = (char*)malloc(entry->data_size + 1);
            if (!*output) return -2;
            memcpy(*output, entry->data, entry->data_size);
            (*output)[entry->data_size] = '\0';
            *output_size = entry->data_size;
            return 0;
        }
        entry = entry->next;
    }
    return 1;
}

int tg_diskcache_put(const char* key, const char* src, const char* data, size_t data_size) {
    if (!key || !src || !data) return -1;
    if (_cachelevel() < 1) return 0;
#ifdef TG_HAVE_SQLITE3
    if (_db_open()==0){
      char tbl[160]; _table_name(key, tbl, sizeof(tbl));
      char sql_create[256]; snprintf(sql_create, sizeof(sql_create), "CREATE TABLE IF NOT EXISTS '%s' (key TEXT PRIMARY KEY, val BLOB)", tbl);
      sqlite3_exec(_db, sql_create, NULL, NULL, NULL);
      char sql[256]; snprintf(sql, sizeof(sql), "REPLACE INTO '%s' (key,val) VALUES (?1,?2)", tbl);
      sqlite3_stmt* st=NULL; if (sqlite3_prepare_v2(_db, sql, -1, &st, NULL)!=SQLITE_OK) return -2;
      sqlite3_bind_text(st, 1, src, -1, SQLITE_TRANSIENT);
      sqlite3_bind_blob(st, 2, data, (int)data_size, SQLITE_TRANSIENT);
      int rc = sqlite3_step(st); sqlite3_finalize(st);
      return (rc==SQLITE_DONE) ? 0 : -2;
    }
#endif
    // fallback in-memory
    cache_entry_t* entry = (cache_entry_t*)malloc(sizeof(cache_entry_t));
    if (!entry) return -2;
    entry->key = strdup(key);
    entry->src = strdup(src);
    entry->data = (char*)malloc(data_size + 1);
    if (!entry->key || !entry->src || !entry->data) {
        if (entry->key) free(entry->key);
        if (entry->src) free(entry->src);
        if (entry->data) free(entry->data);
        free(entry);
        return -2;
    }
    memcpy(entry->data, data, data_size);
    entry->data[data_size] = '\0';
    entry->data_size = data_size;
    entry->next = cache_head;
    cache_head = entry;
    return 0;
}

// ---- Text/ANSI helpers ----
static int _color_index(const char* color){
  if (!color) return -1;
  char buf[16]={0}; size_t n=strlen(color); if (n>15) n=15; for(size_t i=0;i<n;i++) buf[i]=tolower((unsigned char)color[i]);
  const char* names[] = {"black","red","green","yellow","blue","magenta","cyan","white"};
  for (int i=0;i<8;i++) {
    if (strcmp(buf,names[i])==0) return i;
  }
  return -1;
}

char* tg_colored(const char* st, const char* color, int background){
  if (!color) return strdup(st?st:"");
  int idx = _color_index(color); if (idx<0) return strdup(st?st:"");
  int bright = (color[0] && isupper((unsigned char)color[0])) ? 60 : 0;
  int code = 30 + idx + bright + (background?10:0);
  size_t sl = st?strlen(st):0; char* out = (char*)malloc(sl + 16 + 4);
  if (!out) return NULL;
  sprintf(out, "\x1b[%dm%s\x1b[0m", code, st?st:"");
  return out;
}

char* tg_colorize_float(float x){
  const char* color = (x < 0.75f) ? "green" : (x > 1.15f ? "red" : "yellow");
  char buf[64]; snprintf(buf, sizeof(buf), "%7.2fx", x);
  return tg_colored(buf, color, 0);
}

char* tg_ansistrip(const char* s){
  if (!s) return strdup("");
  size_t n=strlen(s); char* out=(char*)malloc(n+1); if(!out) return NULL; size_t j=0;
  for (size_t i=0;i<n;){
    if (s[i]=='\x1b' && i+1<n && s[i+1]=='['){
      // skip escape sequence until 'm' or 'K'
      i+=2;
      while (i<n && s[i] && !(s[i]=='m' || s[i]=='K')) i++;
      if (i<n) i++;
    } else {
      out[j++]=s[i++];
    }
  }
  out[j]='\0'; return out;
}

int tg_ansilen(const char* s){ char* t=tg_ansistrip(s); int l=t?(int)strlen(t):0; free(t); return l; }

char* tg_time_to_str(double t, int w){
  // if t>10s -> seconds; elif t>0.01s -> ms; else us
  char fmt[32]; snprintf(fmt,sizeof(fmt), "%%%d.2f%%s", w);
  char* out=(char*)malloc(64); if(!out) return NULL;
  if (t > 10.0) { snprintf(out,64,fmt, t*1.0, "s "); }
  else if (t > 0.01) { snprintf(out,64,fmt, t*1e3, "ms"); }
  else { snprintf(out,64,fmt, t*1e6, "us"); }
  return out;
}

char* tg_strip_parens(const char* s){
  if (!s) return strdup("");
  size_t n=strlen(s);
  if (n>=2 && s[0]=='(' && s[n-1]==')'){
    // check simple balance condition similar to reference
    int has_open=0, has_close=0; for(size_t i=1;i<n-1;i++){ if(s[i]=='(') {has_open=1; break;} }
    for(size_t i=1;i<n-1;i++){ if(s[i]==')') {has_close=1; break;} }
    if (!has_open || has_open <= has_close) { char* out=(char*)malloc(n-1); if(!out) return NULL; memcpy(out,s+1,n-2); out[n-2]='\0'; return out; }
  }
  return strdup(s);
}

char* tg_word_wrap(const char* s, int wrap){
  if (!s) return strdup("");
  if (wrap<=0) return strdup(s);
  // Strip ANSI for length decisions
  char* plain = tg_ansistrip(s);
  int n = (int)strlen(plain);
  // allocate rough size: original len + newlines
  char* out = (char*)malloc(n*2 + 2); if(!out){ free(plain); return NULL; }
  int oi=0; int i=0; while (i<n){
    int start=i; int col=0; while (i<n && col<wrap){ i++; col++; }
    // break line
    memcpy(out+oi, plain+start, i-start); oi += (i-start);
    if (i<n) out[oi++]='\n';
  }
  out[oi]='\0'; free(plain); return out;
}

// ---- Math/bit helpers ----
int64_t tg_py_floor_div(int64_t a, int64_t b){
  // floor division like Python //
  int64_t q = a / b; int64_t r = a % b;
  if ((r != 0) && ((r>0) != (b>0))) q -= 1; // adjust when signs differ and remainder non-zero
  return q;
}

int64_t tg_ceildiv(int64_t num, int64_t amt){ return -tg_py_floor_div(num, -amt); }
int64_t tg_round_up(int64_t num, int64_t amt){ return tg_ceildiv(num, amt) * amt; }
int64_t tg_round_down(int64_t num, int64_t amt){ return -tg_round_up(-num, amt); }

int64_t tg_cdiv(int64_t x, int64_t y){ if (y==0) return 0; int64_t q = (int64_t)(llabs(x)/llabs(y)); if (x*y<0) q = -q; return q; }
int64_t tg_cmod(int64_t x, int64_t y){ if (y==0) return 0; return x - tg_cdiv(x,y)*y; }

uint32_t tg_lo32(uint64_t x){ return (uint32_t)(x & 0xFFFFFFFFu); }
uint32_t tg_hi32(uint64_t x){ return (uint32_t)((x >> 32) & 0xFFFFFFFFu); }
void tg_data64(uint64_t data, uint32_t* hi, uint32_t* lo){ if(hi) *hi=tg_hi32(data); if(lo) *lo=tg_lo32(data); }
void tg_data64_le(uint64_t data, uint32_t* lo, uint32_t* hi){ if(lo) *lo=tg_lo32(data); if(hi) *hi=tg_hi32(data); }
uint64_t tg_getbits(uint64_t value, int start, int end){ return (value >> start) & ((1ull << (end - start + 1)) - 1ull); }
uint64_t tg_i2u(int bits, int64_t value){ if (value >= 0) return (uint64_t)value; return (1ull<<bits) + (uint64_t)value; }

// ---- wait_cond ----
#ifdef _WIN32
#include <windows.h>
static unsigned long long _now_ms(void){ return GetTickCount64(); }
static void _sleep_ms(unsigned ms){ Sleep(ms); }
#else
#include <sys/time.h>
static unsigned long long _now_ms(void){ struct timeval tv; gettimeofday(&tv,NULL); return (unsigned long long)tv.tv_sec*1000ull + (unsigned long long)(tv.tv_usec/1000); }
static void _sleep_ms(unsigned ms){ struct timespec ts; ts.tv_sec = ms/1000; ts.tv_nsec = (ms%1000)*1000000ull; nanosleep(&ts, NULL); }
#endif

int tg_wait_cond(int (*cb)(void*), void* ctx, int value, int timeout_ms, const char* msg){
  (void)msg; if (!cb) return 0; unsigned long long start = _now_ms();
  while ((_now_ms() - start) < (unsigned long long)(timeout_ms>=0?timeout_ms:0x7FFFFFFF)){
    int v = cb(ctx);
    if (v == value) return 1;
    _sleep_ms(1);
  }
  return 0;
}

// ---- sequence helpers ----
int tg_all_same_int(const int* arr, int n){ if(n<=1) return 1; for(int i=1;i<n;i++) if(arr[i]!=arr[0]) return 0; return 1; }

static const int* _argsort_ctx = NULL;
static int _cmp_idx_int(const void* a, const void* b){
  int ia = *(const int*)a; int ib = *(const int*)b;
  if (_argsort_ctx[ia] < _argsort_ctx[ib]) return -1;
  if (_argsort_ctx[ia] > _argsort_ctx[ib]) return 1;
  return 0;
}

void tg_argsort_int(const int* arr, int n, int* out_idx){
  for(int i=0;i<n;i++) out_idx[i]=i;
  _argsort_ctx = arr;
  qsort(out_idx, n, sizeof(int), _cmp_idx_int);
  _argsort_ctx = NULL;
}

int tg_dedup_str(char** in, int n, char*** out){
  if (n<=0){ *out=NULL; return 0; }
  char** res = (char**)malloc(n*sizeof(char*)); int rc=0;
  for(int i=0;i<n;i++){
    int seen=0; for(int j=0;j<rc;j++){ if (strcmp(in[i], res[j])==0){ seen=1; break; } }
    if (!seen){ res[rc++] = strdup(in[i]); }
  }
  *out = res; return rc;
}

// ---- diskcache clear ----
void tg_diskcache_clear(void){
#ifdef TG_HAVE_SQLITE3
  if (_db){ sqlite3_close(_db); _db=NULL; }
  const char* path = tg_getenv("CACHEDB"); char defpath[256]; if (!path || !*path){ snprintf(defpath,sizeof(defpath),"build/cache/cache.db"); path=defpath; }
  remove(path);
#endif
  // Since cache_head is static in this file, expose clearing behavior here
  cache_entry_t* it = cache_head; cache_entry_t* nx;
  while (it){ nx = it->next; free(it->key); free(it->src); free(it->data); free(it); it = nx; }
  cache_head = NULL;
}

// ---- profiling ----
typedef struct { tg_profile_range_event_t* v; int n, cap; } _vec_prof_t;
static _vec_prof_t _prof = {0};
static int _profile_enabled = -1; // -1=unset, 0=off, 1=on
static unsigned long long _now_us(void){
#ifdef _WIN32
  return GetTickCount64()*1000ull;
#else
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (unsigned long long)ts.tv_sec*1000000ull + ts.tv_nsec/1000ull;
#endif
}
static void _profile_init_from_env(void){ if (_profile_enabled==-1){ _profile_enabled = tg_is_ci() ? 0 : (tg_getenv("PROFILE") && *tg_getenv("PROFILE") ? 1 : 0); } }
void tg_profile_events_clear(void){ free(_prof.v); _prof.v=NULL; _prof.n=_prof.cap=0; }
int  tg_profile_events_count(void){ return _prof.n; }
const tg_profile_range_event_t* tg_profile_events_data(void){ return _prof.v; }
static int _push_prof(const tg_profile_range_event_t* e){ if (_prof.n+1>_prof.cap){ int nc=_prof.cap? _prof.cap*2:16; _prof.v=realloc(_prof.v, nc*sizeof(*_prof.v)); _prof.cap=nc; } _prof.v[_prof.n]=*e; return _prof.n++; }
int tg_cpu_profile_begin(const char* name, const char* device, int is_copy){
  _profile_init_from_env(); if (!_profile_enabled) return -1;
  tg_profile_range_event_t e={device?device:"CPU", name?name:"", _now_us(), 0, is_copy}; return _push_prof(&e);
}
void tg_cpu_profile_end(int handle, int display){
  if (handle<0) return;
  if(handle>=0 && handle<_prof.n){
    _prof.v[handle].en_us=_now_us();
    if(display && _profile_enabled){
      fprintf(stderr, "%s %s %llu us\n", _prof.v[handle].device, _prof.v[handle].name, (unsigned long long)(_prof.v[handle].en_us - _prof.v[handle].st_us));
    }
  }
}
void tg_profile_set_enabled(int enabled){ _profile_enabled = enabled ? 1 : 0; }
int  tg_profile_get_enabled(void){ _profile_init_from_env(); return _profile_enabled; }

// Dump profile events at exit when enabled (JSON for ease of parsing)
static void _mkpath(const char* path){ if(!path) return; char tmp[512]; strncpy(tmp, path, sizeof(tmp)-1); tmp[sizeof(tmp)-1]='\0'; char* p = strrchr(tmp, '/'); if (!p) return; *p='\0';
#ifdef _WIN32
  _mkdir(tmp);
#else
  char t2[512]; strncpy(t2, tmp, sizeof(t2)-1); t2[sizeof(t2)-1]='\0';
  for(char* q=t2+1; *q; q++){ if(*q=='/'){ *q='\0'; mkdir(t2, 0755); *q='/'; } }
  mkdir(t2, 0755);
#endif
}
static void _profile_atexit(void){
  if (!tg_profile_get_enabled()) return;
  const char* out = tg_getenv("PROFILE_OUT");
  char def[128]; if (!out || !*out){ snprintf(def, sizeof(def), "build/profile.json"); out = def; }
  _mkpath(out);
  FILE* f = fopen(out, "w"); if (!f) return;
  fprintf(f, "[\n");
  for (int i=0;i<_prof.n;i++){
    const tg_profile_range_event_t* e = &_prof.v[i];
    unsigned long long dur = (e->en_us>e->st_us)? (e->en_us - e->st_us) : 0ull;
    fprintf(f, " {\"device\":\"%s\",\"name\":\"%s\",\"st_us\":%llu,\"en_us\":%llu,\"dur_us\":%llu,\"is_copy\":%d}%s\n",
            e->device?e->device:"", e->name?e->name:"", (unsigned long long)e->st_us, (unsigned long long)e->en_us, dur, e->is_copy, (i+1<_prof.n)?",":"");
  }
  fprintf(f, "]\n"); fclose(f);
}
#ifdef __GNUC__
__attribute__((constructor)) static void _profile_ctor(void){ atexit(_profile_atexit); }
#else
static void _profile_ctor(void){ atexit(_profile_atexit); }
__attribute__((constructor)) static void _call_ctor(void){ _profile_ctor(); }
#endif

// ---- Fetch helper (high-level policy) ----
// Forward declarations for helpers used below
static int _file_exists(const char* p);
static void _mkdir_p(const char* path);
// Treat anything without a URL scheme as local, or absolute/relative paths
static int _is_local_path(const char* s){
  if (!s) return 0;
  if (s[0]=='/' || s[0]=='.') return 1;
  // windows drive letters like C:\\ (avoid trailing backslash)
#ifdef _WIN32
  if (isalpha((unsigned char)s[0]) && s[1]==':' && (s[2]=='/' || s[2]=='\\')) return 1;
#endif
  // no scheme present -> local
  return strstr(s, "://") == NULL || _file_exists(s);
}
static int _file_exists(const char* p){ struct stat st; return (stat(p,&st)==0 && S_ISREG(st.st_mode)); }
static void _mkdir_p(const char* path){ if(!path) return; char tmp[1024]; strncpy(tmp,path,sizeof(tmp)-1); tmp[sizeof(tmp)-1]='\0'; for(char* p=tmp+1; *p; p++){ if(*p=='/'){ *p='\0'; mkdir(tmp,0755); *p='/'; } } mkdir(tmp,0755); }
static int _is_tinybox(void){ FILE* f=fopen("/etc/tinybox-release","r"); if(f){ fclose(f); return 1;} return 0; }
static const char* _downloads_root(void){ return _is_tinybox()? "/raid/downloads" : "build/downloads"; }
static void _join2(char* out, size_t n, const char* a, const char* b){
  if (!out || n==0) return;
  out[0] = '\0';
  if (!a) a = "";
  if (!b || !*b) {
    size_t la = strnlen(a, n-1);
    memcpy(out, a, la);
    out[la] = '\0';
  } else {
    size_t la = strnlen(a, n-1);
    memcpy(out, a, la);
    size_t pos = la;
    if (pos < n-1) out[pos++] = '/';
    size_t rem = (pos < n) ? (n - 1 - pos) : 0;
    if (rem > 0) {
      size_t lb = strnlen(b, rem);
      memcpy(out + pos, b, lb);
      pos += lb;
    }
    out[pos] = '\0';
  }
}
static void _append_suffix(char* out, size_t n, const char* base, const char* suffix){
  if (!out || n==0) return;
  out[0] = '\0';
  if (!base) base = "";
  size_t lb = strnlen(base, n-1);
  memcpy(out, base, lb);
  size_t pos = lb;
  if (suffix && *suffix && pos < n-1) {
    size_t rem = n - 1 - pos;
    size_t ls = strnlen(suffix, rem);
    memcpy(out + pos, suffix, ls);
    pos += ls;
  }
  out[pos] = '\0';
}
static void __attribute__((unused)) _join3(char* out, size_t n, const char* a, const char* b, const char* c){ char t[1024]; _join2(t,sizeof(t),a,b); _join2(out,n,t,c); }
static const char* _basename(const char* s){ const char* p=strrchr(s,'/'); return p? p+1 : s; }

int tg_fetch(const char* url, const char* name, const char* subdir, int gunzip, int allow_caching,
             char* out_path, size_t out_sz){
  if (!url || !out_path || out_sz==0) return -1;
  const int dbg = tg_getenv("TG_FETCH_DEBUG") && *tg_getenv("TG_FETCH_DEBUG");
  if (dbg) fprintf(stderr, "tg_fetch url='%s' gunzip=%d allow_caching=%d\n", url, gunzip, allow_caching);
  // Local path passthrough
  if (_is_local_path(url)){
    if (dbg) fprintf(stderr, "tg_fetch: treating as local path\n");
    // optional gunzip of local file
    if (gunzip){ char out_gz[1024]; _append_suffix(out_gz,sizeof(out_gz), url, ".gunzip");
      // ensure directory exists
      char dir[1024]; strncpy(dir,out_gz,sizeof(dir)-1); dir[sizeof(dir)-1]='\0'; char* slash=strrchr(dir,'/'); if(slash){ *slash='\0'; _mkdir_p(dir); }
      int rc = tg_gunzip_impl(url, out_gz); if (rc!=0) { /*fprintf(stderr, "tg_fetch local gunzip failed %d from %s to %s\n", rc, url, out_gz);*/ return rc; } snprintf(out_path,out_sz,"%s", out_gz); return 0; }
    snprintf(out_path,out_sz,"%s", url); return 0;
  }
  if (dbg) fprintf(stderr, "tg_fetch: treating as remote URL\n");
  // Determine downloads directory
  char root[1024]; snprintf(root,sizeof(root),"%s", _downloads_root());
  if (subdir && *subdir){ char tmp[1024]; _join2(tmp,sizeof(tmp),root,subdir); strncpy(root,tmp,sizeof(root)-1); root[sizeof(root)-1]='\0'; }
  _mkdir_p(root);
  // Determine filename
  const char* base = name && *name ? name : _basename(url);
  char final[1024]; _join2(final,sizeof(final),root,base);
  if (gunzip){ char tmp[1024]; _append_suffix(tmp,sizeof(tmp), final, ".gunzip"); strncpy(final,tmp,sizeof(final)-1); final[sizeof(final)-1]='\0'; }
  // Caching: if allowed and file exists, return it
  int http_cache_disabled = (tg_getenv("DISABLE_HTTP_CACHE") && *tg_getenv("DISABLE_HTTP_CACHE"));
  if (allow_caching && !http_cache_disabled && _file_exists(final)){ snprintf(out_path,out_sz,"%s", final); return 0; }
  // Temp path for download
  char tmpdl[1024]; _join2(tmpdl,sizeof(tmpdl), root, ".tmp_download");
  // Progress
  int enable_progress = !tg_is_ci(); if (enable_progress){ tg_tqdm_begin(url, 0); }
  int frc = tg_fetch_impl(url, tmpdl, enable_progress);
  if (enable_progress){ tg_tqdm_end(); }
  if (frc!=0) return frc;
  // Gunzip if requested
  if (gunzip){ char tmpout[1024]; _append_suffix(tmpout,sizeof(tmpout), final, ".tmpgunzip"); int grc = tg_gunzip_impl(tmpdl, tmpout); if (grc!=0){ remove(tmpdl); return grc; } remove(tmpdl); rename(tmpout, final); }
  else { rename(tmpdl, final); }
  snprintf(out_path,out_sz,"%s", final); return 0;
}

GPtrArray* tg_flatten_ptr_array(GPtrArray* lists) {
  GPtrArray* out = g_ptr_array_new();
  if (!lists) return out;
  for (guint i = 0; i < lists->len; i++) {
    GPtrArray* sub = g_ptr_array_index(lists, i);
    if (!sub) continue;
    for (guint j = 0; j < sub->len; j++) {
      g_ptr_array_add(out, g_ptr_array_index(sub, j));
    }
  }
  return out;
}

tg_partition_result_t tg_partition_ptr_array(GPtrArray* items, gboolean (*pred)(void*, void*), void* ctx) {
  tg_partition_result_t res;
  res.true_items = g_ptr_array_new();
  res.false_items = g_ptr_array_new();
  if (!items) return res;
  for (guint i = 0; i < items->len; i++) {
    void* item = g_ptr_array_index(items, i);
    gboolean keep = pred ? pred(ctx, item) : (item != NULL);
    if (keep) g_ptr_array_add(res.true_items, item);
    else g_ptr_array_add(res.false_items, item);
  }
  return res;
}

void tg_partition_result_free(tg_partition_result_t* result) {
  if (!result) return;
  if (result->true_items) g_ptr_array_unref(result->true_items);
  if (result->false_items) g_ptr_array_unref(result->false_items);
  result->true_items = result->false_items = NULL;
}

int tg_amx_enabled(void) {
  _amx_init();
  return _amx_enabled;
}

void tg_set_amx_enabled(int enabled) {
  _amx_initialized = 1;
  _amx_enabled = enabled ? 1 : 0;
}

// ---- ctypes-like helpers ----
uintptr_t tg_mv_address(void* p){ return (uintptr_t)p; }
char** tg_to_char_p_p(const char* const* arr, int n){ if (n<=0) return NULL; char** out=(char**)malloc((size_t)n*sizeof(char*)); for(int i=0;i<n;i++) out[i]=strdup(arr[i]); return out; }

// ---- Exec helpers (stubs) ----
void tg_cpu_objdump(const unsigned char* lib, size_t len, const char* tool){
  if (!lib || len==0) return;
  const char* objdump = (tool && *tool) ? tool : "objdump";
  // write to a temp file under build/
  char path[256] = {0}; snprintf(path,sizeof(path),"build/objdump.tmp.bin");
  _mkdir_p("build");
  FILE* f = fopen(path, "wb"); if (!f) return; fwrite(lib,1,len,f); fclose(f);
  char cmd[512]; snprintf(cmd,sizeof(cmd),"%s -d %s", objdump, path);
  FILE* p = popen(cmd, "r");
  if (p){ char buf[1024]; size_t n; while ((n=fread(buf,1,sizeof(buf),p))>0) fwrite(buf,1,n,stdout); pclose(p); }
  remove(path);
  fflush(stdout);
}
int  tg_capstone_flatdump(const unsigned char* lib, size_t len){
#ifdef TG_HAVE_CAPSTONE
  if (!lib || len==0) return -1;
  csh handle; cs_err err;
#if defined(__x86_64__) || defined(_M_X64)
  err = cs_open(CS_ARCH_X86, CS_MODE_64, &handle);
#elif defined(__aarch64__)
  err = cs_open(CS_ARCH_ARM64, CS_MODE_ARM, &handle);
#else
  return -2;
#endif
  if (err != CS_ERR_OK) return -3;
  cs_option(handle, CS_OPT_DETAIL, CS_OPT_OFF);
  cs_insn* insn;
  size_t count = cs_disasm(handle, lib, len, 0, 0, &insn);
  for (size_t i=0;i<count;i++){
    printf("%#08llx: %s\t%s\n", (unsigned long long)insn[i].address, insn[i].mnemonic, insn[i].op_str);
  }
  cs_free(insn, count);
  cs_close(&handle);
  fflush(stdout);
  return 0;
#else
  (void)lib; (void)len; return -10;
#endif
}
