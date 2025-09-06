#include "fetch.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include "tqdm.h"

#ifdef TG_HAVE_CURL
#include <curl/curl.h>
typedef struct { FILE* fp; long last; long total; int enable_progress; unsigned long long written; } dl_ctx_t;
static size_t _write_cb(void* ptr, size_t sz, size_t nm, void* userdata){ dl_ctx_t* c=(dl_ctx_t*)userdata; size_t wr = fwrite(ptr, sz, nm, c->fp); c->written += (unsigned long long)(wr*sz); return wr; }
static int _xfer_cb(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow){ (void)ultotal; (void)ulnow; dl_ctx_t* c=(dl_ctx_t*)clientp; if (!c->enable_progress) return 0; if (dltotal>0){ if (c->last != (long)dlnow){ tg_tqdm_update((long)dlnow); c->last=(long)dlnow; } } return 0; }
int tg_fetch_impl(const char* url, const char* out_tmp, int allow_progress){
  CURL* curl = curl_easy_init(); if (!curl) return -1;
  FILE* f = fopen(out_tmp, "wb"); if (!f){ curl_easy_cleanup(curl); return -2; }
  dl_ctx_t ctx = { .fp=f, .last=-1, .total=0, .enable_progress=allow_progress, .written=0 };
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "tinygrad.c/0.1");
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, _write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
#ifdef CURLOPT_XFERINFOFUNCTION
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, _xfer_cb);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &ctx);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, allow_progress?0L:1L);
#endif
  long code=0; int ret=0;
  CURLcode rc = curl_easy_perform(curl);
  if (rc!=CURLE_OK){ ret=-3; goto done; }
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
  if (code<200 || code>=300){ ret=-4; goto done; }
  // verify length if provided
#ifdef CURLINFO_CONTENT_LENGTH_DOWNLOAD_T
  curl_off_t clen = -1; if (curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &clen)==CURLE_OK && clen>0){
    if ((unsigned long long)clen != ctx.written) { ret=-8; }
  }
#endif
done:
  fclose(f);
  curl_easy_cleanup(curl);
  return ret;
}
#else
int tg_fetch_impl(const char* url, const char* out_tmp, int allow_progress){ (void)url; (void)out_tmp; (void)allow_progress; return -10; }
#endif

#ifdef TG_HAVE_ZLIB
#include <zlib.h>
int tg_gunzip_impl(const char* in_path, const char* out_path){
  gzFile gz = gzopen(in_path, "rb"); if (!gz) return -1;
  FILE* fo = fopen(out_path, "wb"); if (!fo){ gzclose(gz); return -2; }
  int ret = 0; unsigned char buf[1<<15];
  int n;
  while ((n = gzread(gz, buf, sizeof(buf))) > 0){
    size_t off = 0; while (off < (size_t)n){
      size_t wr = fwrite(buf+off,1,(size_t)n - off,fo);
      if (wr == 0){ ret=-3; break; }
      off += wr;
    }
    if (ret!=0) break;
  }
  int zrc = gzclose(gz); fclose(fo);
  if (zrc != Z_OK) return -4;
  return ret;
}
#else
int tg_gunzip_impl(const char* in_path, const char* out_path){ (void)in_path; (void)out_path; return -10; }
#endif
