#ifndef TINYGRADC_FETCH_H
#define TINYGRADC_FETCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Downloads url to out_tmp path. If gunzip!=0 and TG_HAVE_ZLIB, caller may
// pass a .gunzip final path separately and invoke tg_gunzip_impl.
// Returns 0 on success, non-zero on error or unsupported.
int tg_fetch_impl(const char* url, const char* out_tmp, int allow_progress);

// Gunzip in_path to out_path using zlib if available. Returns 0 on success.
int tg_gunzip_impl(const char* in_path, const char* out_path);

#ifdef __cplusplus
}
#endif

#endif // TINYGRADC_FETCH_H
