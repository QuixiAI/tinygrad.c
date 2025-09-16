#include "helpers/string_builder.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void tg_sb_init(tg_string_builder* sb) {
  if (!sb) return;
  sb->data = NULL;
  sb->len = 0;
  sb->cap = 0;
}

void tg_sb_free(tg_string_builder* sb) {
  if (!sb) return;
  if (sb->data) free(sb->data);
  sb->data = NULL;
  sb->len = 0;
  sb->cap = 0;
}

int tg_sb_reserve(tg_string_builder* sb, size_t extra) {
  if (!sb) return -1;
  size_t required = sb->len + extra + 1;
  if (required <= sb->cap) return 0;
  size_t newcap = sb->cap ? sb->cap : 128;
  while (newcap < required) {
    size_t next = newcap * 2;
    if (next <= newcap) { // overflow guard
      newcap = required;
      break;
    }
    newcap = next;
  }
  char* newdata = (char*)realloc(sb->data, newcap);
  if (!newdata) return -1;
  sb->data = newdata;
  sb->cap = newcap;
  return 0;
}

int tg_sb_append_len(tg_string_builder* sb, const char* text, size_t len) {
  if (!sb || !text || len == 0) {
    if (sb && !sb->data) {
      if (tg_sb_reserve(sb, 0) != 0) return -1;
      sb->data[0] = '\0';
    }
    return 0;
  }
  if (tg_sb_reserve(sb, len) != 0) return -1;
  memcpy(sb->data + sb->len, text, len);
  sb->len += len;
  sb->data[sb->len] = '\0';
  return 0;
}

int tg_sb_append(tg_string_builder* sb, const char* text) {
  if (!text) return tg_sb_append_len(sb, "", 0);
  return tg_sb_append_len(sb, text, strlen(text));
}

int tg_sb_vappendf(tg_string_builder* sb, const char* fmt, va_list ap) {
  if (!sb || !fmt) return -1;
  va_list copy;
  va_copy(copy, ap);
  int needed = vsnprintf(NULL, 0, fmt, copy);
  va_end(copy);
  if (needed < 0) return -1;
  if (tg_sb_reserve(sb, (size_t)needed) != 0) return -1;
  va_copy(copy, ap);
  vsnprintf(sb->data + sb->len, sb->cap - sb->len, fmt, copy);
  va_end(copy);
  sb->len += (size_t)needed;
  sb->data[sb->len] = '\0';
  return 0;
}

int tg_sb_appendf(tg_string_builder* sb, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int rv = tg_sb_vappendf(sb, fmt, ap);
  va_end(ap);
  return rv;
}

char* tg_sb_build(tg_string_builder* sb) {
  if (!sb) return NULL;
  if (!sb->data) {
    char* out = (char*)malloc(1);
    if (out) out[0] = '\0';
    return out;
  }
  char* out = sb->data;
  sb->data = NULL;
  sb->len = 0;
  sb->cap = 0;
  return out;
}

const char* tg_sb_data(const tg_string_builder* sb) {
  if (!sb || !sb->data) return "";
  return sb->data;
}

size_t tg_sb_length(const tg_string_builder* sb) {
  return sb ? sb->len : 0;
}

char* tg_sb_new_with(const char* a, const char* b) {
  tg_string_builder sb;
  tg_sb_init(&sb);
  if (a && tg_sb_append(&sb, a) != 0) {
    tg_sb_free(&sb);
    return NULL;
  }
  if (b && tg_sb_append(&sb, b) != 0) {
    tg_sb_free(&sb);
    return NULL;
  }
  char* out = tg_sb_build(&sb);
  if (!out) tg_sb_free(&sb);
  return out;
}

char* tg_sb_append_len_owned(char* base, const char* add, size_t len) {
  if (!add || len == 0) return base ? base : tg_sb_new_with("", "");
  if (!base) {
    char* out = (char*)malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, add, len);
    out[len] = '\0';
    return out;
  }
  size_t la = strlen(base);
  char* out = (char*)realloc(base, la + len + 1);
  if (!out) return base; // leave original untouched on failure
  memcpy(out + la, add, len);
  out[la + len] = '\0';
  return out;
}

char* tg_sb_append_owned(char* base, const char* add) {
  if (!add) return base;
  return tg_sb_append_len_owned(base, add, strlen(add));
}
