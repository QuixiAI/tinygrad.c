#ifndef TG_HELPERS_STRING_BUILDER_H
#define TG_HELPERS_STRING_BUILDER_H

#include <stddef.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tg_string_builder {
  char* data;
  size_t len;
  size_t cap;
} tg_string_builder;

void tg_sb_init(tg_string_builder* sb);
void tg_sb_free(tg_string_builder* sb);
int tg_sb_reserve(tg_string_builder* sb, size_t extra);
int tg_sb_append_len(tg_string_builder* sb, const char* text, size_t len);
int tg_sb_append(tg_string_builder* sb, const char* text);
int tg_sb_appendf(tg_string_builder* sb, const char* fmt, ...);
int tg_sb_vappendf(tg_string_builder* sb, const char* fmt, va_list ap);
char* tg_sb_build(tg_string_builder* sb);
const char* tg_sb_data(const tg_string_builder* sb);
size_t tg_sb_length(const tg_string_builder* sb);

// Convenience helpers matching prior ad-hoc usage (owning char* builder)
char* tg_sb_new_with(const char* a, const char* b);
char* tg_sb_append_owned(char* base, const char* add);
char* tg_sb_append_len_owned(char* base, const char* add, size_t len);

#ifdef __cplusplus
}
#endif

#endif // TG_HELPERS_STRING_BUILDER_H
