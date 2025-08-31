#ifndef SRC_HELPERS_HELPERS_H
#define SRC_HELPERS_HELPERS_H

#ifdef __cplusplus
extern "C" {
#endif

// Environment variable functions
const char* tg_getenv(const char* name);
const char* tg_getenv_default(const char* name, const char* default_val);

// Product function for shapes
int tg_prod(const int* shape, int len);

#ifdef __cplusplus
}
#endif
#endif /* SRC_HELPERS_HELPERS_H */
