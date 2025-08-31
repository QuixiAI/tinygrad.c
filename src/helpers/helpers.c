/* helpers.c
 * Minimal implementation of helper functions needed for dtype.c
 */
#include <stdlib.h>
#include <string.h>
#include "helpers.h"

// Environment variable functions
const char* tg_getenv(const char* name) {
    return getenv(name);
}

const char* tg_getenv_default(const char* name, const char* default_val) {
    const char* val = getenv(name);
    return val ? val : default_val;
}

// Product function for shapes
int tg_prod(const int* shape, int len) {
    int result = 1;
    for (int i = 0; i < len; i++) {
        result *= shape[i];
    }
    return result;
}
