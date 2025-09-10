#ifndef SRC_HELPERS_HELPERS_H
#define SRC_HELPERS_HELPERS_H
/* helpers.h
 * Ported from tinygrad/helpers.py to C.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Argument handling */
typedef struct {
    int64_t *data;
    size_t len;
    bool is_valid;
} tg_argfix_result_t;

tg_argfix_result_t tg_argfix_int64(const int64_t **arrays, const size_t *lens, size_t num_arrays);

/* Type checking */
bool tg_all_int(const void **values, size_t len, const char **type_names);
bool tg_is_numpy_ndarray(const char *type_str);

/* String and formatting utilities */
char* tg_colorize_float(double x);
char* tg_time_to_str(double t, int w);
char* tg_strip_parens(const char *fst);
char* tg_to_function_name(const char *s);

/* Numeric utilities */
uint64_t tg_i2u(int bits, int64_t value);
double tg_polyN(double x, const double *p, size_t len);

/* Safety utilities */
void* tg_unwrap(void *x);

typedef struct {
    void *element;
    bool success;
} tg_single_element_result_t;

tg_single_element_result_t tg_get_single_element(const void **arr, size_t len);

/* Error suppression utilities */
typedef enum {
    TG_SUPPRESS_NONE = 0,
    TG_SUPPRESS_ATTRIBUTE_ERROR = 1,
    TG_SUPPRESS_TYPE_ERROR = 2,
    TG_SUPPRESS_IMPORT_ERROR = 4,
    TG_SUPPRESS_ALL = 7
} tg_suppress_flags_t;

typedef struct {
    int success;
    int error_type;
    char *error_msg;
} tg_suppress_result_t;

tg_suppress_result_t tg_suppress_finalizing(void (*func)(void*), void *arg, tg_suppress_flags_t flags);

/* Core Math Utilities */
int64_t tg_prod(const int64_t *arr, size_t len);
int64_t tg_ceildiv(int64_t num, int64_t amt);
int64_t tg_round_up(int64_t num, int64_t amt);
int64_t tg_round_down(int64_t num, int64_t amt);
int64_t tg_cdiv(int64_t x, int64_t y);
int64_t tg_cmod(int64_t x, int64_t y);

/* Bit Operations */
uint32_t tg_lo32(uint64_t x);
uint32_t tg_hi32(uint64_t x);
void tg_data64(uint64_t data, uint32_t *hi, uint32_t *lo);
void tg_data64_le(uint64_t data, uint32_t *lo, uint32_t *hi);
uint64_t tg_getbits(uint64_t value, int start, int end);

/* String Utilities */
char* tg_colored(const char *st, const char *color);
char* tg_ansistrip(const char *s);
size_t tg_ansilen(const char *s);
char* tg_word_wrap(const char *x, int wrap);
char* tg_pluralize(const char *st, int cnt);

/* Collection Utilities */
int64_t* tg_dedup_int64(const int64_t *arr, size_t len, size_t *out_len);
int64_t* tg_make_tuple(int64_t x, size_t cnt, size_t *out_len);
int64_t* tg_flatten_int64(const int64_t **nested_arr, const size_t *lens, size_t num_arrays, size_t *out_len);
bool tg_all_same_int64(const int64_t *arr, size_t len);
size_t* tg_argsort_int64(const int64_t *arr, size_t len);

/* System Utilities */
char* tg_temp(const char *x, bool append_user);
int64_t tg_getenv_int64(const char *key, int64_t default_value);
bool tg_is_OSX();
bool tg_is_CI();
char* tg_getpass_user();

/* Progress Bar (tqdm) */
typedef struct {
    const char *desc;
    bool disable;
    const char *unit;
    bool unit_scale;
    size_t total;
    size_t current;
    double start_time;
    size_t update_count;
    size_t skip_count;
    size_t rate_limit;
} tg_tqdm_t;

void tg_tqdm_init(tg_tqdm_t *tqdm, const char *desc, bool disable, const char *unit, bool unit_scale, size_t total, size_t rate_limit);
void tg_tqdm_update(tg_tqdm_t *tqdm, size_t n);
void tg_tqdm_finish(tg_tqdm_t *tqdm);
void tg_tqdm_write(const char *s);

/* Range iterator with progress bar */
typedef struct {
    tg_tqdm_t tqdm;
    size_t current;
    size_t end;
} tg_trange_t;

void tg_trange_init(tg_trange_t *trange, size_t n, const char *desc, bool disable, const char *unit);
void tg_trange_next(tg_trange_t *trange);
bool tg_trange_has_next(tg_trange_t *trange);
void tg_trange_finish(tg_trange_t *trange);

/* Internal helper functions (not part of public API) */
char* tg_word_wrap_single_line(const char *line, int wrap);

/* TODO: More function declarations will be added as implementation progresses */

#ifdef __cplusplus
}
#endif
#endif /* SRC_HELPERS_HELPERS_H */
