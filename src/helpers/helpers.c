/* helpers.c
 * Implementation of utility functions ported from tinygrad/helpers.py
 */
#include "helpers.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <regex.h>
#include <unistd.h>
#include <pwd.h>
#include <sys/utsname.h>
#include <time.h>
#include <sys/ioctl.h>
#include <math.h>

/* Core Math Utilities */

/* NOTE: tg_prod returns int64_t 1 if arr is empty regardless of the type of arr */
int64_t tg_prod(const int64_t *arr, size_t len) {
  if (len == 0) return 1;
  int64_t result = 1;
  for (size_t i = 0; i < len; i++) {
    result *= arr[i];
  }
  return result;
}

int64_t tg_ceildiv(int64_t num, int64_t amt) {
  return (num + amt - 1) / amt;
}

int64_t tg_round_up(int64_t num, int64_t amt) {
  return tg_ceildiv(num, amt) * amt;
}

int64_t tg_round_down(int64_t num, int64_t amt) {
  return (num / amt) * amt;
}

/* C-style division and modulo (matches Python's // and % behavior for negative numbers) */
int64_t tg_cdiv(int64_t x, int64_t y) {
  if (y == 0) return 0;
  /* Python // is floor division, not truncation */
  int64_t q = x / y;
  int64_t r = x % y;
  /* Adjust if remainder has wrong sign (need to floor toward -inf) */
  if ((r != 0) && ((x < 0) != (y < 0))) {
    q -= 1;
  }
  return q;
}

int64_t tg_cmod(int64_t x, int64_t y) {
  return x - tg_cdiv(x, y) * y;
}

/* Bit Operations */

uint32_t tg_lo32(uint64_t x) {
  return (uint32_t)(x & 0xFFFFFFFFULL);
}

uint32_t tg_hi32(uint64_t x) {
  return (uint32_t)(x >> 32);
}

void tg_data64(uint64_t data, uint32_t *hi, uint32_t *lo) {
  *hi = tg_hi32(data);
  *lo = tg_lo32(data);
}

void tg_data64_le(uint64_t data, uint32_t *lo, uint32_t *hi) {
  *lo = tg_lo32(data);
  *hi = tg_hi32(data);
}

uint64_t tg_getbits(uint64_t value, int start, int end) {
  return (value >> start) & ((1ULL << (end - start + 1)) - 1);
}

/* String Utilities */

char* tg_colored(const char *st, const char *color) {
  if (!color) return strdup(st);

  const char *colors[] = {"black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"};
  int color_index = -1;
  for (int i = 0; i < 8; i++) {
    if (strcmp(colors[i], color) == 0) {
      color_index = i;
      break;
    }
  }

  if (color_index == -1) return strdup(st);

  bool is_upper = (color[0] >= 'A' && color[0] <= 'Z');
  int color_code = 30 + color_index;
  if (is_upper) color_code += 60; // Bright colors

  char *result = malloc(strlen(st) + 16); // Space for ANSI codes
  sprintf(result, "\033[%dm%s\033[0m", color_code, st);
  return result;
}

char* tg_ansistrip(const char *s) {
  if (!s) return NULL;

  size_t len = strlen(s);
  char *result = malloc(len + 1);
  size_t j = 0;

  for (size_t i = 0; i < len; ) {
    if (s[i] == '\x1b' && s[i+1] == '[') {
      // Skip ANSI escape sequence
      i += 2;
      while (i < len && ((s[i] >= '0' && s[i] <= '9') || s[i] == ';')) i++;
      if (i < len && (s[i] == 'm' || s[i] == 'K')) i++;
    } else {
      result[j++] = s[i++];
    }
  }
  result[j] = '\0';
  result = realloc(result, j + 1); // Shrink to actual size
  return result;
}

size_t tg_ansilen(const char *s) {
  if (!s) return 0;
  char *stripped = tg_ansistrip(s);
  size_t len = strlen(stripped);
  free(stripped);
  return len;
}

char* tg_word_wrap(const char *x, int wrap) {
  if (!x) return NULL;

  char *stripped = tg_ansistrip(x);
  size_t stripped_len = strlen(stripped);

  if (stripped_len <= (size_t)wrap) {
    free(stripped);
    return strdup(x);
  }

  // Check if input has multiple lines
  if (strchr(x, '\n')) {
    // Handle multiple lines: split, wrap each, rejoin
    char *result = NULL;
    size_t result_size = 0;
    char *x_copy = strdup(x);
    char *line = strtok(x_copy, "\n");
    
    while (line) {
      char *wrapped_line = tg_word_wrap(line, wrap); // Recursive call
      size_t wrapped_len = strlen(wrapped_line);
      
      result = realloc(result, result_size + wrapped_len + 2); // +2 for \n and \0
      if (result_size > 0) {
        strcpy(result + result_size, "\n");
        strcpy(result + result_size + 1, wrapped_line);
        result_size += wrapped_len + 1;
      } else {
        strcpy(result, wrapped_line);
        result_size = wrapped_len;
      }
      
      free(wrapped_line);
      line = strtok(NULL, "\n");
    }
    
    result[result_size] = '\0';
    free(x_copy);
    free(stripped);
    return result;
  }

  // Single line: find wrap point and recurse
  size_t i = 0;
  size_t x_len = strlen(x);
  while (i < x_len) {
    char *temp_substr = strndup(x, i + 1);
    size_t temp_len = tg_ansilen(temp_substr);
    free(temp_substr);
    if (temp_len > (size_t)wrap) break;
    i++;
  }
  
  char *first_part = strndup(x, i);
  char *rest = tg_word_wrap(x + i, wrap);
  
  size_t total_len = strlen(first_part) + 1 + strlen(rest) + 1; // +1 for \n, +1 for \0
  char *result = malloc(total_len);
  sprintf(result, "%s\n%s", first_part, rest);
  
  free(first_part);
  free(rest);
  free(stripped);
  return result;
}

char* tg_word_wrap_single_line(const char *line, int wrap) {
  size_t len = strlen(line);
  char *result = malloc(len * 2); // worst case: every char followed by newline
  size_t result_pos = 0;
  size_t i = 0;

  while (i < len) {
    size_t line_start = i;
    size_t line_end = i + wrap;

    if (line_end >= len) {
      // Last segment
      strcpy(result + result_pos, line + i);
      result_pos += len - i;
      break;
    }

    while (line_end > line_start && line[line_end] != ' ') {
      line_end--;
    }

    if (line_end == line_start) {
      line_end = i + wrap;
    }

    size_t segment_len = line_end - line_start;
    memcpy(result + result_pos, line + line_start, segment_len);
    result_pos += segment_len;
    result[result_pos++] = '\n';

    i = line_end;
    if (line[i] == ' ') i++; // Skip the space
  }

  result[result_pos] = '\0';
  result = realloc(result, result_pos + 1);
  return result;
}

char* tg_pluralize(const char *st, int cnt) {
  if (!st) return NULL;
  size_t len = strlen(st);
  char *result = malloc(len + 2); // +1 for 's', +1 for \0
  strcpy(result, st);
  if (cnt != 1) {
    result[len] = 's';
    result[len + 1] = '\0';
  }
  return result;
}

/* Collection Utilities */

int64_t* tg_dedup_int64(const int64_t *arr, size_t len, size_t *out_len) {
  if (!arr || len == 0) {
    *out_len = 0;
    return NULL;
  }

  // TODO(alpin): use a more efficient algorithm
  int64_t *seen = calloc(len, sizeof(int64_t));
  int64_t *result = malloc(len * sizeof(int64_t));
  size_t result_len = 0;

  for (size_t i = 0; i < len; i++) {
    bool found = false;
    for (size_t j = 0; j < result_len; j++) {
      if (seen[j] == arr[i]) {
        found = true;
        break;
      }
    }
    if (!found) {
      seen[result_len] = arr[i];
      result[result_len] = arr[i];
      result_len++;
    }
  }

  free(seen);
  result = realloc(result, result_len * sizeof(int64_t));
  *out_len = result_len;
  return result;
}

int64_t* tg_make_tuple(int64_t x, size_t cnt, size_t *out_len) {
  int64_t *result = malloc(cnt * sizeof(int64_t));
  for (size_t i = 0; i < cnt; i++) {
    result[i] = x;
  }
  *out_len = cnt;
  return result;
}

int64_t* tg_flatten_int64(const int64_t **nested_arr, const size_t *lens, size_t num_arrays, size_t *out_len) {
  if (!nested_arr || !lens || num_arrays == 0) {
    *out_len = 0;
    return NULL;
  }

  size_t total_len = 0;
  for (size_t i = 0; i < num_arrays; i++) {
    total_len += lens[i];
  }

  int64_t *result = malloc(total_len * sizeof(int64_t));
  size_t pos = 0;

  for (size_t i = 0; i < num_arrays; i++) {
    memcpy(result + pos, nested_arr[i], lens[i] * sizeof(int64_t));
    pos += lens[i];
  }

  *out_len = total_len;
  return result;
}

bool tg_all_same_int64(const int64_t *arr, size_t len) {
  if (!arr || len <= 1) return true;
  for (size_t i = 1; i < len; i++) {
    if (arr[i] != arr[0]) return false;
  }
  return true;
}

size_t* tg_argsort_int64(const int64_t *arr, size_t len) {
  if (!arr || len == 0) return NULL;

  size_t *indices = malloc(len * sizeof(size_t));
  for (size_t i = 0; i < len; i++) {
    indices[i] = i;
  }

  // bubble sort
  // TODO(alpin): use a more efficient algorithm if applicable
  for (size_t i = 0; i < len - 1; i++) {
    for (size_t j = 0; j < len - i - 1; j++) {
      if (arr[indices[j]] > arr[indices[j + 1]]) {
        size_t temp = indices[j];
        indices[j] = indices[j + 1];
        indices[j + 1] = temp;
      }
    }
  }

  return indices;
}

/* System Utilities */

char* tg_temp(const char *x, bool append_user) {
  const char *temp_dir = getenv("TMPDIR");
  if (!temp_dir) temp_dir = "/tmp";

  char *user = NULL;
  if (append_user) {
    user = tg_getpass_user();
  }

  size_t path_len = strlen(temp_dir) + strlen(x) + (user ? strlen(user) + 1 : 0) + 2; // +2 for '/' and '\0'
  char *result = malloc(path_len);

  if (user) {
    sprintf(result, "%s/%s.%s", temp_dir, x, user);
    free(user);
  } else {
    sprintf(result, "%s/%s", temp_dir, x);
  }

  return result;
}

int64_t tg_getenv_int64(const char *key, int64_t default_value) {
  const char *env_val = getenv(key);
  if (!env_val) return default_value;

  char *endptr;
  int64_t result = strtoll(env_val, &endptr, 10);
  if (*endptr != '\0') return default_value; // Invalid number

  return result;
}

bool tg_is_OSX() {
  struct utsname uname_data;
  if (uname(&uname_data) != 0) return false;
  return strcmp(uname_data.sysname, "Darwin") == 0;
}

bool tg_is_CI() {
  const char *ci = getenv("CI");
  return ci && strcmp(ci, "") != 0;
}

char* tg_getpass_user() {
  uid_t uid = getuid();
  struct passwd *pw = getpwuid(uid);
  if (pw && pw->pw_name) {
    return strdup(pw->pw_name);
  }
  return strdup("unknown");
}

/* Progress Bar (tqdm) Implementation */

static double tg_get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int tg_get_terminal_width(void) {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_col > 0 ? w.ws_col : 80;
}

static char* tg_format_time(double seconds) {
    if (seconds < 60) {
        char *result = malloc(16);
        sprintf(result, "%.1fs", seconds);
        return result;
    } else if (seconds < 3600) {
        char *result = malloc(16);
        sprintf(result, "%02d:%02d", (int)seconds / 60, (int)seconds % 60);
        return result;
    } else {
        char *result = malloc(16);
        sprintf(result, "%02d:%02d:%02d", (int)seconds / 3600, ((int)seconds % 3600) / 60, (int)seconds % 60);
        return result;
    }
}

static char* tg_format_count(size_t count, bool unit_scale) {
    if (!unit_scale) {
        char *result = malloc(32);
        sprintf(result, "%zu", count);
        return result;
    }

    const char *suffixes = "kMGTPEZY";
    double value = (double)count;
    int suffix_index = 0;

    while (value >= 1000 && suffix_index < 7) {
        value /= 1000;
        suffix_index++;
    }

    char *result = malloc(32);
    if (suffix_index == 0) {
        sprintf(result, "%.0f", value);
    } else {
        sprintf(result, "%.1f%c", value, suffixes[suffix_index - 1]);
    }
    return result;
}

void tg_tqdm_init(tg_tqdm_t *tqdm, const char *desc, bool disable, const char *unit, bool unit_scale, size_t total, size_t rate_limit) {
    tqdm->desc = desc ? strdup(desc) : "";
    tqdm->disable = disable;
    tqdm->unit = unit ? strdup(unit) : "it";
    tqdm->unit_scale = unit_scale;
    tqdm->total = total;
    tqdm->current = 0;
    tqdm->start_time = tg_get_time();
    tqdm->update_count = 0;
    tqdm->skip_count = 1;
    tqdm->rate_limit = rate_limit ? rate_limit : 100;
    tg_tqdm_update(tqdm, 0);
}

void tg_tqdm_update(tg_tqdm_t *tqdm, size_t n) {
    if (tqdm->disable) return;

    tqdm->current += n;
    tqdm->update_count++;

    // fprintf(stderr, "\nDEBUG: update called, current=%zu, n=%zu\n", tqdm->current, n);

    // skip updates to avoid flooding (rate limiting)
    if ((tqdm->update_count % tqdm->skip_count) != 0) {
        // fprintf(stderr, "SKIP: count=%zu, skip=%zu\n", tqdm->update_count, tqdm->skip_count);
        return;
    }

    double elapsed = tg_get_time() - tqdm->start_time;
    double progress = tqdm->total > 0 ? (double)tqdm->current / tqdm->total : 0.0;

    // only apply when we have meaningful elapsed time (dynamic rate limiting)
    if (elapsed > 0.001 && tqdm->update_count / elapsed > tqdm->rate_limit) {  // Minimum 1ms elapsed
        size_t new_skip = (size_t)(tqdm->update_count / elapsed / tqdm->rate_limit);
        if (new_skip < 1) new_skip = 1;
        if (new_skip > 1000) new_skip = 1000;
        if (new_skip != tqdm->skip_count) {
            // fprintf(stderr, "DEBUG: changing skip_count from %zu to %zu (updates=%zu, elapsed=%.3f, rate=%.1f)\n",
            //         tqdm->skip_count, new_skip, tqdm->update_count, elapsed, tqdm->update_count / elapsed);
            tqdm->skip_count = new_skip;
        }
    }

    int terminal_width = tg_get_terminal_width();
    int bar_width = 20;

    int filled = (int)(progress * bar_width);
    char bar[bar_width + 1];
    memset(bar, '#', filled);
    memset(bar + filled, '.', bar_width - filled);
    bar[bar_width] = '\0';

    // Format current/total
    char *current_str = tg_format_count(tqdm->current, tqdm->unit_scale);
    char *total_str = tg_format_count(tqdm->total, tqdm->unit_scale);

    // ETA
    char eta_str[32] = "";
    if (progress > 0 && progress < 1.0) {
        double eta = elapsed / progress - elapsed;
        char *eta_formatted = tg_format_time(eta);
        sprintf(eta_str, "<%s", eta_formatted);
        free(eta_formatted);
    }

    // rate
    double rate = tqdm->current / elapsed;
    char *rate_str = tg_format_count((size_t)rate, tqdm->unit_scale);

    // format elapsed time
    char *elapsed_str = tg_format_time(elapsed);

    // TODO(alpin): not sure about this
    char line[1024];
    int len = 0;

    // Description
    if (tqdm->desc && strlen(tqdm->desc) > 0) {
        len += sprintf(line + len, "%s: ", tqdm->desc);
    }

    // Progress bar
    if (tqdm->total > 0) {
        len += sprintf(line + len, "%3.0f%%|%s| ", progress * 100, bar);
    }

    // Current/total and timing info
    len += sprintf(line + len, "%s", current_str);
    if (tqdm->total > 0) {
        len += sprintf(line + len, "/%s", total_str);
    }
    len += sprintf(line + len, " [%s%s, %s%s/s]", elapsed_str, eta_str, rate_str, tqdm->unit);

    if (len >= terminal_width) {
        line[terminal_width - 1] = '\0';
    }

    fprintf(stderr, "\r%s", line);
    fflush(stderr);

    // fprintf(stderr, "\nDEBUG: printed line: '%s'\n", line);

    free(current_str);
    free(total_str);
    free(rate_str);
    free(elapsed_str);
}

void tg_tqdm_finish(tg_tqdm_t *tqdm) {
    if (!tqdm->disable) {
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    free((char*)tqdm->desc);
    free((char*)tqdm->unit);
}

void tg_tqdm_write(const char *s) {
    fprintf(stderr, "\r\033[K%s", s);
    fflush(stderr);
}

/* Range iterator with progress bar */

void tg_trange_init(tg_trange_t *trange, size_t n, const char *desc, bool disable, const char *unit) {
    tg_tqdm_init(&trange->tqdm, desc, disable, unit, false, n, 100);
    trange->current = 0;
    trange->end = n;
}

void tg_trange_next(tg_trange_t *trange) {
    if (trange->current < trange->end) {
        trange->current++;
        tg_tqdm_update(&trange->tqdm, 1);
    }
}

bool tg_trange_has_next(tg_trange_t *trange) {
    return trange->current < trange->end;
}

void tg_trange_finish(tg_trange_t *trange) {
    tg_tqdm_finish(&trange->tqdm);
}
