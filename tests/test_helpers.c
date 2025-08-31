#include "../src/helpers/helpers.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

void test_math_utils() {
  printf("Testing math utilities...\n");

  // Test tg_prod
  int64_t arr1[] = {2, 3, 4};
  assert(tg_prod(arr1, 3) == 24);
  assert(tg_prod(NULL, 0) == 1); // empty product should be 1

  // Test tg_ceildiv
  assert(tg_ceildiv(7, 3) == 3);
  assert(tg_ceildiv(6, 3) == 2);

  // Test tg_round_up
  assert(tg_round_up(7, 4) == 8);
  assert(tg_round_up(8, 4) == 8);

  // Test tg_cdiv
  assert(tg_cdiv(7, 3) == 2);
  assert(tg_cdiv(-7, 3) == -3);
  assert(tg_cdiv(7, -3) == -3);

  printf("Math utilities: OK\n");
}

void test_bit_operations() {
  printf("Testing bit operations...\n");

  uint64_t test_val = 0x123456789ABCDEF0ULL;

  // Test tg_lo32 and tg_hi32
  assert(tg_lo32(test_val) == 0x9ABCDEF0UL);
  assert(tg_hi32(test_val) == 0x12345678UL);

  // Test tg_data64
  uint32_t lo, hi;
  tg_data64(test_val, &lo, &hi);
  assert(lo == 0x9ABCDEF0UL);
  assert(hi == 0x12345678UL);

  // Test tg_getbits
  assert(tg_getbits(0xFF, 0, 3) == 0xF);
  assert(tg_getbits(0xFF, 4, 7) == 0xF);

  printf("Bit operations: OK\n");
}

void test_string_utils() {
  printf("Testing string utilities...\n");

  // Test tg_colored
  char *colored = tg_colored("test", "red");
  assert(colored != NULL);
  assert(strstr(colored, "test") != NULL);
  free(colored);

  // Test tg_ansistrip
  char *ansi_str = "\x1b[31mHello\x1b[0m";
  char *stripped = tg_ansistrip(ansi_str);
  assert(strcmp(stripped, "Hello") == 0);
  free(stripped);

  // Test tg_ansilen
  assert(tg_ansilen(ansi_str) == 5);

  // Test tg_pluralize
  char *singular = tg_pluralize("apple", 1);
  assert(strcmp(singular, "apple") == 0);
  free(singular);

  char *plural = tg_pluralize("apple", 2);
  assert(strcmp(plural, "apples") == 0);
  free(plural);

  printf("String utilities: OK\n");
}

void test_collection_utils() {
  printf("Testing collection utilities...\n");

  // Test tg_dedup_int64
  int64_t dup_arr[] = {1, 2, 2, 3, 1, 4};
  size_t out_len;
  int64_t *deduped = tg_dedup_int64(dup_arr, 6, &out_len);
  assert(out_len == 4);
  assert(deduped[0] == 1 && deduped[1] == 2 && deduped[2] == 3 && deduped[3] == 4);
  free(deduped);

  // Test tg_make_tuple
  int64_t *tuple = tg_make_tuple(5, 3, &out_len);
  assert(out_len == 3);
  assert(tuple[0] == 5 && tuple[1] == 5 && tuple[2] == 5);
  free(tuple);

  // Test tg_all_same_int64
  int64_t same_arr[] = {3, 3, 3, 3};
  assert(tg_all_same_int64(same_arr, 4) == true);
  int64_t diff_arr[] = {1, 2, 3};
  assert(tg_all_same_int64(diff_arr, 3) == false);

  printf("Collection utilities: OK\n");
}

void test_system_utils() {
  printf("Testing system utilities...\n");

  // Test tg_getenv_int64
  setenv("TEST_VAR", "42", 1);
  assert(tg_getenv_int64("TEST_VAR", 0) == 42);
  assert(tg_getenv_int64("NONEXISTENT_VAR", 99) == 99);
  unsetenv("TEST_VAR");

  // Test tg_temp
  char *temp_path = tg_temp("testfile", false);
  assert(temp_path != NULL);
  assert(strstr(temp_path, "/testfile") != NULL);
  free(temp_path);

  char *temp_path_user = tg_temp("testfile", true);
  assert(temp_path_user != NULL);
  assert(strstr(temp_path_user, "/testfile.") != NULL);
  free(temp_path_user);

  // Test tg_is_CI (should be false in our environment)
  assert(tg_is_CI() == false);

  // Test tg_getpass_user
  char *user = tg_getpass_user();
  assert(user != NULL);
  assert(strlen(user) > 0);
  free(user);

  printf("System utilities: OK\n");
}

void test_progress_bar() {
  printf("Testing progress bar...\n");

  // Test basic tqdm functionality
  tg_tqdm_t tqdm;
  tg_tqdm_init(&tqdm, "Test", false, "it", false, 100, 10);

  for (size_t i = 0; i < 100; i++) {
    tg_tqdm_update(&tqdm, 1);
    // Small delay to make timing visible
    usleep(1000);
  }

  tg_tqdm_finish(&tqdm);

  // Test trange functionality
  tg_trange_t trange;
  tg_trange_init(&trange, 50, "Range test", false, "steps");

  while (tg_trange_has_next(&trange)) {
    tg_trange_next(&trange);
    usleep(2000);
  }

  tg_trange_finish(&trange);

  // Test write function
  tg_tqdm_write("Test message\n");

  printf("Progress bar: OK\n");
}

int main() {
  printf("Running helper function tests...\n");

  test_math_utils();
  test_bit_operations();
  test_string_utils();
  test_collection_utils();
  test_system_utils();
  test_progress_bar();

  printf("All tests passed!\n");
  return 0;
}
