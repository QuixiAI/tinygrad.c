#include "../src/helpers/helpers.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

void test_easy_utils() {

  // Test tg_argfix_int64
  {
    int64_t arr1[] = {1, 2, 3};
    int64_t arr2[] = {4, 5};
    const int64_t *arrays[] = {arr1};
    size_t lens[] = {3};

    tg_argfix_result_t result = tg_argfix_int64(arrays, lens, 1);
    assert(result.is_valid == true);
    assert(result.len == 3);
    assert(result.data[0] == 1 && result.data[1] == 2 && result.data[2] == 3);
    free(result.data);
  }

  // Test tg_all_int
  {
    const char *types1[] = {"int", "int", "int"};
    const void *values1[] = {NULL, NULL, NULL}; // Values don't matter for this test
    assert(tg_all_int(values1, 3, types1) == true);

    const char *types2[] = {"int", "float", "int"};
    assert(tg_all_int(values1, 3, types2) == false);
  }

  // Test tg_colorize_float
  {
    char *green_result = tg_colorize_float(0.5);  // Should be green
    assert(strstr(green_result, "0.50x") != NULL);
    free(green_result);

    char *red_result = tg_colorize_float(1.5);   // Should be red
    assert(strstr(red_result, "1.50x") != NULL);
    free(red_result);

    char *yellow_result = tg_colorize_float(1.0); // Should be yellow
    assert(strstr(yellow_result, "1.00x") != NULL);
    free(yellow_result);
  }

  // Test tg_time_to_str
  {
    char *sec_result = tg_time_to_str(15.0, 8);
    assert(strstr(sec_result, "s") != NULL);
    free(sec_result);

    char *ms_result = tg_time_to_str(0.05, 8);
    assert(strstr(ms_result, "ms") != NULL);
    free(ms_result);

    char *us_result = tg_time_to_str(0.000005, 8);
    assert(strstr(us_result, "us") != NULL);
    free(us_result);
  }

  // Test tg_strip_parens
  {
    char *result1 = tg_strip_parens("(hello world)");
    assert(strcmp(result1, "hello world") == 0);
    free(result1);

    char *result2 = tg_strip_parens("((nested))");
    assert(strcmp(result2, "(nested)") == 0);
    free(result2);

    char *result3 = tg_strip_parens("no_parens");
    assert(strcmp(result3, "no_parens") == 0);
    free(result3);

    char *result4 = tg_strip_parens("(unbalanced()");
    assert(strcmp(result4, "(unbalanced()") == 0);
    free(result4);
  }

  // Test tg_i2u
  {
    assert(tg_i2u(8, 127) == 127);      // Positive stays positive
    assert(tg_i2u(8, -1) == 255);       // -1 becomes 255 in 8-bit
    assert(tg_i2u(16, -1) == 65535);    // -1 becomes 65535 in 16-bit
  }

  // Test tg_is_numpy_ndarray
  {
    assert(tg_is_numpy_ndarray("<class 'numpy.ndarray'>") == true);
    assert(tg_is_numpy_ndarray("<class 'list'>") == false);
    assert(tg_is_numpy_ndarray(NULL) == false);
  }

  // Test tg_unwrap (note: this will abort on NULL, so we only test valid case)
  {
    int test_val = 42;
    void *result = tg_unwrap(&test_val);
    assert(result == &test_val);
  }

  // Test tg_get_single_element
  {
    int test_val = 42;
    const void *arr[] = {&test_val};
    tg_single_element_result_t result = tg_get_single_element(arr, 1);
    assert(result.success == true);
    assert(result.element == &test_val);

    // Test error case
    const void *arr2[] = {&test_val, &test_val};
    tg_single_element_result_t result2 = tg_get_single_element(arr2, 2);
    assert(result2.success == false);
  }

  // Test tg_polyN
  {
    double coeffs[] = {1, 2, 3}; // 1*x^2 + 2*x + 3
    double result = tg_polyN(2.0, coeffs, 3); // 1*4 + 2*2 + 3 = 11
    assert(result == 11.0);

    double coeffs2[] = {2, -1}; // 2*x - 1
    double result2 = tg_polyN(3.0, coeffs2, 2); // 2*3 - 1 = 5
    assert(result2 == 5.0);
  }

  // Test tg_to_function_name
  {
    char *result1 = tg_to_function_name("hello_world123");
    assert(strcmp(result1, "hello_world123") == 0);
    free(result1);

    char *result2 = tg_to_function_name("hello@world!");
    assert(strstr(result2, "hello") != NULL);
    assert(strstr(result2, "world") != NULL);
    // Should contain hex codes for @ and !
    free(result2);

    char *result3 = tg_to_function_name("test-name");
    assert(strstr(result3, "test") != NULL);
    assert(strstr(result3, "name") != NULL);
    free(result3);
  }

  // Test tg_suppress_finalizing (basic functionality)
  {
    // Simplified for now
    tg_suppress_result_t result = tg_suppress_finalizing(NULL, NULL, TG_SUPPRESS_ALL);
    assert(result.success == 1);
  }

}

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

  // Test tg_data64 (hi, lo order)
  uint32_t lo, hi;
  tg_data64(test_val, &hi, &lo);
  assert(hi == 0x12345678UL);
  assert(lo == 0x9ABCDEF0UL);

  // Test tg_data64_le (lo, hi order)
  uint32_t lo_le, hi_le;
  tg_data64_le(test_val, &lo_le, &hi_le);
  assert(lo_le == 0x9ABCDEF0UL);
  assert(hi_le == 0x12345678UL);

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

  test_easy_utils();
  test_math_utils();
  test_bit_operations();
  test_string_utils();
  test_collection_utils();
  test_system_utils();
  test_progress_bar();

  printf("All tests passed!\n");
  return 0;
}
