#include "test_common.h"
#include "helpers/helpers.h"
#include <stdlib.h>

typedef struct { int counter; int target; } ctx_t;
static int inc_until(void* p){ ctx_t* c=(ctx_t*)p; if (c->counter < c->target) c->counter++; return c->counter; }
static int always_zero(void* p){ (void)p; return 0; }

TEST(test_wait_cond_success){
  ctx_t ctx = {0, 5};
  int ok = tg_wait_cond(inc_until, &ctx, /*value*/5, /*timeout_ms*/200, "");
  TEST_ASSERT_TRUE(ok);
}

TEST(test_wait_cond_timeout){
  int ok = tg_wait_cond(always_zero, NULL, /*value*/1, /*timeout_ms*/10, "");
  TEST_ASSERT_FALSE(ok);
}

TEST(test_is_ci_flag){
  const char* prev = getenv("CI");
  setenv("CI", "1", 1);
  TEST_ASSERT_TRUE(tg_is_ci());
  if (prev) setenv("CI", prev, 1); else unsetenv("CI");
}

void setUp(void) {}
void tearDown(void) {}

TEST_MAIN()

