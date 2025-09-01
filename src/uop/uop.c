/* uop.c - Stub implementation for TDD */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "uop/uop.h"

// Global GroupOp instance
const GroupOp group_op = {0};

// Stub implementations
const char* ops_to_string(Ops op) {
    (void)op;
    return "UNIMPLEMENTED";
}

bool ops_is_valid(Ops op) {
    (void)op;
    return false;
}

int ops_get_arity(Ops op) {
    (void)op;
    return 0;
}

void uop_init(void) {
    // Stub
}

void uop_cleanup(void) {
    // Stub
}