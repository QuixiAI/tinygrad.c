#include "codegen/opt/tc.h"
#include "helpers/helpers.h"

#include <stdlib.h>

static int tc_allow_tf32(void) {
  const char* val = tg_getenv("ALLOW_TF32");
  if (!val || !*val) return 0;
  char* end = NULL;
  long parsed = strtol(val, &end, 10);
  if (end == val) return 0;
  return parsed != 0;
}

static const char* const cuda_tc_opts_items[] = {"u0", "l0", "l0", "l1", "l1", "l1", "u1"};
static const TCStrList cuda_tc_opts = {7, cuda_tc_opts_items};

static const char* const cuda_81616_sw0_local_items[] = {"r1", "r2", "l2", "l3", "l4"};
static const char* const cuda_81616_sw0_upcast_items[] = {"u1", "r3"};
static const char* const cuda_81616_sw0_reduce_items[] = {"l0", "l1", "u0", "r0"};
static const char* const cuda_81616_sw1_local_items[] = {"r1", "r2", "u0", "l0", "l1"};
static const char* const cuda_81616_sw1_upcast_items[] = {"r0", "r3"};
static const char* const cuda_81616_sw1_reduce_items[] = {"l2", "l3", "l4", "u1"};
static const TCSwizzlePart cuda_81616_swizzle[2] = {
  {.parts = {{5, cuda_81616_sw0_local_items}, {2, cuda_81616_sw0_upcast_items}, {4, cuda_81616_sw0_reduce_items}}},
  {.parts = {{5, cuda_81616_sw1_local_items}, {2, cuda_81616_sw1_upcast_items}, {4, cuda_81616_sw1_reduce_items}}},
};

static const char* const cuda_8168_f16_sw0_local_items[] = {"r1", "r2", "l2", "l3", "l4"};
static const char* const cuda_8168_f16_sw0_upcast_items[] = {"r0", "u1"};
static const char* const cuda_8168_f16_sw0_reduce_items[] = {"l0", "l1", "u0"};
static const char* const cuda_8168_f16_sw1_local_items[] = {"r1", "r2", "u0", "l0", "l1"};
static const char* const cuda_8168_f16_sw1_upcast_items[] = {"u1", "r0"};
static const char* const cuda_8168_f16_sw1_reduce_items[] = {"l2", "l3", "l4"};
static const TCSwizzlePart cuda_8168_f16_swizzle[2] = {
  {.parts = {{5, cuda_8168_f16_sw0_local_items}, {2, cuda_8168_f16_sw0_upcast_items}, {3, cuda_8168_f16_sw0_reduce_items}}},
  {.parts = {{5, cuda_8168_f16_sw1_local_items}, {2, cuda_8168_f16_sw1_upcast_items}, {3, cuda_8168_f16_sw1_reduce_items}}},
};

static const char* const cuda_8168_tf32_sw0_local_items[] = {"r0", "r1", "l2", "l3", "l4"};
static const char* const cuda_8168_tf32_sw0_upcast_items[] = {"u1", "r2"};
static const char* const cuda_8168_tf32_sw0_reduce_items[] = {"l0", "l1", "u0"};
static const char* const cuda_8168_tf32_sw1_local_items[] = {"r0", "r1", "u0", "l0", "l1"};
static const char* const cuda_8168_tf32_sw1_upcast_items[] = {"u1", "r2"};
static const char* const cuda_8168_tf32_sw1_reduce_items[] = {"l2", "l3", "l4"};
static const TCSwizzlePart cuda_8168_tf32_swizzle[2] = {
  {.parts = {{5, cuda_8168_tf32_sw0_local_items}, {2, cuda_8168_tf32_sw0_upcast_items}, {3, cuda_8168_tf32_sw0_reduce_items}}},
  {.parts = {{5, cuda_8168_tf32_sw1_local_items}, {2, cuda_8168_tf32_sw1_upcast_items}, {3, cuda_8168_tf32_sw1_reduce_items}}},
};

static const char* const amd_rdna3_opts_items[] = {"l0", "l0", "l0", "l0", "l1", "u1", "u1", "u1"};
static const TCStrList amd_rdna3_opts = {8, amd_rdna3_opts_items};

static const char* const amd_rdna3_sw0_local_items[] = {"l4", "u0", "u1", "u2", "l0"};
static const char* const amd_rdna3_sw0_upcast_items[] = {"r1", "r2", "r3"};
static const char* const amd_rdna3_sw0_reduce_items[] = {"l1", "l2", "l3", "r0"};
static const char* const amd_rdna3_sw1_local_items[] = {"l0", "l1", "l2", "l3", "l4"};
static const char* const amd_rdna3_sw1_upcast_items[] = {"r1", "r2", "r3"};
static const char* const amd_rdna3_sw1_reduce_items[] = {"u0", "u1", "u2", "r0"};
static const TCSwizzlePart amd_rdna3_swizzle[2] = {
  {.parts = {{5, amd_rdna3_sw0_local_items}, {3, amd_rdna3_sw0_upcast_items}, {4, amd_rdna3_sw0_reduce_items}}},
  {.parts = {{5, amd_rdna3_sw1_local_items}, {3, amd_rdna3_sw1_upcast_items}, {4, amd_rdna3_sw1_reduce_items}}},
};

static const char* const amd_rdna4_opts_items[] = {"l0", "l0", "l0", "l0", "u1", "u1", "u1", "l1"};
static const TCStrList amd_rdna4_opts = {8, amd_rdna4_opts_items};

static const char* const amd_rdna4_sw0_local_items[] = {"u0", "u1", "u2", "l4", "r2"};
static const char* const amd_rdna4_sw0_upcast_items[] = {"r0", "r1", "r3"};
static const char* const amd_rdna4_sw0_reduce_items[] = {"l0", "l1", "l2", "l3"};
static const char* const amd_rdna4_sw1_local_items[] = {"l0", "l1", "l2", "l3", "r2"};
static const char* const amd_rdna4_sw1_upcast_items[] = {"r0", "r1", "r3"};
static const char* const amd_rdna4_sw1_reduce_items[] = {"l4", "u0", "u1", "u2"};
static const TCSwizzlePart amd_rdna4_swizzle[2] = {
  {.parts = {{5, amd_rdna4_sw0_local_items}, {3, amd_rdna4_sw0_upcast_items}, {4, amd_rdna4_sw0_reduce_items}}},
  {.parts = {{5, amd_rdna4_sw1_local_items}, {3, amd_rdna4_sw1_upcast_items}, {4, amd_rdna4_sw1_reduce_items}}},
};

static const char* const amd_cdna_opts_items[] = {"l0", "l0", "l0", "l0", "u1", "u1", "l1", "l1"};
static const TCStrList amd_cdna_opts = {8, amd_cdna_opts_items};

static const char* const amd_cdna_sw0_local_items[] = {"u0", "u1", "l4", "l5", "r2", "r3"};
static const char* const amd_cdna_sw0_upcast_items[] = {"r0", "r1"};
static const char* const amd_cdna_sw0_reduce_items[] = {"l0", "l1", "l2", "l3"};
static const char* const amd_cdna_sw1_local_items[] = {"l0", "l1", "l2", "l3", "r2", "r3"};
static const char* const amd_cdna_sw1_upcast_items[] = {"r0", "r1"};
static const char* const amd_cdna_sw1_reduce_items[] = {"l4", "l5", "u0", "u1"};
static const TCSwizzlePart amd_cdna_swizzle[2] = {
  {.parts = {{6, amd_cdna_sw0_local_items}, {2, amd_cdna_sw0_upcast_items}, {4, amd_cdna_sw0_reduce_items}}},
  {.parts = {{6, amd_cdna_sw1_local_items}, {2, amd_cdna_sw1_upcast_items}, {4, amd_cdna_sw1_reduce_items}}},
};

static const char* const metal_opts_items[] = {"u0", "l0", "l1", "l1", "l0", "l1"};
static const TCStrList metal_opts = {6, metal_opts_items};

static const char* const metal_sw0_local_items[] = {"r1", "l1", "l2", "r2", "l4"};
static const char* const metal_sw0_upcast_items[] = {"r0"};
static const char* const metal_sw0_reduce_items[] = {"u0", "l0", "l3"};
static const char* const metal_sw1_local_items[] = {"l0", "r0", "r1", "l3", "r2"};
static const char* const metal_sw1_upcast_items[] = {"u0"};
static const char* const metal_sw1_reduce_items[] = {"l1", "l2", "l4"};
static const TCSwizzlePart metal_swizzle[2] = {
  {.parts = {{5, metal_sw0_local_items}, {1, metal_sw0_upcast_items}, {3, metal_sw0_reduce_items}}},
  {.parts = {{5, metal_sw1_local_items}, {1, metal_sw1_upcast_items}, {3, metal_sw1_reduce_items}}},
};

static const char* const amx_opts_items[] = {"u0", "u0", "u0", "u0", "u1", "u1", "u1", "u1"};
static const TCStrList amx_opts = {8, amx_opts_items};

static const char* const amx_sw0_upcast_items[] = {"u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"};
static const char* const amx_sw1_upcast_items[] = {"u4", "u5", "u6", "u7", "u0", "u1", "u2", "u3"};
static const TCSwizzlePart amx_swizzle[2] = {
  {.parts = {{0, NULL}, {8, amx_sw0_upcast_items}, {0, NULL}}},
  {.parts = {{0, NULL}, {8, amx_sw1_upcast_items}, {0, NULL}}},
};

static const char* const intel_opts_items[] = {"l0", "l0", "l0", "u1", "u1", "u1"};
static const TCStrList intel_opts = {6, intel_opts_items};

static const char* const intel_sw0_local_items[] = {"r1", "r2", "r3"};
static const char* const intel_sw0_upcast_items[] = {"u0", "u1", "u2"};
static const char* const intel_sw0_reduce_items[] = {"l0", "l1", "l2", "r0"};
static const char* const intel_sw1_local_items[] = {"l0", "l1", "l2"};
static const char* const intel_sw1_upcast_items[] = {"r1", "r2", "r3"};
static const char* const intel_sw1_reduce_items[] = {"u0", "u1", "u2", "r0"};
static const TCSwizzlePart intel_swizzle[2] = {
  {.parts = {{3, intel_sw0_local_items}, {3, intel_sw0_upcast_items}, {4, intel_sw0_reduce_items}}},
  {.parts = {{3, intel_sw1_local_items}, {3, intel_sw1_upcast_items}, {4, intel_sw1_reduce_items}}},
};

const TensorCore tc_cuda_sm80[] = {
  {
    .dims = {8, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 4, 4},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.float_,
    .opts = cuda_tc_opts,
    .swizzle = {cuda_81616_swizzle[0], cuda_81616_swizzle[1]},
  },
  {
    .dims = {8, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 4, 4},
    .dtype_in = &dtypes.bfloat16,
    .dtype_out = &dtypes.float_,
    .opts = cuda_tc_opts,
    .swizzle = {cuda_81616_swizzle[0], cuda_81616_swizzle[1]},
  },
  {
    .dims = {8, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 4, 4},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.half,
    .opts = cuda_tc_opts,
    .swizzle = {cuda_81616_swizzle[0], cuda_81616_swizzle[1]},
  },
  {
    .dims = {8, 16, 8},
    .threads = 32,
    .elements_per_thread = {4, 2, 4},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.float_,
    .opts = cuda_tc_opts,
    .swizzle = {cuda_8168_f16_swizzle[0], cuda_8168_f16_swizzle[1]},
  },
  {
    .dims = {8, 16, 8},
    .threads = 32,
    .elements_per_thread = {4, 2, 4},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.half,
    .opts = cuda_tc_opts,
    .swizzle = {cuda_8168_f16_swizzle[0], cuda_8168_f16_swizzle[1]},
  },
  {
    .dims = {8, 16, 8},
    .threads = 32,
    .elements_per_thread = {4, 2, 4},
    .dtype_in = &dtypes.float_,
    .dtype_out = &dtypes.float_,
    .opts = cuda_tc_opts,
    .swizzle = {cuda_8168_tf32_swizzle[0], cuda_8168_tf32_swizzle[1]},
  },
};

size_t tc_cuda_sm80_count(void) {
  return 5 + (size_t)tc_allow_tf32();
}

const TensorCore tc_cuda_sm75[] = {
  {
    .dims = {8, 16, 8},
    .threads = 32,
    .elements_per_thread = {4, 2, 4},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.float_,
    .opts = cuda_tc_opts,
    .swizzle = {cuda_8168_f16_swizzle[0], cuda_8168_f16_swizzle[1]},
  },
  {
    .dims = {8, 16, 8},
    .threads = 32,
    .elements_per_thread = {4, 2, 4},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.half,
    .opts = cuda_tc_opts,
    .swizzle = {cuda_8168_f16_swizzle[0], cuda_8168_f16_swizzle[1]},
  },
};

size_t tc_cuda_sm75_count(void) {
  return sizeof(tc_cuda_sm75) / sizeof(tc_cuda_sm75[0]);
}

const TensorCore tc_amd_rdna3[] = {
  {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {16, 16, 8},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.float_,
    .opts = amd_rdna3_opts,
    .swizzle = {amd_rdna3_swizzle[0], amd_rdna3_swizzle[1]},
  },
  {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {16, 16, 8},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.half,
    .opts = amd_rdna3_opts,
    .swizzle = {amd_rdna3_swizzle[0], amd_rdna3_swizzle[1]},
  },
  {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {16, 16, 8},
    .dtype_in = &dtypes.bfloat16,
    .dtype_out = &dtypes.float_,
    .opts = amd_rdna3_opts,
    .swizzle = {amd_rdna3_swizzle[0], amd_rdna3_swizzle[1]},
  },
};

size_t tc_amd_rdna3_count(void) {
  return sizeof(tc_amd_rdna3) / sizeof(tc_amd_rdna3[0]);
}

const TensorCore tc_amd_rdna4[] = {
  {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 8, 8},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.float_,
    .opts = amd_rdna4_opts,
    .swizzle = {amd_rdna4_swizzle[0], amd_rdna4_swizzle[1]},
  },
  {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 8, 8},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.half,
    .opts = amd_rdna4_opts,
    .swizzle = {amd_rdna4_swizzle[0], amd_rdna4_swizzle[1]},
  },
  {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 8, 8},
    .dtype_in = &dtypes.bfloat16,
    .dtype_out = &dtypes.float_,
    .opts = amd_rdna4_opts,
    .swizzle = {amd_rdna4_swizzle[0], amd_rdna4_swizzle[1]},
  },
  {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 8, 8},
    .dtype_in = &dtypes.bfloat16,
    .dtype_out = &dtypes.bfloat16,
    .opts = amd_rdna4_opts,
    .swizzle = {amd_rdna4_swizzle[0], amd_rdna4_swizzle[1]},
  },
};

size_t tc_amd_rdna4_count(void) {
  return sizeof(tc_amd_rdna4) / sizeof(tc_amd_rdna4[0]);
}

const TensorCore tc_amd_cdna[] = {
  {
    .dims = {16, 16, 16},
    .threads = 64,
    .elements_per_thread = {4, 4, 4},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.float_,
    .opts = amd_cdna_opts,
    .swizzle = {amd_cdna_swizzle[0], amd_cdna_swizzle[1]},
  },
  {
    .dims = {16, 16, 16},
    .threads = 64,
    .elements_per_thread = {4, 4, 4},
    .dtype_in = &dtypes.bfloat16,
    .dtype_out = &dtypes.float_,
    .opts = amd_cdna_opts,
    .swizzle = {amd_cdna_swizzle[0], amd_cdna_swizzle[1]},
  },
};

size_t tc_amd_cdna_count(void) {
  return sizeof(tc_amd_cdna) / sizeof(tc_amd_cdna[0]);
}

const TensorCore tc_metal[] = {
  {
    .dims = {8, 8, 8},
    .threads = 32,
    .elements_per_thread = {2, 2, 2},
    .dtype_in = &dtypes.float_,
    .dtype_out = &dtypes.float_,
    .opts = metal_opts,
    .swizzle = {metal_swizzle[0], metal_swizzle[1]},
  },
  {
    .dims = {8, 8, 8},
    .threads = 32,
    .elements_per_thread = {2, 2, 2},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.float_,
    .opts = metal_opts,
    .swizzle = {metal_swizzle[0], metal_swizzle[1]},
  },
  {
    .dims = {8, 8, 8},
    .threads = 32,
    .elements_per_thread = {2, 2, 2},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.half,
    .opts = metal_opts,
    .swizzle = {metal_swizzle[0], metal_swizzle[1]},
  },
  {
    .dims = {8, 8, 8},
    .threads = 32,
    .elements_per_thread = {2, 2, 2},
    .dtype_in = &dtypes.bfloat16,
    .dtype_out = &dtypes.float_,
    .opts = metal_opts,
    .swizzle = {metal_swizzle[0], metal_swizzle[1]},
  },
  {
    .dims = {8, 8, 8},
    .threads = 32,
    .elements_per_thread = {2, 2, 2},
    .dtype_in = &dtypes.bfloat16,
    .dtype_out = &dtypes.bfloat16,
    .opts = metal_opts,
    .swizzle = {metal_swizzle[0], metal_swizzle[1]},
  },
};

size_t tc_metal_count(void) {
  return sizeof(tc_metal) / sizeof(tc_metal[0]);
}

const TensorCore tc_amx[] = {
  {
    .dims = {16, 16, 1},
    .threads = 1,
    .elements_per_thread = {16, 16, 256},
    .dtype_in = &dtypes.float_,
    .dtype_out = &dtypes.float_,
    .opts = amx_opts,
    .swizzle = {amx_swizzle[0], amx_swizzle[1]},
  },
};

size_t tc_amx_count(void) {
  return sizeof(tc_amx) / sizeof(tc_amx[0]);
}

const TensorCore tc_intel[] = {
  {
    .dims = {8, 8, 16},
    .threads = 8,
    .elements_per_thread = {16, 16, 8},
    .dtype_in = &dtypes.half,
    .dtype_out = &dtypes.float_,
    .opts = intel_opts,
    .swizzle = {intel_swizzle[0], intel_swizzle[1]},
  },
};

size_t tc_intel_count(void) {
  return sizeof(tc_intel) / sizeof(tc_intel[0]);
}
