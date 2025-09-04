// TLSF Allocator stub for memory planning
#ifndef TINYGRAD_RUNTIME_SUPPORT_MEMORY_H
#define TINYGRAD_RUNTIME_SUPPORT_MEMORY_H

#include <stdint.h>

typedef struct TLSFAllocator TLSFAllocator;

// Port of: TLSFAllocator(1 << 44, block_size=0x1000, lv2_cnt=32)
TLSFAllocator* tlsf_allocator_new(int64_t size, int block_size, int lv2_cnt);
void tlsf_allocator_free(TLSFAllocator* alloc);

// Port of: alloc(round_up(buf.nbytes, 0x1000))
int tlsf_allocator_alloc(TLSFAllocator* alloc, int64_t size);

// Port of: free(cast(int, buffer_replace[buf][1]))
void tlsf_allocator_free_block(TLSFAllocator* alloc, int offset);

#endif // TINYGRAD_RUNTIME_SUPPORT_MEMORY_H