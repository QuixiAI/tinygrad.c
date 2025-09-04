// TLSF Allocator stub implementation
#include "memory.h"
#include <stdlib.h>

struct TLSFAllocator {
    int64_t size;
    int block_size;
    int lv2_cnt;
    int64_t allocated;
};

TLSFAllocator* tlsf_allocator_new(int64_t size, int block_size, int lv2_cnt) {
    TLSFAllocator* alloc = calloc(1, sizeof(TLSFAllocator));
    alloc->size = size;
    alloc->block_size = block_size;
    alloc->lv2_cnt = lv2_cnt;
    alloc->allocated = 0;
    return alloc;
}

void tlsf_allocator_free(TLSFAllocator* alloc) {
    free(alloc);
}

int tlsf_allocator_alloc(TLSFAllocator* alloc, int64_t size) {
    // Stub: just return the current allocated position
    int offset = (int)alloc->allocated;
    alloc->allocated += size;
    return offset;
}

void tlsf_allocator_free_block(TLSFAllocator* alloc, int offset) {
    // Stub: do nothing for now
    (void)alloc;
    (void)offset;
}