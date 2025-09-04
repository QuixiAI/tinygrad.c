// Port of tinygrad/engine/memory.py
#ifndef TINYGRAD_ENGINE_MEMORY_H
#define TINYGRAD_ENGINE_MEMORY_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Forward declarations
typedef struct Buffer Buffer;
typedef struct ScheduleItem ScheduleItem;
typedef struct TLSFAllocator TLSFAllocator;

// Port of memory planning structures

// Port of: dict[Buffer, Buffer]
typedef struct BufferMapEntry {
    Buffer* key;
    Buffer* value;
    struct BufferMapEntry* next;
} BufferMapEntry;

typedef struct BufferMap {
    BufferMapEntry** buckets;
    size_t bucket_count;
    size_t size;
} BufferMap;

// Port of: dict[Buffer, tuple[Buffer|None, int|None]]
typedef struct BufferReplaceEntry {
    Buffer* key;
    Buffer* base;  // Can be NULL
    int offset;    // -1 means None
    bool has_offset;
    struct BufferReplaceEntry* next;
} BufferReplaceEntry;

typedef struct BufferReplaceMap {
    BufferReplaceEntry** buckets;
    size_t bucket_count;
    size_t size;
} BufferReplaceMap;

// Port of: dict[tuple, list[Buffer]]
typedef struct ReuseBufferKey {
    char* device;
    void* dtype;
    void* options;
    size_t nbytes;
} ReuseBufferKey;

typedef struct ReuseBufferEntry {
    ReuseBufferKey key;
    Buffer** buffers;
    int buffer_count;
    int buffer_capacity;
    struct ReuseBufferEntry* next;
} ReuseBufferEntry;

typedef struct ReuseBufferMap {
    ReuseBufferEntry** buckets;
    size_t bucket_count;
    size_t size;
} ReuseBufferMap;

// Port of: dict[str, tuple[int, TLSFAllocator]]
typedef struct GlobalPlannerEntry {
    char* device;
    int64_t size;
    TLSFAllocator* allocator;
    struct GlobalPlannerEntry* next;
} GlobalPlannerEntry;

typedef struct GlobalPlannerMap {
    GlobalPlannerEntry** buckets;
    size_t bucket_count;
    size_t size;
} GlobalPlannerMap;

// Port of: dict[Buffer, int]
typedef struct BufferIntEntry {
    Buffer* key;
    int value;
    struct BufferIntEntry* next;
} BufferIntEntry;

typedef struct BufferIntMap {
    BufferIntEntry** buckets;
    size_t bucket_count;
    size_t size;
} BufferIntMap;

// **************** memory planning ****************
// Port of lines 10-63

// Port of: def _internal_memory_planner(buffers:list[list[Buffer]], noopt_buffers=None, ignore_checks=False, debug_prefix="") -> dict[Buffer, Buffer]:
BufferMap* _internal_memory_planner(Buffer*** buffers, int* buffer_counts, int num_lists, 
                                    Buffer** noopt_buffers, int noopt_count,
                                    bool ignore_checks, const char* debug_prefix);

// Port of: def memory_planner(schedule:list[ScheduleItem]) -> list[ScheduleItem]:
ScheduleItem** memory_planner(ScheduleItem** schedule, int schedule_count, int* out_count);

// Helper functions for BufferMap
BufferMap* buffer_map_new(void);
void buffer_map_put(BufferMap* map, Buffer* key, Buffer* value);
Buffer* buffer_map_get(BufferMap* map, Buffer* key);
bool buffer_map_contains(BufferMap* map, Buffer* key);
void buffer_map_free(BufferMap* map);

// Helper functions for BufferReplaceMap
BufferReplaceMap* buffer_replace_map_new(void);
void buffer_replace_map_put(BufferReplaceMap* map, Buffer* key, Buffer* base, int offset, bool has_offset);
bool buffer_replace_map_get(BufferReplaceMap* map, Buffer* key, Buffer** out_base, int* out_offset, bool* out_has_offset);
void buffer_replace_map_free(BufferReplaceMap* map);

// Helper functions for ReuseBufferMap  
ReuseBufferMap* reuse_buffer_map_new(void);
void reuse_buffer_map_append(ReuseBufferMap* map, const ReuseBufferKey* key, Buffer* buffer);
Buffer* reuse_buffer_map_pop(ReuseBufferMap* map, const ReuseBufferKey* key);
void reuse_buffer_map_free(ReuseBufferMap* map);

// Helper functions for GlobalPlannerMap
GlobalPlannerMap* global_planner_map_new(void);
GlobalPlannerEntry* global_planner_map_get_or_create(GlobalPlannerMap* map, const char* device);
void global_planner_map_free(GlobalPlannerMap* map);

// Helper functions for BufferIntMap
BufferIntMap* buffer_int_map_new(void);
void buffer_int_map_put(BufferIntMap* map, Buffer* key, int value);
int buffer_int_map_get(BufferIntMap* map, Buffer* key, int default_value);
bool buffer_int_map_contains(BufferIntMap* map, Buffer* key);
void buffer_int_map_free(BufferIntMap* map);

#endif // TINYGRAD_ENGINE_MEMORY_H