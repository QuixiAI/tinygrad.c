// Port of tinygrad/engine/memory.py
#include "memory.h"
#include "../device/device.h"
#include "../device/buffer.h"
#include "schedule.h"
#include "../helpers.h"
#include "uop/uop.h"
#include "../dtype.h"
#include "../runtime/support/memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// Global variable ports
// Port of line 13: if NO_MEMORY_PLANNER: return {}
int NO_MEMORY_PLANNER = 0;  // Default to false
int DEBUG = 0;  // Default to 0

// Hash function for pointer-based keys
static size_t hash_ptr(void* ptr) {
    uintptr_t p = (uintptr_t)ptr;
    p ^= p >> 16;
    p *= 0x85ebca6b;
    p ^= p >> 13;
    p *= 0xc2b2ae35;
    p ^= p >> 16;
    return (size_t)p;
}

// Helper functions for BufferMap
BufferMap* buffer_map_new(void) {
    BufferMap* map = calloc(1, sizeof(BufferMap));
    map->bucket_count = 16;
    map->buckets = calloc(map->bucket_count, sizeof(BufferMapEntry*));
    return map;
}

void buffer_map_put(BufferMap* map, Buffer* key, Buffer* value) {
    size_t idx = hash_ptr(key) % map->bucket_count;
    BufferMapEntry* entry = map->buckets[idx];
    
    // Check if key exists
    while (entry) {
        if (entry->key == key) {
            entry->value = value;
            return;
        }
        entry = entry->next;
    }
    
    // Add new entry
    entry = malloc(sizeof(BufferMapEntry));
    entry->key = key;
    entry->value = value;
    entry->next = map->buckets[idx];
    map->buckets[idx] = entry;
    map->size++;
}

Buffer* buffer_map_get(BufferMap* map, Buffer* key) {
    size_t idx = hash_ptr(key) % map->bucket_count;
    BufferMapEntry* entry = map->buckets[idx];
    
    while (entry) {
        if (entry->key == key) return entry->value;
        entry = entry->next;
    }
    return NULL;
}

bool buffer_map_contains(BufferMap* map, Buffer* key) {
    return buffer_map_get(map, key) != NULL;
}

void buffer_map_free(BufferMap* map) {
    for (size_t i = 0; i < map->bucket_count; i++) {
        BufferMapEntry* entry = map->buckets[i];
        while (entry) {
            BufferMapEntry* next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    free(map);
}

// Helper functions for BufferReplaceMap
BufferReplaceMap* buffer_replace_map_new(void) {
    BufferReplaceMap* map = calloc(1, sizeof(BufferReplaceMap));
    map->bucket_count = 16;
    map->buckets = calloc(map->bucket_count, sizeof(BufferReplaceEntry*));
    return map;
}

void buffer_replace_map_put(BufferReplaceMap* map, Buffer* key, Buffer* base, int offset, bool has_offset) {
    size_t idx = hash_ptr(key) % map->bucket_count;
    BufferReplaceEntry* entry = map->buckets[idx];
    
    // Check if key exists
    while (entry) {
        if (entry->key == key) {
            entry->base = base;
            entry->offset = offset;
            entry->has_offset = has_offset;
            return;
        }
        entry = entry->next;
    }
    
    // Add new entry
    entry = malloc(sizeof(BufferReplaceEntry));
    entry->key = key;
    entry->base = base;
    entry->offset = offset;
    entry->has_offset = has_offset;
    entry->next = map->buckets[idx];
    map->buckets[idx] = entry;
    map->size++;
}

bool buffer_replace_map_get(BufferReplaceMap* map, Buffer* key, Buffer** out_base, int* out_offset, bool* out_has_offset) {
    size_t idx = hash_ptr(key) % map->bucket_count;
    BufferReplaceEntry* entry = map->buckets[idx];
    
    while (entry) {
        if (entry->key == key) {
            if (out_base) *out_base = entry->base;
            if (out_offset) *out_offset = entry->offset;
            if (out_has_offset) *out_has_offset = entry->has_offset;
            return true;
        }
        entry = entry->next;
    }
    return false;
}

void buffer_replace_map_free(BufferReplaceMap* map) {
    for (size_t i = 0; i < map->bucket_count; i++) {
        BufferReplaceEntry* entry = map->buckets[i];
        while (entry) {
            BufferReplaceEntry* next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    free(map);
}

// Helper functions for ReuseBufferMap
static bool reuse_key_equal(const ReuseBufferKey* a, const ReuseBufferKey* b) {
    return strcmp(a->device, b->device) == 0 &&
           a->dtype == b->dtype &&
           a->options == b->options &&
           a->nbytes == b->nbytes;
}

static size_t reuse_key_hash(const ReuseBufferKey* key) {
    size_t h = 0;
    for (const char* p = key->device; *p; p++) h = h * 31 + *p;
    h ^= hash_ptr(key->dtype);
    h ^= hash_ptr(key->options);
    h ^= key->nbytes;
    return h;
}

ReuseBufferMap* reuse_buffer_map_new(void) {
    ReuseBufferMap* map = calloc(1, sizeof(ReuseBufferMap));
    map->bucket_count = 16;
    map->buckets = calloc(map->bucket_count, sizeof(ReuseBufferEntry*));
    return map;
}

void reuse_buffer_map_append(ReuseBufferMap* map, const ReuseBufferKey* key, Buffer* buffer) {
    size_t idx = reuse_key_hash(key) % map->bucket_count;
    ReuseBufferEntry* entry = map->buckets[idx];
    
    // Find existing entry
    while (entry) {
        if (reuse_key_equal(&entry->key, key)) {
            // Append to existing list
            if (entry->buffer_count >= entry->buffer_capacity) {
                entry->buffer_capacity = entry->buffer_capacity ? entry->buffer_capacity * 2 : 4;
                entry->buffers = realloc(entry->buffers, entry->buffer_capacity * sizeof(Buffer*));
            }
            entry->buffers[entry->buffer_count++] = buffer;
            return;
        }
        entry = entry->next;
    }
    
    // Create new entry
    entry = malloc(sizeof(ReuseBufferEntry));
    entry->key = *key;
    entry->key.device = strdup(key->device);
    entry->buffer_capacity = 4;
    entry->buffers = malloc(entry->buffer_capacity * sizeof(Buffer*));
    entry->buffers[0] = buffer;
    entry->buffer_count = 1;
    entry->next = map->buckets[idx];
    map->buckets[idx] = entry;
    map->size++;
}

Buffer* reuse_buffer_map_pop(ReuseBufferMap* map, const ReuseBufferKey* key) {
    size_t idx = reuse_key_hash(key) % map->bucket_count;
    ReuseBufferEntry* entry = map->buckets[idx];
    
    while (entry) {
        if (reuse_key_equal(&entry->key, key)) {
            if (entry->buffer_count > 0) {
                return entry->buffers[--entry->buffer_count];
            }
            return NULL;
        }
        entry = entry->next;
    }
    return NULL;
}

void reuse_buffer_map_free(ReuseBufferMap* map) {
    for (size_t i = 0; i < map->bucket_count; i++) {
        ReuseBufferEntry* entry = map->buckets[i];
        while (entry) {
            ReuseBufferEntry* next = entry->next;
            free(entry->key.device);
            free(entry->buffers);
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    free(map);
}

// Helper functions for GlobalPlannerMap
GlobalPlannerMap* global_planner_map_new(void) {
    GlobalPlannerMap* map = calloc(1, sizeof(GlobalPlannerMap));
    map->bucket_count = 16;
    map->buckets = calloc(map->bucket_count, sizeof(GlobalPlannerEntry*));
    return map;
}

GlobalPlannerEntry* global_planner_map_get_or_create(GlobalPlannerMap* map, const char* device) {
    size_t h = 0;
    for (const char* p = device; *p; p++) h = h * 31 + *p;
    size_t idx = h % map->bucket_count;
    
    GlobalPlannerEntry* entry = map->buckets[idx];
    while (entry) {
        if (strcmp(entry->device, device) == 0) return entry;
        entry = entry->next;
    }
    
    // Create new entry with defaultdict lambda: (0, TLSFAllocator(1 << 44, block_size=0x1000, lv2_cnt=32))
    entry = malloc(sizeof(GlobalPlannerEntry));
    entry->device = strdup(device);
    entry->size = 0;
    entry->allocator = tlsf_allocator_new(1LL << 44, 0x1000, 32);
    entry->next = map->buckets[idx];
    map->buckets[idx] = entry;
    map->size++;
    return entry;
}

void global_planner_map_free(GlobalPlannerMap* map) {
    for (size_t i = 0; i < map->bucket_count; i++) {
        GlobalPlannerEntry* entry = map->buckets[i];
        while (entry) {
            GlobalPlannerEntry* next = entry->next;
            free(entry->device);
            tlsf_allocator_free(entry->allocator);
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    free(map);
}

// Helper functions for BufferIntMap
BufferIntMap* buffer_int_map_new(void) {
    BufferIntMap* map = calloc(1, sizeof(BufferIntMap));
    map->bucket_count = 16;
    map->buckets = calloc(map->bucket_count, sizeof(BufferIntEntry*));
    return map;
}

void buffer_int_map_put(BufferIntMap* map, Buffer* key, int value) {
    size_t idx = hash_ptr(key) % map->bucket_count;
    BufferIntEntry* entry = map->buckets[idx];
    
    // Check if key exists
    while (entry) {
        if (entry->key == key) {
            entry->value = value;
            return;
        }
        entry = entry->next;
    }
    
    // Add new entry
    entry = malloc(sizeof(BufferIntEntry));
    entry->key = key;
    entry->value = value;
    entry->next = map->buckets[idx];
    map->buckets[idx] = entry;
    map->size++;
}

int buffer_int_map_get(BufferIntMap* map, Buffer* key, int default_value) {
    size_t idx = hash_ptr(key) % map->bucket_count;
    BufferIntEntry* entry = map->buckets[idx];
    
    while (entry) {
        if (entry->key == key) return entry->value;
        entry = entry->next;
    }
    return default_value;
}

bool buffer_int_map_contains(BufferIntMap* map, Buffer* key) {
    size_t idx = hash_ptr(key) % map->bucket_count;
    BufferIntEntry* entry = map->buckets[idx];
    
    while (entry) {
        if (entry->key == key) return true;
        entry = entry->next;
    }
    return false;
}

void buffer_int_map_free(BufferIntMap* map) {
    for (size_t i = 0; i < map->bucket_count; i++) {
        BufferIntEntry* entry = map->buckets[i];
        while (entry) {
            BufferIntEntry* next = entry->next;
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    free(map);
}

// Comparison function for sorting buffer requests
typedef struct BufferRequest {
    int time;
    bool is_open;
    Buffer* buf;
} BufferRequest;

static int compare_buffer_requests(const void* a, const void* b) {
    const BufferRequest* ra = a;
    const BufferRequest* rb = b;
    
    // Compare by time first
    if (ra->time != rb->time) return ra->time - rb->time;
    
    // If same time, open events come before close events
    if (ra->is_open != rb->is_open) return ra->is_open ? -1 : 1;
    
    return 0;
}

// Helper to round up to alignment
static int64_t round_up(int64_t x, int64_t alignment) {
    return ((x + alignment - 1) / alignment) * alignment;
}

// Port of lines 12-63: def _internal_memory_planner(...)
BufferMap* _internal_memory_planner(Buffer*** buffers, int* buffer_counts, int num_lists, 
                                    Buffer** noopt_buffers, int noopt_count,
                                    bool ignore_checks, const char* debug_prefix) {
    // Port of line 13: if NO_MEMORY_PLANNER: return {}
    if (NO_MEMORY_PLANNER) return buffer_map_new();
    
    // Port of line 14: first_appearance, last_appearance, buf_to_opt = {}, {}, set()
    BufferIntMap* first_appearance = buffer_int_map_new();
    BufferIntMap* last_appearance = buffer_int_map_new();
    // Note: buf_to_opt as a set would be needed for line 54-56, but we'll handle it differently
    
    // Port of lines 15-21: for i,u in enumerate(buffers):
    for (int i = 0; i < num_lists; i++) {
        for (int j = 0; j < buffer_counts[i]; j++) {
            Buffer* buf = buffers[i][j];
            
            // Port of line 17: should_skip = buf.is_allocated() or buf.base.is_allocated() or buf.uop_refcount > 0 or ...
            bool should_skip = buffer_is_allocated(buf) || 
                              (buffer_get_base(buf) && buffer_is_allocated(buffer_get_base(buf))) ||
                              buffer_get_refcount(buf) > 0;
            
            if (!should_skip && noopt_buffers) {
                Buffer* base = buffer_get_base(buf);
                if (base) {
                    for (int k = 0; k < noopt_count; k++) {
                        if (base == noopt_buffers[k]) {
                            should_skip = true;
                            break;
                        }
                    }
                }
            }
            
            // Port of line 18: if not ignore_checks and should_skip: continue
            if (!ignore_checks && should_skip) continue;
            
            Buffer* base = buffer_get_base(buf);
            if (!base) base = buf;  // Use buf itself if no base
            
            // Port of line 19: if buf.base not in first_appearance: first_appearance[buf.base] = i
            if (!buffer_int_map_contains(first_appearance, base)) {
                buffer_int_map_put(first_appearance, base, i);
            }
            
            // Port of line 20: last_appearance[buf.base] = i
            buffer_int_map_put(last_appearance, base, i);
        }
    }
    
    // Count total buffer requests for allocation
    int request_count = 0;
    for (size_t i = 0; i < first_appearance->bucket_count; i++) {
        BufferIntEntry* entry = first_appearance->buckets[i];
        while (entry) {
            request_count += 2;  // One for open, one for close
            entry = entry->next;
        }
    }
    
    // Port of lines 24-25: buffer_requests = sorted([...])
    BufferRequest* buffer_requests = malloc(request_count * sizeof(BufferRequest));
    int idx = 0;
    
    for (size_t i = 0; i < first_appearance->bucket_count; i++) {
        BufferIntEntry* entry = first_appearance->buckets[i];
        while (entry) {
            Buffer* buf = entry->key;
            int first = entry->value;
            int last = buffer_int_map_get(last_appearance, buf, first);
            
            buffer_requests[idx].time = first;
            buffer_requests[idx].is_open = true;
            buffer_requests[idx].buf = buf;
            idx++;
            
            buffer_requests[idx].time = last + 1;
            buffer_requests[idx].is_open = false;
            buffer_requests[idx].buf = buf;
            idx++;
            
            entry = entry->next;
        }
    }
    
    qsort(buffer_requests, request_count, sizeof(BufferRequest), compare_buffer_requests);
    
    // Port of lines 28-31: Initialize maps
    BufferReplaceMap* buffer_replace = buffer_replace_map_new();
    ReuseBufferMap* reuse_buffers = reuse_buffer_map_new();
    GlobalPlannerMap* global_planner = global_planner_map_new();
    
    // Port of lines 32-41: Process buffer requests
    for (int i = 0; i < request_count; i++) {
        bool is_open_ev = buffer_requests[i].is_open;
        Buffer* buf = buffer_requests[i].buf;
        
        // Port of line 34: if hasattr(Device[buf.device].allocator, "_offset") and not isinstance(buf.dtype, ImageDType):
        Device* dev = device_get(buffer_get_device(buf));
        bool can_suballocate = device_has_offset_allocator(dev) && !dtype_is_image(buffer_get_dtype(buf));
        
        if (can_suballocate) {
            if (is_open_ev) {
                // Port of line 35: buffer_replace[buf] = (None, global_planner[buf.device][1].alloc(...))
                GlobalPlannerEntry* gp = global_planner_map_get_or_create(global_planner, buffer_get_device(buf));
                int offset = tlsf_allocator_alloc(gp->allocator, round_up(buffer_get_nbytes(buf), 0x1000));
                buffer_replace_map_put(buffer_replace, buf, NULL, offset, true);
            } else {
                // Port of line 36: global_planner[buf.device][1].free(cast(int, buffer_replace[buf][1]))
                int offset;
                bool has_offset;
                if (buffer_replace_map_get(buffer_replace, buf, NULL, &offset, &has_offset)) {
                    GlobalPlannerEntry* gp = global_planner_map_get_or_create(global_planner, buffer_get_device(buf));
                    tlsf_allocator_free_block(gp->allocator, offset);
                }
            }
            
            // Port of line 37: global_planner[buf.device] = (max(...), global_planner[buf.device][1])
            GlobalPlannerEntry* gp = global_planner_map_get_or_create(global_planner, buffer_get_device(buf));
            int offset;
            bool has_offset;
            if (buffer_replace_map_get(buffer_replace, buf, NULL, &offset, &has_offset) && has_offset) {
                int64_t end = offset + buffer_get_nbytes(buf);
                if (end > gp->size) gp->size = end;
            }
        } else {
            // Port of line 39: key = (buf.device, buf.dtype, buf.options, buf.nbytes)
            ReuseBufferKey key = {
                .device = buffer_get_device(buf),
                .dtype = buffer_get_dtype(buf),
                .options = buffer_get_options(buf),
                .nbytes = buffer_get_nbytes(buf)
            };
            
            if (is_open_ev) {
                // Port of line 40: buffer_replace[buf] = (reuse_buffers[key].pop(), None) if ... else (buf, None)
                Buffer* reused = reuse_buffer_map_pop(reuse_buffers, &key);
                buffer_replace_map_put(buffer_replace, buf, reused ? reused : buf, -1, false);
            } else {
                // Port of line 41: reuse_buffers[key].append(cast(Buffer, buffer_replace[buf][0]))
                Buffer* base;
                if (buffer_replace_map_get(buffer_replace, buf, &base, NULL, NULL) && base) {
                    reuse_buffer_map_append(reuse_buffers, &key, base);
                }
            }
        }
    }
    
    // Port of line 44: Allocate global buffers based on the memory planner
    BufferMap* global_buffers = buffer_map_new();
    for (size_t i = 0; i < global_planner->bucket_count; i++) {
        GlobalPlannerEntry* entry = global_planner->buckets[i];
        while (entry) {
            if (entry->size > 0) {
                Buffer* global_buf = buffer_new(entry->device, round_up(entry->size, 0x1000), dtype_int8());
                buffer_map_put(global_buffers, (Buffer*)entry->device, global_buf);  // Hack: use device string as key
            }
            entry = entry->next;
        }
    }
    
    // Port of line 45: buffer_resolve:dict[Buffer, tuple[Buffer, int|None]] = {...}
    // This is combined with the assignment phase below
    
    // Port of lines 47-51: Assign buffers. First, assign full buffers (not sub-buffers)
    BufferMap* assigned = buffer_map_new();
    
    for (size_t i = 0; i < buffer_replace->bucket_count; i++) {
        BufferReplaceEntry* entry = buffer_replace->buckets[i];
        while (entry) {
            Buffer* buf = entry->key;
            Buffer* base = entry->base;
            
            if (!base && entry->has_offset) {
                // Use global buffer for this device
                base = buffer_map_get(global_buffers, (Buffer*)buffer_get_device(buf));
            }
            
            // Port of line 50-51: if buf != base:
            if (buf != base) {
                if (entry->has_offset) {
                    // Create sub-buffer with offset
                    Buffer* new_buf = buffer_new_with_base(buffer_get_device(buf), buffer_get_size(buf), 
                                                           buffer_get_dtype(buf), base, entry->offset);
                    buffer_map_put(assigned, buf, new_buf);
                } else {
                    // Use base directly
                    buffer_map_put(assigned, buf, base);
                }
            }
            
            entry = entry->next;
        }
    }
    
    // Port of lines 54-56: Now assign sub-buffers
    // This would require iterating through all buffers again to find sub-buffers
    // For now, we skip this as it needs buf_to_opt tracking
    
    // Port of lines 58-61: Debug output
    if (DEBUG >= 1) {
        // Count memory usage (simplified version)
        int64_t omem = 0, nmem = 0;
        int ocount = 0, ncount = 0;
        
        // We'd need to properly track unique buffers here
        // For now, just count entries
        for (size_t i = 0; i < first_appearance->bucket_count; i++) {
            BufferIntEntry* entry = first_appearance->buckets[i];
            while (entry) {
                omem += buffer_get_nbytes(entry->key);
                ocount++;
                entry = entry->next;
            }
        }
        
        for (size_t i = 0; i < assigned->bucket_count; i++) {
            BufferMapEntry* entry = assigned->buckets[i];
            while (entry) {
                nmem += buffer_get_nbytes(entry->value);
                ncount++;
                entry = entry->next;
            }
        }
        
        if (omem != nmem) {
            printf("%smemory reduced from %.2f MB -> %.2f MB, %d -> %d bufs\n",
                   debug_prefix, omem/1e6, nmem/1e6, ocount, ncount);
        }
    }
    
    // Cleanup
    buffer_int_map_free(first_appearance);
    buffer_int_map_free(last_appearance);
    buffer_replace_map_free(buffer_replace);
    reuse_buffer_map_free(reuse_buffers);
    global_planner_map_free(global_planner);
    buffer_map_free(global_buffers);
    free(buffer_requests);
    
    // Port of line 63: return assigned
    return assigned;
}

// Port of lines 65-69: def memory_planner(schedule:list[ScheduleItem]) -> list[ScheduleItem]:
ScheduleItem** memory_planner(ScheduleItem** schedule, int schedule_count, int* out_count) {
    // Port of line 66-67: Exclude buffers involved in load ops
    // First, collect noopt_buffers
    int noopt_capacity = 100;
    int noopt_count = 0;
    Buffer** noopt_buffers = malloc(noopt_capacity * sizeof(Buffer*));
    
    for (int i = 0; i < schedule_count; i++) {
        // Port of: if si.ast.op is not Ops.SINK for b in si.bufs
        if (schedule_item_get_ast(schedule[i])->op != OPS_SINK) {
            Buffer** bufs = schedule_item_get_bufs(schedule[i]);
            int buf_count = schedule_item_get_buf_count(schedule[i]);
            
            for (int j = 0; j < buf_count; j++) {
                // Add to noopt_buffers if not already there
                bool found = false;
                for (int k = 0; k < noopt_count; k++) {
                    if (noopt_buffers[k] == bufs[j]) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    if (noopt_count >= noopt_capacity) {
                        noopt_capacity *= 2;
                        noopt_buffers = realloc(noopt_buffers, noopt_capacity * sizeof(Buffer*));
                    }
                    noopt_buffers[noopt_count++] = bufs[j];
                }
            }
        }
    }
    
    // Port of line 67: assigned = _internal_memory_planner([list(si.bufs) for si in schedule], ...)
    Buffer*** all_buffers = malloc(schedule_count * sizeof(Buffer**));
    int* buffer_counts = malloc(schedule_count * sizeof(int));
    
    for (int i = 0; i < schedule_count; i++) {
        all_buffers[i] = schedule_item_get_bufs(schedule[i]);
        buffer_counts[i] = schedule_item_get_buf_count(schedule[i]);
    }
    
    BufferMap* assigned = _internal_memory_planner(all_buffers, buffer_counts, schedule_count,
                                                   noopt_buffers, noopt_count, false, "");
    
    // Port of line 69: return [ScheduleItem(si.ast, tuple(assigned.get(x, x) for x in si.bufs), ...) for si in schedule]
    ScheduleItem** result = malloc(schedule_count * sizeof(ScheduleItem*));
    
    for (int i = 0; i < schedule_count; i++) {
        Buffer** old_bufs = schedule_item_get_bufs(schedule[i]);
        int buf_count = schedule_item_get_buf_count(schedule[i]);
        
        Buffer** new_bufs = malloc(buf_count * sizeof(Buffer*));
        for (int j = 0; j < buf_count; j++) {
            Buffer* replacement = buffer_map_get(assigned, old_bufs[j]);
            new_bufs[j] = replacement ? replacement : old_bufs[j];
        }
        
        result[i] = schedule_item_new(schedule_item_get_ast(schedule[i]), 
                                      new_bufs, buf_count,
                                      schedule_item_get_metadata(schedule[i]),
                                      schedule_item_get_metadata_count(schedule[i]),
                                      schedule_item_get_fixedvars_keys(schedule[i]),
                                      schedule_item_get_fixedvars_values(schedule[i]),
                                      schedule_item_get_fixedvars_count(schedule[i]));
    }
    
    *out_count = schedule_count;
    
    // Cleanup
    free(all_buffers);
    free(buffer_counts);
    free(noopt_buffers);
    buffer_map_free(assigned);
    
    return result;
}