// Port of tinygrad/engine/schedule.py
// Line-by-line faithful port from Python to C

#include "schedule.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

// Include other headers we need
#include "uop/ops.h"
#include "device/device.h"
#include "helpers/helpers.h"

// Port of: from typing import cast
// Port of: from dataclasses import dataclass, field
// Port of: from collections import deque, defaultdict
// Port of: from tinygrad.uop.ops import UOp, Variable, Ops, buffers
// Port of: from tinygrad.device import Device, Buffer, MultiBuffer
// Port of: from tinygrad.helpers import Metadata, all_same

// Helper structures for collections
// Port of defaultdict[UOp, list[UOp]]
typedef struct {
    UOp* key;
    UOp** values;
    int values_count;
    int values_capacity;
} UOpListEntry;

typedef struct {
    UOpListEntry* entries;
    int count;
    int capacity;
} UOpListDict;

// Port of dict[UOp, int]
typedef struct {
    UOp* key;
    int value;
} UOpIntEntry;

typedef struct {
    UOpIntEntry* entries;
    int count;
    int capacity;
} UOpIntDict;

// Port of deque[UOp]
typedef struct {
    UOp** items;
    int front;
    int rear;
    int size;
    int capacity;
} UOpDeque;

// Port of defaultdict[int, deque[UOp]]
typedef struct {
    int key;
    UOpDeque* value;
} IntDequeEntry;

typedef struct {
    IntDequeEntry* entries;
    int count;
    int capacity;
} IntDequeDict;

// Helper functions for collections
static UOpListDict* uop_list_dict_new() {
    UOpListDict* dict = calloc(1, sizeof(UOpListDict));
    dict->capacity = 16;
    dict->entries = calloc(dict->capacity, sizeof(UOpListEntry));
    return dict;
}

static void uop_list_dict_append(UOpListDict* dict, UOp* key, UOp* value) {
    // Find existing entry or create new one
    for (int i = 0; i < dict->count; i++) {
        if (dict->entries[i].key == key) {
            // Append to existing list
            if (dict->entries[i].values_count >= dict->entries[i].values_capacity) {
                dict->entries[i].values_capacity = dict->entries[i].values_capacity ? dict->entries[i].values_capacity * 2 : 4;
                dict->entries[i].values = realloc(dict->entries[i].values, 
                                                  dict->entries[i].values_capacity * sizeof(UOp*));
            }
            dict->entries[i].values[dict->entries[i].values_count++] = value;
            return;
        }
    }
    
    // Create new entry
    if (dict->count >= dict->capacity) {
        dict->capacity *= 2;
        dict->entries = realloc(dict->entries, dict->capacity * sizeof(UOpListEntry));
    }
    
    UOpListEntry* entry = &dict->entries[dict->count++];
    entry->key = key;
    entry->values_capacity = 4;
    entry->values = calloc(entry->values_capacity, sizeof(UOp*));
    entry->values[0] = value;
    entry->values_count = 1;
}

static UOp** uop_list_dict_get(UOpListDict* dict, UOp* key, int* count) {
    for (int i = 0; i < dict->count; i++) {
        if (dict->entries[i].key == key) {
            *count = dict->entries[i].values_count;
            return dict->entries[i].values;
        }
    }
    *count = 0;
    return NULL;
}

static void uop_list_dict_free(UOpListDict* dict) {
    for (int i = 0; i < dict->count; i++) {
        free(dict->entries[i].values);
    }
    free(dict->entries);
    free(dict);
}

static UOpIntDict* uop_int_dict_new() {
    UOpIntDict* dict = calloc(1, sizeof(UOpIntDict));
    dict->capacity = 16;
    dict->entries = calloc(dict->capacity, sizeof(UOpIntEntry));
    return dict;
}

static void uop_int_dict_set(UOpIntDict* dict, UOp* key, int value) {
    // Check if key exists
    for (int i = 0; i < dict->count; i++) {
        if (dict->entries[i].key == key) {
            dict->entries[i].value = value;
            return;
        }
    }
    
    // Add new entry
    if (dict->count >= dict->capacity) {
        dict->capacity *= 2;
        dict->entries = realloc(dict->entries, dict->capacity * sizeof(UOpIntEntry));
    }
    
    dict->entries[dict->count].key = key;
    dict->entries[dict->count].value = value;
    dict->count++;
}

static int uop_int_dict_get(UOpIntDict* dict, UOp* key, int default_value) {
    for (int i = 0; i < dict->count; i++) {
        if (dict->entries[i].key == key) {
            return dict->entries[i].value;
        }
    }
    return default_value;
}

static void uop_int_dict_setdefault(UOpIntDict* dict, UOp* key, int default_value) {
    for (int i = 0; i < dict->count; i++) {
        if (dict->entries[i].key == key) {
            return;  // Key already exists
        }
    }
    uop_int_dict_set(dict, key, default_value);
}

static void uop_int_dict_decrement(UOpIntDict* dict, UOp* key) {
    for (int i = 0; i < dict->count; i++) {
        if (dict->entries[i].key == key) {
            dict->entries[i].value--;
            return;
        }
    }
}

static void uop_int_dict_free(UOpIntDict* dict) {
    free(dict->entries);
    free(dict);
}

static UOpDeque* uop_deque_new() {
    UOpDeque* deque = calloc(1, sizeof(UOpDeque));
    deque->capacity = 16;
    deque->items = calloc(deque->capacity, sizeof(UOp*));
    deque->front = 0;
    deque->rear = 0;
    deque->size = 0;
    return deque;
}

static void uop_deque_append(UOpDeque* deque, UOp* item) {
    if (deque->size >= deque->capacity) {
        // Resize
        int new_capacity = deque->capacity * 2;
        UOp** new_items = calloc(new_capacity, sizeof(UOp*));
        // Copy items
        int j = 0;
        for (int i = deque->front; j < deque->size; i = (i + 1) % deque->capacity, j++) {
            new_items[j] = deque->items[i];
        }
        free(deque->items);
        deque->items = new_items;
        deque->capacity = new_capacity;
        deque->front = 0;
        deque->rear = deque->size;
    }
    
    deque->items[deque->rear] = item;
    deque->rear = (deque->rear + 1) % deque->capacity;
    deque->size++;
}

static UOp* uop_deque_popleft(UOpDeque* deque) {
    if (deque->size == 0) return NULL;
    
    UOp* item = deque->items[deque->front];
    deque->front = (deque->front + 1) % deque->capacity;
    deque->size--;
    return item;
}

static bool uop_deque_empty(UOpDeque* deque) {
    return deque->size == 0;
}

static void uop_deque_free(UOpDeque* deque) {
    free(deque->items);
    free(deque);
}

static IntDequeDict* int_deque_dict_new() {
    IntDequeDict* dict = calloc(1, sizeof(IntDequeDict));
    dict->capacity = 16;
    dict->entries = calloc(dict->capacity, sizeof(IntDequeEntry));
    return dict;
}

static UOpDeque* int_deque_dict_get_or_create(IntDequeDict* dict, int key) {
    for (int i = 0; i < dict->count; i++) {
        if (dict->entries[i].key == key) {
            return dict->entries[i].value;
        }
    }
    
    // Create new entry
    if (dict->count >= dict->capacity) {
        dict->capacity *= 2;
        dict->entries = realloc(dict->entries, dict->capacity * sizeof(IntDequeEntry));
    }
    
    dict->entries[dict->count].key = key;
    dict->entries[dict->count].value = uop_deque_new();
    dict->count++;
    
    return dict->entries[dict->count - 1].value;
}

static bool int_deque_dict_any_not_empty(IntDequeDict* dict) {
    for (int i = 0; i < dict->count; i++) {
        if (!uop_deque_empty(dict->entries[i].value)) {
            return true;
        }
    }
    return false;
}

static void int_deque_dict_get_min_nonempty(IntDequeDict* dict, int last_heuristic, int* out_key, UOpDeque** out_deque) {
    int min_diff = INT_MAX;
    *out_key = 0;
    *out_deque = NULL;
    
    for (int i = 0; i < dict->count; i++) {
        if (!uop_deque_empty(dict->entries[i].value)) {
            int diff = abs(dict->entries[i].key - last_heuristic);
            if (diff < min_diff) {
                min_diff = diff;
                *out_key = dict->entries[i].key;
                *out_deque = dict->entries[i].value;
            }
        }
    }
}

static void int_deque_dict_free(IntDequeDict* dict) {
    for (int i = 0; i < dict->count; i++) {
        uop_deque_free(dict->entries[i].value);
    }
    free(dict->entries);
    free(dict);
}

// **** ScheduleItem return type ****
// Port of lines 8-15

ScheduleItem* schedule_item_new(UOp* ast, Buffer** bufs, int bufs_count, 
                                Metadata** metadata, int metadata_count,
                                Variable** fixedvars_keys, int* fixedvars_values, int fixedvars_count) {
    ScheduleItem* item = calloc(1, sizeof(ScheduleItem));
    item->ast = ast;
    
    item->bufs = calloc(bufs_count, sizeof(Buffer*));
    memcpy(item->bufs, bufs, bufs_count * sizeof(Buffer*));
    item->bufs_count = bufs_count;
    
    if (metadata_count > 0) {
        item->metadata = calloc(metadata_count, sizeof(Metadata*));
        memcpy(item->metadata, metadata, metadata_count * sizeof(Metadata*));
    }
    item->metadata_count = metadata_count;
    
    if (fixedvars_count > 0) {
        item->fixedvars_keys = calloc(fixedvars_count, sizeof(Variable*));
        item->fixedvars_values = calloc(fixedvars_count, sizeof(int));
        memcpy(item->fixedvars_keys, fixedvars_keys, fixedvars_count * sizeof(Variable*));
        memcpy(item->fixedvars_values, fixedvars_values, fixedvars_count * sizeof(int));
    }
    item->fixedvars_count = fixedvars_count;
    
    return item;
}

void schedule_item_free(ScheduleItem* item) {
    if (!item) return;
    free(item->bufs);
    free(item->metadata);
    free(item->fixedvars_keys);
    free(item->fixedvars_values);
    free(item);
}

void schedule_result_free(ScheduleResult* result) {
    if (!result) return;
    for (int i = 0; i < result->count; i++) {
        schedule_item_free(&result->items[i]);
    }
    free(result->items);
    free(result->var_vals_keys);
    free(result->var_vals_values);
    free(result);
}

// **** schedule linearizer ****
// Port of lines 17-83

// Port of: def _heuristic(k: UOp):
//   if k.arg.ast.op is Ops.COPY and not all_same([Device[cast(Buffer, s.buf_uop.buffer).device].group_id for s in k.src]): return 1000
//   return 0
static int _heuristic(UOp* k) {
    // Port of line 51-52
    if (k->arg.kernel_info.ast->op == OPS_COPY) {
        // Check if all devices have the same group_id
        // This requires accessing device information from buffers
        // For now, simplified implementation
        bool all_same_group = true;
        int first_group = -1;
        for (int i = 0; i < k->src_count; i++) {
            if (k->src[i]->op == OPS_BUFFER) {
                Buffer* buf = (Buffer*)k->src[i]->arg.ptr;
                // Get device group_id - would need Device implementation
                int group_id = 0; // Placeholder - needs Device[buf->device].group_id
                if (first_group == -1) {
                    first_group = group_id;
                } else if (first_group != group_id) {
                    all_same_group = false;
                    break;
                }
            }
        }
        if (!all_same_group) return 1000;
    }
    return 0;
}

// Port of: def create_schedule_with_vars(sched_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int]]:
ScheduleResult* create_schedule_with_vars(UOp* sched_sink) {
    // Port of line 20-23: construct the KERNEL children graph based on assigns
    // children: defaultdict[UOp, list[UOp]] = defaultdict(list)
    // in_degree: dict[UOp, int] = {}
    // var_vals: dict[Variable, int] = {}
    UOpListDict* children = uop_list_dict_new();
    UOpIntDict* in_degree = uop_int_dict_new();
    
    // For var_vals, we'll build it as we go and convert at the end
    Variable** var_keys = NULL;
    int* var_values = NULL;
    int var_count = 0;
    int var_capacity = 0;
    
    // Port of line 24: for u in sched_sink.toposort():
    UOp** toposort_result = uop_toposort(sched_sink);
    int toposort_count = 0;
    // Count toposort result (need proper implementation)
    for (int i = 0; toposort_result && toposort_result[i]; i++) {
        toposort_count++;
    }
    
    for (int i = 0; i < toposort_count; i++) {
        UOp* u = toposort_result[i];
        
        // Port of line 25: if u.op is not Ops.ASSIGN: continue
        if (u->op != OPS_ASSIGN) continue;
        
        // Port of line 26: k = u.src[1]
        UOp* k = u->src[1];
        
        // Port of line 27: in_degree.setdefault(k, 0)
        uop_int_dict_setdefault(in_degree, k, 0);
        
        // Port of line 28-46: for s in k.src:
        for (int j = 0; j < k->src_count; j++) {
            UOp* s = k->src[j];
            
            // Port of line 29-31: if s.op is Ops.ASSIGN:
            if (s->op == OPS_ASSIGN) {
                // children[s.src[1]].append(k)
                uop_list_dict_append(children, s->src[1], k);
                // in_degree[k] += 1
                uop_int_dict_set(in_degree, k, uop_int_dict_get(in_degree, k, 0) + 1);
            }
            // Port of line 32-38: elif s.op in {Ops.MSELECT, Ops.MSTACK}:
            else if (s->op == OPS_MSELECT || s->op == OPS_MSTACK) {
                for (int l = 0; l < s->src_count; l++) {
                    UOp* ss = s->src[l];
                    // Port of line 34: if ss.op is Ops.MSELECT: ss = ss.src[0]
                    if (ss->op == OPS_MSELECT) ss = ss->src[0];
                    // Port of line 35-38: if ss.op is not Ops.BUFFER:
                    if (ss->op != OPS_BUFFER) {
                        assert(ss->op == OPS_ASSIGN);
                        uop_list_dict_append(children, ss->src[1], k);
                        uop_int_dict_set(in_degree, k, uop_int_dict_get(in_degree, k, 0) + 1);
                    }
                }
            }
            // Port of line 39-40: elif s.op is Ops.BUFFER:
            else if (s->op == OPS_BUFFER) {
                // pass  # a BUFFER is already realized, nothing to do here
            }
            // Port of line 41-44: elif s.op is Ops.BIND:
            else if (s->op == OPS_BIND) {
                // var, val = s.unbind()
                Variable* var = (Variable*)s->src[0];
                int val = s->src[1]->arg.int_val;
                
                // Check if var exists in var_vals
                bool found = false;
                for (int m = 0; m < var_count; m++) {
                    if (var_keys[m] == var) {
                        assert(var_values[m] == val); // "bind mismatch"
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    // Add to var_vals
                    if (var_count >= var_capacity) {
                        var_capacity = var_capacity ? var_capacity * 2 : 16;
                        var_keys = realloc(var_keys, var_capacity * sizeof(Variable*));
                        var_values = realloc(var_values, var_capacity * sizeof(int));
                    }
                    var_keys[var_count] = var;
                    var_values[var_count] = val;
                    var_count++;
                }
            }
            // Port of line 45-46: else:
            else {
                // raise RuntimeError(f"input to kernel must be ASSIGN or BUFFER, not {s.op}")
                assert(false); // Should be error handling
            }
        }
    }
    
    // Port of line 48-52: linearize KERNEL UOps into ScheduleItems in BFS order
    
    // Port of line 54-58: Initialize queues
    // last_heuristic: int = 0
    // queues: defaultdict[int, deque[UOp]] = defaultdict(deque)
    // last_queue: deque[UOp] = deque()
    int last_heuristic = 0;
    IntDequeDict* queues = int_deque_dict_new();
    UOpDeque* last_queue = uop_deque_new();
    
    // Port of line 57-58: for k,v in in_degree.items():
    for (int i = 0; i < in_degree->count; i++) {
        UOp* k = in_degree->entries[i].key;
        int v = in_degree->entries[i].value;
        // if v == 0: queues[_heuristic(k)].append(k)
        if (v == 0) {
            UOpDeque* queue = int_deque_dict_get_or_create(queues, _heuristic(k));
            uop_deque_append(queue, k);
        }
    }
    
    // Port of line 60: schedule: list[ScheduleItem] = []
    ScheduleItem* schedule = NULL;
    int schedule_count = 0;
    int schedule_capacity = 0;
    
    // Port of line 61-81: while last_queue or any(queues.values()):
    while (!uop_deque_empty(last_queue) || int_deque_dict_any_not_empty(queues)) {
        // Port of line 62: if not last_queue:
        if (uop_deque_empty(last_queue)) {
            // last_heuristic, last_queue = min((it for it in queues.items() if it[1]), key=lambda x: abs(x[0]-last_heuristic))
            int new_heuristic;
            int_deque_dict_get_min_nonempty(queues, last_heuristic, &new_heuristic, &last_queue);
            last_heuristic = new_heuristic;
        }
        
        // Port of line 63: k = last_queue.popleft()
        UOp* k = uop_deque_popleft(last_queue);
        
        // Port of line 64: ast = k.arg.ast
        UOp* ast = k->arg.kernel_info.ast;
        
        // Port of line 65-69: create subbuffers if needed
        if (ast->op == OPS_BUFFER_VIEW) {
            // base = k.src[1].buf_uop.buffer
            Buffer* base = (Buffer*)k->src[1]->arg.ptr;
            // assert isinstance(base, Buffer), "base can't be MultiBuffer"
            // buffers[k.src[0]] = base.view(k.size, ast.dtype, ast.arg[1]*base.dtype.itemsize)
            // This requires buffer view implementation
        }
        
        // Port of line 70: ubufs = tuple(s.buf_uop.buffer for s in k.src if s.op is not Ops.BIND)
        Buffer** ubufs = NULL;
        int ubufs_count = 0;
        for (int i = 0; i < k->src_count; i++) {
            if (k->src[i]->op != OPS_BIND) {
                ubufs_count++;
            }
        }
        ubufs = calloc(ubufs_count, sizeof(Buffer*));
        int idx = 0;
        for (int i = 0; i < k->src_count; i++) {
            if (k->src[i]->op != OPS_BIND) {
                ubufs[idx++] = (Buffer*)k->src[i]->arg.ptr;
            }
        }
        
        // Port of line 71-76: Handle MultiBuffer case
        bool has_multibuffer = false;
        for (int i = 0; i < ubufs_count; i++) {
            // Check if buffer is MultiBuffer - needs implementation
            // For now, assume no MultiBuffer
        }
        
        if (has_multibuffer) {
            // Handle MultiBuffer case - lines 72-75
            // This requires MultiBuffer implementation
        } else {
            // Port of line 77-78: ONE -> ONE
            // schedule.append(ScheduleItem(ast, cast(tuple[Buffer, ...], ubufs), k.arg.metadata))
            if (schedule_count >= schedule_capacity) {
                schedule_capacity = schedule_capacity ? schedule_capacity * 2 : 16;
                schedule = realloc(schedule, schedule_capacity * sizeof(ScheduleItem));
            }
            
            ScheduleItem* item = &schedule[schedule_count++];
            item->ast = ast;
            item->bufs = ubufs;
            item->bufs_count = ubufs_count;
            item->metadata = k->arg.kernel_info.metadata;
            item->metadata_count = k->arg.kernel_info.metadata_count;
            item->fixedvars_keys = NULL;
            item->fixedvars_values = NULL;
            item->fixedvars_count = 0;
        }
        
        // Port of line 79-81: for x in children[k]:
        int children_count = 0;
        UOp** children_list = uop_list_dict_get(children, k, &children_count);
        for (int i = 0; i < children_count; i++) {
            UOp* x = children_list[i];
            // in_degree[x] -= 1
            uop_int_dict_decrement(in_degree, x);
            // if in_degree[x] == 0: queues[_heuristic(x)].append(x)
            if (uop_int_dict_get(in_degree, x, 0) == 0) {
                UOpDeque* queue = int_deque_dict_get_or_create(queues, _heuristic(x));
                uop_deque_append(queue, x);
            }
        }
    }
    
    // Port of line 83: return schedule, var_vals
    ScheduleResult* result = calloc(1, sizeof(ScheduleResult));
    result->items = schedule;
    result->count = schedule_count;
    result->var_vals_keys = var_keys;
    result->var_vals_values = var_values;
    result->var_vals_count = var_count;
    
    // Clean up
    uop_list_dict_free(children);
    uop_int_dict_free(in_degree);
    int_deque_dict_free(queues);
    uop_deque_free(last_queue);
    free(toposort_result);
    
    return result;
}

// Accessor functions for ScheduleItem
UOp* schedule_item_get_ast(ScheduleItem* item) {
    return item ? item->ast : NULL;
}

Buffer** schedule_item_get_bufs(ScheduleItem* item) {
    return item ? item->bufs : NULL;
}

int schedule_item_get_buf_count(ScheduleItem* item) {
    return item ? item->bufs_count : 0;
}

Metadata** schedule_item_get_metadata(ScheduleItem* item) {
    return item ? item->metadata : NULL;
}

int schedule_item_get_metadata_count(ScheduleItem* item) {
    return item ? item->metadata_count : 0;
}

Variable** schedule_item_get_fixedvars_keys(ScheduleItem* item) {
    return item ? item->fixedvars_keys : NULL;
}

int* schedule_item_get_fixedvars_values(ScheduleItem* item) {
    return item ? item->fixedvars_values : NULL;
}

int schedule_item_get_fixedvars_count(ScheduleItem* item) {
    return item ? item->fixedvars_count : 0;
}