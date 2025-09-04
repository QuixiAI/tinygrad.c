// Port of tinygrad/engine/schedule.py
#ifndef TINYGRAD_ENGINE_SCHEDULE_H
#define TINYGRAD_ENGINE_SCHEDULE_H

#include <stdint.h>
#include <stdbool.h>

// Forward declarations
typedef struct UOp UOp;
typedef struct Variable Variable;
typedef struct Buffer Buffer;
typedef struct Metadata Metadata;

// Port of: @dataclass(frozen=True)
// Port of: class ScheduleItem:
//   ast: UOp
//   bufs: tuple[Buffer, ...]
//   metadata: tuple[Metadata, ...] = ()
//   fixedvars: dict[Variable, int] = field(default_factory=dict)
typedef struct ScheduleItem {
    UOp* ast;
    Buffer** bufs;
    int bufs_count;
    Metadata** metadata;
    int metadata_count;
    // Port of dict[Variable, int] as arrays
    Variable** fixedvars_keys;
    int* fixedvars_values;
    int fixedvars_count;
} ScheduleItem;

// Port of: def create_schedule_with_vars(sched_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int]]:
typedef struct {
    ScheduleItem* items;
    int count;
    // Port of dict[Variable, int] as arrays
    Variable** var_vals_keys;
    int* var_vals_values;
    int var_vals_count;
} ScheduleResult;

ScheduleResult* create_schedule_with_vars(UOp* sched_sink);

// Helper functions for ScheduleItem
ScheduleItem* schedule_item_new(UOp* ast, Buffer** bufs, int bufs_count, 
                                Metadata** metadata, int metadata_count,
                                Variable** fixedvars_keys, int* fixedvars_values, int fixedvars_count);
void schedule_item_free(ScheduleItem* item);
void schedule_result_free(ScheduleResult* result);

// Accessor functions for ScheduleItem
UOp* schedule_item_get_ast(ScheduleItem* item);
Buffer** schedule_item_get_bufs(ScheduleItem* item);
int schedule_item_get_buf_count(ScheduleItem* item);
Metadata** schedule_item_get_metadata(ScheduleItem* item);
int schedule_item_get_metadata_count(ScheduleItem* item);
Variable** schedule_item_get_fixedvars_keys(ScheduleItem* item);
int* schedule_item_get_fixedvars_values(ScheduleItem* item);
int schedule_item_get_fixedvars_count(ScheduleItem* item);

#endif // TINYGRAD_ENGINE_SCHEDULE_H