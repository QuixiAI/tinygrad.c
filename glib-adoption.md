GLib Adoption Ideas (Tinygrad.c)

Goal
- Introduce GLib where it reduces code, improves robustness, or adds needed primitives — without regressing perf in hot paths.

High-ROI Replacements (Low Blast Radius)
- UOp → Buffer map (src/uop/ops.c)
  - Replace custom chained-bucket map with `GHashTable` (key: `UOp*`, value: `void*`).
  - `g_hash_table_new(g_direct_hash, g_direct_equal)`; use `g_hash_table_replace()` to update; `g_hash_table_remove()` on free.
  - Add a single shutdown hook (or tie into `uop` finalization) to `g_hash_table_destroy()`.
- Gradient dict (src/gradient/gradient.c)
  - Swap bespoke bucket map for `GHashTable` with `g_direct_hash/g_direct_equal`.
  - Optional `_full` variant with key/value destroy to ensure leak‑free teardown.
- ReuseBufferKey map (src/engine/memory.c)
  - Convert to `g_hash_table_new_full(custom_hash, custom_equal, key_destroy, value_destroy)`.
  - Simplifies lifetime and equality semantics, centralizes hashing.
- UOp meta key set (src/uop/ops.c)
  - Replace linked list of strings with `GHashTable` as a set (`g_str_hash/g_str_equal`).
  - Use `g_hash_table_add()` style via boolean value or `GINT_TO_POINTER(1)`; enable `g_free` foreach key in destroy.

Dynamic Collections & Strings
- GArray / GPtrArray
  - Replace ad‑hoc `malloc/realloc` vectors used in e.g., `BoundList`, pattern lists, temporary axes aggregation.
  - Provides bounds checks, append/pop convenience; free with `g_array_free(arr, TRUE)`.
- GString
  - Use for `uop_pretty_str`, AST dumps, graph visualizations.
  - Replaces manual buffer concat; `g_string_free(s, FALSE)` to return char* if needed.

Concurrency (Future)
- GMutex/GCond around any global caches if multi-threading appears (renderer caches, kernel caches).
- GThreadPool for parallelizable preprocessing (e.g., constant propagation or graph partition metrics), if it proves beneficial.

Pragmatic Guidance
- Keep GLib out of tight numeric loops and hot shape math; prefer stack arrays and plain C structs there.
- Prefer `g_direct_hash/g_direct_equal` for pointer-key tables (UOp*, buffers). Use `g_str_hash/g_str_equal` for strings.
- Always use `g_hash_table_new_full` when you own keys/values to ensure clean teardown paths.
- Consider `GPtrArray` for pattern registries to allow dynamic enable/disable without recompilation.

Adoption Order (Suggested)
1) UOp→Buffer map (ops.c) → GHashTable
2) Gradient dict (gradient.c) → GHashTable
3) Reuse buffer map (engine/memory.c) → GHashTable
4) Meta key set → GHashTable
5) GString in pretty printers
6) GArray/GPtrArray for dynamic lists in symbolic helpers

Notes
- Conan provides `glib` imported targets: `glib::glib`, `glib::gobject`, `glib::gio`, `glib::gio-unix`.
- Add compile-time toggles (e.g., `TG_HAVE_GLIB`) so GLib use is optional and easy to disable.

