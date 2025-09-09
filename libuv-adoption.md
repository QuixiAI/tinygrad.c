Libuv Adoption Ideas (Tinygrad.c)

Goal
- Use libuv’s cross‑platform event loop, async I/O, timers, and threadpool where they simplify code or add capabilities, without entangling core math paths.

High-Impact Use Cases
- Async downloads and file I/O (fetch compat)
  - Move blocking file reads/writes and optional decompression orchestration off the main thread.
  - Use `uv_fs_*` for async fs ops, `uv_loop_t` owned by a small runtime helper.
  - Optional: integrate with libcurl multi via libuv FD callbacks if needed.
- Background kernel compilation or cache population
  - Use `uv_queue_work` to offload expensive codegen/compile steps while UI/tests continue.
  - Pair with `uv_async_send` to signal availability.
- Timers for profiling/scheduling
  - `uv_timer_t` to pace periodic maintenance (cache eviction, memory reuse audits) or benchmarking intervals.
- Cross-platform threading
  - If/when we introduce concurrency, prefer `uv_thread_t`, `uv_mutex_t`, `uv_cond_t` abstractions for portability.

Medium-Impact / Optional
- Event-driven device backends
  - If any device benefits from evented file descriptors (e.g., IPC pipes, sockets), use `uv_poll_t` and `uv_async_t` to integrate cleanly.
- File watching for live dev flows
  - `uv_fs_event_t` to watch generated code dirs or test assets for fast edit‑run cycles.

Pragmatic Guidance
- Keep libuv isolated behind a small adapter (e.g., src/runtime/uv_runtime.c) to avoid polluting core layers.
- Own a single `uv_loop_t` per process (or per subsystem) and provide simple APIs: submit work, schedule timer, signal.
- Ensure clean shutdown: stop timers, cancel queued work where safe, and `uv_loop_close` after handles are cleaned up.
- Guard all use with `TG_HAVE_LIBUV` so builds remain optional.

Sketch: Background Work Helper
- uv_runtime.h/c:
  - `int uvrt_init(void); int uvrt_shutdown(void);`
  - `int uvrt_queue_work(void (*work)(void*), void (*after)(int, void*), void* arg);`
  - Internally starts a loop thread or reuses the main loop depending on embedding.

Notes
- Conan exposes CMake target `uv`; include with `find_package(libuv REQUIRED)` and `target_link_libraries(tgt uv)`.
- Headers available as `#include "uv.h"` and platform headers `uv/unix.h`, `uv/win.h` as needed.

