# Backend: vulkan
- Build a shared library exposing `tg_backend_load_fn` (see include/tg_backend.h).
- Implement the `tg_backend_v1` vtable for memory + op execution.
- Keep this plugin standalone: no core changes required to add a backend.
