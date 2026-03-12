# SiMa NEAT Skill

Use this skill for SiMa NEAT application work in C++ or Python (`pyneat`).

## Environment

- DevKit: use installed NEAT directly.
- eLxr SDK: expect `SYSROOT=/opt/toolchain/aarch64/modalix`.
- In SDK, prefer source context from `/neat-resources/core-src` and `/neat-resources/apps-src`.
- In SDK, prefer `dk`/`devkit-run` for runtime checks on a connected DevKit.
- On DevKit, Python usually runs from `$HOME/pyneat/bin/activate`.

## Defaults

- C++: C++20.
- Python: use installed public `pyneat`.
- Public API only.
- Keep pipeline composition explicit and deterministic.
- Keep generated examples runnable with build/run commands.

## C++ Default

Prefer:

```cpp
#include <neat.h>
```

Use narrower public headers only when the task benefits from tighter dependencies.

## Python Default

Use public `pyneat` only.

## Generated CMake

```cmake
cmake_minimum_required(VERSION 3.16)
project(app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE SimaNeat::sima_neat)
```

## Workflow

1. Detect environment first:
   - SDK with `/neat-resources/...`: read source there first.
   - Installed environment only: use installed headers/runtime paths.
2. Run DevKit preflight before coding or testing from SDK:
   - check `dk`/`devkit-run` first
   - confirm the target will be built as ARM64
   - confirm required model/image assets exist when the task depends on them
   - use `references/devkit_preflight.md`
3. Detect language:
   - C++: default to `#include <neat.h>`.
   - Python: default to installed `pyneat`.
4. Pick the closest reference:
   - Environment/runtime: `references/environment_layout.md`, `references/build_test.md`
   - C++ patterns: `references/recipes.md`
   - Python patterns: `references/pyneat_patterns.md`
5. If the user asks to run on DevKit, do not stop at local compile:
   - build the ARM64 target
   - run it with `dk /workspace/...` when available
   - fall back to direct SSH only if `dk` is unavailable
6. Apply `references/api_surface.md` and `references/do_dont.md`.
7. Check `references/tensor_sample_contracts.md` when tensors/samples are involved.

## References

- `references/environment_layout.md`
- `references/devkit_preflight.md`
- `references/minimal_app.md`
- `references/session_pattern.md`
- `references/api_surface.md`
- `references/pyneat_patterns.md`
- `references/recipes.md`
- `references/tensor_sample_contracts.md`
- `references/diagnostics.md`
- `references/build_test.md`
- `references/do_dont.md`
- `references/migration_cpp20.md`
- `references/advanced_patterns.md`
- `references/coverage_map.md`

## Templates

- `templates/session_sync_main.cpp`
- `templates/session_async_main.cpp`
- `templates/model_runner_main.cpp`
- `templates/CMakeLists.txt`
- `templates/pyneat_hello.py`
