# SiMa NEAT Skill

Use this skill when generating, debugging, or refactoring C++ code that builds and runs SiMa NEAT pipelines.

## Goal

Generate production-usable C++20 SiMa NEAT code with correct public APIs, deterministic pipeline composition, and runnable build/test steps.

## Environment Assumptions

- SiMa NEAT is installed and discoverable via CMake config package.
- Project uses CMake and links `SimaNeat::sima_neat`.
- Public headers are available under the install prefix include directory.

## Required Defaults

- Language standard: C++20.
- Public API only: include files from installed public headers (`include/`), never internal implementation headers.
- Deterministic pipeline composition: explicit node/group setup and predictable names/options.
- Keep examples runnable end-to-end with build commands.

## Preferred Include Surface

Use specific public headers, for example:

- `#include <pipeline/Session.h>`
- `#include <pipeline/PipelineRun.h>`
- `#include <pipeline/Tensor.h>`
- `#include <model/Model.h>`
- `#include <nodes/io/InputAppSrc.h>`
- `#include <nodes/common/AppSink.h>`

## Canonical CMake Template

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

## Skill Workflow

1. Classify request by task type (session pipeline, model pipeline, tensor/sample handling, diagnostics, tests).
2. Select the closest recipe from `references/recipes.md`.
3. Apply API constraints from `references/api_surface.md` and `references/do_dont.md`.
4. Validate tensor/sample assumptions via `references/tensor_sample_contracts.md`.
5. Provide build/test commands from `references/build_test.md`.

## References

- `references/minimal_app.md`
- `references/session_pattern.md`
- `references/api_surface.md`
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
