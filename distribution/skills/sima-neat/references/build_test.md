# Build and Test Commands

- DevKit/native: use installed paths directly.
- eLxr SDK: expect `SYSROOT`, prefer `/neat-resources/...` for source context, and use `dk` for runtime checks.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Unit + E2E tests

```bash
ctest --test-dir build/tests --output-on-failure
```

## Python

Activate `pyneat` first:

```bash
source "$HOME/pyneat/bin/activate"
python3 app.py
```

## SDK runtime

When cross-compiling inside eLxr SDK and you need to run the target binary on a connected DevKit from the SDK shell:

```bash
dk /workspace/path/to/build/my_app
```

Use this pattern to validate runtime behavior and collect stdout/stderr during development without manually copying the binary first.

Rules:

- Only use `dk` with targets under `/workspace`.
- Non-Python targets must be ARM64/aarch64.
- Python scripts are allowed:

```bash
dk /workspace/path/to/app.py
```

- `dk` runs remotely on the DevKit over SSH and streams prefixed stdout/stderr back to the SDK shell.

## Tutorial tests

```bash
ctest --test-dir build/tutorials --output-on-failure
```

## Clang-tidy mode

```bash
cmake -S . -B build-tidy -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSIMANEAT_CPP_TIDY=ON
./scripts/run_cpp_tidy.sh
```

## CMake contract for generated examples

- Require C++20 (`CMAKE_CXX_STANDARD 20`).
- Use `find_package(SimaNeat REQUIRED)`.
- Link `SimaNeat::sima_neat`.
- In cross builds, ensure the SDK environment is initialized so `SYSROOT` and toolchain paths are available.
