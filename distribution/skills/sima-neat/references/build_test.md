# Build and Test Commands

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Unit + E2E tests

```bash
ctest --test-dir build/tests --output-on-failure
```

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
