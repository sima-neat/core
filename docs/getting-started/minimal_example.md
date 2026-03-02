---
title: Hello SiMa!
description: Validate your install with a minimal SiMa NEAT CMake app
sidebar_position: 3
---

# Hello SiMa!

## Minimal Example

Use this page to quickly verify that SiMa NEAT is installed correctly on either:

- a Modalix DevKit
- an eLxr SDK cross-compilation environment

The goal is intentionally simple: build and run the smallest possible application that links against NEAT successfully.

Create a new folder with these files:

<div class="minimal-tabs">
  <input type="radio" name="minimal-tabs-lang" id="minimal-tab-cpp" checked>
  <label for="minimal-tab-cpp">C++</label>
  <input type="radio" name="minimal-tabs-lang" id="minimal-tab-py">
  <label for="minimal-tab-py">Python</label>

  <div class="minimal-tab-panel">
    <p>CMakeLists.txt:</p>

```cmake
cmake_minimum_required(VERSION 3.16)
project(sima_neat_hello LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Supports both:
# - DevKit/native installs (system paths)
# - Cross builds with SYSROOT exported (SDK sysroot paths)
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH
    "$ENV{SYSROOT}/usr"
    "$ENV{SYSROOT}/usr/lib"
    "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
endif()

find_package(SimaNeat REQUIRED CONFIG)

add_executable(sima_neat_hello main.cpp)
target_link_libraries(sima_neat_hello PRIVATE SimaNeat::sima_neat)
```

main.cpp:

```cpp
#include <iostream>
#include <pipeline/TensorCore.h>

int main() {
  auto storage = simaai::neat::make_cpu_owned_storage(64);
  if (!storage) {
    std::cerr << "Failed to allocate CPU tensor storage\n";
    return 1;
  }
  std::cout << "Hello from sima-neat\n";
  return 0;
}
```

Set up the cross-compilation environment (eLxr SDK only):

```bash
source /opt/bin/simaai-init-build-env modalix
```

Build the example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run:

```bash
# DevKit/native run
./build/sima_neat_hello
```

:::note Build Target
- DevKit/native: build and run locally.
- eLxr SDK cross-compilation: `./build/sima_neat_hello` is a target binary. Copy it to a Modalix device and run it there.
:::
  </div>

  <div class="minimal-tab-panel">
    <p>hello_neat.py:</p>

```python
from pyneat import DeviceType

def main():
    print("Hello from sima-neat")
    print("DeviceType.CPU =", DeviceType.CPU)

if __name__ == "__main__":
    main()
```

Run:

```bash
# The installer typically creates the pyneat virtual environment in one of:
# - /media/nvme/pyneat
# - $HOME/pyneat
source /media/nvme/pyneat/bin/activate
python3 hello_neat.py
```
  </div>
</div>


## Next Steps
Once this minimal example works, the next step is to explore the broader SiMa NEAT learning material.

- Learn the [core programming model](./programming-model/overview), which explains the main NEAT concepts such as sessions, models, pipeline stages, and graph execution.
- Follow the [tutorials](../tutorials/index), which walk through specific concepts and workflows step by step.
- Curated example applications on the [apps portal](https://apps.sima-neat.com/portal), with source code available in the [apps repository on GitHub](https://github.com/sima-neat/apps)
