---
title: Hello Neat!
description: Validate your install with a minimal SiMa Neat CMake app
sidebar_position: 3
---

# Hello Neat!

## Minimal Example

This guide uses a minimal example to verify that Neat is installed and runnable, while introducing the core application development and validation workflow.

- a Modalix DevKit
- the [Neat eLxr SDK](./installation/neat-elxr-sdk.mdx)

:::note Neat eLxr SDK Prerequisite
To run commands on the DevKit directly from inside the SDK (for example, `dk build/sima_neat_hello` or `dk hello_neat.py`), set up DevKit pairing first:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

If SDK/DevKit pairing is not configured, you can still build inside the Neat eLxr SDK, but you must manually transfer the built binary or script to the DevKit and run it there.
:::

:::tip About `dk` / `devkit-run`
`dk` (alias for `devkit-run`) is a shell function in the SDK container, defined in `~/devkit-sync.rc` and loaded by `~/.bashrc`.

Because it is a shell function, commands such as `which devkit-run` may return nothing in the SDK shell. Use `dk <file>` to execute a built binary or Python entry-point file on the paired DevKit.
:::

Create a working directory with the following files:

<div class="minimal-tabs">
  <input type="radio" name="minimal-tabs-lang" id="minimal-tab-cpp" checked>
  <label for="minimal-tab-cpp">C++</label>
  <input type="radio" name="minimal-tabs-lang" id="minimal-tab-py">
  <label for="minimal-tab-py">Python</label>

  <div class="minimal-tab-panel">
    <p><code>CMakeLists.txt</code>:</p>

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

`main.cpp`:

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

Build the example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run:

**DevKit**

```bash
./build/sima_neat_hello
```

**Neat eLxr SDK**

```bash
dk build/sima_neat_hello
```
  </div>

  <div class="minimal-tab-panel">
    <p><code>hello_neat.py</code>:</p>

```python
from pyneat import DeviceType

def main():
    print("Hello from sima-neat")
    print("DeviceType.CPU =", DeviceType.CPU)

if __name__ == "__main__":
    main()
```

Run:

**DevKit**

```bash
source ~/pyneat/bin/activate
python3 hello_neat.py
```

**Neat eLxr SDK**

```bash
dk hello_neat.py
```

:::note Python Runtime Location
`pyneat` is installed on the DevKit runtime side, even if you run the Neat installer from inside the Neat eLxr SDK container.

When you run `dk hello_neat.py`, `dk` executes the script on the paired DevKit using the DevKit `pyneat` environment.
:::
  </div>
</div>


## Next Steps
Once this minimal example works, continue with broader SiMa Neat learning resources:

- Learn the [core programming model](./programming-model/overview), which explains the main Neat concepts such as sessions, models, pipeline stages, and graph execution.
- Follow the [tutorials](../tutorials/index), which walk through specific concepts and workflows step by step.
- Explore curated applications on the [apps portal](https://apps.sima-neat.com/portal), with source code in the [apps repository on GitHub](https://github.com/sima-neat/apps).
