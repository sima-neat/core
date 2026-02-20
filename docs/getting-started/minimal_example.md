---
title: Hello SiMa!
description: Validate your install with a minimal SiMa NEAT CMake app
sidebar_position: 3
---

# Hello SiMa!

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

find_package(SimaNeat REQUIRED)

add_executable(sima_neat_hello main.cpp)
target_link_libraries(sima_neat_hello PRIVATE SimaNeat::sima_neat)
```

main.cpp:

```cpp
#include <iostream>
#include <pipeline/TensorCore.h>

int main() {
  std::cout << "Hello from sima-neat\n";
  std::cout << "DeviceType::CPU = "
            << static_cast<int>(simaai::neat::DeviceType::CPU) << "\n";
  return 0;
}
```

Build and run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/sima_neat_hello
```
  </div>

  <div class="minimal-tab-panel">
    <p>hello_neat.py:</p>

```python
# Pseudo Python minimal example.
from simaneat import DeviceType

def main():
    print("Hello from sima-neat")
    print("DeviceType.CPU =", DeviceType.CPU)

if __name__ == "__main__":
    main()
```

Run:

```bash
python3 hello_neat.py
```
  </div>
</div>
