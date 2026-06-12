# Minimal SiMa Neat App

```cpp
#include <neat.h>

int main() {
  simaai::neat::Graph graph;
  graph.add(simaai::neat::nodes::InputAppSrc());
  graph.add(simaai::neat::nodes::OutputAppSink());
  return 0;
}
```

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```
