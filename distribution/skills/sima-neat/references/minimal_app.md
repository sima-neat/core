# Minimal SiMa NEAT App

```cpp
#include <pipeline/Session.h>
#include <nodes/io/InputAppSrc.h>
#include <nodes/common/AppSink.h>

int main() {
  simaai::neat::Session session;
  session.add(simaai::neat::nodes::InputAppSrc());
  session.add(simaai::neat::nodes::OutputAppSink());
  return 0;
}
```

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```
