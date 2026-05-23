# Graph Pattern

Typical structure:

1. Construct `simaai::neat::Graph`.
2. Add input node/group.
3. Add model pipeline or processing stages.
4. Add output node/group.
5. Build and run with `PipelineRun`.

Sketch:

```cpp
#include <neat.h>

int main() {
  simaai::neat::Graph s;
  s.add(simaai::neat::nodes::InputAppSrc());
  s.add(simaai::neat::nodes::OutputAppSink());

  // Build when you have a representative input tensor/sample.
  // simaai::neat::Tensor input = ...;
  // auto run = s.build(input, simaai::neat::PipelineRunMode::Sync);
  // simaai::neat::Sample out = run.push_and_pull(input);
  return 0;
}
```

Notes:

- `Graph::run(...)` is convenient for single-shot sync use.
- `Graph::build(...)` + `PipelineRun` is preferred for repeated push/pull.
- For source-like pipelines (no push input), use `build(const PipelineRunOptions&)`.
