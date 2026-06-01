# API Surface Guide

Default include for generated application code:

- `neat.h`

Primary namespaces:

- `simaai::neat`
- `simaai::neat::nodes`

Core public entry points:

- `pipeline/Graph.h`
  - `Graph`: compose nodes/groups and run/validate/build.
- `pipeline/PipelineRun.h`
  - `PipelineRun`: push/pull runtime handle.
- `pipeline/PipelineOptions.h`
  - `Sample`, `PullStatus`, options structs.
- `pipeline/Tensor.h`, `pipeline/TensorCore.h`
  - `Tensor` type and storage/device helpers.
- `model/Model.h`
  - Model pipeline composition and runner utilities.

Frequently used node headers:

- Input: `nodes/io/InputAppSrc.h`, `nodes/io/AppSrcImage.h`, `nodes/io/RTSPInput.h`
- Output: `nodes/common/AppSink.h`, `nodes/io/UdpSink.h`
- Utility: `nodes/common/Caps.h`, `nodes/common/Queue.h`, `nodes/common/VideoConvert.h`, `nodes/common/VideoScale.h`

Graph APIs (advanced):

- `graph/*` and `graph/nodes/*` for stage-style orchestration (`StageNode`, `Map`, `FanOut`, `JoinBundle`, etc.).

Compatibility rule:

- Treat installed headers as stable interface.
- Do not include private or `internal` headers in generated user code.
