# Model Lifecycle

One `pcie::Model` represents one compiled model running on one physical PCIe queue. The constructor
and `build()` have intentionally different responsibilities.

## Lifecycle

1. Construct the model with its archive, model options, and connection options.
2. Call `info()` before allocating inputs. Construction parses the local archive but does not touch
   the card.
3. Call `build(readiness_timeout_ms)` once. It uploads the archive, starts the card-side pipeline,
   waits for readiness, and initializes the host transport.
4. Submit inference through either `run()` or balanced `push()`/`pull()` calls.
5. Call `close()` on every normal and failure path. Closing releases the model's queue and is
   idempotent.

`run()`, `push()`, and `pull()` require a successful `build()`. `running()` reports whether the
model is currently built and active.

## Synchronous Requests

Use `run()` unless overlapping input preparation with inference materially benefits the
application. Pass a finite timeout for bounded failure behavior.

```cpp
namespace pcie = simaai::neat::pcie;

pcie::ConnectionOptions connection;
connection.card_id = 0;
connection.queue = 0;

pcie::Model model("model.tar.gz", {}, connection);
const pcie::ModelInfo info = model.info();
const auto& input_spec = info.inputs.at(0);
if (input_spec.dtype != "FP32" && input_spec.dtype != "FLOAT32") {
  throw std::runtime_error("example expects a float32 input");
}
std::vector<float> values(input_spec.size_bytes / sizeof(float), 0.0F);
pcie::Tensor input = pcie::Tensor::from_vector(
    std::move(values), input_spec.shape, input_spec.name);

model.build(/*readiness_timeout_ms=*/180000);
try {
  pcie::TensorList outputs = model.run(input, /*timeout_ms=*/30000);
  consume(outputs);
} catch (...) {
  model.close();
  throw;
}
model.close();
```

In Python, prefer the context manager. It closes the model when the block exits, including on an
exception:

```python
import numpy as np
import pyneatpcie as pcie

connection = pcie.ConnectionOptions(card_id=0, queue=0)
with pcie.Model("model.tar.gz", connection=connection) as model:
    info = model.info()
    input_spec = info.inputs[0]
    if input_spec.dtype not in {"FP32", "FLOAT32"}:
        raise RuntimeError("example expects a float32 input")
    input_tensor = pcie.Tensor.from_numpy(
        np.zeros(input_spec.shape, dtype=np.float32),
        copy=True,
        route_name=input_spec.name,
    )
    model.build(readiness_timeout_ms=180000)
    outputs = model.run([input_tensor], timeout_ms=30000)
```

## Pipelined Requests

Use `push()` and `pull()` together when the producer should overlap with inference.
`ConnectionOptions.max_inflight` bounds accepted work that has not returned. `push()` applies
backpressure when that window is full, so continue pulling rather than submitting indefinitely.

Maintain an application-side count or FIFO so every accepted push has exactly one pull. Results
for one `Model` arrive in submission order. Drain all pushed work before calling `run()` on the same
model.

A timeout from `run()` or an empty result from timed `pull()` stops waiting; it does not cancel an
input already accepted by the card. After a timeout, either drain the outstanding result with
`pull()` or close the model before beginning a new request sequence.

## Multiple Models

Multiple independent `Model` objects can execute concurrently. Assign each active model a distinct
queue from 0 through 3. Build errors should identify the model and queue, and a partially built set
must close every model that was successfully built.

Do not introduce `pcie::Runtime` to coordinate these models. Separate `Model` instances and
application-owned threads are the supported pattern within this skill.
