# Tensor transfer DMA cache regression

## Prerequisites

Build in the AArch64 SDK and run on a compatible Modalix DevKit through `dk`.
The strict cases require real CMA/DMS DMA heaps and Linux DMA-BUF fdinfo
accounting. Missing providers or telemetry fail the tests. No model or camera
is required. The device E2E lane selects their `e2e` label; each case has an
`accelerator-device` lock and a 60-second deadline.

## Run

```sh
cmake --build build --target tensor_transfer_memory_lifecycle_cma_test tensor_transfer_memory_lifecycle_dms_test tensor_transfer_memory_lifecycle_mixed_test neat_tensor_device_placement_test -j8
dk shell 'ctest --test-dir /workspace/build/tests -R "tensor_transfer_memory_lifecycle|^neat_tensor_device_placement_test$" --output-on-failure'
```

Adapt `/workspace/build` to the shared build directory. Use matching installed
Scheduler/Runner providers and the canonical Neat runtime/plugin paths.

## Assertions

The single-heap cases exercise `Tensor::cvu()` (CMA) and `Tensor::mla(true)` (DMS):

- Fixed-size sequential transfers stabilize after warm-up.
- Thirty-two distinct capacity classes total 530 MiB cumulatively; final idle
  DMA memory fits the shared 256 MiB budget and held payloads remain intact.
- Sixty transfers cycling six nearby sizes make zero heap calls after warm-up,
  preserve changing payloads and expose exact storage/mapping lengths.
- One injected `ENOMEM` drains idle memory before retry; two failures propagate.
  An outstanding tensor remains readable throughout.
- A 256-tensor small-buffer burst retains at most 128 allocations after release.
- Four synchronized workers hold 16 distinct classes totaling 264.5 MiB, forcing
  eviction of borrowed pools. They churn another 128 classes concurrently while
  all protected tensors remain alive, then verify payloads and final bounds.
- A 72-tensor, 288 MiB live burst remains valid; after release, idle memory fits
  both the byte and allocation-count bounds.

The mixed-heap case checks the combined CMA/DMS budget. Device placement also
checks allocation identity during shrink/regrow, exact carrier spans, metadata
reset, zeroed payload tails and concurrent failed-pool-creation recovery.
Accounting deduplicates this process's real DMA-BUF backing objects from
`/proc/self/fdinfo`; it does not infer retention from RSS or global free memory.

Run timings separately with the [transfer benchmark](../perf/transfer_cache/README.md).
Historical results and release-validation status belong in the PR evidence.
