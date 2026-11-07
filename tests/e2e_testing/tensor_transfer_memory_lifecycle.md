# Tensor transfer DMA cache regression

The three strict CTest cases exercise public `Tensor::cvu()` (CMA),
`Tensor::mla(true)` (DMS), and alternating CMA/DMS transfers in separate
processes. They require real DMA heaps and Linux DMA-BUF fdinfo accounting;
missing providers or telemetry fail the tests. No model pack or camera is needed.
The normal device E2E lane selects their `e2e` label. Each case has an
`accelerator-device` lock and a 60-second deadline.

## Policy under test

Core retains at most 256 MiB of rounded transfer capacity and 128 one-buffer
pools across both heaps. Loaned slots count toward these limits while cached.
Retiring a pool releases its idle buffer immediately; a loan stays valid and is
freed on final return. A transfer never waits for an earlier tensor. Requests
larger than the byte budget bypass the cache. Allocation failure with `ENOMEM`
drains the cache and retries at most once, after checking for a concurrently
returned compatible slot. Requests up to 64 KiB round to a system page; larger
cached requests round to 64 KiB (or a system page if larger). Tensors expose only
the requested carrier length. Nearby lengths can therefore reuse one backing
allocation without spending the budget on almost-identical buffers.

The initial investigation used a provisional 32 MiB growth allowance. The final
policy uses 256 MiB to preserve the existing 46 MB input reuse and alternating
40/41 MiB transfer regressions. The size-churn workload is expanded past that
budget; raising the threshold alone would not establish bounded retention.

## Coverage

Each single-heap case:

1. Holds four distinct 1 MiB payloads, warms a fifth allocation, and checks that
   32 same-size transfers stabilize within one page of the warm baseline.
2. Transfers 32 distinct capacity classes, `i * MiB + 4 KiB` for `i=1..32`,
   with at most one transient tensor live. Rounded capacity totals 530 MiB.
3. Checks the four held payloads, releases all tensors, and requires final DMA
   bytes to fit the 256 MiB budget.
4. Warms the 46,080,000-byte capacity class, then cycles 60 times through
   `46,080,000 + i * 4,096` bytes for `i=0..5`. Requires **zero DMA heap calls**
   after warm-up, exact storage/mapping lengths, and correct changing payloads.
5. Injects one real `DMA_HEAP_IOCTL_ALLOC` failure and measures DMA bytes at the
   retry: only the retained 3 MiB tensor may remain, within one MiB of rounding.
   Two consecutive injected failures must propagate after the bounded retry.
   The retained tensor must remain readable in both cases.
6. Holds and verifies 256 simultaneous 4 KiB tensors, then releases them and
   requires at most 128 idle DMA allocations.
7. Runs four concurrent workers through distinct-size transfers while each keeps
   a tensor alive, checking payload integrity and both final cache bounds.
8. Holds 72 simultaneous 4 MiB tensors (288 MiB), checks their payloads, then
   releases them and requires the idle cache to fit both limits.

The mixed-heap case alternates the 32 capacity classes between CMA and DMS, with one
protected tensor per heap, and requires their combined idle bytes to fit the
same 256 MiB budget.

Accounting reads this process's `/proc/self/fdinfo`, selects entries with
`exp_name`, sums `size`, and deduplicates by backing device/inode. A live-buffer
check verifies the accounting sees the real allocations. It does not use RSS or
global free memory. Fault injection replaces allocation errors, never buffers.

## Running

Build in an ARM64 SDK:

```sh
cmake --build build --target tensor_transfer_memory_lifecycle_cma_test tensor_transfer_memory_lifecycle_dms_test tensor_transfer_memory_lifecycle_mixed_test neat_tensor_device_placement_test -j8
```

Run with the SDK wrapper, adapting `/workspace` to the shared source directory:

```sh
dk /workspace/build/tests/tensor_transfer_memory_lifecycle_cma_test cma
dk /workspace/build/tests/tensor_transfer_memory_lifecycle_dms_test dms
dk /workspace/build/tests/tensor_transfer_memory_lifecycle_mixed_test mixed
```

Or invoke registered CTest cases on the device through `dk shell`:

```sh
dk shell 'ctest --test-dir /workspace/build/tests -R "tensor_transfer_memory_lifecycle|^neat_tensor_device_placement_test$" --output-on-failure'
```

## DevKit evidence

Device: `modalix-065848`, AArch64, kernel `6.18.3-modalix #2499`.
The original reproduction and fix validation use the same private-prefix runtime:
DMA adapter/pool libraries from Scheduler
`aa721e9347824291b00fc7769250ad3bd4cc2092`, installed Scheduler frontend
`0.6.0+local.fafd48538df1`, and Runner memory authority
`0.6.0+local.8d4040baa704`. Loaded paths were checked with `ldd`.
No global device packages were changed. This is private-prefix integration,
not qualification of a complete Debian package installation.

Core's reviewed source `896551a25d4ca1772c804980864b127f17fb5ec1` has the same tree
as the PR base `80073726`; the baseline uses its original transfer implementation.
The original fix source and test files were copied byte-for-byte from the PR worktree
into the shared SDK build tree and built with GCC14. GenAI was disabled through
the existing `SIMANEAT_REQUIRE_LLIMA_ARTIFACTS=OFF` option.

The original exact-size test (before capacity classes) used 96 nearby sizes,
`4 MiB + i * 4 KiB` for `i=1..96`. Its historical results were:

| Workload | Original cache | Initial bounded cache |
| --- | ---: | ---: |
| CMA, after 96 sizes and all tensors released | 426,967,040 bytes (fails) | 268,001,280 bytes (passes) |
| DMS, after 96 sizes and all tensors released | 426,967,040 bytes (fails) | 268,001,280 bytes (passes) |
| Mixed CMA/DMS after all tensors released | 423,821,312 bytes (fails) | 268,001,280 bytes (passes) |
| Single-heap 72-tensor burst after release | Not reached after retention failure | 268,435,456 bytes / 64 idle allocations (passes) |

Both single-heap tests passed their held-tensor, small-allocation count,
concurrent eviction, allocation-pressure drain/retry, persistent-error and burst
checks. The existing `neat_tensor_device_placement_test` also passed, including
46 MB reuse, alternating 40/41 MiB reuse, metadata/padding reset and concurrent
failed pool creation recovery. The three new cases failed on the original cache
and passed with the fix.

Temporary sources, builds and verbose logs are in the SDK under
`/workspace/tmp/dma-retention-pr1321`; the key logs are
`budget-before-ctest.log` and `budget-fixed-ctest.log`. For these four CTest cases,
only generated test environment entries were rebased from SDK sysroot paths to
canonical `/usr` runtime paths, with the isolated DMA library prefix prepended.
The committed CTest environment was not modified.

The workload deliberately avoids exhausting the device. It does not qualify
sustained inference throughput, full package installation, or performance for
working sets larger than the cache budget.

## Capacity-class revision

The revised churn test uses distinct classes so quantization cannot hide cache
growth. The initial capacity-class candidate passed all four focused tests on
the same DevKit:

| Check | CMA | DMS / mixed |
| --- | ---: | ---: |
| After 32 classes, no live tensors | 264,830,976 bytes | 264,830,976 bytes |
| Nearby six-size cycle after warm-up | 0 heap calls | 0 heap calls (DMS) |
| 256 small held tensors, after release | 128 allocations / 524,288 bytes | Same (DMS) |
| 72 large held tensors, after release | 64 allocations / 268,435,456 bytes | Same (DMS) |

Both heaps passed pressure recovery, persistent-error propagation, concurrent
eviction and held-payload checks. A timing probe still showed extra access-plan
cost when changing logical lengths. The final revision uses one full-capacity
write plan and trims the buffer after finishing access and releasing its mapping
owners. The placement test additionally checks backing identity, payload and
exact carrier lengths while growing and shrinking within page and 64 KiB
classes, with another tensor still held.

The final revision builds with GCC14. Its device rerun is **pending**: the SDK
container was recreated during validation, removing its authorized SSH key.
The partial timing probe is retained in the local lab and is excluded from the
completed historical benchmark results. No final performance conclusion is
claimed until same-device access is restored.

[The paired performance report](../perf/transfer_cache/README.md) retains the
initial exact-size cache's 3.85× slowdown. The revised comparison is pending.
Capacity classes address that nearby-size regression without raising either
cache bound. Arbitrary classes or concurrency beyond a finite budget can still
require repeated allocations; this is not a promise of universal cache hits.
