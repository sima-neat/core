# EV74 image-transform integration qualification

`image_transform_ev_test` exercises the public `Graph`, `Input`, `Resize`,
`ColorConvert`, `Normalization`, and `Output` APIs with actual EV74 execution.
It needs no external models or media. Missing CMA, EV74, or required plugins fail;
there is no skip, software conversion, or host numerical fallback.

## Catalogue

The default invocation runs 69 graphs and checks 209 completed frames:

| Cases | Coverage |
| ---: | --- |
| 24 | RGB, BGR, GRAY8, NV12, I420, and NV16 Resize with Nearest, Bilinear, Bicubic, and BoxAverage |
| 18 | RGB/BGR/GRAY8 to each YUV output, with BT.601 and BT.709 matrices |
| 6 | Neutral-chroma NV12/I420/NV16 to RGB/BGR |
| 9 | Every direct NV12/I420/NV16 input/output pair |
| 3 | Limited-range direct YUV with centered-to-cosited chroma, including odd-height NV16 |
| 3 | RGB/BGR/GRAY8 normalization with semantic RGB coefficients and exact BF16 truncation |
| 1 | Bounded dynamic Resize → BGR ColorConvert → Normalization, including returning geometries |
| 1 | RGB contain/letterbox Resize |
| 3 | NV12/I420/NV16 contain Resize with distinct raw Y/U/V padding |
| 1 | Encoder-compatible NV12 pitches, row tails, and storage-only rows |

Every ordinary case uses three different deterministic patterns. The dynamic
case uses five frames with three input geometries. A retained first output is
checked during reuse and after closing each run, including its composed affine
metadata. Raw outputs must remain one image, backed by one DMA-BUF and authoritative
`GstVideoMeta`, not a tensor bundle.

The independent scalar reference implements binary16 rounding, interpolation,
Q16 color conversion, and BF16 truncation without calling production numerical
helpers. Comparisons are byte-exact. Fixture writes and completed-output reads
are test preparation/verification only; they do not feed a fallback execution path.
This is a functional test, not a throughput benchmark or an ioctl/copy counter.
Inspection follows the actual materialized segment owners for named-input graphs,
retains those owners and their Gst bins, and verifies every numerical member is
EV74. It does not assume that the root Run owns a single Gst pipeline.

Every submission must match its pulled output in `frame_id`, `input_seq`,
`orig_input_seq`, `stream_id`, and PTS before any numerical assertion or next
submission. Named graphs assign a fresh segment-local `input_seq` starting at
zero. The fixture therefore authors that per-run ordinal for these sequential,
no-drop, one-output-per-input graphs, while its frame ID, original input sequence,
and PTS remain independently unique across runs. This does not assume that an
arbitrary authored `input_seq` survives graph segmentation. A mismatch logs all
five expected and actual fields before retaining the owners as unretired.
Push failure, pull timeout, closure, and runtime error are distinguished in
failure diagnostics. These diagnostics read stored errors and counters from both
the root Run and the retained materialized segment owners; they do not consume
bus messages, submit work, or change pipeline state before parking.

`image_transform_ev_test --encoder` separately requires `neatencoder` and its
matching Codecs backend. Its Input explicitly declares 30 fps, with rational
nanosecond PTS and durations matching that rate; the ColorConvert boundary must
preserve it rather than infer it from the encoder property. It runs one graph
and three frames, checks matching
DMA-BUF backing identity and view at the EV output and encoder input, and parses
the actual H.264 baseline SPS for the visible 130 × 66 dimensions. It does not
claim decoded-pixel accuracy. Probes never map payloads or retain buffer loans;
fixed atomic slots reject duplicate or missing handoffs.

## Build and admitted execution

From the Core checkout in the Palette SDK, with matching dependencies already
configured in an out-of-tree cross-build directory:

```bash
cmake --build "$CORE_BUILD" --target image_transform_ev_test -j8
file "$CORE_BUILD/tests/image_transform_ev_test"
```

The CTest names are `image_transform_ev_test` and
`image_transform_encoder_ev_test`. Both are strict hardware tests with the
`accelerator-device` resource lock. Use the configured admitted hardware
supervisor (`SIMANEAT_GUARDED_HARDWARE_TESTS=ON`) and the repository DevKit wrapper.
A CTest resource lock alone is not device ownership or firmware admission.
Never execute an AArch64 binary in an x86 SDK container. The two native commands
inside an admitted DevKit scope are:

```text
image_transform_ev_test
image_transform_ev_test --encoder
```

A successful run prints `CASE_CLOSED` for every graph followed by
`PASS ... all_runs_closed=1`. Numerical or metadata assertions after a completed
pull close the run and return failure naturally. The encoder case also retains
its Core owner and encoder element and requires `gst_element_get_state(..., 0)` to report success,
current state NULL, and no pending state after `run.close()`. This proves the
codec stop/destroy barrier for prequeued output buffers; a pulled frame alone
does not prove that barrier. A close exception also retains the owners. A
push/pull exception or timeout that does not establish completion instead prints `UNRETIRED image-transform`
and retains the run and all owners. Unverifiable EV setup-probe or encoder startup, or a failed or
incomplete NULL transition also takes this retained-owner path. Do not signal
that accelerator owner or infer
retirement from its disappearance: the admitted supervisor must preserve the
unknown result and perform its guarded recovery. CTest's timeout must supervise
the proxy, not kill an unguarded native owner.
