# 015 Run Multiple Streams in One Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | None |
| Labels | graph, multistream, scheduler, join |

## Concept

Run multiple logical streams through one public `Graph` and combine two named inputs into one deterministic bundle output — the pattern behind multi-camera or multi-source systems where related inputs must be aligned before downstream processing.

## Walkthrough

Earlier chapters pushed one input and pulled one output. Real multi-camera and parallel-branch systems are messier: several streams advance independently, and their results must be *rejoined* correctly before anything downstream can use them. This chapter shows the join primitive that makes that deterministic — a combine graph with two named inputs and one named output that emits a bundle only when both sides have produced the matching frame.

Every sample you push carries a `stream_id` and a `frame_id`. The combine policy `ByFrame` waits until both named inputs (`left` and `right`) have delivered a sample with the same `frame_id`, then emits exactly one combined bundle. By the end you will have built a combine graph, fanned a deterministic per-stream / per-frame workload through its two inputs, and pulled back the joined bundles — verifying both the output count and that each bundle carries two fields.

### Build the combine graph {#step-build-combine-graph}

`graphs::Combine` (C++) / `graphs.combine` (Python) returns a normal public `Graph` fragment — there is nothing special about it beyond its shape: two named inputs, one named output, and a join policy. We pass `["left", "right"]` as the input names, `"combined"` as the output name, and `CombinePolicy.ByFrame` to select frame-id matching. Printing `describe()` shows the resulting topology, and `build()` turns the description into a runnable handle. The graph runs async by default so each stream can make progress on its own.

The output queue is bounded. Instead of allocating enough queue space for the full workload, this example pulls each joined bundle before pushing the next pair. Producer and consumer advance together, so memory use stays bounded as the frame count grows.

`CombinePolicy.ByFrame` matches on `Sample.frame_id`; `CombinePolicy.ByPts` is the alternative that matches on presentation timestamps (`Sample.pts_ns`) when frames don't share a clean frame index.

### Push the streams {#step-push-streams}

Now we drive the workload. For every frame and every stream we synthesize a small deterministic RGB sample tagged with its `stream_id` and a unique `frame_id`, then push it into *both* named inputs. Because the IDs are computed deterministically (`frame * streams + sid`), the join has an unambiguous pairing to find — `left` frame N always has a matching `right` frame N. After the matching `right` push, we drain that pair's joined output before moving to the next pair.

**C++:** Each sample is constructed explicitly as a `Sample` wrapping a `Tensor` (HWC, UInt8, RGB) with `frame_id` and `stream_id` set; `run.push("left", sample)` returns a bool you should check against `run.last_error()`.

**Python:** `make_rgb_sample(...)` builds the `Sample` from a NumPy array via `Tensor.from_numpy(...)`; `run.push("left", [sample])` takes a list of samples.

### Pull each joined bundle {#step-pull-bundles}

Immediately after each matching pair is pushed, we pull once from the named output `"combined"`. Each successful pull returns the bundle that the runtime emitted after both inputs delivered that frame. Draining as we produce prevents the bounded output queue from filling and propagating backpressure to the input side. Both examples verify that every bundle contains the two joined fields, then call `close()` to tear the run down cleanly. The expected bundle count equals `streams * frames`, proving no pairing was dropped.

**C++:** `run.pull("combined", timeout_ms)` returns an optional bundle; we read `bundle.stream_id` and `bundle.fields.size()` and verify that each bundle has two fields.

**Python:** `run.pull("combined", 2000)` returns the bundle or `None`; the example fails immediately on a timeout and verifies each bundle's field count.

## Run

This chapter needs no model archive. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

**Python:**
```bash
python3 share/sima-neat/tutorials/015_run_multiple_streams/run_multiple_streams.py \
  --streams 8 --frames 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_015_run_multiple_streams \
  --streams 8 --frames 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_015_run_multiple_streams
./build/tutorials-standalone/tutorial_015_run_multiple_streams \
  --streams 8 --frames 4
```

Expected output (the C++ build also prints the graph description; both builds print the first few bundles):

```text
received=32 fields=2
[OK] 015_run_multiple_streams
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/015_run_multiple_streams/run_multiple_streams.cpp`
- Python: `tutorials/015_run_multiple_streams/run_multiple_streams.py`
