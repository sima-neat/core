# Qualification harness of the Dataflow backend

The acceptance test of the route-coverage round. Per catalogue package and option view, a program on the
Dataflow backend has exactly one of two outcomes:

- **equal**: every output tensor byte-equal to GStreamer's through the customer's own calls, and every
  customer-visible field equal, except the named rows of `expected_differences.json`, which must hold exactly
  as written;
- **not_carried**: the first build fails with the one error form
  `[<code>] <where>: <subject> is not carried by the Dataflow backend: <fact>.` and
  `Hint: ... simaai::neat::set_backend(simaai::neat::RunBackend::GStreamer) ...`, with the code, subject and
  fact that `expected.json` records, and the same program runs on GStreamer (the hint's remedy).

Anything else fails. It is a bug for the change that caused it, never a row to add.

A frame a backend does not take ends a scenario: both backends must then end it at the same frame,
each with the error an active `stop` row of `expected_differences.json` names (an oriented-box decode
and a frame size whose letterbox has two scales: GStreamer's Run is dead after it, Dataflow refuses
the push and goes on). The size scenario runs last on its Run and pushes the sizes every decode
takes first.

## Files

| file | what |
|---|---|
| `catalogue.json` | the 239 compiled packages (280 components) of the Models registry and, per component, the ModelOptions its `performance.py` sets: the `app` view (1280x720 RGB frames); the `default` view is `Model(package)` with the model's tensors |
| `expected.json` | one row per package and view: the outcome; per component the planner's verdict (a refusal: code, subject, fact, the obstacle units and the PLAN.md phase that lifts them; `form_refusals` where an entry form refuses at another point and so states another subject and fact); `forms`, the state of each entry form; `tiers` |
| `expected_differences.json` | the differences an `equal` row may show: what each backend gives, why, who removes it, and the relation the comparator checks |
| `census.cpp` | `neat_qualification_census`: the planner's verdict per component and view (CPU only, internal headers) |
| `compare.cpp` | `neat_qualification_compare`: one backend's records of one component (public API only) |
| `qualify.py` | runs both tools, compares two processes' records, judges against the two JSON files; `selftest` checks its own rules |
| `make_census_corpus.py` | the census corpus: the packages with the MLA program's section contents zeroed past each section's header |
| `launch_script.py` | runs a catalogue script unchanged after `pyneat.set_backend(...)` |
| `scripts.json` | tier T3: per model, the registry's accuracy dataset, which script takes `--resources`, what `performance.py` asks of the benchmark and how `accuracy.py` builds its Models; the report fields that are measurements; the `--limit` |
| `make_scripts_catalogue.py` | writes the `datasets` and `models` of `scripts.json` from a checkout of the Models registry (reads files, runs nothing) |

## Forms and records

The comparator drives the customer's forms on each backend in a process of its own (`set_backend` first):
`graph` (a Graph of the Model's `preprocess()`, `inference()`, `postprocess()` with the Model's placement
options: lockstep, eight frame sizes, `Run::pull(std::chrono::milliseconds)`, free-running), `model`
(`Model::build(seed)` + Runner, lockstep and sizes; free-running on `{queue_depth 4, Block, ZeroCopy}`),
`model_run`, `plan` (`execution_plan().to_json()`, by hash) and `bench` (the report's filled fields). A form a
later step makes work is `pending` in `expected.json`: it runs in processes of its own, GStreamer must run it,
and Dataflow must refuse it in the one form; when it passes on Dataflow the row fails as `FLIP` until the
step sets it `active`. Every output tensor's dense bytes are hashed with XXH64 (read in place when they are
one contiguous range; the first frames also hash a copy and compare); every `Sample` field, every tensor's
metadata (dtype, layout, shape, strides, offset, device, `read_only`, planes, route, semantic, storage kind,
allocation sharing, Stamp), the `MeasureReport` counters and the process's output (numbers and paths
normalised) are compared.

XXH64 against the byte-wise FNV-1a of the earlier tools, on a 1,228,800-byte output on the A65: 0.88-0.90 ms
in place (1.06-1.10 ms over a copy) against 6.4 ms; the comparator hashed 530 MB in 0.39-0.48 s.

## How a later step uses it

1. run the census (T0) and `qualify.py seed --census <census.jsonl> --obstacles <...>`: the diff of
   `expected.json` is the coverage delta (`not_carried` -> `equal`, or a refusal whose fact changed);
2. a step that makes an entry form work sets it `active` in `forms` (or in the rows it covers);
3. a step that removes a documented difference deletes its row: an interim row is `must_observe`, and fails
   as stale once its fields compare equal; a step that creates a planned difference (the letterbox rule)
   sets it `active`;
4. T1 for the rows it flips, T0 for all, in the same change.

## Tiers (sizes measured on a Modalix DevKit, 2026-11-24)

| tier | what | where | measured | when |
|---|---|---|---|---|
| T0 | `qualify.py census` over the 37 MB census corpus: 280 components, both views, against `expected.json`; `qualify.py selftest` | A65 CPU, no device opened (the devkit, or an arm64 runner); the selftest anywhere | 525 s with 4 processes on 4 cores (464 s of census, the rest waiting for dirty pages); 1,621 CPU-s, median 1.8 s per component and view | every PR |
| T1 | `qualify.py compare --tier T1 --floor`: 9 carried rows, one per route class, and 15 refused rows, one per obstacle class; every form; 200 frames | a devkit on the Kernels firmware | 1,521 s for the 24 rows (the forms Dataflow carries: GStreamer, Dataflow, GStreamer again; the pending ones: GStreamer, Dataflow) | PRs touching `src/pipeline/runtime/Dataflow*`, `src/model/` |
| T2 | `qualify.py compare --floor` over every row, the active forms, 1,000 frames | devkits, sharded by row | the `graph` form at 200 frames (432 frames a process): 81 rows in 3,941 s, median 47 s a row of which about 25 s is the wait for dirty pages; processes median 8.9 s GStreamer, 6.2 s Dataflow. All 90 carried rows: about 75 min on one board; at 1,000 frames about 2 h (estimated) | nightly |
| T3 | `launch_script.py` over every package's `performance.py` and `accuracy.py --limit N`, per backend | devkits with the datasets | not run (the scripts reach Dataflow once `Model::build` does) | weekly, and for the rows a step flips |

## Tier T3: the catalogue's own scripts

The customer programs the catalogue has are each model's `scripts/performance.py` and `scripts/accuracy.py`.
`qualify.py scripts` runs them byte for byte as the registry holds them, each run a process of its own, on
GStreamer and on Dataflow. The backend is chosen outside the script: `launch_script.py` calls
`pyneat.set_backend(...)` and then runs the script with `runpy` as `__main__` with the script's own arguments.
The arguments are the ones the registry's `run_model_performance.sh` and `run_model_accuracy.sh` give:
`--mpk <archive>` (`<component>=<archive>` each for a multi-component model and for every `accuracy.py`),
`--dataset <source id>=<extracted directory>` (the model's first accuracy dataset, looking through a single
top-level directory), `--resources <directory>` where the script requires it, `--output <directory>`, and for
`accuracy.py` `--limit N` (`scripts.json`: 100). `PYTHONPATH` is the registry's `tools` (for `model_report`) and
what `--pythonpath` adds.

The expected outcome of a package's script is the row of the view the script builds: `performance.py` sets the
options `catalogue.json` records (the `app` row); `accuracy.py` feeds the model's own tensors (the `default`
row; NetVLAD and the CLIP and SigLIP 2 encoders build with Tensor-kind options, which the planner answers as it
answers the default view).

- **equal**: the script exits 0 on both backends; every report file (`model-report.json` and the Core report
  beside it) exists on both with the same fields; every field is equal, the accuracy metrics among them, except
  the measurements `scripts.json` names (`measured_fields`: latencies, throughput, durations, telemetry and
  power), which must be filled alike (zero on both or non-zero on both) and whose values are the speed table;
  the lines the two processes print are the same once numbers and paths are normalised, except the `stderr`
  rows of `expected_differences.json`.
- **not_carried**: the script exits 0 on GStreamer; on Dataflow it ends with the one error form, with the code,
  subject and fact `expected.json` records for a refused component of the row.

A step that makes a view run flips these with the row: nothing in `scripts.json` states an outcome.

Each run keeps, under `<out>/runs/<model>__<variant>.<script>.<backend>/`: `run.json` (command, exit status,
seconds, the kernel log's new lines), `launch.json` (the launcher's record: the exception's text when one
escaped), `stdout.txt`, `stderr.txt` and `report/` (the script's `--output`). `results.json` and `results.md`
hold the verdicts and the numbers each report states; `qualify.py speed --results results.json` writes the
latency and throughput table of both backends and names the components slower on Dataflow.
`--backends gstreamer` (or `dataflow`) with `--resume` runs one side and keeps it, so a board session can be cut
anywhere; `--judge-only` judges what is kept (it needs no device and no package).

```
qualify.py scripts --registry <models checkout> --packages <dir of <component id>.tar.gz> \
  --datasets <dir of <source id>/> --resources <dir of <model>__<variant>/> --out <dir> \
  --python <interpreter with the scripts' modules> --pythonpath <dir holding pyneat> \
  --run-prefix "taskset -c 4-7" --alternate [--names "yolov8s w8a8"] [--script performance|accuracy]
```

`performance.py` asks the benchmark for MLA layer statistics. On platform 3.0.1 each MLA model then leaves
kernel-log lines `mla 5000000.mla: non-FTL IRQ tlc=0x200 msc=0x8 with no in-flight job`, on GStreamer and on
Dataflow alike. `--accept-kernel-line <regex>` counts such lines per run (`run.json`) in the runs of a script
that requests those statistics; any other new line, and this line in any other run, stops the tier (exit 22).

### What the scripts need

| | |
|---|---|
| modules | `pyneat` (of the library under test), `numpy`, `cv2` (both in the 3.0.1 image: 2.2.4, 4.10.0) and, pinned by the models' `requirements-accuracy.txt`: `pillow==12.3.0` (30 models), `faster-coco-eval==1.8.0` (the 29 YOLOv8, YOLO11, YOLO26 and YOLOX detection, pose and segmentation models), `shapely==2.1.2` (the 15 oriented-box models), `tokenizers==0.23.3` (CLIP, SigLIP 2), `opencv-python-headless==4.11.0.86` (Depth Anything V2) |
| datasets | `imagenet-2012-val` (6,666,407,650 bytes, 24 models), `coco-2017-val` (898,368,520 bytes, 29 models), `pitts30k-val` (611,858,077 bytes, NetVLAD), `dota-v1-0-val` (327,598,055 bytes, the 15 oriented-box models), `scene-flow-val` (14,868,088,916 bytes, LightStereo S/M/L and RAFT-Stereo), `nyu-depth-v2-val` (156,136,510 bytes, Depth Anything V2), `ade20k-val` (87,700,104 bytes, SegFormer): the pins of each `model.yaml`, resolved with the registry's `dataset-registry resolve <id> --variant val` (SiMa Artifactory; anonymous reads are refused) |
| supporting files | CLIP and SigLIP 2 `accuracy.py` read the tokenizer and the token-embedding table of the package (`--resources`) |

On a devkit this needs no root. The environment the first pass used stays on the board under
`/media/nvme/dataflow-qual` (its `README.txt` says the same):

```
python3 -m venv --system-site-packages --without-pip /media/nvme/dataflow-qual/venv   # the image has no ensurepip
python3 -m pip --python /media/nvme/dataflow-qual/venv/bin/python install --no-index --no-deps --no-compile \
  /media/nvme/dataflow-qual/wheels/*.whl        # cp313 / abi3 manylinux aarch64 wheels of the four pins, from PyPI
# datasets/artifacts/<archive>.tgz as dataset-registry resolved it, extracted to datasets/<source id>/
```

`make_scripts_catalogue.py <registry>` rewrites the model entries when the registry's scripts change or a step
carries a model `scripts.json` does not list yet.

On a board whose EV74 runs the stock firmware every route with an EV74 kernel is (correctly) `not_carried`
with the firmware form, so T1-T3 need the Kernels firmware; the census records the stock-firmware answer.

## Determinism floor

`compare --floor` runs GStreamer twice; a field that differs between the two runs is that package's floor and
is not judged. On the 120 component views of the first pass (2026-11-24) the two GStreamer runs agreed on every
field and every output byte: the floor is empty.

## CI wiring (not done here)

T0: a job on an arm64 runner (or the devkit shard) that builds `neat_qualification_census`, takes the census
corpus from a build artifact (`make_census_corpus.py` over the pinned registry revision), and runs the
`qualification_census` ctest (`-DSIMANEAT_QUALIFICATION_CORPUS=<dir>`) with `qualification_selftest`. T1: the
devkit shard of `plan-devkit-tests` on the Kernels firmware, the T1 packages staged, `qualify.py compare --tier
T1 --floor`. T2/T3: nightly and weekly workflows sharded by row over the devkit fleet. The merge gate reads
`results.json`: every row's verdict equals its expected outcome.
