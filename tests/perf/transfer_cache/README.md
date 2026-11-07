# DMA transfer-cache benchmark

Compare Core shared libraries on one idle Modalix DevKit with identical
Scheduler/Runner providers. The harness uses public Tensor APIs and verifies
payloads outside timing. Results qualify the measured workload and runtime
cohort; they do not qualify a complete package installation or arbitrary cache
working sets. Store release measurements and historical reports in PR evidence.

## Build

In the AArch64 SDK, build the harness against matching installed public headers:

```sh
cmake -S tests/perf/transfer_cache -B build-transfer-perf \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSROOT="$SYSROOT" \
  -DCMAKE_PREFIX_PATH="$SYSROOT/usr"
cmake --build build-transfer-perf -j4
```

Keep each Core variant in a separate directory. Select the same runtime/plugin
providers for every variant and record commits, library hashes and device identity.

## Run

```sh
dk shell 'python3 /workspace/tests/perf/transfer_cache/run_pairs.py \
  --variant develop=/workspace/develop/lib \
  --variant candidate=/workspace/candidate/lib \
  --runtime-library-path /usr/lib/aarch64-linux-gnu/neat/runtime \
  --transfer-bench /workspace/build-transfer-perf/transfer_bench \
  --workload cpp_46MiB --workload cpp_centerpoint_bytes \
  --workload over_budget_six_sizes --cpu 2 \
  --rounds 8 --iterations 100 --pressure-iterations 60 \
  --output /workspace/results/transfer-cache.json'
python3 tests/perf/transfer_cache/summarize.py /workspace/results/transfer-cache.json \
  --baseline develop --candidate candidate
```

These cases cover 46 MiB, 46,080,000 bytes, and six cyclic sizes
`46,080,000 + i * 4,096` bytes (`i=0..5`). Each fresh process warms 12 calls.
Timed regions include transfer creation and destruction. The runner alternates
A/B and B/A order, records loaded libraries and available sensors, and preserves
all samples. The summary bootstraps paired round medians rather than frames.

The runner also supports held-tensor and NumPy transfers. Optional model cases
use `--model /absolute/model_mpk.tar.gz`, `--model-only` and
`--python-executable /absolute/venv/bin/python`. Model construction is excluded;
NumPy/device outputs must match. Use MPK manifests for contracts. A substitute
model does not reproduce the architecture of an unavailable customer model.
