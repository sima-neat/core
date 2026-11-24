import argparse, hashlib, json, os, resource, time
import numpy as np
import pyneat

p = argparse.ArgumentParser()
p.add_argument(
    "--mode", required=True, choices=["numpy_transfer", "model_numpy", "model_device"]
)
p.add_argument("--bytes", type=int, default=48234496)
p.add_argument("--model")
p.add_argument("--iterations", type=int, default=100)
p.add_argument("--warmup", type=int, default=10)
a = p.parse_args()
model = None
if a.mode.startswith("model_"):
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    model = pyneat.Model(a.model, options)
    specs = model.input_specs()
    if len(specs) != 1:
        raise RuntimeError("benchmark expects the reported single-input model")
    spec = specs[0]
    if spec.dtypes[0] != pyneat.TensorDType.Float32:
        raise RuntimeError("benchmark expects FP32 input")
    shape = list(spec.shape)
else:
    shape = [a.bytes // 4]
x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
t = pyneat.Tensor.from_numpy(x)
if a.mode == "numpy_transfer":
    fn = lambda: pyneat.Tensor.from_numpy(x)
elif a.mode == "model_numpy":
    fn = lambda: model.run([x])
else:
    fn = lambda: model.run([t])
for _ in range(a.warmup):
    fn()
before = resource.getrusage(resource.RUSAGE_SELF)
samples = []
for _ in range(a.iterations):
    start = time.perf_counter_ns()
    result = fn()
    del result
    samples.append((time.perf_counter_ns() - start) / 1e6)
after = resource.getrusage(resource.RUSAGE_SELF)


def outputs(r):
    items = r if isinstance(r, (tuple, list)) else [r]
    return [q.to_numpy(copy=True) for q in items]


if model:
    host = outputs(model.run([x]))
    device = outputs(model.run([t]))
    same = len(host) == len(device) and all(
        np.array_equal(h, d) for h, d in zip(host, device)
    )
    if not same:
        raise RuntimeError("NumPy/device outputs differ")
    hashes = [hashlib.sha256(q.tobytes()).hexdigest() for q in host]
else:
    y = pyneat.Tensor.from_numpy(x).to_numpy(copy=True)
    if not np.array_equal(x, y):
        raise RuntimeError("transfer payload mismatch")
    same = True
    hashes = []
paths = sorted(
    set(
        line.split()[-1]
        for line in open("/proc/self/maps")
        if any(
            n in line
            for n in [
                "libsima_neat.so",
                "libneatdmabuf.so",
                "libneatdmabufpool.so",
                "libsimaai_runtime_gstreamer.so",
                "libneatdmabufmemory.so",
            ]
        )
    )
)
print(
    "BENCH_JSON="
    + json.dumps(
        dict(
            mode=a.mode,
            bytes=x.nbytes,
            shape=shape,
            samples_ms=samples,
            minor_faults=after.ru_minflt - before.ru_minflt,
            outputs_identical=same,
            output_sha256=hashes,
            loaded_libraries=paths,
        )
    )
)
