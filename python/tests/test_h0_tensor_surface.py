"""The H0 surface of pyneat, on both backends (Core's Run verbs take and give TensorLists).

A process runs on one backend (``pyneat.set_backend``): the runtime tests run on this process's
backend, and ``test_the_surface_on_the_other_backend`` runs this file again in a child process on
the other one (skipped when the build carries no Dataflow runtime, a build without
``SIMANEAT_DATAFLOW_SOURCE_DIR``). Covers: ``RunBackend``, ``set_backend``/``backend``;
``Stamp``/``TimePoint`` as frozen
values; records (``Tensor.record``, ``is_record``, ``fields``, ``field_names``, ``leaves``,
``t[name]``, ``t[index]``, ``with_stamp``); the ``to_tensor``/``to_sample`` projection;
``Graph.input``/``Graph.output``; ``Run.pull_result`` (``PullResult``: four outcomes, a list of
Tensors carrying the item's Stamp, a pushed list's first stamp, records as their fields);
``Run(tensors)``; ``Run.on_output`` (thread, order, exclusive output, a raising callback, teardown
while a callback needs the GIL); the deprecated ``Run.pull`` and
``Run.pull_tensors`` (once per call site; the C++ runtime's one-time stderr line never reaches
Python); ``Run.alloc``
with ``Tensor.copy_from`` and its commit; and the GIL released by every blocking call.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
import pytest

import pyneat as pn

_DATAFLOW_AVAILABLE: list[bool] = []


def _options(backend: str) -> pn.RunOptions:
  # The Run takes the process's backend; `backend` names it for the test's own checks.
  del backend
  return pn.RunOptions()


def _process_backend() -> str:
  return "dataflow" if pn.backend() == pn.RunBackend.Dataflow else "gstreamer"


def _dataflow_available() -> bool:
  # Asked of a child process: in this one set_backend would change the backend of the tests
  # after it, or be refused once a build fixed it.
  if not _DATAFLOW_AVAILABLE:
    child = subprocess.run(
        [sys.executable, "-c",
         "import pyneat as pn\n"
         "try:\n"
         "  pn.set_backend(pn.RunBackend.Dataflow)\n"
         "except pn.NeatError as exc:\n"
         "  raise SystemExit(3 if 'built without the Dataflow backend' in str(exc) else 1)\n"],
        capture_output=True, text=True, timeout=120)
    if child.returncode not in (0, 3):
      raise AssertionError(child.stdout + child.stderr)
    _DATAFLOW_AVAILABLE.append(child.returncode == 0)
  return _DATAFLOW_AVAILABLE[0]


@pytest.fixture
def backend() -> str:
  return _process_backend()


def _rgb_array(w: int, h: int, tag: int) -> np.ndarray:
  return ((tag + np.arange(h * w * 3, dtype=np.int64) * 7) % 256).astype(np.uint8).reshape(h, w, 3)


def _rgb(w: int, h: int, tag: int) -> pn.Tensor:
  return pn.Tensor.from_numpy(
      _rgb_array(w, h, tag), copy=True, image_format=pn.PixelFormat.RGB, memory=pn.TensorMemory.CPU
  )


def _floats(values, shape=None) -> pn.Tensor:
  arr = np.asarray(values, dtype=np.float32)
  if shape is not None:
    arr = arr.reshape(shape)
  return pn.Tensor.from_numpy(arr, copy=True, memory=pn.TensorMemory.CPU)


def _bytes(t: pn.Tensor) -> bytes:
  return bytes(t.copy_dense_bytes_tight())


def _pull(run: pn.Run, *args):
  """The Sample pull without its DeprecationWarning (tested once, in its own test)."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    return run.pull(*args)


def _named_graph():
  g = pn.Graph()
  image = g.input("image")
  g.output("out", image)
  return g


def _tensor_media_graph():
  opts = pn.InputOptions()
  opts.payload_type = pn.PayloadType.Tensor
  opts.format = pn.Format.FP32
  opts.memory_policy = pn.InputMemoryPolicy.SystemMemory
  g = pn.Graph()
  g.add(pn.nodes.input(opts))
  g.add(pn.nodes.output())
  return g


def _declared_graph(shape, max_shape=None):
  g = pn.Graph()
  x = g.input("x", shape, pn.TensorDType.Float32, max_shape)
  g.output("y", x)
  return g


# ── Types, no Run ──────────────────────────────────────────────────────────────────────────────


def test_backend_setting():
  # A build fixes the process's backend: the same value is then accepted, the other refused with
  # the rule, and nothing changes.
  run = _named_graph().build(pn.RunOptions())
  run.close()
  fixed = pn.backend()
  assert fixed in (pn.RunBackend.GStreamer, pn.RunBackend.Dataflow)
  pn.set_backend(fixed)
  other = pn.RunBackend.Dataflow if fixed == pn.RunBackend.GStreamer else pn.RunBackend.GStreamer
  with pytest.raises(pn.NeatError) as exc:
    pn.set_backend(other)
  assert exc.value.error_code == "misconfig.option_out_of_range"
  assert "a process runs on one backend" in str(exc.value)
  assert pn.backend() == fixed
  assert {m.name for m in pn.RunBackend} == {"Dataflow", "GStreamer"}
  assert not hasattr(pn.RunOptions(), "backend")
  # Before the first build, in a process of its own: the default, and any number of sets.
  values = ["GStreamer"] + (["Dataflow", "GStreamer", "Dataflow"] if _dataflow_available() else [])
  script = "import pyneat as pn\nassert pn.backend() == pn.RunBackend.GStreamer\n" + "".join(
      f"pn.set_backend(pn.RunBackend.{v})\nassert pn.backend() == pn.RunBackend.{v}\n"
      for v in values)
  child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=120)
  assert child.returncode == 0, child.stdout + child.stderr
  inp = pn.InputOptions()
  assert list(inp.shape) == [] and list(inp.max_shape) == [] and inp.dtype == pn.TensorDType.UInt8
  inp.shape, inp.max_shape, inp.dtype = [-1, 4], [8, 4], pn.TensorDType.Float32
  assert list(inp.shape) == [-1, 4] and list(inp.max_shape) == [8, 4]


def test_stamp_and_time_point_are_frozen_values():
  s = pn.Stamp()
  assert (s.stream, s.stream_name, s.seq, s.frame_id) == (0, "", 0, -1)
  assert not s.source_time.valid() and not s.publish_time.valid()
  assert s.flags == pn.StampFlags.None_ and s.version == 1
  with pytest.raises(AttributeError):
    s.frame_id = 3

  t = pn.TimePoint(1234, pn.ClockId.SensorLocal)
  assert t.valid() and t.ns == 1234 and t.clock == pn.ClockId.SensorLocal
  assert t == pn.TimePoint(ns=1234, clock=pn.ClockId.SensorLocal)
  assert hash(t) == hash(pn.TimePoint(1234, pn.ClockId.SensorLocal))
  assert t != pn.TimePoint(1234, pn.ClockId.Monotonic) and t != 1234
  with pytest.raises(AttributeError):
    t.ns = 5
  assert repr(t) == "TimePoint(ns=1234, clock=ClockId.SensorLocal)"
  assert not pn.TimePoint().valid()

  flags = pn.StampFlags.Discontinuity | pn.StampFlags.Gap
  p = pn.Stamp(stream_name="cam-a", frame_id=7, source_time=t, flags=flags)
  assert p.stream_name == "cam-a" and p.frame_id == 7 and p.source_time == t and p.flags == flags
  assert p == pn.Stamp(stream_name="cam-a", frame_id=7, source_time=t, flags=flags)
  assert hash(p) == hash(pn.Stamp(stream_name="cam-a", frame_id=7, source_time=t, flags=flags))
  assert p != s and p != "cam-a"
  q = p.replace(frame_id=8)
  assert q.frame_id == 8 and q.stream_name == "cam-a" and p.frame_id == 7
  with pytest.raises(TypeError):
    p.replace(nope=1)
  with pytest.raises(TypeError):
    pn.Stamp("cam-a")  # keyword-only
  assert "stream_name='cam-a'" in repr(p) and "frame_id=7" in repr(p)


def test_tensor_records_without_a_run():
  a = _floats([1, 2])
  b = _floats([10, 20, 30])
  leaf = a
  assert not leaf.is_record() and leaf.fields() == [] and leaf.field_names() == []
  assert len(leaf.leaves()) == 1 and _bytes(leaf.leaves()[0]) == _bytes(a)
  assert leaf.stamp == pn.Stamp()
  with pytest.raises(IndexError):
    leaf[0]
  with pytest.raises(KeyError):
    leaf["0"]
  with pytest.raises(TypeError):
    iter(leaf)

  r = pn.Tensor.record([a, b])
  assert r.is_record() and r.field_names() == ["0", "1"] and len(r.fields()) == 2
  assert _bytes(r["1"]) == _bytes(b) and _bytes(r[0]) == _bytes(a) and _bytes(r[-1]) == _bytes(b)
  assert [_bytes(x) for x in r.leaves()] == [_bytes(a), _bytes(b)]
  with pytest.raises(KeyError):
    r["2"]
  with pytest.raises(IndexError):
    r[2]
  with pytest.raises(ValueError):
    pn.Tensor.record([])

  stamped = r.with_stamp(pn.Stamp(frame_id=5, stream_name="s"))
  assert stamped.stamp.frame_id == 5 and stamped.is_record() and r.stamp.frame_id == -1

  sample = pn.to_sample(leaf.with_stamp(pn.Stamp(frame_id=9, stream_name="cam-b")))
  assert sample.frame_id == 9 and sample.stream_id == "cam-b"
  back = pn.to_tensor(sample)
  assert back.stamp.frame_id == 9 and back.stamp.stream_name == "cam-b"
  pair = pn.to_tensor(pn.to_sample(r))
  assert pair.is_record() and [_bytes(x) for x in pair.fields()] == [_bytes(a), _bytes(b)]


def test_graph_input_output_errors():
  g = pn.Graph()
  h = g.input("image")
  assert isinstance(h, pn.Graph.Handle) and bool(h) and not bool(pn.Graph.Handle())
  g.output("out", h)
  assert g.inputs() == ["image"] and g.outputs() == ["out"]
  other = pn.Graph()
  for call in (lambda: other.output("o", h), lambda: other.output("o", pn.Graph.Handle()),
               lambda: other.input("")):
    with pytest.raises(pn.NeatError) as exc:
      call()
    assert exc.value.error_code == "misconfig.pipeline_shape"
  for call in (lambda: other.input("v", [0, 4], pn.TensorDType.Float32),
               lambda: other.input("v", [-1, 4], pn.TensorDType.Float32, [8])):
    with pytest.raises(pn.NeatError) as exc:
      call()
    assert exc.value.error_code == "misconfig.input_shape"
  with pytest.raises(ValueError):
    other.input("v", dtype=pn.TensorDType.Float32)


def test_dataflow_refuses_a_graph_it_does_not_carry():
  if _process_backend() != "dataflow":
    pytest.skip("the process runs on GStreamer (the child process of "
                "test_the_surface_on_the_other_backend runs this on Dataflow)")
  # Fan-in, two inputs into one output.
  g = pn.Graph()
  g.add(pn.nodes.input("left"))
  g.add(pn.nodes.input("right"))
  g.add(pn.nodes.output("combined"))
  g.connect("left", "combined")
  g.connect("right", "combined")
  with pytest.raises(pn.NeatError) as exc:
    g.build(pn.RunOptions())
  assert "is not carried by the Dataflow backend" in str(exc.value) and "fan-in" in str(exc.value)
  assert "pyneat.set_backend(pyneat.RunBackend.GStreamer)" in str(exc.value)


def test_the_surface_on_the_other_backend():
  # This file again, in a child process that selects the other backend before anything is built.
  other = "GStreamer" if _process_backend() == "dataflow" else "Dataflow"
  if other == "Dataflow" and not _dataflow_available():
    pytest.skip("this build of libsima_neat has no Dataflow backend (SIMANEAT_DATAFLOW_SOURCE_DIR)")
  script = (
      "import sys, pytest, pyneat as pn\n"
      f"pn.set_backend(pn.RunBackend.{other})\n"
      f"sys.exit(pytest.main([{__file__!r}, '-q', '-p', 'no:cacheprovider', '-k',\n"
      "                       'not test_the_surface_on_the_other_backend']))\n")
  child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=1800)
  sys.stdout.write(child.stdout[-4000:])
  assert child.returncode == 0, f"the surface on {other}:\n{child.stdout[-6000:]}{child.stderr[-2000:]}"


def test_graph_of_several_endpoints(backend):
  # Two inputs, each feeding its own output, and a fan-out: one handle published twice.
  g = pn.Graph()
  g.output("a_out", g.input("a"))
  b = g.input("b")
  g.output("b_one", b)
  g.output("b_two", b)
  run = g.build(_options(backend))
  try:
    assert run.input_names() == ["a", "b"]
    assert run.output_names() == ["a_out", "b_one", "b_two"]
    assert run.push("a", [_rgb(16, 8, 1)])
    assert run.push("b", [_floats([1.0, 2.0, 3.0])])
    a = run.pull_result("a_out", 5000)
    assert a.ok and _bytes(a.value()[0]) == _rgb_array(16, 8, 1).tobytes()
    for name in ("b_one", "b_two"):
      r = run.pull_result(name, 5000)
      assert r.ok and _bytes(r.value()[0]) == np.asarray([1.0, 2.0, 3.0], np.float32).tobytes()
    unnamed = run.pull_result(30)
    assert unnamed.status == pn.PullStatus.Error
    assert "no unambiguous default output" in unnamed.error.message
  finally:
    run.close()


# ── Runs, both backends ────────────────────────────────────────────────────────────────────────


def test_graph_input_output_run(backend):
  run = _named_graph().build(_options(backend))
  try:
    assert run.input_names() == ["image"] and run.output_names() == ["out"]
    assert run.push("image", [_rgb(32, 24, 1)])
    assert run.push([_rgb(32, 24, 2)])
    for tag in (1, 2):
      s = _pull(run, "out", 5000) if tag == 1 else _pull(run, 5000)
      assert s is not None
      t = pn.to_tensor(s)  # a pulled item is a TensorSet Sample; its Tensor view is the leaf
      assert list(t.shape) == [24, 32, 3] and not t.is_record()
      assert _bytes(t) == _rgb_array(32, 24, tag).tobytes()
  finally:
    run.close()


def test_stamp_through_a_run(backend):
  run = _named_graph().build(_options(backend))
  try:
    proposed = pn.Stamp(stream_name="cam-a", frame_id=7,
                        source_time=pn.TimePoint(1234, pn.ClockId.SensorLocal))
    assert run.push_samples(pn.to_sample(_rgb(16, 8, 1).with_stamp(proposed)))
    assert run.push("image", [_rgb(16, 8, 2)])
    first = pn.to_tensor(_pull(run, "out", 5000))
    second = pn.to_tensor(_pull(run, "out", 5000))
    s1, s2 = first.stamp, second.stamp
    assert s1.frame_id == 7 and s1.stream_name == "cam-a"
    assert s2.frame_id == -1
    assert (s1.seq, s2.seq) == (1, 2)
    assert s1.source_time.valid() and s1.source_time.clock == pn.ClockId.SensorLocal
    if backend == "dataflow":
      assert s1.source_time == pn.TimePoint(1234, pn.ClockId.SensorLocal)
    assert _bytes(first) == _rgb_array(16, 8, 1).tobytes()
  finally:
    run.close()


def test_positional_records_through_a_run(backend):
  a = _floats([1, 2])
  b = _floats([10, 11])
  run = _tensor_media_graph().build([a, b], _options(backend))
  try:
    assert run.push([_floats([5, 6]), _floats([20, 21])])  # one item of two tensors
    t = pn.to_tensor(_pull(run, 5000))
    assert t.is_record() and t.field_names() == ["0", "1"]
    assert _bytes(t["1"]) == _bytes(_floats([20, 21])) and _bytes(t[0]) == _bytes(_floats([5, 6]))
    assert [list(x.shape) for x in t.leaves()] == [[2], [2]]
  finally:
    run.close()


def test_pull_is_deprecated_once_per_call_site(backend):
  run = _named_graph().build(_options(backend))
  try:
    for tag in range(4):
      assert run.push([_rgb(8, 8, tag)])
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("default")
      pulled = []
      for _ in range(2):
        pulled.append(run.pull("out", 5000))  # one call site, called twice
      pulled.append(run.pull(5000))  # a second call site
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 2, [str(w.message) for w in caught]
    assert all("pull_result" in str(w.message) for w in deprecations)
    assert all(w.filename == __file__ for w in deprecations)
    assert all(isinstance(s, pn.Sample) for s in pulled)
    assert [_bytes(pn.to_tensor(s)) for s in pulled] == [
        _rgb_array(8, 8, tag).tobytes() for tag in range(3)]

    with warnings.catch_warnings():
      warnings.simplefilter("error", DeprecationWarning)
      with pytest.raises(DeprecationWarning):
        run.pull("out", 5000)
    # The warning is raised before the pull: the fourth output is still there.
    last = _pull(run, "out", 5000)
    assert last is not None and _bytes(pn.to_tensor(last)) == _rgb_array(8, 8, 3).tobytes()

    run.close_input()
    assert _pull(run, "out", 5000) is None  # end of stream: still None on the Sample form
  finally:
    run.close()


def test_pull_tensors_is_deprecated_once_per_call_site(backend, capfd):
  run = _named_graph().build(_options(backend))
  try:
    for tag in range(4):
      assert run.push([_rgb(8, 8, tag)])
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("default")
      pulled = []
      for _ in range(2):
        pulled.append(run.pull_tensors("out", 5000))  # one call site, called twice
      pulled.append(run.pull_tensors(5000))  # a second call site
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 2, [str(w.message) for w in caught]
    assert all("Run.pull_tensors()" in str(w.message) for w in deprecations)
    assert all("pull_result" in str(w.message) for w in deprecations)
    assert all(w.filename == __file__ for w in deprecations)
    # Unchanged: a list of Tensors per item.
    assert [[_bytes(t) for t in ts] for ts in pulled] == [
        [_rgb_array(8, 8, tag).tobytes()] for tag in range(3)]

    with warnings.catch_warnings():
      warnings.simplefilter("error", DeprecationWarning)
      with pytest.raises(DeprecationWarning):
        run.pull_tensors("out", 5000)
    # The warning is raised before the pull: the fourth output is still there.
    last = run.pull_result("out", 5000)
    assert last.ok and _bytes(last.value()[0]) == _rgb_array(8, 8, 3).tobytes()
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", DeprecationWarning)
      assert run.pull_tensors("out", 0) == []  # timeout: still an empty list
  finally:
    run.close()
  # The C++ runtime's one-time "[WARN] Run::... is deprecated" line is for C++ callers.
  assert "[WARN] Run::" not in capfd.readouterr().err


def test_python_never_sees_the_cpp_deprecation_line(backend, capfd):
  run = _named_graph().build(_options(backend))
  try:
    for tag in range(3):
      assert run.push([_rgb(8, 8, tag)])
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      first = run.pull("out", 5000)
      second = run.pull_tensors("out", 5000)
      third = run.pull_samples("out", 5000)
    deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    # Run.pull (C6) and Run.pull_tensors (C5) warn; Run.pull_samples does not (it is the only
    # Python Sample pull without a deprecation: Sample.attributes readers need it).
    assert len(deprecations) == 2, deprecations
    assert "Run.pull()" in deprecations[0] and "Run.pull_tensors()" in deprecations[1]
    assert _bytes(pn.to_tensor(first)) == _rgb_array(8, 8, 0).tobytes()
    assert [_bytes(t) for t in second] == [_rgb_array(8, 8, 1).tobytes()]
    assert _bytes(pn.to_tensor(third)) == _rgb_array(8, 8, 2).tobytes()
  finally:
    run.close()
  assert "[WARN] Run::" not in capfd.readouterr().err


def test_blocking_pull_releases_the_gil(backend):
  run = _named_graph().build(_options(backend))
  try:
    assert run.push([_rgb(8, 8, 0)])
    assert _pull(run, "out", 5000) is not None  # first push and pull: every segment is up

    def push_later() -> None:
      time.sleep(0.3)
      run.push([_rgb(8, 8, 1)])

    pusher = threading.Thread(target=push_later)
    t0 = time.monotonic()
    pusher.start()
    s = _pull(run, "out", 8000)  # the pusher needs the GIL while this waits
    elapsed = time.monotonic() - t0
    pusher.join(5)
    assert s is not None and _bytes(pn.to_tensor(s)) == _rgb_array(8, 8, 1).tobytes()
    assert 0.2 <= elapsed < 5, elapsed
  finally:
    run.close()


def test_declared_input_alloc_and_copy_from(backend):
  run = _declared_graph([4, 8]).build(_options(backend))
  try:
    for bad in (_floats(np.zeros(36), (4, 9)), _floats(np.zeros(32), (32,)),
                pn.Tensor.from_numpy(np.zeros((4, 8), dtype=np.int32), copy=True,
                                     memory=pn.TensorMemory.CPU)):
      with pytest.raises(pn.NeatError) as exc:
        run.push("x", [bad])
      assert exc.value.error_code == "misconfig.input_shape"

    t = run.alloc("x")
    assert t.dtype == pn.TensorDType.Float32 and list(t.shape) == [4, 8] and not t.read_only
    values = np.arange(32, dtype=np.float32).reshape(4, 8) + 100
    t.copy_from(values)
    assert np.array_equal(np.from_dlpack(t), values)
    with pytest.raises(ValueError):
      t.copy_from(np.zeros((4, 9), dtype=np.float32))
    with pytest.raises(ValueError):
      t.copy_from(np.zeros((4, 8), dtype=np.int32))
    with pytest.raises(RuntimeError, match="TensorList"):
      run.push("x", t)  # Run.push keeps the list-only rule
    assert run.try_push("x", [t])  # try_push reads it like any CPU tensor: a copy
    first = run.pull_result("y", 5000)
    assert first.ok and _bytes(first.value()[0]) == values.tobytes()
    t.copy_from(values + 1)  # still writable
    view = np.from_dlpack(t)  # a live mapping of the allocation
    with pytest.raises(pn.NeatError) as exc:
      run.push("x", [t])
    assert exc.value.error_code == "misconfig.input_shape" and "mapped" in str(exc.value)
    del view
    assert run.push("x", [t])  # a blocking push of the list holding just it: the commit
    with pytest.raises(pn.NeatError) as exc:
      run.push("x", [t])
    assert exc.value.error_code == "misconfig.input_shape"  # pushed already
    with pytest.raises(RuntimeError):
      t.copy_from(values)  # empty after its commit
    second = run.pull_result("y", 5000)
    assert second.ok and _bytes(second.value()[0]) == (values + 1).tobytes()

    undeclared = _named_graph().build(_options(backend))
    with pytest.raises(pn.NeatError) as exc:
      undeclared.alloc("image")
    assert exc.value.error_code == "misconfig.input_shape"
    undeclared.close()
    with pytest.raises(Exception, match="unknown input"):
      run.alloc("nope")
    with pytest.raises(ValueError):
      _floats([1, 2]).copy_from(np.zeros(3, dtype=np.float32))
    readonly = pn.Tensor.from_numpy(np.zeros(2, dtype=np.float32), memory=pn.TensorMemory.CPU)
    assert readonly.read_only
    with pytest.raises(ValueError):
      readonly.copy_from(np.zeros(2, dtype=np.float32))
  finally:
    run.close()
  with pytest.raises(Exception):
    run.alloc("x")  # a closed Run lends nothing


def test_bounded_dynamic_axis_alloc(backend):
  run = _declared_graph([-1, 4], [10, 4]).build(_options(backend))
  try:
    t = run.alloc("x")
    assert list(t.shape) == [10, 4]
    t.shape = [3, 4]
    rows = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    t.copy_from(rows)
    assert run.push("x", [t])
    out = run.pull_result("y", 5000).value()[0]
    assert list(out.shape) == [3, 4] and _bytes(out) == rows.tobytes()
    with pytest.raises(pn.NeatError) as exc:
      run.push("x", [_floats(np.zeros(44), (11, 4))])
    assert exc.value.error_code == "misconfig.input_shape"
  finally:
    run.close()


def test_blocking_alloc_releases_the_gil(backend):
  run = _declared_graph([4, 8]).build(_options(backend))
  held: list[pn.Tensor] = []
  finished = threading.Event()

  def release_slots() -> None:
    while not finished.is_set():
      time.sleep(0.2)
      held.clear()  # drops the allocations (needs the GIL while alloc waits)

  helper = threading.Thread(target=release_slots)
  helper.start()
  try:
    t0 = time.monotonic()
    for _ in range(24):
      held.append(run.alloc("x"))
    elapsed = time.monotonic() - t0
    if backend == "dataflow":
      # A Dataflow allocation is a slot of the input's pool: alloc waits for a free one.
      assert elapsed >= 0.15, f"alloc never waited for a slot ({elapsed:.3f} s)"
  finally:
    finished.set()
    helper.join(5)
    held.clear()
    run.close()


# ── The TensorList verbs: pull_result, Run(...), on_output ────────────────────────────────────


def test_push_list_and_pull_result(backend):
  run = _named_graph().build(_options(backend))
  try:
    proposed = pn.Stamp(stream_name="cam-a", frame_id=7,
                        source_time=pn.TimePoint(1234, pn.ClockId.SensorLocal),
                        flags=pn.StampFlags.Discontinuity)
    assert run.push("image", [_rgb(16, 8, 1).with_stamp(proposed)])  # the first tensor's stamp
    assert run.push([_rgb(16, 8, 2)])
    assert run.push("image", [_rgb_array(16, 8, 3)], image_format=pn.PixelFormat.RGB)
    r1 = run.pull_result("out", 5000)
    r2 = run.pull_result(5000)
    r3 = run.pull_result("out", 5000)
    for r in (r1, r2, r3):
      assert isinstance(r, pn.PullResult) and r.ok and bool(r) and r.status == pn.PullStatus.Ok
      assert isinstance(r.value(), list) and len(r.value()) == 1
    t1, t2, t3 = r1.value()[0], r2.value()[0], r3.value()[0]
    assert [_bytes(t) for t in (t1, t2, t3)] == [_rgb_array(16, 8, k).tobytes() for k in (1, 2, 3)]
    assert list(t1.shape) == [8, 16, 3] and not t1.is_record()
    s1, s2 = t1.stamp, t2.stamp
    assert s1.frame_id == 7 and s1.stream_name == "cam-a" and s2.frame_id == -1
    assert (s1.seq, s2.seq, t3.stamp.seq) == (1, 2, 3)
    assert s1.stream != 0 and s1.stream != s2.stream
    assert s1.source_time.valid() and s1.version == 1
    if backend == "dataflow":
      assert s1.source_time == pn.TimePoint(1234, pn.ClockId.SensorLocal)
      assert s1.flags == pn.StampFlags.Discontinuity and s2.flags == pn.StampFlags.None_
      assert s1.publish_time.valid() and s1.publish_time.clock == pn.ClockId.Monotonic
      assert s2.publish_time.ns >= s1.publish_time.ns
    else:
      assert not s1.publish_time.valid()  # GStreamer records no publish time (c2.md section 4)
  finally:
    run.close()


def test_pull_result_outcomes_and_call(backend):
  run = _named_graph().build(_options(backend))
  try:
    timeout = run.pull_result("out", 30)
    assert timeout.status == pn.PullStatus.Timeout and not timeout and not timeout.ok
    with pytest.raises(pn.NeatError) as exc:
      timeout.value()
    assert exc.value.error_code == "runtime.pull"
    assert "Timeout" in repr(timeout)

    error = run.pull_result("nope", 30)
    assert error.status == pn.PullStatus.Error
    assert error.error.code == "runtime.pull" and "unknown output" in error.error.message

    y = run([_rgb(8, 8, 5)])  # Run(tensors): push, then the output
    assert isinstance(y, list) and len(y) == 1 and _bytes(y[0]) == _rgb_array(8, 8, 5).tobytes()
    with pytest.raises(RuntimeError, match="TensorList"):
      run(_rgb(8, 8, 5))  # no single-Tensor form

    assert run.push([_rgb(8, 8, 6)])
    run.close_input()
    last = run.pull_result("out", 5000)
    assert last.ok and _bytes(last.value()[0]) == _rgb_array(8, 8, 6).tobytes()
    closed = run.pull_result("out", 5000)
    assert closed.status == pn.PullStatus.Closed and not closed
  finally:
    run.close()
  assert run.pull_result(10).status == pn.PullStatus.Closed


def test_records_through_pull_result(backend):
  a = _floats([1, 2])
  b = _floats([10, 11])
  run = _tensor_media_graph().build([a, b], _options(backend))
  try:
    record = pn.Tensor.record([_floats([5, 6]), _floats([20, 21])]).with_stamp(pn.Stamp(frame_id=3))
    assert run.push([record])  # a list of one record: its fields, with its stamp
    assert run.push([_floats([7, 8]).with_stamp(pn.Stamp(frame_id=4)), _floats([30, 31])])
    first = run.pull_result(5000).value()
    second = run.pull_result(5000).value()
    assert [_bytes(t) for t in first] == [_bytes(_floats([5, 6])), _bytes(_floats([20, 21]))]
    assert [t.stamp.frame_id for t in first] == [3, 3]  # each field carries the item's stamp
    assert [_bytes(t) for t in second] == [_bytes(_floats([7, 8])), _bytes(_floats([30, 31]))]
    assert [t.stamp.frame_id for t in second] == [4, 4]  # the first tensor's stamp
    with pytest.raises(pn.NeatError) as exc:
      run.push([record, _floats([1, 2])])  # a record inside a list of several
    assert exc.value.error_code == "misconfig.input_shape"
  finally:
    run.close()


def test_blocking_pull_result_releases_the_gil(backend):
  run = _named_graph().build(_options(backend))
  try:
    assert len(run([_rgb(8, 8, 0)])) == 1  # first push and pull: every segment is up

    def push_later() -> None:
      time.sleep(0.3)
      run.push([_rgb(8, 8, 1)])

    pusher = threading.Thread(target=push_later)
    t0 = time.monotonic()
    pusher.start()
    r = run.pull_result("out", 8000)  # the pusher needs the GIL while this waits
    elapsed = time.monotonic() - t0
    pusher.join(5)
    assert r.ok and _bytes(r.value()[0]) == _rgb_array(8, 8, 1).tobytes()
    assert 0.2 <= elapsed < 5, elapsed
  finally:
    run.close()


def test_on_output_delivers_lists_in_order_on_a_run_thread(backend):
  run = _named_graph().build(_options(backend))
  seen: list[int] = []
  threads: set[int] = set()
  cv = threading.Condition()

  def callback(tensors) -> None:
    with cv:
      assert isinstance(tensors, list) and len(tensors) == 1
      seen.append(tensors[0].stamp.frame_id)
      threads.add(threading.get_ident())
      cv.notify_all()

  try:
    run.on_output("out", callback)
    with pytest.raises(pn.NeatError) as exc:
      run.on_output("out", lambda ts: None)
    assert exc.value.error_code == "misconfig.pipeline_shape"
    with pytest.raises(pn.NeatError):
      run.on_output("nope", lambda ts: None)
    for i in range(5):
      assert run.push([_rgb(8, 8, i).with_stamp(pn.Stamp(frame_id=100 + i))])
    with cv:
      assert cv.wait_for(lambda: len(seen) == 5, timeout=10), seen
    assert seen == [100, 101, 102, 103, 104]
    assert threading.get_ident() not in threads and len(threads) == 1
    owned = run.pull_result("out", 10)
    assert owned.status == pn.PullStatus.Error and "on_output" in owned.error.message
    run.close_input()
    t0 = time.monotonic()
    run.stop()
    assert time.monotonic() - t0 < 3
  finally:
    run.close()


def test_on_output_callback_that_raises(backend, monkeypatch):
  run = _named_graph().build(_options(backend))
  calls: list[int] = []
  reported = []
  monkeypatch.setattr(sys, "unraisablehook", lambda info: reported.append(
      (info.exc_type, str(info.exc_value), info.object)))

  def callback(tensors) -> None:
    calls.append(1)
    raise ValueError("boom")

  try:
    run.on_output(callback)
    assert run.push([_rgb(8, 8, 1)]) and run.push([_rgb(8, 8, 2)])
    deadline = time.monotonic() + 10
    while "boom" not in run.last_error() and time.monotonic() < deadline:
      time.sleep(0.005)
    assert "callback threw: ValueError: boom" in run.last_error(), run.last_error()
    time.sleep(0.1)
    assert len(calls) == 1  # the delivery ended at the exception
    assert reported and reported[0][0] is ValueError and reported[0][1] == "boom"
    assert "Run.on_output callback" in str(reported[0][2])  # the context it was raised in
  finally:
    run.close()


def _busy_callback(started: threading.Event, done: list[int]):
  def callback(tensors) -> None:
    started.set()
    total = 0
    for i in range(2_000_000):  # pure Python: needs the GIL throughout
      total += i
    done.append(total)

  return callback


def test_close_while_a_callback_needs_the_gil(backend, capfd):
  run = _named_graph().build(_options(backend))
  started = threading.Event()
  done: list[int] = []
  run.on_output("out", _busy_callback(started, done))
  for i in range(3):
    assert run.push([_rgb(8, 8, i)])
  assert started.wait(10)
  t0 = time.monotonic()
  run.close()  # waits for the callback in progress, which needs the GIL: close releases it
  elapsed = time.monotonic() - t0
  # Holding the GIL here would leave the callback stuck until the 2 s stop budget expired, then
  # print a warning and detach it (RunTensor.cpp stop_output_callbacks).
  assert "did not return within" not in capfd.readouterr().err
  assert done, "the callback in progress finished before close returned"
  assert elapsed < 10, elapsed


def test_releasing_a_run_while_a_callback_needs_the_gil(backend, capfd):
  run = _named_graph().build(_options(backend))
  started = threading.Event()
  done: list[int] = []
  run.on_output("out", _busy_callback(started, done))
  for i in range(3):
    assert run.push([_rgb(8, 8, i)])
  assert started.wait(10)
  t0 = time.monotonic()
  del run  # the last reference: the binding closes it without the GIL before destroying it
  gc.collect()
  elapsed = time.monotonic() - t0
  assert "did not return within" not in capfd.readouterr().err
  assert done, "the callback in progress finished before the Run was destroyed"
  assert elapsed < 10, elapsed
