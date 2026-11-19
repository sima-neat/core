import numpy as np
import pytest
import warnings

import pyneat as pn


def _assert_not_type_error(call):
  try:
    call()
  except Exception as exc:  # Runtime is expected to fail for an unbuilt Run.
    assert not isinstance(exc, TypeError), str(exc)


def test_run_accepts_numpy_without_explicit_tensor_wrap():
  run = pn.Run()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: run.push([arr]))
  _assert_not_type_error(lambda: run.push([arr], copy=False))
  _assert_not_type_error(lambda: run.push([arr], copy=True))
  _assert_not_type_error(lambda: run.run([arr], timeout_ms=1))


def test_run_rejects_single_tensor_or_sample_inputs():
  run = pn.Run()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)
  tensor = pn.Tensor.from_numpy(
      arr,
      copy=True,
      image_format=pn.PixelFormat.RGB,
      memory=pn.TensorMemory.CPU,
  )
  sample = pn.Sample()

  with pytest.raises(Exception, match="expected list/tuple"):
    run.push(arr)
  with pytest.raises(Exception, match="TensorList"):
    run.push(tensor)
  with pytest.raises(Exception, match="Sample"):
    run.push(sample)


def test_model_runner_accepts_numpy_without_explicit_tensor_wrap():
  runner = pn.ModelRunner()
  arr = np.zeros((8, 8, 3), dtype=np.uint8)

  _assert_not_type_error(lambda: runner.push([arr]))
  _assert_not_type_error(lambda: runner.push([arr], copy=False))
  _assert_not_type_error(lambda: runner.run([arr], timeout_ms=1))


@pytest.mark.parametrize("method", ["try_push_samples", "try_push_tensors"])
def test_model_runner_try_push_matches_run(method):
  tensor = pn.Tensor.from_numpy(
      np.zeros((2, 2), dtype=np.float32), copy=True, memory=pn.TensorMemory.CPU
  )
  sample = pn.Sample()
  sample.kind = pn.SampleKind.TensorSet
  sample.tensors = [tensor, tensor]
  inputs = sample if method == "try_push_samples" else [tensor, tensor]
  for value in (inputs, pn.Sample() if method == "try_push_samples" else []):
    with pytest.raises(RuntimeError) as expected:
      getattr(pn.Run(), method)(value)
    with pytest.raises(RuntimeError) as actual:
      getattr(pn.ModelRunner(), method)(value)
    assert str(actual.value) == str(expected.value)
  with pytest.raises(TypeError):
    getattr(pn.ModelRunner(), method)(object())


def test_run_chw_image_autoconverts_to_hwc_with_warning():
  run = pn.Run()
  arr = np.zeros((3, 8, 8), dtype=np.uint8)

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _assert_not_type_error(lambda: run.push([arr], image_format=pn.PixelFormat.RGB))

  if caught:
    assert any(
        "Auto-converting to HWC" in str(item.message) and "full memory copy" in str(item.message)
        for item in caught
    )


@pytest.mark.parametrize("named", [False, True])
def test_run_pull_status_distinguishes_timeout_output_and_closed(named):
  import threading

  options = pn.InputOptions()
  options.payload_type = pn.PayloadType.Image
  options.format = pn.Format.RGB
  options.memory_policy = pn.InputMemoryPolicy.SystemMemory
  options.width, options.height, options.depth = 8, 8, 3
  graph = pn.Graph()
  graph.add(pn.nodes.input(options))
  graph.add(pn.nodes.output("frames"))
  array = np.full((8, 8, 3), 17, dtype=np.uint8)
  tensor = pn.Tensor.from_numpy(array, copy=True, image_format=pn.PixelFormat.RGB,
                                memory=pn.TensorMemory.CPU)
  run_options = pn.RunOptions()
  run_options.startup_preflight = False
  run = graph.build([tensor], run_options)
  pull = (lambda timeout: run.pull_status("frames", timeout)) if named else run.pull_status
  sender = None
  try:
    status, sample, error = pull(0)
    assert status == pn.PullStatus.Timeout
    assert isinstance(sample, pn.Sample) and isinstance(error, pn.PullError)
    assert run.pull("frames", 0) is None

    # The producer must acquire the GIL while pull_status is blocked.
    sender = threading.Timer(0.05, lambda: run.push([tensor]))
    sender.start()
    status, sample, error = pull(2000)
    assert status == pn.PullStatus.Ok
    assert error.code == "" and error.message == "" and error.report is None
    output = sample.tensor if sample.tensor is not None else sample.tensors[0]
    np.testing.assert_array_equal(output.to_numpy(copy=True), array)
    del sample

    status, _, error = run.pull_status("missing", 0)
    assert status == pn.PullStatus.Error
    assert error.code == "runtime.pull" and "unknown output" in error.message
    with pytest.raises(RuntimeError, match="unknown output"):
      run.pull("missing", 0)

    run.close_input()
    status, _, error = pull(2000)
    assert status == pn.PullStatus.Closed
    assert error.message
    assert run.pull("frames", 0) is None
  finally:
    if sender is not None:
      sender.join(timeout=3)
    run.close()


def test_run_pull_status_reports_one_source_eos_while_other_input_runs():
  graph = pn.Graph()
  graph.connect(
      pn.nodes.custom("videotestsrc num-buffers=1 ! "
                      "video/x-raw,format=RGB,width=8,height=8,framerate=30/1",
                      pn.InputRole.Source),
      pn.nodes.output("finite"),
  )
  options = pn.InputOptions()
  options.payload_type = pn.PayloadType.Image
  options.format = pn.Format.RGB
  options.memory_policy = pn.InputMemoryPolicy.SystemMemory
  options.width, options.height, options.depth = 8, 8, 3
  graph.connect(pn.nodes.input("live", options), pn.nodes.output("live"))
  run = graph.build()
  try:
    assert run.pull_status("finite", 2000)[0] == pn.PullStatus.Ok
    status, _, error = run.pull_status("finite", 2000)
    assert status == pn.PullStatus.Closed
    assert error.code == "io.source_ended"
    assert run.pull_status("live", 0)[0] == pn.PullStatus.Timeout
    tensor = pn.Tensor.from_numpy(np.zeros((8, 8, 3), dtype=np.uint8), copy=True,
                                  image_format=pn.PixelFormat.RGB, memory=pn.TensorMemory.CPU)
    assert run.push("live", [tensor])
    assert run.pull_status("live", 2000)[0] == pn.PullStatus.Ok
  finally:
    run.close()
