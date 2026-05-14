from __future__ import annotations


def _shape_tuple(obj):
  if not hasattr(obj, "shape"):
    return None
  try:
    shape = tuple(int(x) for x in obj.shape)
  except Exception:
    return None
  return shape


def _normalize_memory(core, memory):
  if memory is None:
    return core.TensorMemory.EV74
  if isinstance(memory, str):
    token = memory.strip().lower().replace("-", "_")
    if token in ("auto",):
      return core.TensorMemory.Auto
    if token in ("cpu", "a65"):
      return core.TensorMemory.CPU if token == "cpu" else core.TensorMemory.A65
    if token in ("ev74", "cvu", "sima_cvu"):
      return core.TensorMemory.EV74
    if token in ("mla", "sima_mla"):
      return core.TensorMemory.MLA
    raise ValueError(f"unsupported tensor memory placement: {memory!r}")
  return memory


def _infer_layout(core, obj, *, prefer_chw: bool = False):
  if not hasattr(obj, "ndim"):
    return core.TensorLayout.Unknown

  try:
    ndim = int(obj.ndim)
  except Exception:
    return core.TensorLayout.Unknown

  if ndim == 2:
    return core.TensorLayout.HW
  if ndim != 3:
    return core.TensorLayout.Unknown

  shape = _shape_tuple(obj)
  if shape is not None and len(shape) == 3:
    c_first = shape[0]
    c_last = shape[2]
    channel_like_first = c_first in (1, 3, 4)
    channel_like_last = c_last in (1, 3, 4)

    if prefer_chw and channel_like_first and not channel_like_last:
      return core.TensorLayout.CHW
    if channel_like_last and not channel_like_first:
      return core.TensorLayout.HWC
    if channel_like_first and not channel_like_last:
      return core.TensorLayout.CHW
    if channel_like_first and channel_like_last:
      return core.TensorLayout.Unknown
    if not channel_like_first and not channel_like_last:
      return core.TensorLayout.Unknown

  return core.TensorLayout.Unknown


def install_wrappers(core) -> None:
  raw_model_build = core.Model.build
  raw_model_build_with_session_options = core.Model.build_with_session_options
  has_native_build_overloads = bool(getattr(core, "_HAS_NATIVE_BUILD_OBJECT_OVERLOADS", False))

  def _model_image_format_hint(model):
    try:
      spec = model.input_spec()
      return getattr(spec, "image_format", None)
    except Exception:
      return None

  def tensor_from_dlpack(cls, obj, copy: bool = False, layout=None, image_format=None, memory=None):
    if hasattr(obj, "__dlpack__"):
      capsule = obj.__dlpack__()
    else:
      capsule = obj

    if layout is None:
      layout = _infer_layout(core, obj, prefer_chw=False)

    return cls._from_dlpack_capsule(
        capsule,
        copy=copy,
        layout=layout,
        image_format=image_format,
        memory=_normalize_memory(core, memory),
    )

  def tensor_from_numpy(cls, array, copy: bool = False, layout=None, image_format=None, memory=None):
    import numpy as np

    arr = np.asarray(array)
    if layout is None:
      layout = _infer_layout(core, arr, prefer_chw=False)
    return cls._from_dlpack_capsule(
        arr.__dlpack__(),
        copy=copy,
        layout=layout,
        image_format=image_format,
        memory=_normalize_memory(core, memory),
    )

  def tensor_to_numpy(self, copy: bool = False):
    import numpy as np

    out = np.from_dlpack(self)
    if copy:
      return out.copy()
    return out

  def tensor_from_torch(cls, tensor, copy: bool = False, layout=None, image_format=None, memory=None):
    if hasattr(tensor, "device"):
      device_type = getattr(getattr(tensor, "device"), "type", "cpu")
      if device_type != "cpu":
        tensor = tensor.cpu()

    if layout is None:
      layout = _infer_layout(core, tensor, prefer_chw=True)

    return cls._from_dlpack_capsule(
        tensor.__dlpack__(),
        copy=copy,
        layout=layout,
        image_format=image_format,
        memory=_normalize_memory(core, memory),
    )

  def tensor_to_torch(self, copy: bool = False):
    import torch

    out = torch.from_dlpack(self)
    if copy:
      return out.clone()
    return out

  core.Tensor.from_dlpack = classmethod(tensor_from_dlpack)
  core.Tensor.from_numpy = classmethod(tensor_from_numpy)
  core.Tensor.to_numpy = tensor_to_numpy
  core.Tensor.from_torch = classmethod(tensor_from_torch)
  core.Tensor.from_pytorch = classmethod(tensor_from_torch)
  core.Tensor.to_torch = tensor_to_torch
  core.Tensor.to_pytorch = tensor_to_torch

  if not hasattr(core.Run, "push"):
    def run_push(self, value, copy: bool = False, layout=None, image_format=None):
      if isinstance(value, core.Sample):
        return self.push_sample(value)
      if isinstance(value, core.Tensor):
        return self.push_tensor(value)
      tensor = core.Tensor.from_dlpack(value, copy=copy, layout=layout, image_format=image_format)
      return self.push_tensor(tensor)

    core.Run.push = run_push

  if not hasattr(core.Run, "try_push"):
    def run_try_push(self, value, copy: bool = False, layout=None, image_format=None):
      if isinstance(value, core.Sample):
        return self.try_push_sample(value)
      if isinstance(value, core.Tensor):
        return self.try_push_tensor(value)
      tensor = core.Tensor.from_dlpack(value, copy=copy, layout=layout, image_format=image_format)
      return self.try_push_tensor(tensor)

    core.Run.try_push = run_try_push

  if not hasattr(core.Run, "run"):
    def run_execute(self, value, timeout_ms: int = -1):
      if isinstance(value, core.Sample):
        return self.run_sample(value, timeout_ms)
      if isinstance(value, core.Tensor):
        return self.run_tensor(value, timeout_ms)
      raise TypeError("Run.run expects Tensor or Sample")

    core.Run.run = run_execute

  def session_add(self, item):
    if isinstance(item, core.NodeGroup):
      return self.add_group(item)
    return self.add_node(item)

  def session_build(
      self,
      input_value=None,
      mode=None,
      options=None,
      copy: bool = False,
      layout=None,
      image_format=None,
  ):
    if mode is None:
      mode = core.RunMode.Async
    if options is None:
      options = core.RunOptions()

    if input_value is None:
      return self.build_source(options)
    if isinstance(input_value, core.Sample):
      return self.build_sample(input_value, mode, options)
    if isinstance(input_value, core.Tensor):
      return self.build_tensor(input_value, mode, options)

    tensor = core.Tensor.from_dlpack(
        input_value,
        copy=copy,
        layout=layout,
        image_format=image_format,
    )
    return self.build_tensor(tensor, mode, options)

  def session_run(self, input_value=None, options=None):
    if input_value is None:
      return self.run_source()
    if options is None:
      options = core.RunOptions()
    if isinstance(input_value, core.Tensor):
      return self.run_tensor(input_value, options)
    raise TypeError("Session.run expects Tensor or None")

  core.Session.add = session_add
  if not has_native_build_overloads:
    core.Session.build = session_build
  core.Session.run = session_run

  def model_session(self, options=None):
    if options is None:
      return self.session_group()
    return self.session_group_with_options(options)

  def model_build(
      self,
      input_value=None,
      session_options=None,
      run_options=None,
      copy: bool = False,
  ):
    if input_value is None:
      if session_options is None:
        return raw_model_build(self)
      return raw_model_build_with_session_options(self, session_options)

    if session_options is None:
      session_options = core.ModelSessionOptions()
    if run_options is None:
      run_options = core.RunOptions()

    if isinstance(input_value, core.Tensor):
      return self.build_tensor(input_value, session_options, run_options)
    if isinstance(input_value, core.Sample):
      return self.build_sample(input_value, session_options, run_options)

    tensor = core.Tensor.from_dlpack(
        input_value,
        copy=copy,
        image_format=_model_image_format_hint(self),
    )
    return self.build_tensor(tensor, session_options, run_options)

  core.Model.session = model_session
  if not has_native_build_overloads:
    core.Model.build = model_build
  if not hasattr(core.Model, "run"):
    def model_run(
        self,
        input_value,
        timeout_ms: int = -1,
        copy: bool = False,
    ):
      if isinstance(input_value, core.Tensor):
        return self.run_tensor(input_value, timeout_ms)
      if isinstance(input_value, core.Sample):
        return self.run_sample(input_value, timeout_ms)
      image_format = _model_image_format_hint(self)
      if isinstance(input_value, (list, tuple)):
        tensors = []
        for value in input_value:
          if isinstance(value, core.Tensor):
            tensors.append(value)
          else:
            tensors.append(
                core.Tensor.from_dlpack(value, copy=copy, image_format=image_format)
            )
        return self.run_batch(tensors, timeout_ms)

      tensor = core.Tensor.from_dlpack(
          input_value,
          copy=copy,
          image_format=image_format,
      )
      return self.run_tensor(tensor, timeout_ms)

    core.Model.run = model_run

  if not hasattr(core.ModelRunner, "push"):
    def model_runner_push(self, value, copy: bool = False, layout=None, image_format=None):
      if isinstance(value, core.Tensor):
        return self.push_tensor(value)
      if isinstance(value, core.Sample):
        return self.push_sample(value)
      tensor = core.Tensor.from_dlpack(value, copy=copy, layout=layout, image_format=image_format)
      return self.push_tensor(tensor)

    core.ModelRunner.push = model_runner_push

  if not hasattr(core.ModelRunner, "run"):
    def model_runner_run(self, value, timeout_ms: int = -1):
      if isinstance(value, core.Tensor):
        return self.run_tensor(value, timeout_ms)
      if isinstance(value, core.Sample):
        return self.run_sample(value, timeout_ms)
      raise TypeError("ModelRunner.run expects Tensor or Sample")

    core.ModelRunner.run = model_runner_run
