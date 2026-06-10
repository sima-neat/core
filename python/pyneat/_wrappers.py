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


def _is_sequence(value) -> bool:
  return isinstance(value, (list, tuple))


def _reject_single_tensor_or_sample(core, value, where: str) -> None:
  if isinstance(value, core.Sample):
    raise TypeError(f"{where} expects a Sample; pass [sample] instead of a single Sample")
  if isinstance(value, core.Tensor):
    raise TypeError(f"{where} expects a TensorList; pass [tensor] instead of a single Tensor")


def _sequence_all_samples(core, value) -> bool:
  return _is_sequence(value) and len(value) > 0 and all(isinstance(v, core.Sample) for v in value)


def _tensor_list_from_sequence(core, value, *, copy: bool = False, layout=None,
                               image_format=None, memory=None):
  if not _is_sequence(value):
    raise TypeError("expected a list/tuple of Tensor or DLPack-compatible inputs")
  out = []
  for item in value:
    if isinstance(item, core.Tensor):
      out.append(item)
    else:
      out.append(core.Tensor.from_dlpack(
          item,
          copy=copy,
          layout=layout,
          image_format=image_format,
          memory=memory,
      ))
  return out


def install_wrappers(core) -> None:
  raw_model_build = core.Model.build
  raw_model_build_with_route_options = core.Model.build_with_route_options
  has_native_build_overloads = bool(getattr(core, "_HAS_NATIVE_BUILD_OBJECT_OVERLOADS", False))

  def _model_image_format_hint(model):
    try:
      specs = model.input_specs()
      spec = specs[0] if specs else None
      return getattr(spec, "image_format", None)
    except Exception:
      return None

  def tensor_from_dlpack(
      cls, obj, copy: bool = False, layout=None, image_format=None, memory=None, byte_format=None
  ):
    if hasattr(obj, "__dlpack__"):
      capsule = obj.__dlpack__()
    else:
      capsule = obj

    if byte_format is not None:
      layout = core.TensorLayout.Unknown
    elif layout is None:
      layout = _infer_layout(core, obj, prefer_chw=False)

    return cls._from_dlpack_capsule(
        capsule,
        copy=copy,
        layout=layout,
        image_format=image_format,
        byte_format=byte_format,
        memory=_normalize_memory(core, memory),
    )

  def tensor_from_numpy(
      cls, array, copy: bool = False, layout=None, image_format=None, memory=None, byte_format=None
  ):
    import numpy as np

    arr = np.asarray(array)
    if byte_format is not None:
      layout = core.TensorLayout.Unknown
    elif layout is None:
      layout = _infer_layout(core, arr, prefer_chw=False)
    return cls._from_dlpack_capsule(
        arr.__dlpack__(),
        copy=copy,
        layout=layout,
        image_format=image_format,
        byte_format=byte_format,
        memory=_normalize_memory(core, memory),
    )

  def tensor_to_numpy(self, copy: bool = False):
    import numpy as np

    out = np.from_dlpack(self)
    if copy:
      return out.copy()
    return out

  def tensor_from_torch(
      cls, tensor, copy: bool = False, layout=None, image_format=None, memory=None, byte_format=None
  ):
    if hasattr(tensor, "device"):
      device_type = getattr(getattr(tensor, "device"), "type", "cpu")
      if device_type != "cpu":
        tensor = tensor.cpu()

    if byte_format is not None:
      layout = core.TensorLayout.Unknown
    elif layout is None:
      layout = _infer_layout(core, tensor, prefer_chw=True)

    return cls._from_dlpack_capsule(
        tensor.__dlpack__(),
        copy=copy,
        layout=layout,
        image_format=image_format,
        byte_format=byte_format,
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
      _reject_single_tensor_or_sample(core, value, "Run.push")
      if _sequence_all_samples(core, value):
        return self.push_samples(list(value))
      return self.push_tensors(_tensor_list_from_sequence(
          core, value, copy=copy, layout=layout, image_format=image_format))

    core.Run.push = run_push

  if not hasattr(core.Run, "try_push"):
    def run_try_push(self, value, copy: bool = False, layout=None, image_format=None):
      _reject_single_tensor_or_sample(core, value, "Run.try_push")
      if _sequence_all_samples(core, value):
        return self.try_push_samples(list(value))
      return self.try_push_tensors(_tensor_list_from_sequence(
          core, value, copy=copy, layout=layout, image_format=image_format))

    core.Run.try_push = run_try_push

  if not hasattr(core.Run, "run"):
    def run_execute(self, value, timeout_ms: int = -1):
      _reject_single_tensor_or_sample(core, value, "Run.run")
      if _sequence_all_samples(core, value):
        return self.run_samples(list(value), timeout_ms)
      return self.run_tensors(_tensor_list_from_sequence(core, value), timeout_ms)

    core.Run.run = run_execute

  def graph_add(self, item):
    if isinstance(item, core.Graph):
      return self.add_graph(item)
    if isinstance(item, core.Model):
      return self.add_model(item)
    return self.add_node(item)

  def graph_build(
      self,
      input_value=None,
      options=None,
      copy: bool = False,
      layout=None,
      image_format=None,
  ):
    if options is None:
      options = core.RunOptions()

    if input_value is None:
      return self.build_source(options)
    _reject_single_tensor_or_sample(core, input_value, "Graph.build")
    if _sequence_all_samples(core, input_value):
      return self.build_samples(list(input_value), options)
    return self.build_tensors(_tensor_list_from_sequence(
        core, input_value, copy=copy, layout=layout, image_format=image_format), options)

  def graph_run(self, input_value=None, options=None):
    if input_value is None:
      return self.run_source()
    if options is None:
      options = core.RunOptions()
    _reject_single_tensor_or_sample(core, input_value, "Graph.run")
    if _sequence_all_samples(core, input_value):
      return self.run_samples(list(input_value), options)
    return self.run_tensors(_tensor_list_from_sequence(core, input_value), options)

  core.Graph.add = graph_add
  if not has_native_build_overloads:
    core.Graph.build = graph_build
    core.Graph.run = graph_run


  def model_build(
      self,
      input_value=None,
      route_options=None,
      run_options=None,
      copy: bool = False,
  ):
    if input_value is None:
      if route_options is None:
        return raw_model_build(self)
      return raw_model_build_with_route_options(self, route_options)

    if route_options is None:
      route_options = core.ModelRouteOptions()
    if run_options is None:
      run_options = core.RunOptions()

    _reject_single_tensor_or_sample(core, input_value, "Model.build")
    if _sequence_all_samples(core, input_value):
      return self.build_samples(list(input_value), route_options, run_options)
    return self.build_tensors(
        _tensor_list_from_sequence(
            core,
            input_value,
            copy=copy,
            image_format=_model_image_format_hint(self),
        ),
        route_options,
        run_options,
    )

  if not has_native_build_overloads:
    core.Model.build = model_build
  if not hasattr(core.Model, "run"):
    def model_run(
        self,
        input_value,
        timeout_ms: int = -1,
        copy: bool = False,
    ):
      _reject_single_tensor_or_sample(core, input_value, "Model.run")
      image_format = _model_image_format_hint(self)
      if _sequence_all_samples(core, input_value):
        return self.run_samples(list(input_value), timeout_ms)
      return self.run_tensors(
          _tensor_list_from_sequence(core, input_value, copy=copy, image_format=image_format),
          timeout_ms,
      )

    core.Model.run = model_run

  if not hasattr(core.ModelRunner, "push"):
    def model_runner_push(self, value, copy: bool = False, layout=None, image_format=None):
      _reject_single_tensor_or_sample(core, value, "ModelRunner.push")
      if _sequence_all_samples(core, value):
        return self.push_samples(list(value))
      return self.push_tensors(_tensor_list_from_sequence(
          core, value, copy=copy, layout=layout, image_format=image_format))

    core.ModelRunner.push = model_runner_push

  if not hasattr(core.ModelRunner, "run"):
    def model_runner_run(self, value, timeout_ms: int = -1):
      _reject_single_tensor_or_sample(core, value, "ModelRunner.run")
      if _sequence_all_samples(core, value):
        return self.run_samples(list(value), timeout_ms)
      return self.run_tensors(_tensor_list_from_sequence(core, value), timeout_ms)

    core.ModelRunner.run = model_runner_run
