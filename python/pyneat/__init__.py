"""SiMa NEAT Python bindings."""

from . import _pyneat_core as _core
from ._pyneat_core import *
from ._wrappers import install_wrappers

def _set_doc(obj, doc: str) -> None:
  try:
    obj.__doc__ = doc
  except Exception:
    pass


install_wrappers(_core)

_set_doc(
  _core.Model,
  """Model loads a compiled MPK package and exposes reusable stage fragments.

  Use Model to assemble Sessions or run inference directly via build/run helpers.
  """,
)
_set_doc(
  _core.Session,
  """Session assembles Nodes/NodeGroups into a deterministic pipeline."""
)
_set_doc(
  _core.Run,
  """Run executes a built pipeline in sync or async mode."""
)
_set_doc(
  _core.Tensor,
  """Tensor represents typed payload data (shape, dtype, layout, device)."""
)
_set_doc(
  _core.Sample,
  """Sample wraps Tensor payloads with stream metadata."""
)

__version__ = _core.__version__
