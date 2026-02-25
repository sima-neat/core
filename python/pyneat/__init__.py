"""SiMa NEAT Python bindings."""

import os as _os
import sys as _sys

# Load _pyneat_core with RTLD_GLOBAL so pyneat and NEAT GStreamer plugins share
# a single symbol namespace in this process. With RTLD_LOCAL (CPython default
# on POSIX), allocator/buffer-pool GObject types may be registered in separate
# loader scopes, which can cause plugin initialization failures and timeouts.
_old_dl_flags = _sys.getdlopenflags()
_sys.setdlopenflags(_os.RTLD_NOW | _os.RTLD_GLOBAL)

from . import _pyneat_core as _core

# Restore the previous dlopen flags for subsequent imports.
_sys.setdlopenflags(_old_dl_flags)
del _old_dl_flags

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
