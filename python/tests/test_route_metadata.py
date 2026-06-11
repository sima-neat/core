"""Phase 1 high-value slice: output-identity route metadata ("which output is which").

Binds Tensor.route (a nested TensorRouteMeta) and the canonical Sample route fields. These let
Python correlate a returned tensor/sample back to its logical model output — the review's #2
priority. Round-trip tests verify the binding without needing inference (the runtime populates
these on pull; that path is covered by the runtime tests).
"""

from __future__ import annotations

import pyneat


def test_route_surface_present():
  assert hasattr(pyneat, "TensorRouteMeta")
  assert hasattr(pyneat.Tensor, "route")
  for field in ("logical_output_index", "memory_index", "route_slot", "segment_name"):
    assert hasattr(pyneat.Sample, field), field


def test_tensor_route_default_and_assign():
  tensor = pyneat.Tensor()
  # Every tensor carries a route metadata object (unset by default).
  assert isinstance(tensor.route, pyneat.TensorRouteMeta)
  assert tensor.route.logical_index == -1

  meta = pyneat.TensorRouteMeta()
  meta.logical_index = 2
  meta.name = "boxes"
  meta.backend_name = "detessdequant_out"
  meta.route_slot = 5
  meta.memory_index = 1
  meta.segment_name = "seg0"
  meta.physical_byte_offset = 256
  tensor.route = meta

  assert tensor.route.logical_index == 2
  assert tensor.route.name == "boxes"
  assert tensor.route.backend_name == "detessdequant_out"
  assert tensor.route.route_slot == 5
  assert tensor.route.memory_index == 1
  assert tensor.route.segment_name == "seg0"
  assert tensor.route.physical_byte_offset == 256


def test_tensor_route_member_mutation():
  tensor = pyneat.Tensor()
  tensor.route.logical_index = 3
  tensor.route.name = "scores"
  # Mutating the nested route object persists on the parent tensor.
  assert tensor.route.logical_index == 3
  assert tensor.route.name == "scores"


def test_sample_route_fields_roundtrip():
  sample = pyneat.Sample()
  assert sample.logical_output_index == -1
  sample.logical_output_index = 1
  sample.memory_index = 4
  sample.route_slot = 7
  sample.segment_name = "post"
  assert sample.logical_output_index == 1
  assert sample.memory_index == 4
  assert sample.route_slot == 7
  assert sample.segment_name == "post"
