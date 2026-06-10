"""Phase 1 (plan S2/S3/S4): AdvancedExecutionOptions — jargon-free execution surface.

The merge semantics (resolve into legacy fields; explicit false beats truthy default) are
verified at the C++ level (header-only resolve_advanced_execution); these tests cover the Python
binding surface, the S2 "no raw fields on GraphOptions" tiering, and an end-to-end Graph build.
"""

from __future__ import annotations

import pyneat


def test_advanced_execution_options_roundtrip():
  ae = pyneat.AdvancedExecutionOptions()
  # Optionals default to unset (None), not a value.
  assert ae.inference_async is None
  assert ae.preprocess_target is None
  ae.preprocess_target = "EV74"
  ae.postprocess_target = "A65"
  ae.preprocess_async = True
  ae.inference_async = False  # explicit false must round-trip as False, not None
  ae.inference_output_buffers = 8
  ae.defer_output_cache_sync = False
  ae.internal_queue_depth = 3
  assert ae.preprocess_target == "EV74"
  assert ae.postprocess_target == "A65"
  assert ae.preprocess_async is True
  assert ae.inference_async is False
  assert ae.inference_output_buffers == 8
  assert ae.defer_output_cache_sync is False
  assert ae.internal_queue_depth == 3


def test_prepared_runner_options():
  pr = pyneat.PreparedRunnerOptions()
  pr.mode = "dequant"
  pr.ring_depth = 2
  pr.profile = True
  pr.dequant_flags = "fused,half"
  assert pr.mode == "dequant"
  assert pr.ring_depth == 2
  assert pr.profile is True
  ae = pyneat.AdvancedExecutionOptions()
  ae.prepared_runner = pr
  assert ae.prepared_runner.mode == "dequant"


def test_graph_options_execution_surface_is_advanced_only():
  go = pyneat.GraphOptions()
  assert hasattr(go, "advanced_execution")
  assert hasattr(go, "verbose")
  go.advanced_execution.inference_async = False
  assert go.advanced_execution.inference_async is False  # nested in-place mutation persists
  # S2: raw legacy execution fields are NOT bound on GraphOptions.
  for raw in ("processcvu", "processmla", "async_queue_depth", "prepared_runner"):
    assert not hasattr(go, raw), raw


def test_model_options_have_advanced_execution():
  assert hasattr(pyneat.ModelOptions(), "advanced_execution")
  assert hasattr(pyneat.ModelRouteOptions(), "advanced_execution")


def test_graph_builds_with_advanced_execution():
  # End-to-end: the Graph constructor folds advanced_execution into the legacy fields (resolve).
  # This exercises that path without error.
  go = pyneat.GraphOptions()
  go.advanced_execution.preprocess_target = "EV74"
  go.advanced_execution.inference_async = False
  go.advanced_execution.internal_queue_depth = 4
  graph = pyneat.Graph(options=go)
  graph.add(pyneat.nodes.input())
  graph.add(pyneat.nodes.output())
  assert isinstance(graph.describe_backend(), str)
