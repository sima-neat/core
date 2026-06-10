"""Phase 3 (plan slice): model / preprocess introspection bindings.

Surface tests assert the new types/methods exist and that the S1 tiering holds (the deep
resolved-plan contract tree lives under ``pyneat.advanced``, NOT at top level). The functional
test exercises the methods against a real model fixture and skips when none is available.
"""

from __future__ import annotations

import pytest

import model_fixture_helpers as model_fixtures
import pyneat


# ── Surface (no model required) ──────────────────────────────────────────────────────────────


def test_primary_introspection_types_present():
  # Direct returns of the primary Model methods live at top level.
  assert hasattr(pyneat, "ModelInfo")
  assert hasattr(pyneat, "PreprocessRequirements")
  # ModelInfo's nested diagnostic structs are scoped under ModelInfo, not the top level.
  for nested in ("RouteNeeds", "RouteCapabilities", "RouteSelection", "OutputTopology"):
    assert hasattr(pyneat.ModelInfo, nested), nested


def test_model_introspection_methods_present():
  for method in (
      "info",
      "compiled_batch_size",
      "preprocess_requirements",
      "resolved_preprocess_plan",
      "preprocess_plan",
  ):
    assert hasattr(pyneat.Model, method), method


def test_resolved_plan_tree_is_advanced_tier_only():
  # S1: "advanced = by namespace, not by docs." The contract tree is reachable only via
  # pyneat.advanced, never leaked to the top-level beginner surface.
  assert hasattr(pyneat, "advanced")
  for name in (
      "ResolvedPreprocessPlan",
      "PreprocessContract",
      "PreprocessMetaContract",
      "PreprocessExplicitKnobs",
      "PreprocessGraphFamily",
  ):
    assert hasattr(pyneat.advanced, name), name
    assert not hasattr(pyneat, name), f"{name} must not leak to the top-level surface (S1)"


def test_preprocess_graph_family_values():
  fam = pyneat.advanced.PreprocessGraphFamily
  # PreprocessPlan.h has FIVE values including Disabled=0 (plan correction S12).
  for member in ("Disabled", "Preproc", "Quant", "Tess", "QuantTess"):
    assert hasattr(fam, member), member


# ── Functional (requires a real model fixture; skips otherwise) ──────────────────────────────


@pytest.fixture(scope="module")
def resnet_model():
  path = model_fixtures.strict_model_tar_path("SIMA_RESNET50_TAR")
  return pyneat.Model(str(path))


def test_info_snapshot(resnet_model):
  info = resnet_model.info()
  assert isinstance(info, pyneat.ModelInfo)
  assert isinstance(info.model_name, str)
  assert isinstance(info.mpk_json_path, str)
  assert isinstance(info.pre_kernels, list)
  assert isinstance(info.post_kernels, list)
  # Topology sanity: a classifier emits at least one logical output.
  assert info.output_topology.logical_outputs >= 1
  # Nested structs are real typed objects, not opaque handles.
  assert isinstance(info.needs.pre_quantization, bool)
  assert isinstance(info.capabilities.has_post_boxdecode, bool)
  assert isinstance(info.selection.preprocess_graph, str)


def test_compiled_batch_size(resnet_model):
  assert resnet_model.compiled_batch_size() >= 1


def test_preprocess_requirements(resnet_model):
  req = resnet_model.preprocess_requirements()
  assert isinstance(req, pyneat.PreprocessRequirements)
  assert isinstance(req.input_format, str)
  assert isinstance(req.output_dtype, str)
  assert isinstance(req.output_shape, list)
  assert isinstance(req.quant_needed, bool)


def test_preprocess_plan_alias_and_tree(resnet_model):
  plan = resnet_model.preprocess_plan()
  assert isinstance(plan, pyneat.advanced.ResolvedPreprocessPlan)
  # The friendly alias returns the same plan type as the canonical method.
  assert isinstance(resnet_model.resolved_preprocess_plan(), type(plan))
  assert isinstance(plan.graph_family, pyneat.advanced.PreprocessGraphFamily)
  assert isinstance(plan.mla_contract, pyneat.advanced.PreprocessContract)
  assert isinstance(plan.ingress_contracts, list)
  # to_debug_string()/__repr__ render a human-readable summary.
  assert isinstance(plan.to_debug_string(), str)
  assert plan.to_debug_string() == repr(plan)
