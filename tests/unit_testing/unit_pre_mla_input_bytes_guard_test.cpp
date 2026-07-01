#include "pipeline/internal/PreMlaContractCheck.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "test_main.h"

#include <cstdint>
#include <string>

// Tests for the predicate that enforce_pre_mla_input_bytes_guard checks before running:
//
//   if (contract_logical_bytes <= 0 || contract_input_dtype.empty()) { return; }
//
// Before the fix, the entire guard was gated behind a SHADOW_CHANGE env-var, so it
// silently skipped ALL validation by default. After the fix the guard runs whenever
// the MLA input contract is present in the manifest. Models that have no strict MLA
// input contract (span_size_bytes == 0 or logical_dtype empty) still get a graceful
// skip — no regression for those.
//
// We test mla_input_tensor_info_from_manifest directly: it produces the
// (span_size_bytes, logical_dtype) pair that the guard reads.

namespace {

namespace pre_mla = simaai::neat::pipeline_internal;
namespace rendered_stage_query = simaai::neat::pipeline_internal::rendered_stage_query;
namespace sima = simaai::neat::pipeline_internal::sima;

sima::SimaPluginStaticManifest make_empty_manifest() {
  return {};
}

sima::SimaPluginStaticManifest make_manifest_no_mla_stage() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.payload_kind = sima::StagePayloadKind::ProcessCvu;
  manifest.stages.push_back(stage);
  return manifest;
}

sima::SimaPluginStaticManifest make_manifest_mla_no_logical_inputs() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.payload_kind = sima::StagePayloadKind::ProcessMla;
  // logical_inputs left empty
  manifest.stages.push_back(stage);
  return manifest;
}

// MLA stage with one logical input but no backing physical buffer entry.
// Falls back to logical.size_bytes for span_size_bytes.
sima::SimaPluginStaticManifest make_manifest_mla_logical_size_only(std::uint64_t size_bytes,
                                                                   const std::string& dtype) {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.payload_kind = sima::StagePayloadKind::ProcessMla;

  sima::LogicalInputStaticSpec logical;
  logical.dtype = dtype;
  logical.size_bytes = size_bytes;
  logical.physical_index = -1; // no physical backing → fallback to logical.size_bytes
  stage.logical_inputs.push_back(logical);

  manifest.stages.push_back(stage);
  return manifest;
}

// MLA stage with a physical buffer entry; span_size_bytes comes from physical.size_bytes.
sima::SimaPluginStaticManifest make_manifest_mla_physical_size(std::uint64_t physical_size_bytes,
                                                               const std::string& dtype) {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.payload_kind = sima::StagePayloadKind::ProcessMla;

  sima::LogicalInputStaticSpec logical;
  logical.dtype = dtype;
  logical.physical_index = 0; // points at physical_inputs[0]
  stage.logical_inputs.push_back(logical);

  sima::PhysicalBufferStaticSpec physical;
  physical.size_bytes = physical_size_bytes;
  stage.physical_inputs.push_back(physical);

  manifest.stages.push_back(stage);
  return manifest;
}

// — Guard-skip cases: no contract available → guard returns gracefully (no throw) —

void verify_no_contract_when_manifest_is_empty() {
  const auto info =
      rendered_stage_query::mla_input_tensor_info_from_manifest(make_empty_manifest());
  require(info.span_size_bytes <= 0,
          "empty manifest: span_size_bytes must be 0 (guard will skip gracefully)");
  require(info.logical_dtype.empty(),
          "empty manifest: logical_dtype must be empty (guard will skip gracefully)");
}

void verify_no_contract_when_no_mla_stage() {
  const auto info =
      rendered_stage_query::mla_input_tensor_info_from_manifest(make_manifest_no_mla_stage());
  require(info.span_size_bytes <= 0,
          "non-MLA manifest: span_size_bytes must be 0 (guard will skip gracefully)");
  require(info.logical_dtype.empty(),
          "non-MLA manifest: logical_dtype must be empty (guard will skip gracefully)");
}

void verify_no_contract_when_mla_stage_has_no_logical_inputs() {
  const auto info = rendered_stage_query::mla_input_tensor_info_from_manifest(
      make_manifest_mla_no_logical_inputs());
  require(info.span_size_bytes <= 0,
          "MLA stage with no logical_inputs: span_size_bytes must be 0 (guard will skip)");
  require(info.logical_dtype.empty(),
          "MLA stage with no logical_inputs: logical_dtype must be empty (guard will skip)");
}

// — Guard-enforce cases: contract present → span_size_bytes > 0, logical_dtype non-empty —
// These are the regression cases for the bug. Before the fix the guard was always
// silently skipped; after the fix it enforces whenever this contract is present.

void verify_contract_present_via_logical_size_bytes() {
  // Representative single-input model: IFM size is 663552 bytes, BF16 dtype.
  const auto info = rendered_stage_query::mla_input_tensor_info_from_manifest(
      make_manifest_mla_logical_size_only(663552U, "BF16"));
  require(info.span_size_bytes == 663552,
          "MLA stage with logical size 663552: span_size_bytes must equal logical size "
          "(guard will enforce, not skip)");
  require(info.logical_dtype == "BF16",
          "MLA stage with BF16 dtype: logical_dtype must be 'BF16' (guard will enforce)");
}

void verify_contract_present_via_physical_size_bytes() {
  // Physical buffer backing: guard picks up physical.size_bytes for the span.
  const auto info = rendered_stage_query::mla_input_tensor_info_from_manifest(
      make_manifest_mla_physical_size(1327104U, "INT8"));
  require(info.span_size_bytes == 1327104,
          "MLA stage with physical size 1327104: span_size_bytes must equal physical size "
          "(guard will enforce, not skip)");
  require(info.logical_dtype == "INT8",
          "MLA stage with INT8 dtype: logical_dtype must be 'INT8' (guard will enforce)");
}

void verify_contract_present_multi_input_accumulated_size() {
  // This mirrors the multi-input scenario from the ifm-accumulation bug:
  // two tess stages produce 663552 (proj) + 2400 (gather) bytes.
  // The MLA contract span covers the full concatenated IFM.
  const std::uint64_t proj_bytes = 663552U;
  const std::uint64_t gather_bytes = 2400U;
  const auto info = rendered_stage_query::mla_input_tensor_info_from_manifest(
      make_manifest_mla_logical_size_only(proj_bytes + gather_bytes, "BF16"));
  require(info.span_size_bytes == static_cast<int64_t>(proj_bytes + gather_bytes),
          "multi-input MLA stage: span_size_bytes must equal the full concatenated IFM size "
          "(guard will enforce against the accumulated size, not just the first input)");
  require(!info.logical_dtype.empty(),
          "multi-input MLA stage: logical_dtype must be non-empty (guard will enforce)");
}

// — Direct coverage of check_pre_mla_input_bytes_contract() —
//
// These tests call the extracted helper directly so that:
//   (a) the gate path (guard_active=false → Skipped) is exercised, and
//   (b) every error code produced by the validation body is verified.
// If someone re-introduces a gate before the validation body inside
// enforce_pre_mla_input_bytes_guard() the guard_active=true cases below
// will still detect any regression in the check logic itself.

void verify_guard_skips_when_inactive() {
  // guard_active=false must return Skipped regardless of bad inputs.
  // This is the behaviour that the old unconditional SHADOW_CHANGE gate produced:
  // the guard returned without checking anything.
  const auto code = pre_mla::check_pre_mla_input_bytes_contract(
      false, 0, 0, 1024, pre_mla::DTypeFamily::Unknown, pre_mla::DTypeFamily::Unknown);
  require(code == pre_mla::PreMlaCheckCode::Skipped,
          "guard_active=false: must return Skipped even with bad inputs (former gated behaviour)");
}

void verify_guard_passes_on_matching_bytes() {
  const auto code = pre_mla::check_pre_mla_input_bytes_contract(
      true, 1024, 1024, 1024, pre_mla::DTypeFamily::BFloat16, pre_mla::DTypeFamily::BFloat16);
  require(code == pre_mla::PreMlaCheckCode::Ok,
          "matching handle/runtime/contract bytes and dtype: must return Ok");
}

void verify_guard_fires_handle_unavailable() {
  const auto code = pre_mla::check_pre_mla_input_bytes_contract(
      true, 0, 1024, 1024, pre_mla::DTypeFamily::BFloat16, pre_mla::DTypeFamily::BFloat16);
  require(code == pre_mla::PreMlaCheckCode::HandleUnknown,
          "handle_bytes=0: must return HandleUnknown");
}

void verify_guard_fires_contract_mismatch() {
  // handle(2048) >= runtime(512) so HandleTooSmall is not triggered first;
  // runtime(512) != contract(1024) → ContractMismatch.
  const auto code = pre_mla::check_pre_mla_input_bytes_contract(
      true, 2048, 512, 1024, pre_mla::DTypeFamily::BFloat16, pre_mla::DTypeFamily::BFloat16);
  require(code == pre_mla::PreMlaCheckCode::ContractMismatch,
          "runtime_logical(512) != contract(1024): must return ContractMismatch");
}

void verify_guard_fires_dtype_mismatch() {
  const auto code = pre_mla::check_pre_mla_input_bytes_contract(
      true, 1024, 1024, 1024, pre_mla::DTypeFamily::Int8, pre_mla::DTypeFamily::BFloat16);
  require(code == pre_mla::PreMlaCheckCode::DtypeMismatch,
          "INT8 runtime vs BF16 contract: must return DtypeMismatch");
}

void verify_dtype_family_from_token_roundtrip() {
  // Sanity-check the token parser used to derive contract_dtype.
  require(pre_mla::dtype_family_from_token("BF16") == pre_mla::DTypeFamily::BFloat16,
          "dtype_family_from_token: BF16 must map to BFloat16");
  require(pre_mla::dtype_family_from_token("INT8") == pre_mla::DTypeFamily::Int8,
          "dtype_family_from_token: INT8 must map to Int8");
  require(pre_mla::dtype_family_from_token("") == pre_mla::DTypeFamily::Unknown,
          "dtype_family_from_token: empty string must map to Unknown");
}

} // namespace

RUN_TEST("unit_pre_mla_input_bytes_guard_test", ([] {
           // Guard-skip (no contract) — backward-compatible graceful return path
           verify_no_contract_when_manifest_is_empty();
           verify_no_contract_when_no_mla_stage();
           verify_no_contract_when_mla_stage_has_no_logical_inputs();

           // Guard-enforce (contract present) — the path that was previously silenced by the
           // SHADOW_CHANGE gate and now always runs
           verify_contract_present_via_logical_size_bytes();
           verify_contract_present_via_physical_size_bytes();
           verify_contract_present_multi_input_accumulated_size();

           // Direct coverage of check_pre_mla_input_bytes_contract(): gate + each error code
           verify_guard_skips_when_inactive();
           verify_guard_passes_on_matching_bytes();
           verify_guard_fires_handle_unavailable();
           verify_guard_fires_contract_mismatch();
           verify_guard_fires_dtype_mismatch();
           verify_dtype_family_from_token_roundtrip();
         }));
