#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/TensorMath.h"

#include <cstdint>
#include <string>

namespace simaai::neat::pipeline_internal {

enum class DTypeFamily {
  Unknown = 0,
  Int8,
  Int16,
  Int32,
  BFloat16,
  Float32,
  Float64,
};

inline const char* dtype_family_name(DTypeFamily family) {
  switch (family) {
  case DTypeFamily::Int8:
    return "INT8";
  case DTypeFamily::Int16:
    return "INT16";
  case DTypeFamily::Int32:
    return "INT32";
  case DTypeFamily::BFloat16:
    return "BF16";
  case DTypeFamily::Float32:
    return "FP32";
  case DTypeFamily::Float64:
    return "FP64";
  case DTypeFamily::Unknown:
    break;
  }
  return "UNKNOWN";
}

inline DTypeFamily dtype_family_from_token(std::string token) {
  token = upper_copy(std::move(token));
  if (token.empty()) {
    return DTypeFamily::Unknown;
  }
  if (token.find("BF16") != std::string::npos || token.find("BFLOAT16") != std::string::npos) {
    return DTypeFamily::BFloat16;
  }
  if (token.find("INT8") != std::string::npos || token.find("UINT8") != std::string::npos) {
    return DTypeFamily::Int8;
  }
  if (token.find("INT16") != std::string::npos || token.find("UINT16") != std::string::npos) {
    return DTypeFamily::Int16;
  }
  if (token.find("INT32") != std::string::npos) {
    return DTypeFamily::Int32;
  }
  if (token.find("FP64") != std::string::npos || token.find("FLOAT64") != std::string::npos) {
    return DTypeFamily::Float64;
  }
  if (token.find("FP32") != std::string::npos || token.find("FLOAT32") != std::string::npos) {
    return DTypeFamily::Float32;
  }
  return DTypeFamily::Unknown;
}

// Result codes from check_pre_mla_input_bytes_contract().
enum class PreMlaCheckCode {
  Ok,
  Skipped,          // guard_active was false — former SHADOW_CHANGE gate behaviour
  HandleUnknown,    // handle_bytes <= 0
  RuntimeUnknown,   // runtime_logical_bytes <= 0
  DtypeMismatch,    // runtime_dtype != contract_dtype (both known)
  HandleTooSmall,   // handle_bytes < runtime_logical_bytes
  ContractMismatch, // runtime_logical_bytes != contract_logical_bytes
};

// Pure validation of the pre-MLA input byte contract.
//
// guard_active is the former shadow_change_env_enabled() gate, now a parameter so the
// decision logic — including the skip path — is directly testable without env-var side effects.
// Returns Skipped when !guard_active (matching the old gated behaviour), Ok on success, or the
// first failing code. Callers handle error reporting and throwing.
inline PreMlaCheckCode check_pre_mla_input_bytes_contract(bool guard_active, int64_t handle_bytes,
                                                          int64_t runtime_logical_bytes,
                                                          int64_t contract_logical_bytes,
                                                          DTypeFamily runtime_dtype,
                                                          DTypeFamily contract_dtype) {
  if (!guard_active) {
    return PreMlaCheckCode::Skipped;
  }
  if (handle_bytes <= 0) {
    return PreMlaCheckCode::HandleUnknown;
  }
  if (runtime_logical_bytes <= 0) {
    return PreMlaCheckCode::RuntimeUnknown;
  }
  if (contract_dtype != DTypeFamily::Unknown && runtime_dtype != DTypeFamily::Unknown &&
      runtime_dtype != contract_dtype) {
    return PreMlaCheckCode::DtypeMismatch;
  }
  if (handle_bytes < runtime_logical_bytes) {
    return PreMlaCheckCode::HandleTooSmall;
  }
  if (runtime_logical_bytes != contract_logical_bytes) {
    return PreMlaCheckCode::ContractMismatch;
  }
  return PreMlaCheckCode::Ok;
}

} // namespace simaai::neat::pipeline_internal
