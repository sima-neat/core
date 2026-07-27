#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

inline constexpr std::uint32_t kDecoderAdmissionPolicyZeroCopyOutput = 1u << 8;
inline constexpr std::uint32_t kDecoderAdmissionPolicyNoOutputCopy = 1u << 9;
inline constexpr std::uint32_t kDecoderAdmissionCodecH264 = 101;
inline constexpr std::uint32_t kDecoderAdmissionCodecH265 = 102;

struct DecoderAdmissionStreamRequest {
  std::uint32_t stream_index = 0;
  std::uint32_t codec = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps_num = 0;
  std::uint32_t fps_den = 1;
  std::uint32_t stream_mode = 0;
  std::uint32_t requested_policy = 0;
};

struct DecoderAdmissionLease {
  std::uint32_t stream_index = 0;
  std::uint32_t resolved_output_buffers = 0;
  std::uint32_t resolved_input_buffers = 0;
  std::uint32_t resolved_tuning = 0;
  std::uint64_t lease_token_hi = 0;
  std::uint64_t lease_token_lo = 0;
  std::uint64_t estimated_reserved_bytes = 0;
};

struct DecoderAdmissionResult {
  bool admitted = false;
  bool endpoint_missing = false;
  std::string error;
  std::array<std::uint8_t, 16> group_uuid{};
  std::uint64_t estimated_reserved_bytes = 0;
  std::vector<DecoderAdmissionLease> leases;
};

bool decoder_admission_endpoint_available();
DecoderAdmissionResult
admit_decoder_graph(const std::vector<DecoderAdmissionStreamRequest>& streams,
                    bool dry_run = false);
bool release_decoder_graph(const std::array<std::uint8_t, 16>& group_uuid, std::string* error);
std::string decoder_admission_uuid_to_string(const std::array<std::uint8_t, 16>& uuid);
const char* decoder_admission_tuning_name(std::uint32_t tuning);
const char* decoder_admission_status_name(std::uint32_t status);

} // namespace simaai::neat::pipeline_internal
