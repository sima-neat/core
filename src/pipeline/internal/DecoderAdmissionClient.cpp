#include "pipeline/internal/DecoderAdmissionClient.h"

#include <iomanip>
#include <sstream>

namespace simaai::neat::pipeline_internal {

std::string decoder_admission_uuid_to_string(const std::array<std::uint8_t, 16>& uuid) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (std::size_t i = 0; i < uuid.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      oss << '-';
    }
    oss << std::setw(2) << static_cast<unsigned>(uuid[i]);
  }
  return oss.str();
}

const char* decoder_admission_tuning_name(std::uint32_t tuning) {
  switch (tuning) {
  case 0:
    return "default";
  case 1:
    return "low-memory";
  case 2:
    return "throughput-low-latency";
  case 3:
    return "auto";
  default:
    return "auto";
  }
}

} // namespace simaai::neat::pipeline_internal
