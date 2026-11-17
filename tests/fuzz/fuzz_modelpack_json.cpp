#include "pipeline/internal/sima/MpkContract.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (!data || size == 0 || size > (1U << 20)) {
    return 0;
  }

  const std::string_view payload(reinterpret_cast<const char*>(data), size);
  std::string diagnostic;
  try {
    (void)simaai::neat::pipeline_internal::sima::load_mpk_contract_from_json(
        payload, "fuzz_mpk.json", &diagnostic);
  } catch (const nlohmann::json::exception&) {
    // Existing field/type validation can throw as well as return nullopt.
  } catch (const std::runtime_error&) {
    // Graph validation rejects incomplete contracts through this exception.
    // Allocation failures, logic errors and sanitizer failures remain visible.
  }

  return 0;
}
