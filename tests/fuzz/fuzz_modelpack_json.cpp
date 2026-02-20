#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (!data || size == 0 || size > (1U << 20)) {
    return 0;
  }

  try {
    const std::string payload(reinterpret_cast<const char*>(data), size);
    nlohmann::json j = nlohmann::json::parse(payload, nullptr, true, true);
    if (j.is_object()) {
      (void)j.dump();
      if (j.contains("pipelines") && j["pipelines"].is_array()) {
        for (const auto& p : j["pipelines"]) {
          if (p.is_object() && p.contains("sequence")) {
            (void)p["sequence"].is_array();
          }
        }
      }
    }
  } catch (...) {
    // Fuzz harness must be exception-safe.
  }

  return 0;
}
