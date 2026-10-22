#include "gst/internal/GstParseLaunch.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (!data || size == 0 || size > (1U << 15)) {
    return 0;
  }

  static const bool gst_ready = [] {
    gst_init(nullptr, nullptr);
    return true;
  }();
  (void)gst_ready;

  const std::string_view pipeline(reinterpret_cast<const char*>(data), size);
  // Parse and inspect only; ParsedLaunch retires partial roots and errors too.
  // No pipeline is started and no model is loaded.
  const auto parsed = simaai::neat::gst::parse_launch(pipeline);
  (void)simaai::neat::gst::inventory_elements(parsed.get());

  return 0;
}
