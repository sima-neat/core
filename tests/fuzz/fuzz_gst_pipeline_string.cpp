#include <gst/gst.h>

#include <cstddef>
#include <cstdint>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (!data || size == 0 || size > (1U << 15)) {
    return 0;
  }

  static bool gst_ready = false;
  if (!gst_ready) {
    gst_init(nullptr, nullptr);
    gst_ready = true;
  }

  const std::string pipeline(reinterpret_cast<const char*>(data), size);

  GError* err = nullptr;
  GstElement* element = gst_parse_launch(pipeline.c_str(), &err);
  if (err) {
    g_error_free(err);
  }
  if (element) {
    gst_object_unref(element);
  }

  return 0;
}
