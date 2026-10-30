#pragma once

#include <gst/gst.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Test-only, cold preparation diagnostics. No payload ownership or hashing work
// survives into the encoder's measured frame loop.
namespace sima_codec_diagnostics {
using Json = nlohmann::json;
using Checksum = std::unique_ptr<GChecksum, decltype(&g_checksum_free)>;

inline Checksum checksum() {
  Checksum result(g_checksum_new(G_CHECKSUM_SHA256), g_checksum_free);
  if (!result)
    throw std::runtime_error("cannot allocate codec diagnostic checksum");
  return result;
}

inline std::string payload_hash(const std::vector<std::uint8_t>& bytes) {
  auto sum = checksum();
  g_checksum_update(sum.get(), bytes.data(), bytes.size());
  return g_checksum_get_string(sum.get());
}

inline Json file_identity(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read codec diagnostic file: " + path.string());
  auto sum = checksum();
  std::array<char, 8192> buffer{};
  std::uint64_t size = 0;
  while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
    const auto count = input.gcount();
    g_checksum_update(sum.get(), reinterpret_cast<const guchar*>(buffer.data()), count);
    size += static_cast<std::uint64_t>(count);
  }
  if (!input.eof())
    throw std::runtime_error("failed reading codec diagnostic file: " + path.string());
  return {{"path", path.string()}, {"size", size}, {"sha256", g_checksum_get_string(sum.get())}};
}

inline Json parser_identity(GstElement* parser) {
  if (!parser)
    throw std::runtime_error("codec diagnostics require the actual parser element");
  auto* factory = gst_element_get_factory(parser);
  if (!factory)
    throw std::runtime_error("codec parser has no factory");
  std::unique_ptr<GstPlugin, decltype(&gst_object_unref)> plugin(
      gst_plugin_feature_get_plugin(GST_PLUGIN_FEATURE(factory)), gst_object_unref);
  if (!plugin)
    throw std::runtime_error("codec parser factory has no plugin");
  std::unique_ptr<gchar, decltype(&g_free)> runtime_version(gst_version_string(), g_free);
  Json result = {{"factory", gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory))},
                 {"plugin", gst_plugin_get_name(plugin.get())},
                 {"plugin_version", gst_plugin_get_version(plugin.get())},
                 {"gstreamer_version", runtime_version.get()}};
  const char* filename = gst_plugin_get_filename(plugin.get());
  result["file"] = filename ? file_identity(filename) : Json(nullptr);
  return result;
}

// Preserve the emitted string separately. Structured fields are explanatory
// only: never feed them back into caps or the accepted input fingerprint.
inline Json caps_fields(const std::string& text) {
  if (text.empty())
    return nullptr;
  std::unique_ptr<GstCaps, decltype(&gst_caps_unref)> caps(gst_caps_from_string(text.c_str()),
                                                           gst_caps_unref);
  if (!caps)
    return {{"parse_error", true}};
  Json structures = Json::array();
  for (guint i = 0; i < gst_caps_get_size(caps.get()); ++i) {
    const auto* structure = gst_caps_get_structure(caps.get(), i);
    Json fields = Json::object();
    for (int j = 0; j < gst_structure_n_fields(structure); ++j) {
      const char* name = gst_structure_nth_field_name(structure, j);
      const auto* value = gst_structure_get_value(structure, name);
      std::unique_ptr<gchar, decltype(&g_free)> serialized(gst_value_serialize(value), g_free);
      fields[name] = {{"type", G_VALUE_TYPE_NAME(value)},
                      {"value", serialized ? Json(serialized.get()) : Json(nullptr)}};
    }
    structures.push_back({{"name", gst_structure_get_name(structure)}, {"fields", fields}});
  }
  return structures;
}

template <typename Frames>
Json access_units(const Frames& frames, const std::vector<std::string>& emitted_caps) {
  if (frames.size() != emitted_caps.size())
    throw std::runtime_error("codec diagnostic AU/caps count mismatch");
  Json result = Json::array();
  for (std::size_t i = 0; i < frames.size(); ++i) {
    result.push_back({{"index", i},
                      {"size", frames[i].bytes.size()},
                      {"sha256", payload_hash(frames[i].bytes)},
                      {"emitted_caps", emitted_caps[i]},
                      {"effective_caps", frames[i].caps},
                      {"caps_fields", caps_fields(emitted_caps[i])}});
  }
  return result;
}
} // namespace sima_codec_diagnostics
