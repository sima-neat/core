#include "pcie_genai_models.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <system_error>

namespace simaai::neat::pcie::genai::tools {

namespace {

// Drop a trailing '#' comment and surrounding whitespace from one config line.
std::string strip(std::string s) {
  const std::size_t hash = s.find('#');
  if (hash != std::string::npos) {
    s.erase(hash);
  }
  const std::size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const std::size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

// Total size of a folder's files, best-effort: unreadable entries are skipped.
std::uintmax_t folder_size(const std::filesystem::path& dir) {
  std::uintmax_t total = 0;
  std::error_code ec;
  std::filesystem::recursive_directory_iterator it(dir, ec), end;
  for (; !ec && it != end; it.increment(ec)) {
    if (it->is_regular_file(ec)) {
      const std::uintmax_t bytes = it->file_size(ec);
      if (!ec) {
        total += bytes;
      }
    }
  }
  return total;
}

std::string human_size(std::uintmax_t bytes) {
  constexpr std::array<const char*, 5> units{"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < units.size()) {
    value /= 1024.0;
    ++unit;
  }
  std::array<char, 32> buf{};
  std::snprintf(buf.data(), buf.size(), unit == 0 ? "%.0f %s" : "%.1f %s", value, units[unit]);
  return buf.data();
}

} // namespace

std::optional<std::string> serve_root_path(const std::string& conf_text,
                                           const std::string& serve_name) {
  std::istringstream in(conf_text);
  std::string line;
  bool in_serve = false;
  while (std::getline(in, line)) {
    const std::string trimmed = strip(line);
    if (trimmed.empty()) {
      continue;
    }
    if (trimmed.front() == '[' && trimmed.back() == ']') {
      in_serve = trimmed == "[serve]";
      continue;
    }
    if (!in_serve) {
      continue;
    }
    const std::size_t eq = trimmed.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    if (strip(trimmed.substr(0, eq)) == serve_name) {
      return strip(trimmed.substr(eq + 1));
    }
  }
  return std::nullopt;
}

std::string model_type_from_vlm_config(const std::string& json_text) {
  const nlohmann::json j = nlohmann::json::parse(json_text, nullptr, /*allow_exceptions=*/false);
  if (j.is_object()) {
    const auto it = j.find("model_type");
    if (it != j.end() && it->is_string()) {
      return it->get<std::string>();
    }
  }
  return {};
}

std::vector<ModelListing> list_models_in(const std::filesystem::path& serve_root) {
  std::vector<ModelListing> models;
  std::error_code ec;
  std::filesystem::directory_iterator it(serve_root, ec), end;
  for (; !ec && it != end; it.increment(ec)) {
    const std::filesystem::path dir = it->path();
    // A model folder has the same layout `llima run` requires.
    if (!std::filesystem::is_directory(dir / "devkit", ec) ||
        !std::filesystem::is_directory(dir / "elf_files", ec)) {
      continue;
    }
    ModelListing model;
    model.name = dir.filename().string();
    std::ifstream cfg(dir / "devkit" / "vlm_config.json");
    if (cfg) {
      const std::string body((std::istreambuf_iterator<char>(cfg)),
                             std::istreambuf_iterator<char>());
      model.type = model_type_from_vlm_config(body);
    }
    model.size_bytes = folder_size(dir);
    models.push_back(std::move(model));
  }
  std::sort(models.begin(), models.end(),
            [](const ModelListing& a, const ModelListing& b) { return a.name < b.name; });
  return models;
}

std::string format_model_listing(const std::vector<ModelListing>& models,
                                 const std::string& serve_root) {
  std::ostringstream out;
  if (models.empty()) {
    out << "No models served over PCIe under " << serve_root << ".\n";
  } else {
    out << "Models served over PCIe (from " << serve_root << "):\n\n";
    // Width the name column to the longest name, so the table stays aligned.
    std::size_t name_w = 4; // len("NAME")
    for (const ModelListing& m : models) {
      name_w = std::max(name_w, m.name.size());
    }
    out << "  " << std::left << std::setw(static_cast<int>(name_w)) << "NAME"
        << "  " << std::setw(14) << "TYPE"
        << "SIZE\n";
    for (const ModelListing& m : models) {
      out << "  " << std::left << std::setw(static_cast<int>(name_w)) << m.name << "  "
          << std::setw(14) << (m.type.empty() ? "unknown" : m.type) << human_size(m.size_bytes)
          << "\n";
    }
  }
  out << "\nCopy a model folder into " << serve_root
      << " to make it available,\nthen run: pcie-genai --model <NAME> ...\n";
  return out.str();
}

std::string format_final_stats(const simaai::neat::genai::GenerationMetrics& metrics,
                               const std::uint32_t dropped_events,
                               const std::string& finish_reason) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2);
  out << "[" << finish_reason << " | " << metrics.generated_tokens << " tokens | TTFT "
      << metrics.time_to_first_token_s << " s | " << metrics.tokens_per_second
      << " tok/s | dropped " << dropped_events << "]";
  if (dropped_events > 0) {
    out << "\npcie-genai: WARNING: " << dropped_events
        << " token event(s) were dropped in transit; the answer above is missing text.";
  }
  return out.str();
}

} // namespace simaai::neat::pcie::genai::tools
