#include "mpk/MpKLoader.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace simaai::neat::mpk {
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

enum class TarMode { Gzip, Plain };

enum class EntryClass { Directory, Json, SharedObject, Elf, Auxiliary };

struct TarEntry {
  std::string raw_path;
  std::string normalized_path;
  char type = '?';
  std::uint64_t size_bytes = 0;
  EntryClass entry_class = EntryClass::Directory;
};

struct ValidatedArchive {
  TarMode mode = TarMode::Gzip;
  std::vector<TarEntry> entries;
  MpKManifest manifest;
};

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (char c : value) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out.push_back(c);
    }
  }
  out += "'";
  return out;
}

bool parse_uint64(const std::string& s, std::uint64_t* out) {
  if (!out || s.empty())
    return false;
  std::uint64_t value = 0;
  for (char c : s) {
    if (c < '0' || c > '9')
      return false;
    const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10ULL) {
      return false;
    }
    value = value * 10ULL + digit;
  }
  *out = value;
  return true;
}

std::string strip_suffix(std::string value, const std::string& suffix) {
  if (value.size() < suffix.size())
    return value;
  if (value.compare(value.size() - suffix.size(), suffix.size(), suffix) != 0)
    return value;
  value.resize(value.size() - suffix.size());
  return value;
}

std::string package_name_from_archive(const fs::path& archive_path) {
  std::string name = archive_path.filename().string();
  name = strip_suffix(name, ".tar.gz");
  name = strip_suffix(name, ".tgz");
  name = strip_suffix(name, ".mpk");
  name = strip_suffix(name, ".tar");
  if (name.empty()) {
    name = "package";
  }
  return name;
}

[[noreturn]] void throw_mpk(ErrorClass code, const std::string& message) {
  throw MpKError(code, message);
}

bool mode_likely_gzip(const fs::path& archive_path) {
  const std::string name = archive_path.filename().string();
  return name.size() >= 4 &&
         (name.ends_with(".tgz") || name.ends_with(".gz") || name.ends_with(".mpk"));
}

std::vector<std::string> tar_list_verbose(const std::string& archive_path, TarMode mode) {
  const std::string cmd = std::string("tar --numeric-owner --full-time -tv") +
                          ((mode == TarMode::Gzip) ? "z" : "") + "f " + shell_quote(archive_path) +
                          " 2>/dev/null";

  FILE* pipe = ::popen(cmd.c_str(), "r");
  if (!pipe) {
    throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: failed to open tar listing pipe");
  }

  std::vector<std::string> lines;
  char buffer[4096];
  while (::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    lines.emplace_back(buffer);
  }

  const int rc = ::pclose(pipe);
  if (rc != 0) {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: tar listing failed for " + archive_path);
  }
  return lines;
}

std::vector<std::string> tar_list_verbose_autodetect(const std::string& archive_path,
                                                     TarMode* mode_out) {
  const bool prefer_gzip = mode_likely_gzip(fs::path(archive_path));

  for (int attempt = 0; attempt < 2; ++attempt) {
    TarMode mode = TarMode::Plain;
    if (prefer_gzip) {
      mode = (attempt == 0) ? TarMode::Gzip : TarMode::Plain;
    } else {
      mode = (attempt == 0) ? TarMode::Plain : TarMode::Gzip;
    }

    try {
      auto lines = tar_list_verbose(archive_path, mode);
      if (mode_out) {
        *mode_out = mode;
      }
      return lines;
    } catch (const MpKError&) {
      if (attempt == 1)
        throw;
    }
  }

  throw_mpk(ErrorClass::InvalidArchive,
            "invalid_archive: unable to read tar stream for " + archive_path);
}

std::string trim(std::string s) {
  std::size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
    ++b;
  std::size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
    --e;
  return s.substr(b, e - b);
}

bool is_valid_utf8(const std::string& text) {
  std::size_t i = 0;
  while (i < text.size()) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (c <= 0x7F) {
      ++i;
      continue;
    }

    std::size_t need = 0;
    std::uint32_t code = 0;
    std::uint32_t min_value = 0;
    if (c >= 0xC2 && c <= 0xDF) {
      need = 1;
      code = c & 0x1FU;
      min_value = 0x80U;
    } else if (c >= 0xE0 && c <= 0xEF) {
      need = 2;
      code = c & 0x0FU;
      min_value = 0x800U;
    } else if (c >= 0xF0 && c <= 0xF4) {
      need = 3;
      code = c & 0x07U;
      min_value = 0x10000U;
    } else {
      return false;
    }

    if (i + need >= text.size())
      return false;
    for (std::size_t k = 0; k < need; ++k) {
      const unsigned char cc = static_cast<unsigned char>(text[i + 1 + k]);
      if ((cc & 0xC0U) != 0x80U)
        return false;
      code = (code << 6U) | static_cast<std::uint32_t>(cc & 0x3FU);
    }

    if (code < min_value)
      return false; // overlong
    if (code >= 0xD800U && code <= 0xDFFFU)
      return false; // surrogate
    if (code > 0x10FFFFU)
      return false;
    i += need + 1;
  }
  return true;
}

bool contains_utf8_sequence(const std::string& text, const char* seq) {
  return text.find(seq) != std::string::npos;
}

bool contains_tar_octal_escape(const std::string& path) {
  if (path.size() < 4)
    return false;
  for (std::size_t i = 0; i + 3 < path.size(); ++i) {
    if (path[i] != '\\')
      continue;
    const bool d0 = (path[i + 1] >= '0' && path[i + 1] <= '7');
    const bool d1 = (path[i + 2] >= '0' && path[i + 2] <= '7');
    const bool d2 = (path[i + 3] >= '0' && path[i + 3] <= '7');
    if (d0 && d1 && d2)
      return true;
  }
  return false;
}

void replace_utf8_sequence(std::string& text, const char* seq, char replacement) {
  const std::string needle(seq);
  for (;;) {
    const std::size_t pos = text.find(needle);
    if (pos == std::string::npos)
      break;
    text.replace(pos, needle.size(), std::string(1, replacement));
  }
}

void normalize_unicode_path_confusables(std::string& path, const MpKLoaderOptions& opt) {
  struct Utf8Confusable {
    const char* bytes;
    char normalized;
    const char* label;
  };
  static const Utf8Confusable kConfusables[] = {
      {"\xEF\xBC\x8F", '/', "fullwidth slash U+FF0F"},
      {"\xE2\x88\x95", '/', "division slash U+2215"},
      {"\xEF\xBC\xBC", '\\', "fullwidth backslash U+FF3C"},
      {"\xEF\xBC\x8E", '.', "fullwidth dot U+FF0E"},
      {"\xE2\x80\xA4", '.', "dot leader U+2024"},
      {"\xEF\xB9\x92", '.', "small full stop U+FE52"},
  };

  for (const auto& c : kConfusables) {
    if (!contains_utf8_sequence(path, c.bytes))
      continue;
    if (opt.reject_unicode_path_confusables) {
      throw_mpk(ErrorClass::PathTraversal,
                std::string("path_traversal: unicode path confusable detected (") + c.label + ")");
    }
    replace_utf8_sequence(path, c.bytes, c.normalized);
  }
}

std::string normalize_entry_path(const std::string& raw_path, const MpKLoaderOptions& opt) {
  if (raw_path.empty()) {
    throw_mpk(ErrorClass::PathTraversal, "path_traversal: empty archive entry path");
  }

  if (opt.reject_invalid_utf8_paths && contains_tar_octal_escape(raw_path)) {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: archive entry path contains escaped non-UTF8 bytes");
  }

  if (opt.reject_invalid_utf8_paths && !is_valid_utf8(raw_path)) {
    throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: archive entry path is not valid UTF-8");
  }

  std::string normalized = raw_path;
  normalize_unicode_path_confusables(normalized, opt);
  for (char& c : normalized) {
    if (c == '\\')
      c = '/';
  }

  while (normalized.rfind("./", 0) == 0) {
    normalized.erase(0, 2);
  }

  if (normalized.empty()) {
    if (raw_path == "." || raw_path == "./") {
      return ".";
    }
    throw_mpk(ErrorClass::PathTraversal, "path_traversal: empty normalized archive entry path");
  }
  if (normalized.front() == '/') {
    throw_mpk(ErrorClass::PathTraversal,
              "path_traversal: absolute archive path is forbidden: " + raw_path);
  }

  if (normalized.size() >= 2 && std::isalpha(static_cast<unsigned char>(normalized[0])) &&
      normalized[1] == ':') {
    throw_mpk(ErrorClass::PathTraversal,
              "path_traversal: drive-prefixed archive path is forbidden: " + raw_path);
  }

  std::stringstream ss(normalized);
  std::string segment;
  while (std::getline(ss, segment, '/')) {
    if (segment.empty() || segment == ".")
      continue;
    if (segment == "..") {
      throw_mpk(ErrorClass::PathTraversal,
                "path_traversal: parent traversal segment forbidden: " + raw_path);
    }
  }

  return normalized;
}

EntryClass classify_entry(char type, const std::string& normalized_path,
                          bool reject_unsupported_file_types) {
  if (type == 'd') {
    return EntryClass::Directory;
  }
  if (type == 'l' || type == 'h') {
    throw_mpk(ErrorClass::PathTraversal,
              "path_traversal: link entries are forbidden: " + normalized_path);
  }
  if (type != '-') {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: unsupported tar entry type for " + normalized_path);
  }

  const std::string ext = fs::path(normalized_path).extension().string();
  if (ext == ".json")
    return EntryClass::Json;
  if (ext == ".so")
    return EntryClass::SharedObject;
  if (ext == ".elf")
    return EntryClass::Elf;
  if (ext == ".yaml" || ext == ".yml")
    return EntryClass::Auxiliary;

  if (reject_unsupported_file_types) {
    throw_mpk(ErrorClass::SchemaError,
              "schema_error: unsupported file type in archive: " + normalized_path);
  }
  return EntryClass::Directory;
}

std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

// Canonical allow-list shared by loader validation and pipeline sequence validation.
bool is_supported_kernel_impl(const std::string& kernel) {
  static const std::unordered_set<std::string> kSupportedKernels = {
      "preproc", "quant", "tess", "tessellate", "quanttess", "cast",
      "infer",   "mla",   "detessdequant", "detessellate", "dequantize",
      "boxdecode", "buffer_concat",
  };
  return kSupportedKernels.find(lower_copy(kernel)) != kSupportedKernels.end();
}

std::string require_nonempty_string(const json& obj, const char* key) {
  if (!obj.contains(key) || !obj[key].is_string()) {
    throw_mpk(ErrorClass::SchemaError,
              std::string("schema_error: pipeline_sequence stage missing string field '") + key +
                  "'");
  }
  const std::string value = obj[key].get<std::string>();
  if (value.empty()) {
    throw_mpk(ErrorClass::SchemaError,
              std::string("schema_error: pipeline_sequence stage has empty field '") + key + "'");
  }
  return value;
}

void ensure_safe_config_path(const std::string& path) {
  std::string normalized = path;
  for (char& c : normalized) {
    if (c == '\\')
      c = '/';
  }

  if (normalized.empty()) {
    throw_mpk(ErrorClass::SchemaError,
              "schema_error: pipeline_sequence stage has empty configPath");
  }
  if (normalized.front() == '/') {
    throw_mpk(ErrorClass::SchemaError,
              "schema_error: pipeline_sequence configPath must be relative");
  }
  if (normalized.size() >= 2 && std::isalpha(static_cast<unsigned char>(normalized[0])) &&
      normalized[1] == ':') {
    throw_mpk(ErrorClass::SchemaError,
              "schema_error: pipeline_sequence configPath must not use drive prefix");
  }

  std::stringstream ss(normalized);
  std::string segment;
  while (std::getline(ss, segment, '/')) {
    if (segment.empty() || segment == ".")
      continue;
    if (segment == "..") {
      throw_mpk(ErrorClass::SchemaError,
                "schema_error: pipeline_sequence configPath must not traverse parent");
    }
  }
}

TarEntry parse_tar_line(const std::string& line, const MpKLoaderOptions& opt,
                        bool reject_unsupported_file_types) {
  std::istringstream iss(line);

  std::string perms;
  std::string owner;
  std::string size_token;
  std::string date_token;
  std::string time_token;

  if (!(iss >> perms >> owner >> size_token >> date_token >> time_token)) {
    throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: malformed tar listing line: " + line);
  }

  std::string path;
  std::getline(iss, path);
  path = trim(path);

  const char type = perms.empty() ? '?' : perms.front();
  if ((type == 'l' || type == 'h')) {
    const std::size_t arrow = path.find(" -> ");
    if (arrow != std::string::npos) {
      path = path.substr(0, arrow);
      path = trim(path);
    }
  }

  std::uint64_t size_bytes = 0;
  if (!parse_uint64(size_token, &size_bytes)) {
    throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: invalid tar size token: " + size_token);
  }

  TarEntry entry;
  entry.raw_path = path;
  entry.normalized_path = normalize_entry_path(path, opt);
  entry.type = type;
  entry.size_bytes = size_bytes;
  entry.entry_class = classify_entry(type, entry.normalized_path, reject_unsupported_file_types);
  return entry;
}

class StrictJsonSax final : public nlohmann::json_sax<json> {
public:
  StrictJsonSax(std::size_t max_depth, bool reject_duplicate_keys)
      : max_depth_(max_depth), reject_duplicate_keys_(reject_duplicate_keys) {}

  bool null() override {
    return true;
  }
  bool boolean(bool) override {
    return true;
  }
  bool number_integer(number_integer_t) override {
    return true;
  }
  bool number_unsigned(number_unsigned_t) override {
    return true;
  }
  bool number_float(number_float_t, const string_t&) override {
    return true;
  }
  bool string(string_t&) override {
    return true;
  }
  bool binary(binary_t&) override {
    return fail("schema_error: binary JSON payload is not supported");
  }

  bool start_object(std::size_t) override {
    if (!enter_container())
      return false;
    object_keys_.emplace_back();
    return true;
  }

  bool key(string_t& val) override {
    if (object_keys_.empty()) {
      return fail("schema_error: malformed JSON object key state");
    }
    if (!reject_duplicate_keys_)
      return true;
    auto& keys = object_keys_.back();
    if (!keys.insert(val).second) {
      return fail("schema_error: duplicate key in JSON object: " + val);
    }
    return true;
  }

  bool end_object() override {
    if (object_keys_.empty()) {
      return fail("schema_error: malformed JSON object closure");
    }
    object_keys_.pop_back();
    return leave_container();
  }

  bool start_array(std::size_t) override {
    return enter_container();
  }

  bool end_array() override {
    return leave_container();
  }

  bool parse_error(std::size_t, const std::string&,
                   const nlohmann::detail::exception& ex) override {
    return fail(std::string("schema_error: ") + ex.what());
  }

  const std::string& error_message() const {
    return error_message_;
  }

private:
  bool enter_container() {
    ++depth_;
    if (depth_ > max_depth_) {
      return fail("schema_error: JSON depth exceeds limit");
    }
    return true;
  }

  bool leave_container() {
    if (depth_ == 0) {
      return fail("schema_error: malformed JSON depth accounting");
    }
    --depth_;
    return true;
  }

  bool fail(std::string message) {
    if (error_message_.empty()) {
      error_message_ = std::move(message);
    }
    return false;
  }

  std::size_t max_depth_ = 0;
  bool reject_duplicate_keys_ = true;
  std::size_t depth_ = 0;
  std::vector<std::unordered_set<std::string>> object_keys_;
  std::string error_message_;
};

json parse_json_entry_strict(const std::vector<std::uint8_t>& bytes, const std::string& entry_name,
                             const MpKLoaderOptions& opt) {
  StrictJsonSax sax(opt.max_json_depth, opt.reject_duplicate_json_keys);
  const bool sax_ok = json::sax_parse(bytes.begin(), bytes.end(), &sax);
  if (!sax_ok) {
    const std::string reason = sax.error_message().empty()
                                   ? "schema_error: failed strict JSON parse"
                                   : sax.error_message();
    throw_mpk(ErrorClass::SchemaError, reason + " for entry '" + entry_name + "'");
  }

  try {
    return json::parse(bytes.begin(), bytes.end());
  } catch (const std::exception& e) {
    throw_mpk(ErrorClass::SchemaError, std::string("schema_error: failed to parse JSON entry '") +
                                           entry_name + "': " + e.what());
  }
}

std::vector<std::uint8_t> read_tar_entry(const std::string& archive_path, TarMode mode,
                                         const std::string& raw_entry_path, std::size_t max_bytes) {
  const std::string cmd = std::string("tar -xO") + ((mode == TarMode::Gzip) ? "z" : "") + "f " +
                          shell_quote(archive_path) + " -- " + shell_quote(raw_entry_path) +
                          " 2>/dev/null";

  FILE* pipe = ::popen(cmd.c_str(), "r");
  if (!pipe) {
    throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: failed to read tar entry stream");
  }

  std::vector<std::uint8_t> out;
  std::array<std::uint8_t, 8192> buf{};

  while (true) {
    const std::size_t n = ::fread(buf.data(), 1, buf.size(), pipe);
    if (n > 0) {
      if (out.size() + n > max_bytes) {
        ::pclose(pipe);
        throw_mpk(ErrorClass::SizeLimitExceeded,
                  "size_limit_exceeded: tar entry exceeds maximum allowed bytes");
      }
      out.insert(out.end(), buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n));
    }

    if (n < buf.size()) {
      if (::feof(pipe))
        break;
      if (::ferror(pipe)) {
        ::pclose(pipe);
        throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: failed reading tar entry bytes");
      }
    }
  }

  const int rc = ::pclose(pipe);
  if (rc != 0) {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: tar entry extraction failed for " + raw_entry_path);
  }

  return out;
}

std::uint64_t stream_tar_entry_to_file(const std::string& archive_path, TarMode mode,
                                       const std::string& raw_entry_path, const fs::path& out_path,
                                       std::size_t max_bytes) {
  const std::string cmd = std::string("tar -xO") + ((mode == TarMode::Gzip) ? "z" : "") + "f " +
                          shell_quote(archive_path) + " -- " + shell_quote(raw_entry_path) +
                          " 2>/dev/null";

  FILE* pipe = ::popen(cmd.c_str(), "r");
  if (!pipe) {
    throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: failed to read tar entry stream");
  }

  std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    ::pclose(pipe);
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: failed to open extraction output file: " + out_path.string());
  }

  std::array<char, 8192> buf{};
  std::uint64_t total = 0;

  while (true) {
    const std::size_t n = ::fread(buf.data(), 1, buf.size(), pipe);
    if (n > 0) {
      total += static_cast<std::uint64_t>(n);
      if (total > max_bytes) {
        ::pclose(pipe);
        throw_mpk(ErrorClass::SizeLimitExceeded,
                  "size_limit_exceeded: extracted entry exceeds per-entry size limit");
      }
      out.write(buf.data(), static_cast<std::streamsize>(n));
      if (!out.good()) {
        ::pclose(pipe);
        throw_mpk(ErrorClass::InvalidArchive,
                  "invalid_archive: failed writing extracted entry: " + out_path.string());
      }
    }

    if (n < buf.size()) {
      if (::feof(pipe))
        break;
      if (::ferror(pipe)) {
        ::pclose(pipe);
        throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: failed reading tar entry bytes");
      }
    }
  }

  const int rc = ::pclose(pipe);
  if (rc != 0) {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: tar entry extraction failed for " + raw_entry_path);
  }

  return total;
}

void validate_pipeline_sequence_json(const json& j) {
  if (!j.is_object()) {
    throw_mpk(ErrorClass::SchemaError,
              "schema_error: pipeline_sequence.json must be a JSON object");
  }
  if (!j.contains("pipelines") || !j["pipelines"].is_array() || j["pipelines"].empty()) {
    throw_mpk(ErrorClass::SchemaError,
              "schema_error: pipeline_sequence.json missing non-empty 'pipelines' array");
  }
  const auto& first = j["pipelines"][0];
  if (!first.is_object() || !first.contains("sequence") || !first["sequence"].is_array() ||
      first["sequence"].empty()) {
    throw_mpk(ErrorClass::SchemaError,
              "schema_error: pipeline_sequence.json first pipeline missing non-empty 'sequence'");
  }

  const auto& seq = first["sequence"];
  std::unordered_set<std::string> seen_names;
  seen_names.reserve(seq.size());

  for (const auto& stage : seq) {
    if (!stage.is_object()) {
      throw_mpk(ErrorClass::SchemaError,
                "schema_error: pipeline_sequence stage entry must be an object");
    }

    if (!stage.contains("sequence_id") || !stage["sequence_id"].is_number_integer()) {
      throw_mpk(ErrorClass::SchemaError,
                "schema_error: pipeline_sequence stage missing integer field 'sequence_id'");
    }
    std::int64_t sequence_id = 0;
    try {
      sequence_id = stage["sequence_id"].get<std::int64_t>();
    } catch (const std::exception& e) {
      throw_mpk(ErrorClass::SchemaError,
                std::string("schema_error: invalid sequence_id: ") + e.what());
    }
    if (sequence_id <= 0 || sequence_id > std::numeric_limits<int>::max()) {
      throw_mpk(ErrorClass::SchemaError, "schema_error: sequence_id must be in range [1, INT_MAX]");
    }

    const std::string name = require_nonempty_string(stage, "name");
    (void)require_nonempty_string(stage, "pluginId");
    const std::string config_path = require_nonempty_string(stage, "configPath");
    (void)require_nonempty_string(stage, "processor");
    const std::string kernel = require_nonempty_string(stage, "kernel");

    ensure_safe_config_path(config_path);

    if (!seen_names.insert(name).second) {
      throw_mpk(ErrorClass::SchemaError,
                "schema_error: duplicate stage name in pipeline_sequence: " + name);
    }
    if (!is_supported_kernel_impl(kernel)) {
      throw_mpk(ErrorClass::SchemaError,
                "schema_error: unsupported kernel in pipeline_sequence: " + kernel);
    }

    if (stage.contains("input")) {
      if (stage["input"].is_string()) {
        (void)stage["input"].get<std::string>();
      } else if (stage["input"].is_array()) {
        for (const auto& dep_json : stage["input"]) {
          if (!dep_json.is_string()) {
            throw_mpk(ErrorClass::SchemaError,
                      "schema_error: stage input array must contain only strings");
          }
          (void)dep_json.get<std::string>();
        }
      } else {
        throw_mpk(ErrorClass::SchemaError,
                  "schema_error: stage input must be string or array of strings");
      }
    }
  }
}

void validate_version_json(const json& j) {
  if (!j.is_object() || !j.contains("version")) {
    return;
  }

  std::string version;
  if (j["version"].is_number_integer()) {
    version = std::to_string(j["version"].get<int>());
  } else if (j["version"].is_string()) {
    version = j["version"].get<std::string>();
  } else {
    throw_mpk(ErrorClass::SchemaError, "schema_error: manifest version must be string or integer");
  }

  if (version != "1") {
    throw_mpk(ErrorClass::UnsupportedVersion,
              "unsupported_version: manifest version is not supported: " + version);
  }
}

ValidatedArchive validate_archive(const std::string& archive_path, const MpKLoaderOptions& opt) {
  ValidatedArchive out;

  const fs::path archive_fs_path(archive_path);
  std::error_code ec;
  if (!fs::exists(archive_fs_path, ec) || ec || !fs::is_regular_file(archive_fs_path, ec)) {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: archive path does not exist or is not a regular file: " +
                  archive_path);
  }

  const std::uint64_t archive_size = fs::file_size(archive_fs_path, ec);
  if (ec) {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: failed to read archive size: " + archive_path);
  }
  if (archive_size > opt.max_archive_bytes) {
    throw_mpk(ErrorClass::SizeLimitExceeded,
              "size_limit_exceeded: archive exceeds configured maximum size");
  }

  out.manifest.archive_path = archive_path;
  out.manifest.archive_size_bytes = archive_size;
  out.manifest.package_name = package_name_from_archive(archive_fs_path);

  std::vector<std::string> listing = tar_list_verbose_autodetect(archive_path, &out.mode);
  if (listing.empty()) {
    throw_mpk(ErrorClass::InvalidArchive, "invalid_archive: archive contains no entries");
  }
  if (listing.size() > opt.max_entries) {
    throw_mpk(ErrorClass::SizeLimitExceeded, "size_limit_exceeded: archive has too many entries");
  }

  std::unordered_set<std::string> seen_paths;
  std::size_t total_json_bytes = 0;
  std::size_t json_file_count = 0;

  for (const auto& line : listing) {
    TarEntry entry = parse_tar_line(line, opt, opt.reject_unsupported_file_types);

    if (!seen_paths.insert(entry.normalized_path).second) {
      throw_mpk(ErrorClass::InvalidArchive,
                "invalid_archive: duplicate archive entry path: " + entry.normalized_path);
    }

    if (entry.type == '-' && entry.size_bytes > opt.max_entry_bytes) {
      throw_mpk(ErrorClass::SizeLimitExceeded,
                "size_limit_exceeded: archive entry exceeds maximum allowed size: " +
                    entry.normalized_path);
    }

    if (entry.entry_class == EntryClass::Json) {
      ++json_file_count;
      total_json_bytes += static_cast<std::size_t>(entry.size_bytes);
      if (total_json_bytes > opt.max_total_json_bytes) {
        throw_mpk(ErrorClass::SizeLimitExceeded,
                  "size_limit_exceeded: total JSON bytes exceed configured limit");
      }
    } else if (entry.entry_class == EntryClass::SharedObject ||
               entry.entry_class == EntryClass::Elf) {
      out.manifest.has_model_binary = true;
    }

    ArchiveEntry manifest_entry;
    manifest_entry.path = entry.raw_path;
    manifest_entry.normalized_path = entry.normalized_path;
    manifest_entry.type = entry.type;
    manifest_entry.size_bytes = entry.size_bytes;
    out.manifest.entries.push_back(std::move(manifest_entry));

    out.entries.push_back(std::move(entry));
  }

  for (const auto& entry : out.entries) {
    if (entry.entry_class != EntryClass::Json)
      continue;
    const std::vector<std::uint8_t> bytes =
        read_tar_entry(archive_path, out.mode, entry.raw_path, opt.max_entry_bytes);

    json parsed = parse_json_entry_strict(bytes, entry.normalized_path, opt);

    const std::string base = fs::path(entry.normalized_path).filename().string();
    if (base == "pipeline_sequence.json") {
      try {
        validate_pipeline_sequence_json(parsed);
        out.manifest.has_pipeline_sequence = true;
      } catch (const MpKError&) {
        if (opt.require_pipeline_sequence) {
          throw;
        }
        // Runtime extraction can tolerate legacy/partial pipeline_sequence JSON
        // because ModelPack has a fallback synthesizer.
      }
    }
    if (base == "manifest.json" || base == "mpk_manifest.json") {
      validate_version_json(parsed);
    }
  }

  if (opt.require_pipeline_sequence && !out.manifest.has_pipeline_sequence) {
    throw_mpk(ErrorClass::SchemaError, "schema_error: required pipeline_sequence.json is missing");
  }
  if (json_file_count == 0) {
    throw_mpk(ErrorClass::SchemaError,
              "schema_error: archive contains no JSON configuration files");
  }
  if (opt.require_model_binary && !out.manifest.has_model_binary) {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: archive missing model binary artifacts (*.elf or *.so)");
  }

  return out;
}

fs::path extract_destination_for(const fs::path& package_root, const TarEntry& entry) {
  const std::string ext = fs::path(entry.normalized_path).extension().string();
  const fs::path name = fs::path(entry.normalized_path).filename();

  if (entry.entry_class == EntryClass::Json) {
    return package_root / "etc" / name;
  }
  if (ext == ".so") {
    return package_root / "lib" / name;
  }
  if (ext == ".elf") {
    return package_root / "share" / name;
  }

  return package_root / "etc" / name;
}

} // namespace

bool is_supported_kernel(const std::string& kernel) {
  return is_supported_kernel_impl(kernel);
}

const char* error_class_name(ErrorClass code) {
  switch (code) {
  case ErrorClass::InvalidArchive:
    return "invalid_archive";
  case ErrorClass::PathTraversal:
    return "path_traversal";
  case ErrorClass::SchemaError:
    return "schema_error";
  case ErrorClass::UnsupportedVersion:
    return "unsupported_version";
  case ErrorClass::SizeLimitExceeded:
    return "size_limit_exceeded";
  }
  return "invalid_archive";
}

MpKError::MpKError(ErrorClass code, const std::string& message)
    : std::runtime_error(std::string(error_class_name(code)) + ": " + message), code_(code) {}

MpKManifest MpKLoader::inspect(const std::string& archive_path, const MpKLoaderOptions& opt) {
  return validate_archive(archive_path, opt).manifest;
}

MpKExtractResult MpKLoader::extract(const std::string& archive_path, const std::string& output_root,
                                    const MpKLoaderOptions& opt) {
  ValidatedArchive validated = validate_archive(archive_path, opt);

  const fs::path root(output_root);
  std::error_code ec;
  fs::create_directories(root, ec);
  if (ec) {
    throw_mpk(ErrorClass::InvalidArchive,
              "invalid_archive: failed to create output root: " + root.string());
  }

  const fs::path package_root = root / validated.manifest.package_name;
  fs::remove_all(package_root, ec);
  ec.clear();
  try {
    fs::create_directories(package_root / "etc", ec);
    if (ec) {
      throw_mpk(ErrorClass::InvalidArchive,
                "invalid_archive: failed to create etc dir under output root");
    }
    fs::create_directories(package_root / "lib", ec);
    if (ec) {
      throw_mpk(ErrorClass::InvalidArchive,
                "invalid_archive: failed to create lib dir under output root");
    }
    fs::create_directories(package_root / "share", ec);
    if (ec) {
      throw_mpk(ErrorClass::InvalidArchive,
                "invalid_archive: failed to create share dir under output root");
    }

    for (const auto& entry : validated.entries) {
      if (entry.type != '-')
        continue;
      if (entry.entry_class == EntryClass::Directory)
        continue;

      const fs::path dst = extract_destination_for(package_root, entry);
      (void)stream_tar_entry_to_file(archive_path, validated.mode, entry.raw_path, dst,
                                     opt.max_entry_bytes);
    }
  } catch (...) {
    fs::remove_all(package_root, ec);
    throw;
  }

  MpKExtractResult out;
  out.package_root = package_root.string();
  out.etc_dir = (package_root / "etc").string();
  out.lib_dir = (package_root / "lib").string();
  out.share_dir = (package_root / "share").string();
  out.manifest = std::move(validated.manifest);
  return out;
}

} // namespace simaai::neat::mpk
