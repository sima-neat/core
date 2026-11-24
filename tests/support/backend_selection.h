#pragma once

// A test process's backend. A process runs on one backend (neat/runtime.h: set_backend), so a test
// that runs on both is registered once per backend with --backend=gstreamer|dataflow, and a test
// that compares the two is two processes: the one on GStreamer writes a reference record
// (--reference=<file>), the one on Dataflow reads it and compares what it pulls, key by key.

#include "neat/runtime.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sima_test {

// The value of --<name>=<value> in argv, or `fallback`.
inline std::string arg_value(int argc, char** argv, const std::string& name,
                             const std::string& fallback = {}) {
  const std::string flag = "--" + name + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind(flag, 0) == 0) {
      return arg.substr(flag.size());
    }
  }
  return fallback;
}

// Selects the process's backend from --backend=gstreamer|dataflow (GStreamer without one), as an
// application's first call does, before anything is built.
inline simaai::neat::RunBackend select_backend(int argc, char** argv) {
  const std::string value = arg_value(argc, argv, "backend", "gstreamer");
  if (value != "gstreamer" && value != "dataflow") {
    throw std::runtime_error("--backend=" + value + ": the backends are gstreamer and dataflow");
  }
  const simaai::neat::RunBackend backend = value == "dataflow"
                                               ? simaai::neat::RunBackend::Dataflow
                                               : simaai::neat::RunBackend::GStreamer;
  simaai::neat::set_backend(backend);
  return backend;
}

inline bool on_dataflow() {
  return simaai::neat::backend() == simaai::neat::RunBackend::Dataflow;
}

inline const char* backend_name() {
  return on_dataflow() ? "dataflow" : "gstreamer";
}

// A value as the record writes it.
template <class T> std::string text(const T& value) {
  if constexpr (std::is_same_v<T, std::string>) {
    return value;
  } else if constexpr (std::is_same_v<T, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_enum_v<T>) {
    return std::to_string(static_cast<long long>(value));
  } else if constexpr (std::is_arithmetic_v<T>) {
    return std::to_string(value);
  } else if constexpr (std::is_convertible_v<T, std::string>) {
    return std::string(value);
  } else {
    std::string out = "[";
    bool first = true;
    for (const auto& item : value) {
      out += (first ? "" : ",") + text(item);
      first = false;
    }
    return out + "]";
  }
}

template <class T> std::string text(const std::optional<T>& value) {
  return value ? text(*value) : std::string("none");
}

// Bytes as the record keeps them: their count and an FNV-1a 64 digest.
inline std::string bytes_digest(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const std::uint8_t b : bytes) {
    h = (h ^ b) * 1099511628211ULL;
  }
  std::ostringstream out;
  out << bytes.size() << ":" << std::hex << h;
  return out.str();
}

// Fields written in order and read back in the same order: a structure the GStreamer run observed,
// for the Dataflow run to compare with its own (Reference::blob, Reference::recorded).
class Archive {
public:
  Archive() = default;
  explicit Archive(std::string data) : data_(std::move(data)) {}

  Archive& put(const std::string& value) {
    data_ += std::to_string(value.size()) + ":" + value;
    return *this;
  }
  Archive& put(const std::vector<std::uint8_t>& bytes) {
    return put(std::string(bytes.begin(), bytes.end()));
  }
  template <class T> Archive& put_number(T value) {
    return put(std::to_string(static_cast<long long>(value)));
  }

  std::string get() {
    const std::size_t colon = data_.find(':', at_);
    require(colon != std::string::npos, "the archive ends early");
    const std::size_t size = std::stoull(data_.substr(at_, colon - at_));
    require(colon + 1 + size <= data_.size(), "the archive ends early");
    std::string value = data_.substr(colon + 1, size);
    at_ = colon + 1 + size;
    return value;
  }
  std::vector<std::uint8_t> get_bytes() {
    const std::string value = get();
    return std::vector<std::uint8_t>(value.begin(), value.end());
  }
  template <class T> T get_number() {
    return static_cast<T>(std::stoll(get()));
  }

  const std::string& data() const {
    return data_;
  }

private:
  std::string data_;
  std::size_t at_ = 0;
};

// What the run on GStreamer records and the run on Dataflow compares. Keys are unique; a value is
// one line once escaped.
class Reference {
public:
  Reference() = default;

  // --reference=<file>. On GStreamer the file is written by finish(); on Dataflow it is read now,
  // and a comparing test needs it.
  static Reference from_args(int argc, char** argv) {
    Reference r;
    r.path_ = arg_value(argc, argv, "reference");
    r.dataflow_ = on_dataflow();
    if (r.dataflow_ && !r.path_.empty()) {
      std::ifstream in(r.path_);
      require(static_cast<bool>(in), "the reference record " + r.path_ +
                                         " is missing: run the test with --backend=gstreamer "
                                         "--reference=" +
                                         r.path_ + " first");
      std::string line;
      while (std::getline(in, line)) {
        const std::size_t tab = line.find('\t');
        const std::size_t tab2 = line.find('\t', tab + 1);
        require(tab != std::string::npos && tab2 != std::string::npos,
                "the reference record's line: " + line);
        r.recorded_[unescape(line.substr(tab + 1, tab2 - tab - 1))] = {line.substr(0, tab),
                                                                       unescape(line.substr(tab2 + 1))};
      }
      require(!r.recorded_.empty(), "the reference record " + r.path_ + " is empty");
    }
    return r;
  }

  // True on Dataflow: the comparisons need GStreamer's record.
  bool comparing() const {
    return dataflow_;
  }

  // A value both backends give: GStreamer's is recorded; Dataflow's must equal it.
  void equal(const std::string& key, const std::string& value) {
    put("=", key, value, [](const std::string& gst, const std::string& df) { return gst == df; });
  }
  template <class T> void equal(const std::string& key, const T& value) {
    equal(key, text(value));
  }

  // GStreamer's value starts with Dataflow's: an element name GStreamer numbers per process.
  void prefix(const std::string& key, const std::string& value) {
    put("^", key, value,
        [](const std::string& gst, const std::string& df) { return gst.rfind(df, 0) == 0; });
  }

  // A value that is each backend's own: recorded, and on Dataflow printed beside its own.
  void note(const std::string& key, const std::string& value) {
    if (!dataflow_) {
      add("~", key, value);
      return;
    }
    const auto it = recorded_.find(key);
    std::cout << "[" << key << "] GStreamer '" << (it == recorded_.end() ? "?" : it->second.second)
              << "' / Dataflow '" << value << "'\n";
    if (it != recorded_.end()) {
      met_.insert(key);
    }
  }
  template <class T> void note(const std::string& key, const T& value) {
    note(key, text(value));
  }

  // A structure the GStreamer run observed (an Archive's data), recorded for the Dataflow run.
  void blob(const std::string& key, const std::string& data) {
    if (!dataflow_) {
      add("B", key, data);
    }
  }

  // On Dataflow: what the GStreamer run recorded under `key` (blob or value).
  std::string recorded(const std::string& key) {
    require(!path_.empty(), "on Dataflow a comparing test needs --reference=<file>, which a run "
                            "with --backend=gstreamer writes");
    const auto it = recorded_.find(key);
    require(it != recorded_.end(), "'" + key + "': the GStreamer run recorded nothing");
    met_.insert(key);
    return it->second.second;
  }

  // GStreamer: writes the record. Dataflow: every key GStreamer recorded was met.
  void finish() {
    if (!dataflow_) {
      if (path_.empty()) {
        return;
      }
      std::ofstream out(path_, std::ios::trunc);
      require(static_cast<bool>(out), "the reference record " + path_ + " cannot be written");
      for (const auto& [key, line] : lines_) {
        (void)key;
        out << line << "\n";
      }
      require(static_cast<bool>(out), "the reference record " + path_ + " was not written");
      std::cout << "[INFO] reference record: " << lines_.size() << " values in " << path_ << "\n";
      return;
    }
    require(!path_.empty(), "on Dataflow a comparing test needs --reference=<file>, which a run "
                            "with --backend=gstreamer writes");
    for (const auto& [key, recorded] : recorded_) {
      require(met_.count(key) == 1U, "the Dataflow run gave no value for '" + key +
                                         "', which the GStreamer run recorded");
    }
    std::cout << "[INFO] reference record: " << met_.size() << " values compared with " << path_
              << "\n";
  }

private:
  template <class Same>
  void put(const char* kind, const std::string& key, const std::string& value, Same same) {
    if (!dataflow_) {
      add(kind, key, value);
      return;
    }
    require(!path_.empty(), "on Dataflow a comparing test needs --reference=<file>, which a run "
                            "with --backend=gstreamer writes");
    const auto it = recorded_.find(key);
    require(it != recorded_.end(), "'" + key + "': the GStreamer run recorded no such value");
    require(same(it->second.second, value),
            key + ": GStreamer '" + it->second.second + "' / Dataflow '" + value + "'");
    require(met_.insert(key).second, "'" + key + "' compared twice");
  }

  void add(const char* kind, const std::string& key, const std::string& value) {
    require(lines_.emplace(key, std::string(kind) + "\t" + escape(key) + "\t" + escape(value)).second,
            "'" + key + "' recorded twice");
  }

  static std::string escape(const std::string& s) {
    std::string out;
    for (const char c : s) {
      if (c == '\\') {
        out += "\\\\";
      } else if (c == '\n') {
        out += "\\n";
      } else if (c == '\t') {
        out += "\\t";
      } else {
        out += c;
      }
    }
    return out;
  }

  static std::string unescape(const std::string& s) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (s[i] == '\\' && i + 1 < s.size()) {
        const char n = s[++i];
        out += n == 'n' ? '\n' : (n == 't' ? '\t' : n);
      } else {
        out += s[i];
      }
    }
    return out;
  }

  std::string path_;
  bool dataflow_ = false;
  std::map<std::string, std::string> lines_;                            // GStreamer: key -> line
  std::map<std::string, std::pair<std::string, std::string>> recorded_; // Dataflow: key -> kind, value
  std::set<std::string> met_;
};

} // namespace sima_test

// A test's main() for a test registered once per backend: --backend selects the process's
// backend before the body runs.
#define RUN_BACKEND_TEST(name_literal, body_lambda)                                                \
  int main(int argc, char** argv) {                                                                \
    try {                                                                                          \
      sima_test::select_backend(argc, argv);                                                       \
    } catch (const std::exception& e) {                                                            \
      std::cerr << "[FAIL] " << e.what() << "\n";                                                  \
      return 1;                                                                                    \
    }                                                                                              \
    return sima_test::run_test(name_literal, body_lambda);                                         \
  }
