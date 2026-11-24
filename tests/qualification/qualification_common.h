// Shared by the qualification harness's two tools (census.cpp, compare.cpp): a small JSON writer,
// the catalogue's option sets read from key=value files, and the output hash.
//
// An option file holds one `key=value` per line, the ModelOptions a catalogue script sets
// (catalogue.json; qualify.py writes the file). `frame_format=<RGB|BGR|NV12|GRAY8>` names the
// format of the frames the script pushes and is not a ModelOptions field. A key this Core's
// Model::Options has no field for is reported, never ignored silently.
#pragma once

#include "model/Model.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::qualification {

// ---- JSON writing -------------------------------------------------------------------------------
inline std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 2);
  for (const unsigned char c : in) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "\\u%04x", c);
        out += buf;
      } else {
        out += static_cast<char>(c);
      }
    }
  }
  return out;
}

inline std::string json_string(const std::string& v) {
  return "\"" + json_escape(v) + "\"";
}

inline std::string json_number(double v) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

// An object built key by key: Obj o; o.str("a", "x").num("b", 1); o.text() == {"a":"x","b":1}.
class Obj {
public:
  Obj& raw(const std::string& key, const std::string& json) {
    body_ += (body_.empty() ? "" : ",") + json_string(key) + ":" + json;
    return *this;
  }
  Obj& str(const std::string& key, const std::string& v) {
    return raw(key, json_string(v));
  }
  Obj& num(const std::string& key, double v) {
    return raw(key, json_number(v));
  }
  Obj& integer(const std::string& key, long long v) {
    return raw(key, std::to_string(v));
  }
  Obj& boolean(const std::string& key, bool v) {
    return raw(key, v ? "true" : "false");
  }
  std::string text() const {
    return "{" + body_ + "}";
  }

private:
  std::string body_;
};

inline std::string json_array(const std::vector<std::string>& items_json) {
  std::string s = "[";
  for (std::size_t i = 0; i < items_json.size(); ++i) {
    s += (i ? "," : "") + items_json[i];
  }
  return s + "]";
}

inline std::string json_strings(const std::vector<std::string>& v) {
  std::vector<std::string> q;
  q.reserve(v.size());
  for (const auto& x : v) {
    q.push_back(json_string(x));
  }
  return json_array(q);
}

template <class T> std::string json_integers(const std::vector<T>& v) {
  std::vector<std::string> q;
  q.reserve(v.size());
  for (const auto& x : v) {
    q.push_back(std::to_string(static_cast<long long>(x)));
  }
  return json_array(q);
}

// ---- XXH64 ------------------------------------------------------------------------------------
// XXH64 (seed 0) of a byte range, written from the published algorithm. The harness hashes every
// output tensor's dense bytes with it: about one multiply per eight bytes, against the byte-wise
// FNV-1a that costs one per byte. Little-endian reads (the A65 and x86 both are).
namespace xxh64_detail {
constexpr std::uint64_t kP1 = 11400714785074694791ULL;
constexpr std::uint64_t kP2 = 14029467366897019727ULL;
constexpr std::uint64_t kP3 = 1609587929392839161ULL;
constexpr std::uint64_t kP4 = 9650029242287828579ULL;
constexpr std::uint64_t kP5 = 2870177450012600261ULL;
inline std::uint64_t rotl(std::uint64_t x, int r) {
  return (x << r) | (x >> (64 - r));
}
inline std::uint64_t read64(const unsigned char* p) {
  std::uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}
inline std::uint32_t read32(const unsigned char* p) {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
inline std::uint64_t round(std::uint64_t acc, std::uint64_t input) {
  acc += input * kP2;
  acc = rotl(acc, 31);
  return acc * kP1;
}
inline std::uint64_t merge(std::uint64_t acc, std::uint64_t value) {
  acc ^= round(0, value);
  return acc * kP1 + kP4;
}
} // namespace xxh64_detail

inline std::uint64_t xxh64(const void* data, std::size_t len, std::uint64_t seed = 0) {
  using namespace xxh64_detail;
  const auto* p = static_cast<const unsigned char*>(data);
  const unsigned char* const end = p + len;
  std::uint64_t h;
  if (len >= 32) {
    const unsigned char* const limit = end - 32;
    std::uint64_t v1 = seed + kP1 + kP2, v2 = seed + kP2, v3 = seed, v4 = seed - kP1;
    do {
      v1 = round(v1, read64(p));
      v2 = round(v2, read64(p + 8));
      v3 = round(v3, read64(p + 16));
      v4 = round(v4, read64(p + 24));
      p += 32;
    } while (p <= limit);
    h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    h = merge(h, v1);
    h = merge(h, v2);
    h = merge(h, v3);
    h = merge(h, v4);
  } else {
    h = seed + kP5;
  }
  h += static_cast<std::uint64_t>(len);
  while (p + 8 <= end) {
    h ^= round(0, read64(p));
    h = rotl(h, 27) * kP1 + kP4;
    p += 8;
  }
  if (p + 4 <= end) {
    h ^= static_cast<std::uint64_t>(read32(p)) * kP1;
    h = rotl(h, 23) * kP2 + kP3;
    p += 4;
  }
  while (p < end) {
    h ^= static_cast<std::uint64_t>(*p) * kP5;
    h = rotl(h, 11) * kP1;
    ++p;
  }
  h ^= h >> 33;
  h *= kP2;
  h ^= h >> 29;
  h *= kP3;
  h ^= h >> 32;
  return h;
}

// The byte-wise FNV-1a 64 the earlier measurements used, kept to report its cost beside XXH64's.
inline std::uint64_t fnv1a64(const void* data, std::size_t len) {
  const auto* b = static_cast<const unsigned char*>(data);
  std::uint64_t h = 14695981039346656037ULL;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

inline std::string hex64(std::uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

// ---- option sets
// ---------------------------------------------------------------------------------
inline std::vector<float> floats_of(const std::string& v) {
  std::vector<float> out;
  std::string cur;
  for (const char c : v + ",") {
    if (c == ',' || c == '[' || c == ']' || c == ' ') {
      if (!cur.empty()) {
        out.push_back(std::stof(cur));
        cur.clear();
      }
    } else {
      cur += c;
    }
  }
  return out;
}

// One key=value of a catalogue script's ModelOptions: "" when applied, else why not.
inline std::string apply_option(Model::Options& o, const std::string& key, const std::string& v) {
  const auto flag = [&](AutoFlag& f) {
    f = v == "On" ? AutoFlag::On : v == "Off" ? AutoFlag::Off : AutoFlag::Auto;
  };
  const auto resize_mode = [&] {
    return v == "Stretch" ? ResizeMode::Stretch
           : v == "Crop"  ? ResizeMode::Crop
                          : ResizeMode::Letterbox;
  };
  const auto color = [&](PreprocessColorFormat& f) -> std::string {
    static const std::map<std::string, PreprocessColorFormat> m{
        {"Auto", PreprocessColorFormat::Auto}, {"RGB", PreprocessColorFormat::RGB},
        {"BGR", PreprocessColorFormat::BGR},   {"GRAY8", PreprocessColorFormat::GRAY8},
        {"NV12", PreprocessColorFormat::NV12}, {"I420", PreprocessColorFormat::I420},
        {"NV16", PreprocessColorFormat::NV16}};
    const auto it = m.find(v);
    if (it == m.end()) {
      return "unknown colour format " + v;
    }
    f = it->second;
    return {};
  };
  if (key == "preprocess.kind") {
    o.preprocess.kind = v == "Image"    ? InputKind::Image
                        : v == "Tensor" ? InputKind::Tensor
                                        : InputKind::Auto;
  } else if (key == "preprocess.enable") {
    flag(o.preprocess.enable);
  } else if (key == "preprocess.input_max_width") {
    o.preprocess.input_max_width = std::stoi(v);
  } else if (key == "preprocess.input_max_height") {
    o.preprocess.input_max_height = std::stoi(v);
  } else if (key == "preprocess.resize.enable") {
    flag(o.preprocess.resize.enable);
  } else if (key == "preprocess.resize.mode") {
    o.preprocess.resize.mode = resize_mode();
  } else if (key == "preprocess.resize.width") {
    o.preprocess.resize.width = std::stoi(v);
  } else if (key == "preprocess.resize.height") {
    o.preprocess.resize.height = std::stoi(v);
  } else if (key == "preprocess.resize.pad_value") {
    o.preprocess.resize.pad_value = std::stoi(v);
  } else if (key == "preprocess.normalize.enable") {
    flag(o.preprocess.normalize.enable);
  } else if (key == "preprocess.normalize.has_explicit_stats") {
    o.preprocess.normalize.has_explicit_stats = v == "true" || v == "True" || v == "1";
  } else if (key == "preprocess.normalize.mean" || key == "preprocess.normalize.stddev") {
    const std::vector<float> f = floats_of(v);
    if (f.size() != 3U) {
      return "expected three numbers";
    }
    auto& dst = key == "preprocess.normalize.mean" ? o.preprocess.normalize.mean
                                                   : o.preprocess.normalize.stddev;
    dst = {f[0], f[1], f[2]};
  } else if (key == "preprocess.preset") {
    if (v != "ImageNet") {
      return "unknown preset " + v;
    }
    o.preprocess.preset = NormalizePreset::ImageNet;
  } else if (key == "preprocess.color_convert.enable") {
    flag(o.preprocess.color_convert.enable);
  } else if (key == "preprocess.color_convert.input_format") {
    return color(o.preprocess.color_convert.input_format);
  } else if (key == "preprocess.color_convert.output_format") {
    return color(o.preprocess.color_convert.output_format);
  } else if (key == "decode_type") {
    static const std::map<std::string, BoxDecodeType> m{{"Unspecified", BoxDecodeType::Unspecified},
                                                        {"Yolo", BoxDecodeType::Yolo},
                                                        {"YoloV5", BoxDecodeType::YoloV5},
                                                        {"YoloV8", BoxDecodeType::YoloV8},
                                                        {"YoloV8Seg", BoxDecodeType::YoloV8Seg},
                                                        {"YoloV8Pose", BoxDecodeType::YoloV8Pose},
                                                        {"Detr", BoxDecodeType::Detr},
                                                        {"YoloV26", BoxDecodeType::YoloV26},
                                                        {"YoloV26Pose", BoxDecodeType::YoloV26Pose},
                                                        {"YoloV26Seg", BoxDecodeType::YoloV26Seg},
                                                        {"YoloX", BoxDecodeType::YoloX},
                                                        {"Ssd", BoxDecodeType::Ssd},
                                                        {"SuperPoint", BoxDecodeType::SuperPoint},
                                                        {"YoloV26Obb", BoxDecodeType::YoloV26Obb},
                                                        {"YoloV8Obb", BoxDecodeType::YoloV8Obb}};
    const auto it = m.find(v);
    if (it == m.end()) {
      return "unknown decode type " + v;
    }
    o.decode_type = it->second;
  } else if (key == "decode_type_option") {
    static const std::map<std::string, BoxDecodeTypeOption> m{
        {"Auto", BoxDecodeTypeOption::Auto},
        {"PackedPerHead", BoxDecodeTypeOption::PackedPerHead},
        {"InterleavedByHead", BoxDecodeTypeOption::InterleavedByHead},
        {"GroupedByRole", BoxDecodeTypeOption::GroupedByRole},
        {"Split3Interleaved", BoxDecodeTypeOption::Split3Interleaved},
        {"Split3Grouped", BoxDecodeTypeOption::Split3Grouped},
        {"InterleavedByHeadProbability", BoxDecodeTypeOption::InterleavedByHeadProbability},
        {"InterleavedByHeadLogit", BoxDecodeTypeOption::InterleavedByHeadLogit},
        {"GroupedByRoleProbability", BoxDecodeTypeOption::GroupedByRoleProbability},
        {"GroupedByRoleLogit", BoxDecodeTypeOption::GroupedByRoleLogit}};
    const auto it = m.find(v);
    if (it == m.end()) {
      return "unknown decode type option " + v;
    }
    o.decode_type_option = it->second;
  } else if (key == "score_threshold") {
    o.score_threshold = std::stof(v);
  } else if (key == "nms_iou_threshold") {
    o.nms_iou_threshold = std::stof(v);
  } else if (key == "top_k") {
    o.top_k = std::stoi(v);
  } else if (key == "num_classes") {
    o.num_classes = std::stoi(v);
  } else if (key == "boxdecode_original_width") {
    o.boxdecode_original_width = std::stoi(v);
  } else if (key == "boxdecode_original_height") {
    o.boxdecode_original_height = std::stoi(v);
  } else if (key == "boxdecode_resize_mode") {
    o.boxdecode_resize_mode = resize_mode();
  } else if (key == "superpoint.profile") {
    static const std::map<std::string, SuperPointProfile> m{
        {"Auto", SuperPointProfile::Auto},
        {"LightGlueV1", SuperPointProfile::LightGlueV1},
        {"MagicLeapDemoV1", SuperPointProfile::MagicLeapDemoV1},
        {"PaperBicubicV1", SuperPointProfile::PaperBicubicV1},
        {"A65V1", SuperPointProfile::A65V1}};
    const auto it = m.find(v);
    if (it == m.end()) {
      return "unknown SuperPoint profile " + v;
    }
    o.superpoint.profile = it->second;
  } else if (key == "processcvu.pre_run_target") {
    o.processcvu.pre_run_target = v;
  } else if (key == "processcvu.post_run_target") {
    o.processcvu.post_run_target = v;
  } else if (key == "advanced_execution.preprocess_target") {
    o.advanced_execution.preprocess_target = v;
  } else if (key == "advanced_execution.postprocess_target") {
    o.advanced_execution.postprocess_target = v;
  } else {
    return "this Core's Model::Options has no such field";
  }
  return {};
}

// What an option file says: the Model's options, the format of the frames, and every line that
// was not applied (as "key=value: why").
struct OptionSet {
  Model::Options options;
  bool given = false;               // a file with at least one key: the view's own options
  bool image = false;               // the script pushes frames (preprocess.kind == Image)
  std::string frame_format = "RGB"; // of those frames
  std::vector<std::string> not_applied;
};

// Reads `path` ("" or "-": no file, the Model's default options). Mirrors what the script does
// before it builds: an image model whose options name no input colour takes the frames' colour.
inline OptionSet read_option_set(const std::string& path) {
  OptionSet set;
  if (path.empty() || path == "-") {
    return set;
  }
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read option file " + path);
  }
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t eq = line.find('=');
    if (line.empty() || line[0] == '#' || eq == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, eq), value = line.substr(eq + 1);
    set.given = true;
    if (key == "frame_format") {
      set.frame_format = value;
      continue;
    }
    std::string why;
    try {
      why = apply_option(set.options, key, value);
    } catch (const std::exception& e) {
      why = e.what();
    }
    if (!why.empty()) {
      set.not_applied.push_back(key + "=" + value + ": " + why);
    }
  }
  set.image = set.given && set.options.preprocess.kind == InputKind::Image;
  if (set.image &&
      set.options.preprocess.color_convert.input_format == PreprocessColorFormat::Auto) {
    set.options.preprocess.color_convert.input_format =
        set.frame_format == "BGR"    ? PreprocessColorFormat::BGR
        : set.frame_format == "NV12" ? PreprocessColorFormat::NV12
                                     : PreprocessColorFormat::RGB;
  }
  return set;
}

// ---- arguments
// -----------------------------------------------------------------------------------
// --key value pairs and bare --flags.
struct Args {
  std::map<std::string, std::string> kv;
  std::string get(const std::string& k, const std::string& d = {}) const {
    const auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  long num(const std::string& k, long d) const {
    const auto it = kv.find(k);
    return it == kv.end() ? d : std::stol(it->second);
  }
  bool has(const std::string& k) const {
    return kv.count(k) != 0U;
  }
  static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0) {
        continue;
      }
      k = k.substr(2);
      const std::size_t eq = k.find('=');
      if (eq != std::string::npos) {
        a.kv[k.substr(0, eq)] = k.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        a.kv[k] = argv[++i];
      } else {
        a.kv[k] = "1";
      }
    }
    return a;
  }
};

} // namespace simaai::neat::qualification
