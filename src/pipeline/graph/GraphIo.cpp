/**
 * @file SessionIo.cpp
 * @brief Graph save/load and JSON helpers.
 *
 * This is a mechanical split from the original monolithic Graph.cpp.
 * No behavior is intended to change.
 */
#include "pipeline/Graph.h"
#include "GraphDetail.h"
#include "internal/GraphBuildInternal.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/NeatError.h"
#include "pipeline/GraphReport.h"
#include "pipeline/ErrorCodes.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorUtil.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "builder/GraphPrinter.h"
#include "graph/Node.h"
#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "nodes/sima/Cast.h"
#include "nodes/sima/CastTess.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/Quant.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/Tess.h"
#include "model/internal/ModelRouteRetarget.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::DiagCtx;

// =====================================================================================
// Simple JSON helpers for pipeline save/load (schema is controlled by us).
// =====================================================================================

namespace {

[[noreturn]] void throw_io_error(const std::string& code, const std::string& where,
                                 const std::string& path, const std::string& detail,
                                 const std::string& hint = {}) {
  std::ostringstream oss;
  oss << where << ": " << detail;
  if (!path.empty()) {
    oss << " (path='" << path << "')";
  }
  if (!hint.empty()) {
    oss << "\nHint: " << hint;
  }

  GraphReport rep = pipeline_internal::error_util::make_report(code, oss.str());
  throw NeatError(pipeline_internal::error_util::decorate_error(rep.error_code, rep.repro_note),
                  std::move(rep));
}

struct JsonValue {
  struct JsonArray;
  struct JsonObject;

  enum class Type {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object,
  };

  Type type = Type::Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::unique_ptr<JsonArray> arr;
  std::unique_ptr<JsonObject> obj;

  JsonValue() = default;
  JsonValue(const JsonValue& other);
  JsonValue& operator=(const JsonValue& other);
  JsonValue(JsonValue&&) noexcept = default;
  JsonValue& operator=(JsonValue&&) noexcept = default;
  ~JsonValue() = default;
};

struct JsonValue::JsonArray : std::vector<JsonValue> {
  using std::vector<JsonValue>::vector;
};

struct JsonValue::JsonObject : std::unordered_map<std::string, JsonValue> {
  using std::unordered_map<std::string, JsonValue>::unordered_map;
};

JsonValue::JsonValue(const JsonValue& other)
    : type(other.type), b(other.b), num(other.num), str(other.str),
      arr(other.arr ? std::make_unique<JsonArray>(*other.arr) : nullptr),
      obj(other.obj ? std::make_unique<JsonObject>(*other.obj) : nullptr) {}

JsonValue& JsonValue::operator=(const JsonValue& other) {
  if (this == &other)
    return *this;
  type = other.type;
  b = other.b;
  num = other.num;
  str = other.str;
  arr = other.arr ? std::make_unique<JsonArray>(*other.arr) : nullptr;
  obj = other.obj ? std::make_unique<JsonObject>(*other.obj) : nullptr;
  return *this;
}

class JsonParser {
public:
  explicit JsonParser(const std::string& s) : s_(s) {}

  JsonValue parse() {
    skip_ws();
    JsonValue v = parse_value();
    skip_ws();
    if (pos_ != s_.size()) {
      fail_here("trailing characters after JSON value");
    }
    return v;
  }

private:
  const std::string& s_;
  size_t pos_ = 0;

  static std::string render_char(char c) {
    switch (c) {
    case '\0':
      return "<eof>";
    case '\n':
      return "\\n";
    case '\r':
      return "\\r";
    case '\t':
      return "\\t";
    default:
      return std::string(1, c);
    }
  }

  std::string near_token(size_t at) const {
    if (at >= s_.size())
      return "<eof>";
    const size_t kPreview = 24;
    const size_t end = std::min(s_.size(), at + kPreview);
    std::string token = s_.substr(at, end - at);
    for (char& c : token) {
      if (c == '\n' || c == '\r' || c == '\t')
        c = ' ';
    }
    if (end < s_.size())
      token += "...";
    return token;
  }

  [[noreturn]] void fail_at(size_t at, const std::string& detail) const {
    std::ostringstream oss;
    oss << "JSON parse error: offset=" << at << " near='" << near_token(at) << "'"
        << " detail=" << detail;
    throw std::runtime_error(oss.str());
  }

  [[noreturn]] void fail_here(const std::string& detail) const {
    fail_at(pos_, detail);
  }

  void skip_ws() {
    while (pos_ < s_.size() &&
           (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) {
      ++pos_;
    }
  }

  char peek() const {
    if (pos_ >= s_.size())
      return '\0';
    return s_[pos_];
  }

  char get() {
    if (pos_ >= s_.size()) {
      fail_here("unexpected end of input");
    }
    return s_[pos_++];
  }

  void expect(char c) {
    if (pos_ >= s_.size()) {
      fail_here("expected '" + render_char(c) + "' but found <eof>");
    }
    const char got = s_[pos_];
    if (got != c) {
      fail_here("expected '" + render_char(c) + "' but found '" + render_char(got) + "'");
    }
    ++pos_;
  }

  JsonValue parse_value() {
    skip_ws();
    const char c = peek();
    if (c == '"')
      return parse_string();
    if (c == '{')
      return parse_object();
    if (c == '[')
      return parse_array();
    if (c == 't' || c == 'f')
      return parse_bool();
    if (c == 'n')
      return parse_null();
    if ((c >= '0' && c <= '9') || c == '-')
      return parse_number();
    fail_here("invalid JSON value starting with '" + render_char(c) + "'");
  }

  JsonValue parse_string() {
    JsonValue v;
    v.type = JsonValue::Type::String;
    expect('"');
    std::string out;
    bool terminated = false;
    while (pos_ < s_.size()) {
      char c = get();
      if (c == '"') {
        terminated = true;
        break;
      }
      if (c == '\\') {
        char esc = get();
        switch (esc) {
        case '"':
          out.push_back('"');
          break;
        case '\\':
          out.push_back('\\');
          break;
        case '/':
          out.push_back('/');
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        default:
          fail_at(pos_ - 1, "unsupported escape sequence '\\" + render_char(esc) + "'");
        }
      } else {
        if (static_cast<unsigned char>(c) < 0x20) {
          fail_at(pos_ - 1, "control character in string");
        }
        out.push_back(c);
      }
    }
    if (!terminated) {
      fail_here("unterminated string literal");
    }
    v.str = std::move(out);
    return v;
  }

  JsonValue parse_number() {
    JsonValue v;
    v.type = JsonValue::Type::Number;
    size_t start = pos_;
    if (peek() == '-')
      ++pos_;
    bool has_int_digits = false;
    while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
      has_int_digits = true;
      ++pos_;
    }
    if (!has_int_digits) {
      fail_at(start, "invalid number: expected digits");
    }
    if (pos_ < s_.size() && s_[pos_] == '.') {
      ++pos_;
      bool has_fraction_digits = false;
      while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
        has_fraction_digits = true;
        ++pos_;
      }
      if (!has_fraction_digits) {
        fail_here("invalid number: expected digits after decimal point");
      }
    }
    if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-'))
        ++pos_;
      bool has_exponent_digits = false;
      while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
        has_exponent_digits = true;
        ++pos_;
      }
      if (!has_exponent_digits) {
        fail_here("invalid number: expected exponent digits");
      }
    }
    const std::string num = s_.substr(start, pos_ - start);
    try {
      v.num = std::stod(num);
    } catch (const std::exception&) {
      fail_at(start, "invalid numeric token '" + num + "'");
    }
    return v;
  }

  JsonValue parse_bool() {
    JsonValue v;
    v.type = JsonValue::Type::Bool;
    if (s_.compare(pos_, 4, "true") == 0) {
      v.b = true;
      pos_ += 4;
      return v;
    }
    if (s_.compare(pos_, 5, "false") == 0) {
      v.b = false;
      pos_ += 5;
      return v;
    }
    fail_here("invalid boolean token");
  }

  JsonValue parse_null() {
    if (s_.compare(pos_, 4, "null") != 0) {
      fail_here("invalid null token");
    }
    pos_ += 4;
    JsonValue v;
    v.type = JsonValue::Type::Null;
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.type = JsonValue::Type::Array;
    v.arr = std::make_unique<JsonValue::JsonArray>();
    expect('[');
    skip_ws();
    if (peek() == ']') {
      get();
      return v;
    }
    while (true) {
      v.arr->push_back(parse_value());
      skip_ws();
      char c = get();
      if (c == ']')
        break;
      if (c != ',') {
        fail_at(pos_ - 1, "expected ',' or ']' after array item, got '" + render_char(c) + "'");
      }
      skip_ws();
    }
    return v;
  }

  JsonValue parse_object() {
    JsonValue v;
    v.type = JsonValue::Type::Object;
    v.obj = std::make_unique<JsonValue::JsonObject>();
    expect('{');
    skip_ws();
    if (peek() == '}') {
      get();
      return v;
    }
    while (true) {
      JsonValue key = parse_string();
      skip_ws();
      expect(':');
      JsonValue val = parse_value();
      v.obj->emplace(key.str, std::move(val));
      skip_ws();
      char c = get();
      if (c == '}')
        break;
      if (c != ',') {
        fail_at(pos_ - 1, "expected ',' or '}' after object field, got '" + render_char(c) + "'");
      }
      skip_ws();
    }
    return v;
  }
};

static std::string json_escape(const std::string& in) {
  std::ostringstream oss;
  for (char c : in) {
    switch (c) {
    case '"':
      oss << "\\\"";
      break;
    case '\\':
      oss << "\\\\";
      break;
    case '\b':
      oss << "\\b";
      break;
    case '\f':
      oss << "\\f";
      break;
    case '\n':
      oss << "\\n";
      break;
    case '\r':
      oss << "\\r";
      break;
    case '\t':
      oss << "\\t";
      break;
    default:
      oss << c;
      break;
    }
  }
  return oss.str();
}

const JsonValue* object_field(const JsonValue::JsonObject& obj, const char* key) {
  auto it = obj.find(key);
  return it == obj.end() ? nullptr : &it->second;
}

std::string string_field(const JsonValue::JsonObject& obj, const char* key,
                         const std::string& fallback = {}) {
  const JsonValue* value = object_field(obj, key);
  return value && value->type == JsonValue::Type::String ? value->str : fallback;
}

int int_field(const JsonValue::JsonObject& obj, const char* key, int fallback = 0) {
  const JsonValue* value = object_field(obj, key);
  return value && value->type == JsonValue::Type::Number ? static_cast<int>(value->num) : fallback;
}

double double_field(const JsonValue::JsonObject& obj, const char* key, double fallback = 0.0) {
  const JsonValue* value = object_field(obj, key);
  return value && value->type == JsonValue::Type::Number ? value->num : fallback;
}

bool bool_field(const JsonValue::JsonObject& obj, const char* key, bool fallback = false) {
  const JsonValue* value = object_field(obj, key);
  return value && value->type == JsonValue::Type::Bool ? value->b : fallback;
}

std::vector<int> int_array_field(const JsonValue::JsonObject& obj, const char* key) {
  std::vector<int> out;
  const JsonValue* value = object_field(obj, key);
  if (!value || value->type != JsonValue::Type::Array || !value->arr) {
    return out;
  }
  out.reserve(value->arr->size());
  for (const auto& item : *value->arr) {
    if (item.type == JsonValue::Type::Number) {
      out.push_back(static_cast<int>(item.num));
    }
  }
  return out;
}

std::array<float, 3> float3_field(const JsonValue::JsonObject& obj, const char* key,
                                  std::array<float, 3> fallback) {
  const JsonValue* value = object_field(obj, key);
  if (!value || value->type != JsonValue::Type::Array || !value->arr) {
    return fallback;
  }
  for (std::size_t i = 0; i < fallback.size() && i < value->arr->size(); ++i) {
    const JsonValue& item = (*value->arr)[i];
    if (item.type == JsonValue::Type::Number) {
      fallback[i] = static_cast<float>(item.num);
    }
  }
  return fallback;
}

const char* combine_policy_name(CombinePolicy policy) {
  switch (policy) {
  case CombinePolicy::None:
    return "None";
  case CombinePolicy::ByFrame:
    return "ByFrame";
  case CombinePolicy::ByPts:
    return "ByPts";
  }
  return "None";
}

CombinePolicy parse_combine_policy(const std::string& value) {
  if (value == "ByFrame") {
    return CombinePolicy::ByFrame;
  }
  if (value == "ByPts") {
    return CombinePolicy::ByPts;
  }
  return CombinePolicy::None;
}

void write_input_options_json(std::ostream& oss, const InputOptions& opt) {
  oss << "{\"payload_type\":" << static_cast<int>(opt.payload_type) << ","
      << "\"format\":\"" << json_escape(opt.format.str()) << "\","
      << "\"width\":" << opt.width << ","
      << "\"height\":" << opt.height << ","
      << "\"depth\":" << opt.depth << ","
      << "\"max_width\":" << opt.max_width << ","
      << "\"max_height\":" << opt.max_height << ","
      << "\"max_depth\":" << opt.max_depth << ","
      << "\"fps_n\":" << opt.fps_n << ","
      << "\"fps_d\":" << opt.fps_d << ","
      << "\"caps_override\":\"" << json_escape(opt.caps_override) << "\","
      << "\"is_live\":" << (opt.is_live ? "true" : "false") << ","
      << "\"do_timestamp\":" << (opt.do_timestamp ? "true" : "false") << ","
      << "\"block\":" << (opt.block ? "true" : "false") << ","
      << "\"stream_type\":" << opt.stream_type << ","
      << "\"max_bytes\":" << opt.max_bytes << ","
      << "\"use_simaai_pool\":" << (opt.use_simaai_pool ? "true" : "false") << ","
      << "\"pool_min_buffers\":" << opt.pool_min_buffers << ","
      << "\"pool_max_buffers\":" << opt.pool_max_buffers << ","
      << "\"buffer_name\":\"" << json_escape(opt.buffer_name) << "\"}";
}

InputOptions parse_input_options(const JsonValue::JsonObject& obj) {
  InputOptions opt;
  const int payload_type = int_field(obj, "payload_type", int_field(obj, "input_type", -1));
  if (payload_type >= 0) {
    opt.payload_type = static_cast<PayloadType>(payload_type);
  } else {
    opt.payload_type = input_type_from_media_type(string_field(obj, "media_type", ""));
  }
  opt.format = string_field(obj, "format", opt.format.str());
  opt.width = int_field(obj, "width", opt.width);
  opt.height = int_field(obj, "height", opt.height);
  opt.depth = int_field(obj, "depth", opt.depth);
  opt.max_width = int_field(obj, "max_width", opt.max_width);
  opt.max_height = int_field(obj, "max_height", opt.max_height);
  opt.max_depth = int_field(obj, "max_depth", opt.max_depth);
  opt.fps_n = int_field(obj, "fps_n", opt.fps_n);
  opt.fps_d = int_field(obj, "fps_d", opt.fps_d);
  opt.caps_override = string_field(obj, "caps_override", opt.caps_override);
  opt.is_live = bool_field(obj, "is_live", opt.is_live);
  opt.do_timestamp = bool_field(obj, "do_timestamp", opt.do_timestamp);
  opt.block = bool_field(obj, "block", opt.block);
  opt.stream_type = int_field(obj, "stream_type", opt.stream_type);
  opt.max_bytes = static_cast<std::uint64_t>(int_field(obj, "max_bytes", 0));
  opt.use_simaai_pool = bool_field(obj, "use_simaai_pool", opt.use_simaai_pool);
  opt.pool_min_buffers = int_field(obj, "pool_min_buffers", opt.pool_min_buffers);
  opt.pool_max_buffers = int_field(obj, "pool_max_buffers", opt.pool_max_buffers);
  opt.buffer_name = string_field(obj, "buffer_name", opt.buffer_name);
  return opt;
}

void write_output_options_json(std::ostream& oss, const OutputOptions& opt) {
  oss << "{\"max_buffers\":" << opt.max_buffers << ","
      << "\"drop\":" << (opt.drop ? "true" : "false") << ","
      << "\"sync\":" << (opt.sync ? "true" : "false") << ","
      << "\"combine_policy\":\"" << combine_policy_name(opt.combine_policy) << "\"}";
}

OutputOptions parse_output_options(const JsonValue::JsonObject& obj) {
  OutputOptions opt;
  opt.max_buffers = int_field(obj, "max_buffers", opt.max_buffers);
  opt.drop = bool_field(obj, "drop", opt.drop);
  opt.sync = bool_field(obj, "sync", opt.sync);
  opt.combine_policy = parse_combine_policy(
      string_field(obj, "combine_policy", combine_policy_name(opt.combine_policy)));
  return opt;
}

template <typename EnumT> int enum_int(EnumT value) {
  return static_cast<int>(value);
}

void write_int_vector_json(std::ostream& oss, const std::vector<int>& values) {
  oss << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
}

void write_float3_json(std::ostream& oss, const std::array<float, 3>& values) {
  oss << "[" << values[0] << "," << values[1] << "," << values[2] << "]";
}

void write_resize_json(std::ostream& oss, const ResizeSpec& spec) {
  oss << "{\"enable\":" << enum_int(spec.enable) << ","
      << "\"width\":" << spec.width << ","
      << "\"height\":" << spec.height << ","
      << "\"mode\":" << enum_int(spec.mode) << ","
      << "\"pad_value\":" << spec.pad_value << ","
      << "\"scaling_type\":\"" << json_escape(spec.scaling_type) << "\"}";
}

ResizeSpec parse_resize_json(const JsonValue::JsonObject& obj) {
  ResizeSpec spec;
  spec.enable = static_cast<AutoFlag>(int_field(obj, "enable", enum_int(spec.enable)));
  spec.width = int_field(obj, "width", spec.width);
  spec.height = int_field(obj, "height", spec.height);
  spec.mode = static_cast<ResizeMode>(int_field(obj, "mode", enum_int(spec.mode)));
  spec.pad_value = int_field(obj, "pad_value", spec.pad_value);
  spec.scaling_type = string_field(obj, "scaling_type", spec.scaling_type);
  return spec;
}

void write_color_convert_json(std::ostream& oss, const ColorConvertSpec& spec) {
  oss << "{\"enable\":" << enum_int(spec.enable) << ","
      << "\"input_format\":" << enum_int(spec.input_format) << ","
      << "\"output_format\":" << enum_int(spec.output_format) << "}";
}

ColorConvertSpec parse_color_convert_json(const JsonValue::JsonObject& obj) {
  ColorConvertSpec spec;
  spec.enable = static_cast<AutoFlag>(int_field(obj, "enable", enum_int(spec.enable)));
  spec.input_format = static_cast<PreprocessColorFormat>(
      int_field(obj, "input_format", enum_int(spec.input_format)));
  spec.output_format = static_cast<PreprocessColorFormat>(
      int_field(obj, "output_format", enum_int(spec.output_format)));
  return spec;
}

void write_layout_convert_json(std::ostream& oss, const LayoutConvertSpec& spec) {
  oss << "{\"enable\":" << enum_int(spec.enable) << ",\"perm\":";
  write_int_vector_json(oss, spec.perm);
  oss << "}";
}

LayoutConvertSpec parse_layout_convert_json(const JsonValue::JsonObject& obj) {
  LayoutConvertSpec spec;
  spec.enable = static_cast<AutoFlag>(int_field(obj, "enable", enum_int(spec.enable)));
  spec.perm = int_array_field(obj, "perm");
  return spec;
}

void write_normalize_json(std::ostream& oss, const NormalizeSpec& spec) {
  oss << "{\"enable\":" << enum_int(spec.enable) << ",\"mean\":";
  write_float3_json(oss, spec.mean);
  oss << ",\"stddev\":";
  write_float3_json(oss, spec.stddev);
  oss << ",\"has_explicit_stats\":" << (spec.has_explicit_stats ? "true" : "false") << "}";
}

NormalizeSpec parse_normalize_json(const JsonValue::JsonObject& obj) {
  NormalizeSpec spec;
  spec.enable = static_cast<AutoFlag>(int_field(obj, "enable", enum_int(spec.enable)));
  spec.mean = float3_field(obj, "mean", spec.mean);
  spec.stddev = float3_field(obj, "stddev", spec.stddev);
  spec.has_explicit_stats = bool_field(obj, "has_explicit_stats", spec.has_explicit_stats);
  return spec;
}

void write_quantize_json(std::ostream& oss, const QuantizeSpec& spec) {
  oss << "{\"enable\":" << enum_int(spec.enable) << ","
      << "\"zero_point\":" << spec.zero_point << ","
      << "\"scale\":" << spec.scale << ","
      << "\"output_dtype\":\"" << json_escape(spec.output_dtype) << "\"}";
}

QuantizeSpec parse_quantize_json(const JsonValue::JsonObject& obj) {
  QuantizeSpec spec;
  spec.enable = static_cast<AutoFlag>(int_field(obj, "enable", enum_int(spec.enable)));
  spec.zero_point = int_field(obj, "zero_point", spec.zero_point);
  spec.scale = double_field(obj, "scale", spec.scale);
  spec.output_dtype = string_field(obj, "output_dtype", spec.output_dtype);
  return spec;
}

void write_tessellate_json(std::ostream& oss, const TessellateSpec& spec) {
  oss << "{\"enable\":" << enum_int(spec.enable) << ",\"slice_shape\":";
  write_int_vector_json(oss, spec.slice_shape);
  oss << "}";
}

TessellateSpec parse_tessellate_json(const JsonValue::JsonObject& obj) {
  TessellateSpec spec;
  spec.enable = static_cast<AutoFlag>(int_field(obj, "enable", enum_int(spec.enable)));
  spec.slice_shape = int_array_field(obj, "slice_shape");
  return spec;
}

void write_transform_json(std::ostream& oss, const Transform& transform) {
  oss << "{\"type\":" << enum_int(transform.type) << ",\"resize\":";
  write_resize_json(oss, transform.resize);
  oss << ",\"color_convert\":";
  write_color_convert_json(oss, transform.color_convert);
  oss << ",\"layout_convert\":";
  write_layout_convert_json(oss, transform.layout_convert);
  oss << ",\"normalize\":";
  write_normalize_json(oss, transform.normalize);
  oss << ",\"quantize\":";
  write_quantize_json(oss, transform.quantize);
  oss << ",\"tessellate\":";
  write_tessellate_json(oss, transform.tessellate);
  oss << "}";
}

Transform parse_transform_json(const JsonValue::JsonObject& obj) {
  Transform transform;
  transform.type = static_cast<TransformType>(int_field(obj, "type", enum_int(transform.type)));
  if (const JsonValue* v = object_field(obj, "resize");
      v && v->type == JsonValue::Type::Object && v->obj) {
    transform.resize = parse_resize_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "color_convert");
      v && v->type == JsonValue::Type::Object && v->obj) {
    transform.color_convert = parse_color_convert_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "layout_convert");
      v && v->type == JsonValue::Type::Object && v->obj) {
    transform.layout_convert = parse_layout_convert_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "normalize");
      v && v->type == JsonValue::Type::Object && v->obj) {
    transform.normalize = parse_normalize_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "quantize");
      v && v->type == JsonValue::Type::Object && v->obj) {
    transform.quantize = parse_quantize_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "tessellate");
      v && v->type == JsonValue::Type::Object && v->obj) {
    transform.tessellate = parse_tessellate_json(*v->obj);
  }
  return transform;
}

void write_preprocess_options_json(std::ostream& oss, const PreprocessOptions& opt) {
  oss << "{\"kind\":" << enum_int(opt.kind) << ","
      << "\"enable\":" << enum_int(opt.enable) << ","
      << "\"input_max_width\":" << opt.input_max_width << ","
      << "\"input_max_height\":" << opt.input_max_height << ","
      << "\"input_max_depth\":" << opt.input_max_depth << ",\"resize\":";
  write_resize_json(oss, opt.resize);
  oss << ",\"color_convert\":";
  write_color_convert_json(oss, opt.color_convert);
  oss << ",\"layout_convert\":";
  write_layout_convert_json(oss, opt.layout_convert);
  oss << ",\"normalize\":";
  write_normalize_json(oss, opt.normalize);
  oss << ",\"quantize\":";
  write_quantize_json(oss, opt.quantize);
  oss << ",\"tessellate\":";
  write_tessellate_json(oss, opt.tessellate);
  oss << ",\"transforms\":[";
  for (std::size_t i = 0; i < opt.transforms.size(); ++i) {
    if (i) {
      oss << ",";
    }
    write_transform_json(oss, opt.transforms[i]);
  }
  oss << "],\"preset\":" << enum_int(opt.preset) << "}";
}

PreprocessOptions parse_preprocess_options_json(const JsonValue::JsonObject& obj) {
  PreprocessOptions opt;
  opt.kind = static_cast<InputKind>(int_field(obj, "kind", enum_int(opt.kind)));
  opt.enable = static_cast<AutoFlag>(int_field(obj, "enable", enum_int(opt.enable)));
  opt.input_max_width = int_field(obj, "input_max_width", opt.input_max_width);
  opt.input_max_height = int_field(obj, "input_max_height", opt.input_max_height);
  opt.input_max_depth = int_field(obj, "input_max_depth", opt.input_max_depth);
  if (const JsonValue* v = object_field(obj, "resize");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.resize = parse_resize_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "color_convert");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.color_convert = parse_color_convert_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "layout_convert");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.layout_convert = parse_layout_convert_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "normalize");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.normalize = parse_normalize_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "quantize");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.quantize = parse_quantize_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "tessellate");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.tessellate = parse_tessellate_json(*v->obj);
  }
  opt.transforms.clear();
  if (const JsonValue* v = object_field(obj, "transforms");
      v && v->type == JsonValue::Type::Array && v->arr) {
    for (const auto& item : *v->arr) {
      if (item.type == JsonValue::Type::Object && item.obj) {
        opt.transforms.push_back(parse_transform_json(*item.obj));
      }
    }
  }
  opt.preset = static_cast<NormalizePreset>(int_field(obj, "preset", enum_int(opt.preset)));
  return opt;
}

void write_processcvu_options_json(std::ostream& oss, const ProcessCvuOptions& opt) {
  oss << "{\"pre_run_target\":\"" << json_escape(opt.pre_run_target) << "\","
      << "\"post_run_target\":\"" << json_escape(opt.post_run_target) << "\","
      << "\"async\":" << (opt.async ? "true" : "false") << "}";
}

ProcessCvuOptions parse_processcvu_options_json(const JsonValue::JsonObject& obj) {
  ProcessCvuOptions opt;
  opt.pre_run_target = string_field(obj, "pre_run_target", opt.pre_run_target);
  opt.post_run_target = string_field(obj, "post_run_target", opt.post_run_target);
  opt.async = bool_field(obj, "async", opt.async);
  return opt;
}

void write_processmla_options_json(std::ostream& oss, const ProcessMlaOptions& opt) {
  oss << "{\"async\":" << (opt.async ? "true" : "false") << ","
      << "\"output_pool_buffers\":" << opt.output_pool_buffers << ","
      << "\"defer_output_invalidate\":" << (opt.defer_output_invalidate ? "true" : "false") << "}";
}

ProcessMlaOptions parse_processmla_options_json(const JsonValue::JsonObject& obj) {
  ProcessMlaOptions opt;
  opt.async = bool_field(obj, "async", opt.async);
  opt.output_pool_buffers = int_field(obj, "output_pool_buffers", opt.output_pool_buffers);
  opt.defer_output_invalidate =
      bool_field(obj, "defer_output_invalidate", opt.defer_output_invalidate);
  return opt;
}

void write_prepared_runner_options_json(std::ostream& oss, const PreparedRunnerOptions& opt) {
  oss << "{\"mode\":\"" << json_escape(opt.mode) << "\","
      << "\"ring_depth\":" << opt.ring_depth << ","
      << "\"profile\":" << (opt.profile ? "true" : "false") << ","
      << "\"dequant_flags\":\"" << json_escape(opt.dequant_flags) << "\"}";
}

PreparedRunnerOptions parse_prepared_runner_options_json(const JsonValue::JsonObject& obj) {
  PreparedRunnerOptions opt;
  opt.mode = string_field(obj, "mode", opt.mode);
  opt.ring_depth = int_field(obj, "ring_depth", opt.ring_depth);
  opt.profile = bool_field(obj, "profile", opt.profile);
  opt.dequant_flags = string_field(obj, "dequant_flags", opt.dequant_flags);
  return opt;
}

void write_model_route_options_json(std::ostream& oss,
                                    const runtime::ModelRouteOptionsProvenance& opt) {
  oss << "{\"upstream_name\":\"" << json_escape(opt.upstream_name) << "\","
      << "\"name_suffix\":\"" << json_escape(opt.name_suffix) << "\","
      << "\"buffer_name\":\"" << json_escape(opt.buffer_name) << "\","
      << "\"processcvu_requested_run_target\":\""
      << json_escape(opt.processcvu_requested_run_target) << "\","
      << "\"processcvu\":";
  write_processcvu_options_json(oss, opt.processcvu);
  oss << ",\"processmla\":";
  write_processmla_options_json(oss, opt.processmla);
  oss << ",\"prepared_runner\":";
  write_prepared_runner_options_json(oss, opt.prepared_runner);
  oss << ",\"async_queue_depth\":" << opt.async_queue_depth
      << ",\"expose_all_outputs\":" << (opt.expose_all_outputs ? "true" : "false") << "}";
}

Model::RouteOptions parse_model_route_options_json(const JsonValue::JsonObject& obj) {
  Model::RouteOptions opt;
  opt.upstream_name = string_field(obj, "upstream_name", opt.upstream_name);
  opt.name_suffix = string_field(obj, "name_suffix", opt.name_suffix);
  opt.buffer_name = string_field(obj, "buffer_name", opt.buffer_name);
  opt.processcvu_requested_run_target =
      string_field(obj, "processcvu_requested_run_target", opt.processcvu_requested_run_target);
  if (const JsonValue* v = object_field(obj, "processcvu");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.processcvu = parse_processcvu_options_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "processmla");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.processmla = parse_processmla_options_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "prepared_runner");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.prepared_runner = parse_prepared_runner_options_json(*v->obj);
  }
  opt.async_queue_depth = int_field(obj, "async_queue_depth", opt.async_queue_depth);
  opt.expose_all_outputs = bool_field(obj, "expose_all_outputs", opt.expose_all_outputs);
  return opt;
}

void write_model_options_json(std::ostream& oss, const Model::Options& opt) {
  oss << "{\"preprocess\":";
  write_preprocess_options_json(oss, opt.preprocess);
  oss << ",\"decode_type\":" << enum_int(opt.decode_type) << ","
      << "\"score_threshold\":" << opt.score_threshold << ","
      << "\"nms_iou_threshold\":" << opt.nms_iou_threshold << ","
      << "\"top_k\":" << opt.top_k << ","
      << "\"boxdecode_original_width\":" << opt.boxdecode_original_width << ","
      << "\"boxdecode_original_height\":" << opt.boxdecode_original_height << ","
      << "\"boxdecode_resize_mode\":";
  if (opt.boxdecode_resize_mode.has_value()) {
    oss << enum_int(*opt.boxdecode_resize_mode);
  } else {
    oss << "null";
  }
  oss << ","
      << "\"upstream_name\":\"" << json_escape(opt.upstream_name) << "\","
      << "\"name_suffix\":\"" << json_escape(opt.name_suffix) << "\","
      << "\"cleanup_extracted_model_data\":"
      << (opt.cleanup_extracted_model_data ? "true" : "false") << ","
      << "\"inference_terminal\":{\"mla_only\":"
      << (opt.inference_terminal.mla_only ? "true" : "false");
  if (opt.inference_terminal.last_stage_index.has_value()) {
    oss << ",\"last_stage_index\":" << *opt.inference_terminal.last_stage_index;
  }
  if (opt.inference_terminal.last_stage_name.has_value()) {
    oss << ",\"last_stage_name\":\"" << json_escape(*opt.inference_terminal.last_stage_name)
        << "\"";
  }
  if (opt.inference_terminal.last_plugin_id.has_value()) {
    oss << ",\"last_plugin_id\":\"" << json_escape(*opt.inference_terminal.last_plugin_id) << "\"";
  }
  if (opt.inference_terminal.last_processor.has_value()) {
    oss << ",\"last_processor\":\"" << json_escape(*opt.inference_terminal.last_processor) << "\"";
  }
  oss << "},\"processcvu\":";
  write_processcvu_options_json(oss, opt.processcvu);
  oss << ",\"processmla\":";
  write_processmla_options_json(oss, opt.processmla);
  oss << ",\"prepared_runner\":";
  write_prepared_runner_options_json(oss, opt.prepared_runner);
  oss << ",\"async_queue_depth\":" << opt.async_queue_depth << "}";
}

std::string model_options_json_string(const Model::Options& opt) {
  std::ostringstream oss;
  write_model_options_json(oss, opt);
  return oss.str();
}

Model::Options parse_model_options_json(const JsonValue::JsonObject& obj) {
  Model::Options opt;
  if (const JsonValue* v = object_field(obj, "preprocess");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.preprocess = parse_preprocess_options_json(*v->obj);
  }
  opt.decode_type =
      static_cast<BoxDecodeType>(int_field(obj, "decode_type", enum_int(opt.decode_type)));
  opt.score_threshold =
      static_cast<float>(double_field(obj, "score_threshold", opt.score_threshold));
  opt.nms_iou_threshold =
      static_cast<float>(double_field(obj, "nms_iou_threshold", opt.nms_iou_threshold));
  opt.top_k = int_field(obj, "top_k", opt.top_k);
  opt.boxdecode_original_width =
      int_field(obj, "boxdecode_original_width", opt.boxdecode_original_width);
  opt.boxdecode_original_height =
      int_field(obj, "boxdecode_original_height", opt.boxdecode_original_height);
  if (const JsonValue* v = object_field(obj, "boxdecode_resize_mode");
      v && v->type == JsonValue::Type::Number) {
    opt.boxdecode_resize_mode = static_cast<ResizeMode>(static_cast<int>(v->num));
  }
  opt.upstream_name = string_field(obj, "upstream_name", opt.upstream_name);
  opt.name_suffix = string_field(obj, "name_suffix", opt.name_suffix);
  opt.cleanup_extracted_model_data =
      bool_field(obj, "cleanup_extracted_model_data", opt.cleanup_extracted_model_data);
  if (const JsonValue* v = object_field(obj, "inference_terminal");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.inference_terminal.mla_only =
        bool_field(*v->obj, "mla_only", opt.inference_terminal.mla_only);
    if (const JsonValue* stage = object_field(*v->obj, "last_stage_index");
        stage && stage->type == JsonValue::Type::Number) {
      opt.inference_terminal.last_stage_index = static_cast<std::size_t>(stage->num);
    }
    if (const std::string s = string_field(*v->obj, "last_stage_name"); !s.empty()) {
      opt.inference_terminal.last_stage_name = s;
    }
    if (const std::string s = string_field(*v->obj, "last_plugin_id"); !s.empty()) {
      opt.inference_terminal.last_plugin_id = s;
    }
    if (const std::string s = string_field(*v->obj, "last_processor"); !s.empty()) {
      opt.inference_terminal.last_processor = s;
    }
  }
  if (const JsonValue* v = object_field(obj, "processcvu");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.processcvu = parse_processcvu_options_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "processmla");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.processmla = parse_processmla_options_json(*v->obj);
  }
  if (const JsonValue* v = object_field(obj, "prepared_runner");
      v && v->type == JsonValue::Type::Object && v->obj) {
    opt.prepared_runner = parse_prepared_runner_options_json(*v->obj);
  }
  opt.async_queue_depth = int_field(obj, "async_queue_depth", opt.async_queue_depth);
  return opt;
}

const internal::ModelLineageBinding*
node_model_lineage_binding_for_io(const std::shared_ptr<Node>& node) {
  if (!node) {
    return nullptr;
  }
  if (auto* pre = dynamic_cast<Preproc*>(node.get())) {
    return pre->options().model_lineage.get();
  }
  if (auto* quant = dynamic_cast<Quant*>(node.get())) {
    return quant->options().model_lineage.get();
  }
  if (auto* tess = dynamic_cast<Tess*>(node.get())) {
    return tess->options().model_lineage.get();
  }
  if (auto* quanttess = dynamic_cast<QuantTess*>(node.get())) {
    return quanttess->options().model_lineage.get();
  }
  if (auto* cast = dynamic_cast<Cast*>(node.get())) {
    return cast->options().model_lineage.get();
  }
  if (auto* casttess = dynamic_cast<CastTess*>(node.get())) {
    return casttess->options().model_lineage.get();
  }
  if (auto* box = dynamic_cast<SimaBoxDecode*>(node.get())) {
    return box->model_lineage_binding_internal().get();
  }
  if (auto* provider = dynamic_cast<const internal::ModelLineageProvider*>(node.get())) {
    return provider->model_lineage_binding();
  }
  return nullptr;
}

template <typename Vertices>
const internal::ModelLineageBinding*
lineage_for_fragment_range(const Vertices& vertices, const runtime::FragmentPlan& fragment) {
  for (std::size_t i = fragment.graph_start; i < fragment.graph_end && i < vertices.size(); ++i) {
    if (const auto* binding = node_model_lineage_binding_for_io(vertices[i])) {
      return binding;
    }
  }
  return nullptr;
}

bool has_model_fragment_provenance(const runtime::FragmentPlan& fragment) {
  return fragment.boundary_hints.has_value() && !fragment.provenance.model_source_path.empty();
}

bool has_any_model_fragment_provenance(const std::vector<runtime::FragmentPlan>& fragments) {
  return std::any_of(fragments.begin(), fragments.end(), has_model_fragment_provenance);
}

template <typename Vertices>
void write_model_fragments_json(std::ostream& oss,
                                const std::vector<runtime::FragmentPlan>& fragments,
                                const Vertices& vertices) {
  oss << "  \"model_fragments\": [\n";
  bool first = true;
  for (const auto& fragment : fragments) {
    if (!has_model_fragment_provenance(fragment)) {
      continue;
    }
    if (!first) {
      oss << ",\n";
    }
    first = false;
    oss << "    {\"start\":" << fragment.graph_start << ","
        << "\"end\":" << fragment.graph_end << ","
        << "\"model_id\":\"" << json_escape(fragment.provenance.model_id) << "\","
        << "\"source_path\":\"" << json_escape(fragment.provenance.model_source_path) << "\","
        << "\"stage_role\":\"" << json_escape(fragment.provenance.model_stage_role) << "\"";
    if (const auto* binding = lineage_for_fragment_range(vertices, fragment)) {
      oss << ",\"model_options\":";
      write_model_options_json(oss, binding->base_options);
    } else if (!fragment.provenance.model_options_json.empty()) {
      oss << ",\"model_options\":" << fragment.provenance.model_options_json;
    }
    if (fragment.provenance.model_route.present) {
      oss << ",\"route_options\":";
      write_model_route_options_json(oss, fragment.provenance.model_route);
    }
    oss << "}";
  }
  oss << "\n  ]";
}

// Node wrapper used for load() when only the fragment is available.
class ConfiguredNode final : public Node {
public:
  ConfiguredNode(std::string kind, std::string label, std::string fragment,
                 std::vector<std::string> elements)
      : kind_(std::move(kind)), label_(std::move(label)), fragment_(std::move(fragment)),
        elements_(std::move(elements)) {}

  std::string kind() const override {
    return kind_;
  }
  std::string user_label() const override {
    return label_;
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }
  std::string backend_fragment(int /*node_index*/) const override {
    return fragment_;
  }
  std::vector<std::string> element_names(int /*node_index*/) const override {
    return elements_;
  }

private:
  std::string kind_;
  std::string label_;
  std::string fragment_;
  std::vector<std::string> elements_;
};

} // namespace

std::string Graph::describe(const GraphPrinter::Options& opt) const {
  if (composition_ && !composition_->is_linear()) {
    (void)opt;
    std::ostringstream oss;
    oss << "Graph";
    if (!endpoint_name_.empty()) {
      oss << " \"" << endpoint_name_ << "\"";
    }
    oss << " {\n";
    oss << "  nodes:\n";
    for (std::size_t i = 0; i < composition_->vertices.size(); ++i) {
      const auto& node = composition_->vertices[i];
      const auto runtime_node = composition_->runtime_node(i);
      oss << "    n" << i << ": "
          << (node ? node->kind() : (runtime_node ? runtime_node->kind() : std::string("<null>")));
      if (node && !node->user_label().empty()) {
        oss << " label=\"" << node->user_label() << "\"";
      } else if (runtime_node && !runtime_node->user_label().empty()) {
        oss << " label=\"" << runtime_node->user_label() << "\"";
      }
      if (const auto* input = dynamic_cast<const Input*>(node.get())) {
        if (!input->endpoint_name().empty()) {
          oss << " input=\"" << input->endpoint_name() << "\"";
        }
      }
      if (const auto* output = dynamic_cast<const Output*>(node.get())) {
        if (!output->endpoint_name().empty()) {
          oss << " output=\"" << output->endpoint_name() << "\"";
        }
        if (output->options().combine_policy != CombinePolicy::None) {
          oss << " combine=";
          switch (output->options().combine_policy) {
          case CombinePolicy::ByFrame:
            oss << "ByFrame";
            break;
          case CombinePolicy::ByPts:
            oss << "ByPts";
            break;
          case CombinePolicy::None:
            oss << "None";
            break;
          }
        }
      }
      oss << "\n";
    }
    oss << "  edges:\n";
    for (const auto& edge : composition_->edges) {
      oss << "    n" << edge.from << " -> n" << edge.to << " ";
      switch (edge.kind) {
      case CompositionEdgeKind::ImplicitLinear:
        oss << "(implicit)";
        break;
      case CompositionEdgeKind::RuntimePort:
        oss << "(runtime-port " << edge.from_port << " -> " << edge.to_port << ")";
        break;
      case CompositionEdgeKind::PublicEndpoint:
        oss << "(endpoint";
        if (edge.endpoint.has_value()) {
          oss << " " << edge.endpoint->from_endpoint << " -> " << edge.endpoint->to_endpoint;
        }
        oss << ")";
        break;
      }
      oss << "\n";
    }
    oss << "}\n";
    return oss.str();
  }

  const NameTransform name_transform = make_name_transform(opt_);
  const auto nodes = linear_nodes_snapshot("Graph::describe");
  const std::vector<std::shared_ptr<Node>> describe_nodes =
      session_build_materialize_model_bound_nodes(nodes, false);
  if (!name_transform_enabled(name_transform)) {
    return GraphPrinter::to_text(describe_nodes, opt);
  }

  std::vector<std::shared_ptr<Node>> renamed;
  renamed.reserve(describe_nodes.size());
  for (size_t i = 0; i < describe_nodes.size(); ++i) {
    const auto& node = describe_nodes[i];
    if (!node) {
      renamed.push_back(nullptr);
      continue;
    }
    NodeFragment frag = make_node_fragment(node, static_cast<int>(i), name_transform);
    renamed.push_back(std::make_shared<ConfiguredNode>(node->kind(), node->user_label(),
                                                       frag.fragment, frag.element_names));
  }
  return GraphPrinter::to_text(renamed, opt);
}

void Graph::save(const std::string& path) const {
  if (composition_ &&
      (!composition_->is_linear() || has_any_model_fragment_provenance(composition_->fragments))) {
    if (composition_->has_runtime_vertices()) {
      throw std::runtime_error(
          "Graph::save does not serialize runtime-stage Graph fragments yet; rebuild this Graph "
          "from the original factory (for example genai::graphs::VisionLanguage(...)) before "
          "saving, or export a built Run with save_run_json(...) for visualization/metrics.");
    }

    std::ostringstream oss;
    oss << "{\n  \"version\": 2,\n";
    oss << "  \"graph_name\":\"" << json_escape(endpoint_name_) << "\",\n";
    oss << "  \"nodes\": [\n";

    const NameTransform name_transform = make_name_transform(opt_);
    for (std::size_t i = 0; i < composition_->vertices.size(); ++i) {
      const auto& n = composition_->vertices[i];
      if (i) {
        oss << ",\n";
      }
      const std::string kind = n ? n->kind() : "<null>";
      const std::string label = n ? n->user_label() : "";
      NodeFragment frag =
          n ? make_node_fragment(n, static_cast<int>(i), name_transform) : NodeFragment{};

      oss << "    {\"kind\":\"" << json_escape(kind) << "\","
          << "\"label\":\"" << json_escape(label) << "\",";
      if (const auto* input = dynamic_cast<const Input*>(n.get())) {
        oss << "\"endpoint\":\"" << json_escape(input->endpoint_name()) << "\",";
        oss << "\"input_options\":";
        write_input_options_json(oss, input->options());
        oss << ",";
      }
      if (const auto* output = dynamic_cast<const Output*>(n.get())) {
        oss << "\"endpoint\":\"" << json_escape(output->endpoint_name()) << "\",";
        oss << "\"output_options\":";
        write_output_options_json(oss, output->options());
        oss << ",";
      }
      oss << "\"fragment\":\"" << json_escape(frag.fragment) << "\","
          << "\"elements\":[";
      for (std::size_t e = 0; e < frag.element_names.size(); ++e) {
        if (e) {
          oss << ",";
        }
        oss << "\"" << json_escape(frag.element_names[e]) << "\"";
      }
      oss << "]}";
    }

    auto edge_kind_name = [](CompositionEdgeKind kind) -> const char* {
      switch (kind) {
      case CompositionEdgeKind::ImplicitLinear:
        return "implicit";
      case CompositionEdgeKind::RuntimePort:
        return "runtime_port";
      case CompositionEdgeKind::PublicEndpoint:
        return "public_endpoint";
      }
      return "implicit";
    };

    oss << "\n  ],\n  \"edges\": [\n";
    for (std::size_t i = 0; i < composition_->edges.size(); ++i) {
      const auto& edge = composition_->edges[i];
      if (i) {
        oss << ",\n";
      }
      oss << "    {\"from\":" << edge.from << ","
          << "\"to\":" << edge.to << ","
          << "\"kind\":\"" << edge_kind_name(edge.kind) << "\","
          << "\"from_port\":\"" << json_escape(edge.from_port) << "\","
          << "\"to_port\":\"" << json_escape(edge.to_port) << "\"";
      if (edge.endpoint.has_value()) {
        oss << ",\"from_endpoint\":\"" << json_escape(edge.endpoint->from_endpoint) << "\","
            << "\"to_endpoint\":\"" << json_escape(edge.endpoint->to_endpoint) << "\"";
      }
      oss << "}";
    }
    oss << "\n  ],\n";
    write_model_fragments_json(oss, composition_->fragments, composition_->vertices);
    oss << "\n}\n";

    std::ofstream out(path, std::ios::binary);
    if (!out) {
      throw_io_error(error_codes::kIoOpen, "Graph::save", path, "failed to open file for writing",
                     "Check parent directory exists and is writable.");
    }
    out << oss.str();
    out.flush();
    if (!out.good()) {
      throw_io_error(error_codes::kIoOpen, "Graph::save", path, "failed while writing Graph JSON",
                     "Check disk space and file permissions.");
    }
    return;
  }

  std::ostringstream oss;
  oss << "{\n  \"version\": 1,\n  \"nodes\": [\n";

  const auto nodes = linear_nodes_snapshot("Graph::save");
  const std::vector<std::shared_ptr<Node>> save_nodes =
      session_build_materialize_model_bound_nodes(nodes, false);
  const NameTransform name_transform = make_name_transform(opt_);
  for (size_t i = 0; i < save_nodes.size(); ++i) {
    const auto& n = save_nodes[i];
    if (!n)
      continue;
    if (i)
      oss << ",\n";

    const std::string kind = n->kind();
    const std::string label = n->user_label();
    NodeFragment frag = make_node_fragment(n, static_cast<int>(i), name_transform);
    const std::string& fragment = frag.fragment;
    const auto& elements = frag.element_names;

    oss << "    {\"kind\":\"" << json_escape(kind) << "\","
        << "\"label\":\"" << json_escape(label) << "\","
        << "\"fragment\":\"" << json_escape(fragment) << "\","
        << "\"elements\":[";

    for (size_t e = 0; e < elements.size(); ++e) {
      if (e)
        oss << ",";
      oss << "\"" << json_escape(elements[e]) << "\"";
    }
    oss << "]}";
  }

  oss << "\n  ]\n}\n";

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw_io_error(error_codes::kIoOpen, "Graph::save", path, "failed to open file for writing",
                   "Check parent directory exists and is writable.");
  }
  out << oss.str();
  out.flush();
  if (!out.good()) {
    throw_io_error(error_codes::kIoOpen, "Graph::save", path, "failed while writing Graph JSON",
                   "Check disk space and file permissions.");
  }
}

Graph Graph::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw_io_error(error_codes::kIoOpen, "Graph::load", path, "failed to open file for reading",
                   "Check file path and read permissions.");
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  if (!in.good() && !in.eof()) {
    throw_io_error(error_codes::kIoOpen, "Graph::load", path, "failed while reading Graph JSON",
                   "Check storage health and file permissions.");
  }

  std::string content = buf.str();
  JsonValue root;
  try {
    JsonParser parser(content);
    root = parser.parse();
  } catch (const std::exception& e) {
    throw_io_error(error_codes::kIoParse, "Graph::load", path,
                   std::string("failed to parse JSON: ") + e.what(),
                   "Ensure JSON is valid and follows Graph::save schema.");
  }
  if (root.type != JsonValue::Type::Object) {
    throw_io_error(error_codes::kIoParse, "Graph::load", path,
                   "invalid JSON root: expected object");
  }
  if (!root.obj) {
    throw_io_error(error_codes::kIoParse, "Graph::load", path, "invalid JSON root object storage");
  }

  auto it_nodes = root.obj->find("nodes");
  if (it_nodes == root.obj->end() || it_nodes->second.type != JsonValue::Type::Array ||
      !it_nodes->second.arr) {
    throw_io_error(error_codes::kIoParse, "Graph::load", path, "missing required 'nodes' array");
  }

  auto it_edges = root.obj->find("edges");
  if (it_edges != root.obj->end()) {
    if (it_edges->second.type != JsonValue::Type::Array || !it_edges->second.arr) {
      throw_io_error(error_codes::kIoParse, "Graph::load", path,
                     "'edges' must be an array for connected Graph schema");
    }

    Graph graph(string_field(*root.obj, "graph_name"));
    graph.opt_.element_name_prefix.clear();
    graph.opt_.element_name_suffix.clear();

    const auto& nodes = *it_nodes->second.arr;
    graph.composition_->vertices.reserve(nodes.size());
    for (std::size_t idx = 0; idx < nodes.size(); ++idx) {
      const auto& n = nodes[idx];
      if (n.type != JsonValue::Type::Object || !n.obj) {
        throw_io_error(error_codes::kIoParse, "Graph::load", path,
                       "node entry must be object (node_index=" + std::to_string(idx) + ")");
      }
      const auto& nobj = *n.obj;
      const std::string kind = string_field(nobj, "kind", "Gst");
      const std::string label = string_field(nobj, "label");
      const std::string endpoint = string_field(nobj, "endpoint");
      const std::string fragment = string_field(nobj, "fragment");

      std::shared_ptr<Node> node;
      if (kind == "Input") {
        InputOptions opt;
        if (const JsonValue* input_opt = object_field(nobj, "input_options");
            input_opt && input_opt->type == JsonValue::Type::Object && input_opt->obj) {
          opt = parse_input_options(*input_opt->obj);
        }
        node = endpoint.empty() ? simaai::neat::nodes::Input(std::move(opt))
                                : simaai::neat::nodes::Input(endpoint, std::move(opt));
      } else if (kind == "Output") {
        OutputOptions opt;
        if (const JsonValue* output_opt = object_field(nobj, "output_options");
            output_opt && output_opt->type == JsonValue::Type::Object && output_opt->obj) {
          opt = parse_output_options(*output_opt->obj);
        }
        node = endpoint.empty() ? simaai::neat::nodes::Output(std::move(opt))
                                : simaai::neat::nodes::Output(endpoint, std::move(opt));
      } else {
        std::vector<std::string> elements;
        if (const JsonValue* elems_value = object_field(nobj, "elements");
            elems_value && elems_value->type == JsonValue::Type::Array && elems_value->arr) {
          for (std::size_t eidx = 0; eidx < elems_value->arr->size(); ++eidx) {
            const auto& e = (*elems_value->arr)[eidx];
            if (e.type != JsonValue::Type::String) {
              throw_io_error(
                  error_codes::kIoParse, "Graph::load", path,
                  "node 'elements' entry must be string (node_index=" + std::to_string(idx) +
                      ", element_index=" + std::to_string(eidx) + ")");
            }
            elements.push_back(e.str);
          }
        }
        if (fragment.empty()) {
          throw_io_error(error_codes::kIoParse, "Graph::load", path,
                         "node fragment is empty (node_index=" + std::to_string(idx) + ", kind='" +
                             kind + "')",
                         "Only built-in Input/Output endpoint nodes can omit generic fragments.");
        }
        node = std::make_shared<ConfiguredNode>(kind, label, fragment, std::move(elements));
      }

      graph.composition_->vertices.push_back(node);
      graph.groups_.push_back(Graph::GroupMeta{
          .start = idx,
          .end = idx + 1U,
          .caps_behavior = node ? node->caps_behavior() : NodeCapsBehavior::Dynamic,
          .label = "",
      });
    }

    auto parse_kind = [](const std::string& value) {
      if (value == "runtime_port") {
        return CompositionEdgeKind::RuntimePort;
      }
      if (value == "public_endpoint") {
        return CompositionEdgeKind::PublicEndpoint;
      }
      return CompositionEdgeKind::ImplicitLinear;
    };

    const auto& edges = *it_edges->second.arr;
    graph.composition_->edges.reserve(edges.size());
    for (std::size_t idx = 0; idx < edges.size(); ++idx) {
      const auto& e = edges[idx];
      if (e.type != JsonValue::Type::Object || !e.obj) {
        throw_io_error(error_codes::kIoParse, "Graph::load", path,
                       "edge entry must be object (edge_index=" + std::to_string(idx) + ")");
      }
      const auto& eobj = *e.obj;
      const int from_i = int_field(eobj, "from", -1);
      const int to_i = int_field(eobj, "to", -1);
      if (from_i < 0 || to_i < 0 ||
          static_cast<std::size_t>(from_i) >= graph.composition_->vertices.size() ||
          static_cast<std::size_t>(to_i) >= graph.composition_->vertices.size()) {
        throw_io_error(error_codes::kIoParse, "Graph::load", path,
                       "edge references invalid vertex (edge_index=" + std::to_string(idx) + ")");
      }
      CompositionEdge edge;
      edge.from = static_cast<std::size_t>(from_i);
      edge.to = static_cast<std::size_t>(to_i);
      edge.kind = parse_kind(string_field(eobj, "kind", "implicit"));
      edge.from_port = string_field(eobj, "from_port");
      edge.to_port = string_field(eobj, "to_port");
      if (edge.kind == CompositionEdgeKind::PublicEndpoint) {
        edge.endpoint = EndpointEdgeMeta{
            .from_endpoint = string_field(eobj, "from_endpoint"),
            .to_endpoint = string_field(eobj, "to_endpoint"),
        };
        graph.composition_->endpoint_mode = true;
      }
      graph.composition_->edges.push_back(std::move(edge));
    }

    if (const auto it_model_fragments = root.obj->find("model_fragments");
        it_model_fragments != root.obj->end()) {
      if (it_model_fragments->second.type != JsonValue::Type::Array ||
          !it_model_fragments->second.arr) {
        throw_io_error(error_codes::kIoParse, "Graph::load", path,
                       "'model_fragments' must be an array");
      }
      for (std::size_t midx = 0; midx < it_model_fragments->second.arr->size(); ++midx) {
        const auto& item = (*it_model_fragments->second.arr)[midx];
        if (item.type != JsonValue::Type::Object || !item.obj) {
          throw_io_error(error_codes::kIoParse, "Graph::load", path,
                         "model_fragments entry must be object (index=" + std::to_string(midx) +
                             ")");
        }
        const auto& mobj = *item.obj;
        const int start_i = int_field(mobj, "start", -1);
        const int end_i = int_field(mobj, "end", -1);
        const std::string source_path = string_field(mobj, "source_path");
        if (start_i < 0 || end_i <= start_i ||
            static_cast<std::size_t>(end_i) > graph.composition_->vertices.size()) {
          throw_io_error(error_codes::kIoParse, "Graph::load", path,
                         "model_fragments entry has invalid range (index=" + std::to_string(midx) +
                             ")");
        }
        if (source_path.empty()) {
          throw_io_error(
              error_codes::kIoParse, "Graph::load", path,
              "model_fragments entry is missing source_path (index=" + std::to_string(midx) + ")");
        }

        runtime::FragmentBoundaryHints hints;
        runtime::Provenance provenance;
        Model::RouteOptions route_options;
        Model::Options model_options;
        const std::string stage_role = string_field(mobj, "stage_role", "route");
        const bool route_like_fragment =
            stage_role.empty() || stage_role == "route" || stage_role == "full";
        bool has_route_options = false;
        try {
          if (const JsonValue* opt_value = object_field(mobj, "model_options");
              opt_value && opt_value->type == JsonValue::Type::Object && opt_value->obj) {
            model_options = parse_model_options_json(*opt_value->obj);
          }
          if (const JsonValue* route_value = object_field(mobj, "route_options");
              route_value && route_value->type == JsonValue::Type::Object && route_value->obj) {
            route_options = parse_model_route_options_json(*route_value->obj);
            has_route_options = true;
          }
          const Model model(source_path, model_options);
          if (route_like_fragment) {
            (void)internal::ModelAccess::build_graph_fragment(model, route_options, &hints);
          }
        } catch (const std::exception& e) {
          throw_io_error(error_codes::kIoParse, "Graph::load", path,
                         "failed to rehydrate model fragment from source_path '" + source_path +
                             "': " + e.what(),
                         "Ensure the original model archive is still available, or rebuild the "
                         "Graph from Model objects instead of loading this saved file.");
        }

        provenance.model_id = string_field(mobj, "model_id");
        provenance.model_source_path = source_path;
        provenance.model_stage_role = stage_role;
        provenance.model_options_json = model_options_json_string(model_options);
        if (route_like_fragment || has_route_options) {
          provenance.model_route.present = true;
          provenance.model_route.upstream_name = route_options.upstream_name;
          provenance.model_route.name_suffix = route_options.name_suffix;
          provenance.model_route.buffer_name = route_options.buffer_name;
          provenance.model_route.processcvu_requested_run_target =
              route_options.processcvu_requested_run_target;
          provenance.model_route.processcvu = route_options.processcvu;
          provenance.model_route.processmla = route_options.processmla;
          provenance.model_route.prepared_runner = route_options.prepared_runner;
          provenance.model_route.async_queue_depth = route_options.async_queue_depth;
          provenance.model_route.expose_all_outputs = route_options.expose_all_outputs;
        }

        graph.attach_fragment_boundary_hints_(static_cast<std::size_t>(start_i),
                                              static_cast<std::size_t>(end_i), std::move(hints),
                                              std::move(provenance));
      }
    }

    if (!graph.endpoint_name_.empty() && !graph.composition_->vertices.empty()) {
      graph.composition_->named_fragments.push_back(NamedFragment{
          .start = 0,
          .end = graph.composition_->vertices.size(),
          .name = graph.endpoint_name_,
          .user_named = true,
      });
    }
    graph.composition_->recompute_unique_tail();
    graph.mark_composition_changed();
    return graph;
  }

  Graph sess;
  // Loaded fragments already carry any name transforms applied at save time.
  sess.opt_.element_name_prefix.clear();
  sess.opt_.element_name_suffix.clear();
  const auto& arr = *it_nodes->second.arr;
  for (size_t idx = 0; idx < arr.size(); ++idx) {
    const auto& n = arr[idx];
    if (n.type != JsonValue::Type::Object || !n.obj) {
      throw_io_error(error_codes::kIoParse, "Graph::load", path,
                     "node entry must be object (node_index=" + std::to_string(idx) + ")");
    }
    const auto& nobj = *n.obj;

    const auto it_kind = nobj.find("kind");
    const auto it_label = nobj.find("label");
    const auto it_frag = nobj.find("fragment");
    const auto it_elems = nobj.find("elements");

    if (it_kind != nobj.end() && it_kind->second.type != JsonValue::Type::String) {
      throw_io_error(error_codes::kIoParse, "Graph::load", path,
                     "node 'kind' must be string (node_index=" + std::to_string(idx) + ")");
    }
    if (it_label != nobj.end() && it_label->second.type != JsonValue::Type::String) {
      throw_io_error(error_codes::kIoParse, "Graph::load", path,
                     "node 'label' must be string (node_index=" + std::to_string(idx) + ")");
    }
    if (it_frag != nobj.end() && it_frag->second.type != JsonValue::Type::String) {
      throw_io_error(error_codes::kIoParse, "Graph::load", path,
                     "node 'fragment' must be string (node_index=" + std::to_string(idx) + ")");
    }
    if (it_elems != nobj.end() &&
        (it_elems->second.type != JsonValue::Type::Array || !it_elems->second.arr)) {
      throw_io_error(error_codes::kIoParse, "Graph::load", path,
                     "node 'elements' must be array (node_index=" + std::to_string(idx) + ")");
    }

    std::string kind = (it_kind != nobj.end() && it_kind->second.type == JsonValue::Type::String)
                           ? it_kind->second.str
                           : "Gst";
    std::string label = (it_label != nobj.end() && it_label->second.type == JsonValue::Type::String)
                            ? it_label->second.str
                            : "";
    std::string fragment =
        (it_frag != nobj.end() && it_frag->second.type == JsonValue::Type::String)
            ? it_frag->second.str
            : "";

    std::vector<std::string> elements;
    if (it_elems != nobj.end() && it_elems->second.type == JsonValue::Type::Array &&
        it_elems->second.arr) {
      const auto& elems = *it_elems->second.arr;
      for (size_t eidx = 0; eidx < elems.size(); ++eidx) {
        const auto& e = elems[eidx];
        if (e.type != JsonValue::Type::String) {
          throw_io_error(error_codes::kIoParse, "Graph::load", path,
                         "node 'elements' entry must be string (node_index=" + std::to_string(idx) +
                             ", element_index=" + std::to_string(eidx) + ")");
        }
        elements.push_back(e.str);
      }
    }

    if (fragment.empty()) {
      throw_io_error(error_codes::kIoParse, "Graph::load", path,
                     "node fragment is empty (node_index=" + std::to_string(idx) + ", kind='" +
                         kind + "')",
                     "Ensure each node has a non-empty 'fragment' field.");
    }

    sess.add(std::make_shared<ConfiguredNode>(kind, label, fragment, elements));
  }

  return sess;
}

} // namespace simaai::neat
