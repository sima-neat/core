/**
 * @file SessionIo.cpp
 * @brief Session save/load and JSON helpers.
 *
 * This is a mechanical split from the original monolithic Session.cpp.
 * No behavior is intended to change.
 */
#include "pipeline/Session.h"
#include "SessionDetail.h"
#include "internal/SessionBuildInternal.h"

#include "gst/GstInit.h"
#include "gst/GstParseLaunch.h"
#include "gst/GstBusWatch.h"
#include "gst/GstHelpers.h"

#include "pipeline/SessionError.h"
#include "pipeline/SessionReport.h"
#include "pipeline/ErrorCodes.h"
#include "internal/InputStream.h"
#include "internal/InputStreamUtil.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/SimaaiGuard.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorUtil.h"
#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "builder/OutputSpec.h"
#include "builder/GraphPrinter.h"
#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"

#include <gst/gst.h>
#include <gst/gstdebugutils.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/sdp/sdp.h>
#include <gst/video/video.h>
#include <glib.h>

#include <algorithm>
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

  SessionReport rep = pipeline_internal::error_util::make_report(code, oss.str());
  throw SessionError(pipeline_internal::error_util::decorate_error(rep.error_code, rep.repro_note),
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

std::string Session::describe(const GraphPrinter::Options& opt) const {
  const NameTransform name_transform = make_name_transform(opt_);
  const std::vector<std::shared_ptr<Node>> describe_nodes =
      session_build_materialize_model_bound_nodes(nodes_, false);
  if (!name_transform_enabled(name_transform)) {
    NodeGroup group(describe_nodes);
    return GraphPrinter::to_text(group, opt);
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
  NodeGroup group(std::move(renamed));
  return GraphPrinter::to_text(group, opt);
}

void Session::save(const std::string& path) const {
  std::ostringstream oss;
  oss << "{\n  \"version\": 1,\n  \"nodes\": [\n";

  const std::vector<std::shared_ptr<Node>> save_nodes =
      session_build_materialize_model_bound_nodes(nodes_, false);
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
    throw_io_error(error_codes::kIoOpen, "Session::save", path, "failed to open file for writing",
                   "Check parent directory exists and is writable.");
  }
  out << oss.str();
  out.flush();
  if (!out.good()) {
    throw_io_error(error_codes::kIoOpen, "Session::save", path, "failed while writing session JSON",
                   "Check disk space and file permissions.");
  }
}

Session Session::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw_io_error(error_codes::kIoOpen, "Session::load", path, "failed to open file for reading",
                   "Check file path and read permissions.");
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  if (!in.good() && !in.eof()) {
    throw_io_error(error_codes::kIoOpen, "Session::load", path, "failed while reading session JSON",
                   "Check storage health and file permissions.");
  }

  std::string content = buf.str();
  JsonValue root;
  try {
    JsonParser parser(content);
    root = parser.parse();
  } catch (const std::exception& e) {
    throw_io_error(error_codes::kIoParse, "Session::load", path,
                   std::string("failed to parse JSON: ") + e.what(),
                   "Ensure JSON is valid and follows Session::save schema.");
  }
  if (root.type != JsonValue::Type::Object) {
    throw_io_error(error_codes::kIoParse, "Session::load", path,
                   "invalid JSON root: expected object");
  }
  if (!root.obj) {
    throw_io_error(error_codes::kIoParse, "Session::load", path,
                   "invalid JSON root object storage");
  }

  auto it_nodes = root.obj->find("nodes");
  if (it_nodes == root.obj->end() || it_nodes->second.type != JsonValue::Type::Array ||
      !it_nodes->second.arr) {
    throw_io_error(error_codes::kIoParse, "Session::load", path, "missing required 'nodes' array");
  }

  Session sess;
  // Loaded fragments already carry any name transforms applied at save time.
  sess.opt_.element_name_prefix.clear();
  sess.opt_.element_name_suffix.clear();
  const auto& arr = *it_nodes->second.arr;
  for (size_t idx = 0; idx < arr.size(); ++idx) {
    const auto& n = arr[idx];
    if (n.type != JsonValue::Type::Object || !n.obj) {
      throw_io_error(error_codes::kIoParse, "Session::load", path,
                     "node entry must be object (node_index=" + std::to_string(idx) + ")");
    }
    const auto& nobj = *n.obj;

    const auto it_kind = nobj.find("kind");
    const auto it_label = nobj.find("label");
    const auto it_frag = nobj.find("fragment");
    const auto it_elems = nobj.find("elements");

    if (it_kind != nobj.end() && it_kind->second.type != JsonValue::Type::String) {
      throw_io_error(error_codes::kIoParse, "Session::load", path,
                     "node 'kind' must be string (node_index=" + std::to_string(idx) + ")");
    }
    if (it_label != nobj.end() && it_label->second.type != JsonValue::Type::String) {
      throw_io_error(error_codes::kIoParse, "Session::load", path,
                     "node 'label' must be string (node_index=" + std::to_string(idx) + ")");
    }
    if (it_frag != nobj.end() && it_frag->second.type != JsonValue::Type::String) {
      throw_io_error(error_codes::kIoParse, "Session::load", path,
                     "node 'fragment' must be string (node_index=" + std::to_string(idx) + ")");
    }
    if (it_elems != nobj.end() &&
        (it_elems->second.type != JsonValue::Type::Array || !it_elems->second.arr)) {
      throw_io_error(error_codes::kIoParse, "Session::load", path,
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
          throw_io_error(error_codes::kIoParse, "Session::load", path,
                         "node 'elements' entry must be string (node_index=" + std::to_string(idx) +
                             ", element_index=" + std::to_string(eidx) + ")");
        }
        elements.push_back(e.str);
      }
    }

    if (fragment.empty()) {
      throw_io_error(error_codes::kIoParse, "Session::load", path,
                     "node fragment is empty (node_index=" + std::to_string(idx) + ", kind='" +
                         kind + "')",
                     "Ensure each node has a non-empty 'fragment' field.");
    }

    sess.add(std::make_shared<ConfiguredNode>(kind, label, fragment, elements));
  }

  return sess;
}

} // namespace simaai::neat
