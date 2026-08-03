#include "nodes/common/Caps.h"

#include "builder/ConfigJsonWire.h"
#include "builder/OutputSpec.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/internal/TempJsonFileUtil.h"
#include "gst/internal/GstLaunchBindings.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

class CustomNode final : public simaai::neat::Node {
public:
  explicit CustomNode(std::string fragment, simaai::neat::InputRole role)
      : fragment_(std::move(fragment)), role_(role) {
    if (fragment_.empty())
      fragment_ = "identity silent=true";
    const auto analysis = simaai::neat::gst::launch::analyze(fragment_);
    has_config_json_ =
        std::any_of(analysis.assignments.begin(), analysis.assignments.end(),
                    [](const auto& assignment) { return assignment.key == "config"; });
  }
  ~CustomNode() override {
    for (const auto& path : temp_paths_) {
      if (!path.empty())
        std::remove(path.c_str());
    }
  }

  std::string kind() const override {
    return "CustomNode";
  }
  std::string user_label() const override {
    return fragment_;
  }
  simaai::neat::InputRole input_role() const override {
    return role_;
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }
  bool has_config_json() const override {
    return has_config_json_;
  }
  bool wire_input_names(const std::vector<std::string>& upstream_names,
                        const std::string& tag) override {
    (void)tag;
    if (!has_config_json_ || upstream_names.empty() || upstream_names[0].empty()) {
      return false;
    }
    using namespace simaai::neat::gst::launch;
    const Analysis analysis = analyze(fragment_);
    if (!analysis.complete)
      return false;
    std::vector<AssignmentValueReplacement> replacements;
    const std::size_t created_start = temp_paths_.size();
    bool changed = false;
    std::size_t config_index = 0;
    for (const auto& assignment : analysis.assignments) {
      if (assignment.key != "config")
        continue;
      const std::string& path = assignment.canonical_value;
      if (path.empty())
        continue;
      const std::string& name = (config_index < upstream_names.size())
                                    ? upstream_names[config_index]
                                    : upstream_names.back();
      ++config_index;
      if (name.empty())
        continue;
      nlohmann::json j = load_json_file(path, "CustomNode");
      if (!simaai::neat::set_input_buffer_name_if_exists(j, name))
        continue;
      const std::string new_path = write_json_temp(j);
      replacements.push_back(AssignmentValueReplacement{
          .value_span = assignment.value_span,
          .canonical_value = new_path,
          .quote = assignment.quote,
      });
      temp_paths_.push_back(new_path);
      changed = true;
    }
    if (changed) {
      const RewriteResult rewritten = rewrite_assignment_values(fragment_, analysis, replacements);
      if (!rewritten.complete) {
        for (std::size_t i = created_start; i < temp_paths_.size(); ++i) {
          std::remove(temp_paths_[i].c_str());
        }
        temp_paths_.resize(created_start);
        return false;
      }
      fragment_ = rewritten.text;
    }
    return changed;
  }

  std::string backend_fragment(int node_index) const override {
    const std::string frag = trim_(fragment_);
    const auto analysis = simaai::neat::gst::launch::analyze(frag);
    const bool has_name = !simaai::neat::gst::launch::explicit_name_bindings(analysis).empty();
    const bool looks_complex = analysis.has_topology_syntax || !analysis.complete;

    if (has_name || looks_complex)
      return frag;

    std::string factory = first_token_(frag);
    const std::string elname = "n" + std::to_string(node_index) + "_" +
                               simaai::neat::pipeline_internal::sanitize_name(factory, "gst");
    return frag + " name=" + elname;
  }

  std::vector<std::string> element_names(int node_index) const override {
    const std::string frag = trim_(fragment_);
    const auto analysis = simaai::neat::gst::launch::analyze(frag);
    std::vector<std::string> names;
    for (const auto* binding : simaai::neat::gst::launch::explicit_name_bindings(analysis)) {
      names.push_back(binding->canonical_value);
    }
    if (!names.empty()) {
      return names;
    }

    const bool looks_complex = analysis.has_topology_syntax || !analysis.complete;
    if (looks_complex)
      return {};

    std::string factory = first_token_(frag);
    return {"n" + std::to_string(node_index) + "_" +
            simaai::neat::pipeline_internal::sanitize_name(factory, "gst")};
  }

private:
  static nlohmann::json load_json_file(const std::string& path, const char* label) {
    std::ifstream in(path);
    if (!in.is_open()) {
      throw std::runtime_error(std::string(label) + ": failed to open config: " + path);
    }
    nlohmann::json j;
    in >> j;
    return j;
  }

  static std::string make_temp_json_path() {
    return simaai::neat::pipeline_internal::make_temp_json_path("/tmp", "sima_gstnode",
                                                                "CustomNode");
  }

  static std::string write_json_temp(const nlohmann::json& j) {
    const std::string path = make_temp_json_path();
    std::ofstream out(path);
    if (!out.is_open()) {
      throw std::runtime_error("CustomNode: failed to write config override");
    }
    out << j.dump(2);
    return path;
  }

  static std::string trim_(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])))
      b++;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
      e--;
    return s.substr(b, e - b);
  }

  static std::string first_token_(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
      i++;
    size_t j = i;
    while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j])))
      j++;
    return (j > i) ? s.substr(i, j - i) : "gst";
  }

  std::string fragment_;
  simaai::neat::InputRole role_ = simaai::neat::InputRole::None;
  bool has_config_json_ = false;
  std::vector<std::string> temp_paths_;
};

class CapsRawNode final : public simaai::neat::Node, public simaai::neat::OutputSpecProvider {
public:
  CapsRawNode(std::string format, int w, int h, int fps, simaai::neat::CapsMemory mem)
      : format_(std::move(format)), w_(w), h_(h), fps_(fps), mem_(mem) {}

  std::string kind() const override {
    return "CapsRaw";
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }

  std::string backend_fragment(int node_index) const override {
    const std::string name = "n" + std::to_string(node_index) + "_caps";

    std::ostringstream caps;
    caps << "video/x-raw";
    if (mem_ == simaai::neat::CapsMemory::SystemMemory)
      caps << "(memory:SystemMemory)";
    if (!format_.empty())
      caps << ",format=" << format_;
    if (w_ > 0)
      caps << ",width=" << w_;
    if (h_ > 0)
      caps << ",height=" << h_;
    if (fps_ > 0)
      caps << ",framerate=" << fps_ << "/1";

    std::ostringstream ss;
    ss << "capsfilter name=" << name << " caps=\"" << caps.str() << "\"";
    return ss.str();
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_caps"};
  }

  simaai::neat::OutputSpec output_spec(const simaai::neat::OutputSpec& input) const override {
    simaai::neat::OutputSpec out = input;
    out.media_type = "video/x-raw";
    if (!format_.empty())
      out.format = format_;
    if (w_ > 0)
      out.width = w_;
    if (h_ > 0)
      out.height = h_;
    if (fps_ > 0) {
      out.fps_num = fps_;
      out.fps_den = 1;
    }
    out.memory = (mem_ == simaai::neat::CapsMemory::SystemMemory) ? "SystemMemory" : out.memory;
    out.dtype = "UInt8";
    if (out.format == "RGB" || out.format == "BGR") {
      out.layout = "HWC";
      out.depth = 3;
    }
    if (out.format == "GRAY8") {
      out.layout = "HW";
      out.depth = 1;
    }
    if (out.format == "NV12" || out.format == "I420") {
      out.layout = "Planar";
    }
    out.certainty = simaai::neat::SpecCertainty::Derived;
    out.note = "CapsRaw";
    out.byte_size = 0;
    out.byte_size = simaai::neat::expected_byte_size(out);
    return out;
  }

private:
  std::string format_;
  int w_ = -1;
  int h_ = -1;
  int fps_ = -1;
  simaai::neat::CapsMemory mem_ = simaai::neat::CapsMemory::Any;
};

} // namespace

namespace simaai::neat::nodes {

std::shared_ptr<simaai::neat::Node> Custom(std::string fragment, simaai::neat::InputRole role) {
  return std::make_shared<CustomNode>(std::move(fragment), role);
}

std::shared_ptr<simaai::neat::Node> CapsRaw(std::string format, int width, int height, int fps,
                                            simaai::neat::CapsMemory memory) {
  return std::make_shared<CapsRawNode>(std::move(format), width, height, fps, memory);
}

std::shared_ptr<simaai::neat::Node> CapsNV12SysMem(int w, int h, int fps) {
  return CapsRaw("NV12", w, h, fps, simaai::neat::CapsMemory::SystemMemory);
}

std::shared_ptr<simaai::neat::Node> CapsI420(int w, int h, int fps,
                                             simaai::neat::CapsMemory memory) {
  return CapsRaw("I420", w, h, fps, memory);
}

} // namespace simaai::neat::nodes
