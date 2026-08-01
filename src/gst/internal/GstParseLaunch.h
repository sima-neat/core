#pragma once

#include <gst/gst.h>

#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::gst {

struct ElementObjectInfo {
  std::string short_name;
  std::string object_path;
  std::string parent_path;
  std::string type_name;
  std::string factory_name;
};

class ParsedLaunch final {
public:
  ParsedLaunch() = default;
  ~ParsedLaunch();

  ParsedLaunch(const ParsedLaunch&) = delete;
  ParsedLaunch& operator=(const ParsedLaunch&) = delete;
  ParsedLaunch(ParsedLaunch&& other) noexcept;
  ParsedLaunch& operator=(ParsedLaunch&& other) noexcept;

  GstElement* get() const noexcept {
    return root_;
  }
  GstElement* release() noexcept;
  const GError* error() const noexcept {
    return error_;
  }
  bool had_error() const noexcept {
    return error_ != nullptr;
  }
  std::string_view error_message() const noexcept {
    return error_ && error_->message ? std::string_view(error_->message) : std::string_view{};
  }

private:
  friend ParsedLaunch parse_launch(std::string_view launch, GstParseFlags flags);
  GstElement* root_ = nullptr;
  GError* error_ = nullptr;
};

ParsedLaunch parse_launch(std::string_view launch,
                          GstParseFlags flags = GST_PARSE_FLAG_FATAL_ERRORS);

std::vector<ElementObjectInfo> inventory_elements(GstElement* root);

} // namespace simaai::neat::gst
