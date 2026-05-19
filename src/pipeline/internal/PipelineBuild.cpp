#include "pipeline/internal/PipelineBuild.h"

#include <cctype>
#include <string_view>
#include <unordered_set>

namespace simaai::neat {

std::string apply_name_transform_once(const std::string& value, const NameTransform& transform) {
  if (value.empty() || !name_transform_enabled(transform))
    return value;
  std::string out = value;
  if (!transform.prefix.empty() && out.rfind(transform.prefix, 0) != 0) {
    out = transform.prefix + out;
  }
  if (!transform.suffix.empty()) {
    if (out.size() < transform.suffix.size() ||
        out.compare(out.size() - transform.suffix.size(), transform.suffix.size(),
                    transform.suffix) != 0) {
      out += transform.suffix;
    }
  }
  return pipeline_internal::sanitize_name(out);
}

} // namespace simaai::neat

namespace simaai::neat::pipeline_internal {
namespace {

std::vector<std::string> split_buffer_names(const std::string& raw) {
  std::vector<std::string> out;
  const std::string trimmed = trim_copy(raw);
  if (trimmed.empty())
    return out;

  size_t start = 0;
  while (start < trimmed.size()) {
    size_t end = trimmed.find(',', start);
    if (end == std::string::npos)
      end = trimmed.size();
    std::string item = trim_copy(std::string_view(trimmed).substr(start, end - start));
    if (!item.empty())
      out.push_back(std::move(item));
    start = end + 1;
  }
  return out;
}

} // namespace

PipelineBuildContext::PipelineBuildContext(const SessionOptions& opt)
    : name_transform_(make_name_transform(opt)) {}

PipelineBuildContext::PipelineBuildContext(const NameTransform& transform)
    : name_transform_(transform) {}

std::string PipelineBuildContext::resolve_buffer_name(const std::string& raw) const {
  return pipeline_internal::resolve_buffer_name(raw, name_transform_);
}

std::vector<std::string>
PipelineBuildContext::resolve_expected_buffer_names(const std::string& raw) const {
  return pipeline_internal::resolve_expected_buffer_names(raw, name_transform_);
}

std::string resolve_buffer_name(const std::string& raw, const NameTransform& transform) {
  const std::string base = raw.empty() ? "decoder" : raw;
  if (base.find(',') == std::string::npos) {
    return simaai::neat::apply_name_transform_once(base, transform);
  }

  std::string out;
  size_t pos = 0;
  while (pos <= base.size()) {
    const size_t comma = base.find(',', pos);
    const size_t end = (comma == std::string::npos) ? base.size() : comma;
    const std::string token = trim_copy(std::string_view(base).substr(pos, end - pos));
    if (!token.empty()) {
      if (!out.empty())
        out += ",";
      out += simaai::neat::apply_name_transform_once(token, transform);
    }
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }

  if (out.empty()) {
    return simaai::neat::apply_name_transform_once("decoder", transform);
  }
  return out;
}

std::vector<std::string> resolve_expected_buffer_names(const std::string& raw,
                                                       const NameTransform& transform) {
  std::vector<std::string> names = split_buffer_names(raw);
  if (names.empty()) {
    names.push_back("decoder");
  }

  std::vector<std::string> out;
  out.reserve(names.size() * 2);
  std::unordered_set<std::string> seen;
  for (const auto& name : names) {
    const std::string base = name.empty() ? "decoder" : name;
    const std::string transformed = simaai::neat::apply_name_transform_once(base, transform);
    if (seen.insert(transformed).second)
      out.push_back(transformed);
    if (transformed != base && seen.insert(base).second)
      out.push_back(base);
  }
  return out;
}

} // namespace simaai::neat::pipeline_internal
