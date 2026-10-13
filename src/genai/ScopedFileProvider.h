#pragma once
#include <sima_lmm/file_provider.hpp>
#include <stdexcept>

namespace simaai::neat::genai::internal {
// Target and draft share one transfer channel but use disjoint asset names.
class ScopedFileProvider final : public simaai::llima::FileProvider {
public:
  ScopedFileProvider(std::shared_ptr<simaai::llima::FileProvider> files,
                     std::filesystem::path prefix)
      : files_(std::move(files)), prefix_(checked(prefix.generic_string())) {
    if (!files_)
      throw std::invalid_argument("GenAI asset provider is required");
  }
  std::filesystem::path get_path(std::string_view name) override {
    return files_->get_path(prefixed(name));
  }
  std::filesystem::path reserve(std::string_view name) override {
    return files_->reserve(prefixed(name));
  }
  std::unique_ptr<std::istream> open_stream(std::string_view name) override {
    return files_->open_stream(prefixed(name));
  }
  bool exists(std::string_view name) override {
    return files_->exists(prefixed(name));
  }
  void release(std::string_view name) override {
    files_->release(prefixed(name));
  }
  void fetch(const std::filesystem::path& path) override {
    files_->fetch(path);
  }
  void evict(const std::filesystem::path& path) override {
    files_->evict(path);
  }
  bool pulls_files() const override {
    return files_->pulls_files();
  }

private:
  static std::filesystem::path checked(std::string_view name) {
    const std::filesystem::path path(name);
    if (name.empty() || path.is_absolute() || name.find('\0') != name.npos ||
        name.find('\\') != name.npos)
      throw std::invalid_argument("Expected a relative model asset path");
    for (const auto& part : path)
      if (part == ".." || part == ".")
        throw std::invalid_argument("Unsafe model asset path");
    return path;
  }
  std::string prefixed(std::string_view name) const {
    return (prefix_ / checked(name)).generic_string();
  }
  std::shared_ptr<simaai::llima::FileProvider> files_;
  std::filesystem::path prefix_;
};
} // namespace simaai::neat::genai::internal
