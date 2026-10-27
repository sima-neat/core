#pragma once

#include <filesystem>

namespace simaai::neat::internal {

// A unique extraction directory whose held lock identifies its owner across PID namespaces.
class ModelPackProcessRoot {
public:
  explicit ModelPackProcessRoot(const std::filesystem::path& base, bool collect_stale);
  ~ModelPackProcessRoot();
  ModelPackProcessRoot(const ModelPackProcessRoot&) = delete;
  ModelPackProcessRoot& operator=(const ModelPackProcessRoot&) = delete;

  const std::filesystem::path& path() const {
    return path_;
  }
  void preserve();

private:
  std::filesystem::path path_;
  int lock_fd_ = -1;
  bool preserved_ = false;
};

} // namespace simaai::neat::internal
