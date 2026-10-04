#pragma once
#include <sima_lmm/file_provider.hpp>
#include "Service.h"
#include "Protocol.h"
#include <fstream>
#include <set>
#include <sys/stat.h>

namespace simaai::neat::pcie::genai::wire {
// Used only by the worker execution thread. All destinations belong to this
// session; the daemon's receive root is never swept or removed.
class PcieFileProvider final : public simaai::llima::FileProvider {
public:
  PcieFileProvider(std::filesystem::path recv, std::string session, std::string serve,
                   std::string model)
      : service_(0, true), recv_(std::filesystem::canonical(recv)),
        prefix_("neat-genai/" + session), serve_(relative_name(serve)),
        model_(relative_name(model)) {
    validate_session(session);
    root_ = recv_ / prefix_;
    std::filesystem::create_directories(root_.parent_path());
    if (std::filesystem::canonical(root_.parent_path()) != root_.parent_path())
      throw std::runtime_error("Receive session parent must not contain symlinks");
    if (!std::filesystem::create_directory(root_))
      throw std::runtime_error("GenAI receive session already exists");
    chmod(root_.c_str(), 0700);
  }
  ~PcieFileProvider() override {
    std::error_code ec;
    std::filesystem::remove_all(root_, ec);
  }
  std::filesystem::path model_root() const {
    return root_ / "model";
  }
  std::filesystem::path reserve(std::string_view name) override {
    return model_root() / relative_name(std::string(name));
  }
  std::filesystem::path get_path(std::string_view name) override {
    auto path = reserve(name);
    fetch(path);
    return path;
  }
  bool exists(std::string_view name) override {
    const auto rel = relative_name(std::string(name));
    auto path = reserve(rel);
    if (present_.contains(path))
      return true;
    prepare(path);
    if (!service_.fetch(serve_, model_ + "/" + rel, prefix_ + "/model/" + rel, true))
      return false;
    present_.insert(path);
    return true;
  }
  std::unique_ptr<std::istream> open_stream(std::string_view name) override {
    auto p = get_path(name);
    auto stream = std::make_unique<std::ifstream>(p, std::ios::binary);
    if (!*stream)
      throw std::runtime_error("Cannot open fetched model asset");
    release(name); // the open descriptor retains the contents until consumed
    return stream;
  }
  void release(std::string_view name) override {
    evict(reserve(name));
  }
  void fetch(const std::filesystem::path& path) override {
    auto rel = relative_name(path.lexically_relative(model_root()).generic_string());
    auto p = reserve(rel);
    if (present_.contains(p))
      return;
    prepare(p);
    service_.fetch(serve_, model_ + "/" + rel, prefix_ + "/model/" + rel);
    present_.insert(p);
  }
  void evict(const std::filesystem::path& path) override {
    auto p = reserve(relative_name(path.lexically_relative(model_root()).generic_string()));
    std::filesystem::remove(p);
    present_.erase(p);
  }
  bool pulls_files() const override {
    return true;
  }
  std::filesystem::path media(uint64_t request, const std::string& root, const std::string& remote,
                              std::size_t index) {
    auto rel = "requests/" + std::to_string(request) + "/" + std::to_string(index);
    auto p = root_ / rel;
    prepare(p);
    service_.fetch(relative_name(root), relative_name(remote), prefix_ + "/" + rel);
    return p;
  }
  void release_media(uint64_t request) {
    std::filesystem::remove_all(root_ / "requests" / std::to_string(request));
  }

private:
  void prepare(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    if (std::filesystem::weakly_canonical(path).lexically_relative(root_).empty() ||
        std::filesystem::weakly_canonical(path)
            .lexically_relative(root_)
            .generic_string()
            .starts_with(".."))
      throw std::runtime_error("Asset destination escapes session directory");
  }
  Service service_;
  std::filesystem::path recv_, root_;
  std::string prefix_, serve_, model_;
  std::set<std::filesystem::path> present_;
};
} // namespace simaai::neat::pcie::genai::wire
