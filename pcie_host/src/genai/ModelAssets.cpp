#include "genai/ModelAssets.h"
#include "HostPlatform.h"
#include <chrono>

namespace simaai::neat::pcie::genai::internal {
std::filesystem::path host_model_path(const std::filesystem::path& model) {
  if (model.empty() || model.native().find('\0') != std::string::npos)
    throw std::invalid_argument("Model path must name a host directory");
  std::error_code error;
  const auto path = std::filesystem::canonical(model, error);
  if (error || !std::filesystem::is_directory(path))
    throw std::invalid_argument("Model path must name an existing host directory: " +
                                pcie::internal::host_path_string(model));
  return path;
}
std::optional<std::filesystem::path> model_asset_path(const std::filesystem::path& model,
                                                      const std::string& name) {
  // Compare canonical forms: Windows may expand an 8.3 short name in one path and not the other.
  const auto root = std::filesystem::weakly_canonical(model);
  // Worker names are UTF-8; convert explicitly so Windows does not use the ANSI code page.
  const auto path = std::filesystem::weakly_canonical(
      root / pcie::internal::host_path_from_string(wire::relative_name(name)));
  const auto relative = path.lexically_relative(root);
  if (relative.empty() || relative == "." || *relative.begin() == "..")
    throw std::invalid_argument("Model asset escapes its host directory: " + name);
  if (!std::filesystem::exists(path))
    return std::nullopt;
  if (!std::filesystem::is_regular_file(path))
    throw std::invalid_argument("Model asset must be a regular file: " + name);
  return path;
}
// Shared with the worker so close() can leak a worker stuck in put() instead of hanging.
struct ModelAssets::State {
  State(std::filesystem::path model, std::string session, int card)
      : model_(std::move(model)), session_(std::move(session)), service_(card) {}
  void receive();
  std::filesystem::path model_;
  std::string session_;
  wire::Service service_;
  std::atomic<bool> stop_{false}, done_{false};
  mutable std::mutex mutex_;
  std::exception_ptr error_;
};
ModelAssets::ModelAssets(std::filesystem::path model, std::string session, int card)
    : state_(std::make_shared<State>(std::move(model), std::move(session), card)) {
  state_->service_.subscribe(wire::asset_tag(state_->session_, false));
  thread_ = std::thread([state = state_] { state->receive(); });
}
ModelAssets::~ModelAssets() {
  close();
}
void ModelAssets::close() {
  if (!thread_.joinable())
    return;
  state_->stop_ = true;
  // Windows cancels only I/O already pending, so a put() that starts waiting
  // after the first interrupt needs another one. Leak a put() that ignores it.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!state_->done_ && std::chrono::steady_clock::now() < deadline) {
    state_->service_.interrupt();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (state_->done_)
    thread_.join();
  else
    thread_.detach();
}
void ModelAssets::check() const {
  std::lock_guard lock(state_->mutex_);
  if (state_->error_)
    std::rethrow_exception(state_->error_);
}
void ModelAssets::State::receive() {
  uint64_t last = 0;
  wire::Json reply;
  try {
    while (!stop_) {
      auto payload = service_.receive(100);
      if (!payload)
        continue;
      const auto request = wire::parse(*payload, session_);
      const uint64_t id = request.at("request");
      if (request.at("kind") != "asset" || id == 0 || id > last + 1)
        throw std::runtime_error("Invalid model asset request");
      if (id < last)
        continue;
      if (id != last) {
        reply = wire::envelope(session_, id, "asset");
        try {
          const auto name = wire::relative_name(request.at("name").get<std::string>());
          const auto source = model_asset_path(model_, name);
          reply["found"] = source.has_value();
          if (source) {
            if (stop_)
              break;
            reply["bytes"] = std::filesystem::file_size(*source);
            service_.put(pcie::internal::host_path_string(*source),
                         "neat-genai/" + session_ + "/model/" + name);
          }
        } catch (const std::exception& e) {
          reply["error"] = std::string(e.what()).substr(0, 4096);
        }
        last = id;
      }
      // Replay only the last reply: a lost notification must not re-upload a
      // file the worker may already have consumed and evicted.
      service_.send(wire::asset_tag(session_, true), reply.dump());
    }
  } catch (...) {
    if (!stop_) {
      std::lock_guard lock(mutex_);
      error_ = std::current_exception();
    }
  }
  done_ = true;
}
} // namespace simaai::neat::pcie::genai::internal
