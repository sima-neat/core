#include "genai/ModelAssets.h"

namespace simaai::neat::pcie::genai::internal {
std::filesystem::path host_model_path(const std::filesystem::path& model) {
  if (model.empty() || model.native().find('\0') != std::string::npos)
    throw std::invalid_argument("Model path must name a host directory");
  std::error_code error;
  const auto path = std::filesystem::canonical(model, error);
  if (error || !std::filesystem::is_directory(path))
    throw std::invalid_argument("Model path must name an existing host directory: " +
                                model.string());
  return path;
}
std::optional<std::filesystem::path> model_asset_path(const std::filesystem::path& model,
                                                      const std::string& name) {
  const auto path = std::filesystem::weakly_canonical(model / wire::relative_name(name));
  const auto relative = path.lexically_relative(model);
  if (relative.empty() || relative == "." || *relative.begin() == "..")
    throw std::invalid_argument("Model asset escapes its host directory: " + name);
  if (!std::filesystem::exists(path))
    return std::nullopt;
  if (!std::filesystem::is_regular_file(path))
    throw std::invalid_argument("Model asset must be a regular file: " + name);
  return path;
}
ModelAssets::ModelAssets(std::filesystem::path model, std::string session, int card)
    : model_(std::move(model)), session_(std::move(session)), service_(card) {
  service_.subscribe(wire::asset_tag(session_, false));
  thread_ = std::thread([this] { receive(); });
}
ModelAssets::~ModelAssets() {
  close();
}
void ModelAssets::close() {
  stop_ = true;
  service_.interrupt();
  if (thread_.joinable())
    thread_.join();
}
void ModelAssets::check() const {
  std::lock_guard lock(mutex_);
  if (error_)
    std::rethrow_exception(error_);
}
void ModelAssets::receive() {
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
            reply["bytes"] = std::filesystem::file_size(*source);
            service_.put(source->string(), "neat-genai/" + session_ + "/model/" + name);
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
}
} // namespace simaai::neat::pcie::genai::internal
