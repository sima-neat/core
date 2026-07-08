#include "simaai/neat/pcie/Model.h"

#include "HostPcieChannel.h"
#include "ModelOptionsJsonWriter.h"
#include "PcieModelFactsReader.h"
#include "RemoteRuntime.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

namespace simaai::neat::pcie {
namespace {

constexpr auto kPostReadyStabilizationDelay = std::chrono::seconds(5);

std::string write_temp_model_options(const std::string& contents) {
  std::string tmpl = (fs::temp_directory_path() / "sima-neat-pcie-options-XXXXXX.json").string();
  std::vector<char> chars(tmpl.begin(), tmpl.end());
  chars.push_back('\0');
  const int fd = ::mkstemps(chars.data(), 5);
  if (fd < 0) {
    throw std::runtime_error(std::string("mkstemps failed: ") + std::strerror(errno));
  }
  const std::string path(chars.data());
  {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
      ::close(fd);
      throw std::runtime_error("failed to open temp model-options file: " + path);
    }
    out << contents;
  }
  ::close(fd);
  return path;
}

} // namespace

class Model::Impl {
public:
  Impl(std::string model_path, ModelOptions options, ConnectionOptions connection)
      : model_path_(std::move(model_path)), options_(std::move(options)),
        connection_(std::move(connection)), remote_(connection_) {
    validate_queue(connection_.queue);
    validate_max_inflight(connection_.max_inflight);
    (void)internal::write_model_options_json(options_);
    facts_ = internal::read_model_facts(model_path_);
    model_info_ = internal::to_public_model_info(facts_);
  }

  ~Impl() noexcept {
    try {
      close();
    } catch (...) {
    }
  }

  ModelInfo info() const {
    std::lock_guard<std::mutex> lock(mu_);
    return model_info_;
  }

  std::vector<TensorInfo> input_specs() const {
    std::lock_guard<std::mutex> lock(mu_);
    return model_info_.inputs;
  }

  std::vector<TensorInfo> output_specs() const {
    std::lock_guard<std::mutex> lock(mu_);
    return model_info_.outputs;
  }

  void build(const int readiness_timeout_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    if (readiness_timeout_ms <= 0) {
      throw std::invalid_argument("readiness_timeout_ms must be positive");
    }
    if (state_ == PipelineState::Ready) {
      return;
    }
    if (channel_.is_running() || state_ == PipelineState::Starting) {
      close_locked();
    }

    auto model_options = internal::write_model_options_json(options_);
    const std::string remote_model_path = remote_.upload_file(model_path_);

    std::optional<std::string> remote_options_path;
    std::string local_options_path;
    if (model_options.json.has_value()) {
      local_options_path = write_temp_model_options(*model_options.json);
      remote_options_path = remote_.upload_file(local_options_path);
      std::error_code ec;
      fs::remove(local_options_path, ec);
    }

    state_ = PipelineState::Starting;
    bool remote_started = false;
    try {
      remote_.start(connection_.queue, remote_model_path, remote_options_path);
      remote_started = true;
      (void)remote_.wait_ready(connection_.queue, readiness_timeout_ms);
      std::this_thread::sleep_for(kPostReadyStabilizationDelay);
      channel_.configure(facts_, connection_.queue, connection_.card_id,
                         connection_.max_inflight);
      state_ = PipelineState::Ready;
    } catch (...) {
      state_ = PipelineState::Failed;
      channel_.stop();
      if (remote_started) {
        try {
          remote_.stop(connection_.queue);
        } catch (...) {
        }
      }
      throw;
    }
  }

  bool running() const {
    std::lock_guard<std::mutex> lock(mu_);
    return state_ == PipelineState::Ready;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    close_locked();
  }

  bool push(const Tensor& tensor) {
    return push(TensorList{tensor});
  }

  bool push(const TensorList& tensors) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      ensure_ready();
    }
    return channel_.push(tensors);
  }

  std::optional<TensorList> pull(const int timeout_ms) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      ensure_ready();
    }
    return channel_.pull(timeout_ms);
  }

  TensorList run(const Tensor& tensor, const int timeout_ms) {
    ensure_ready_for_run();
    (void)push(tensor);
    return pull_strict(timeout_ms);
  }

  TensorList run(const TensorList& tensors, const int timeout_ms) {
    ensure_ready_for_run();
    (void)push(tensors);
    return pull_strict(timeout_ms);
  }

private:
  static void validate_queue(const int queue) {
    if (queue < 0 || queue > 5) {
      throw std::invalid_argument("queue must be in range 0..5");
    }
  }

  static void validate_max_inflight(const int max_inflight) {
    if (max_inflight < 0 || max_inflight > 256) {
      throw std::invalid_argument("max_inflight must be in range 0..256");
    }
  }

  void ensure_ready() const {
    if (state_ != PipelineState::Ready) {
      throw std::runtime_error("PCIe model is not built; call model.build() before run/push/pull");
    }
  }

  void ensure_ready_for_run() const {
    std::lock_guard<std::mutex> lock(mu_);
    ensure_ready();
  }

  TensorList pull_strict(const int timeout_ms) {
    auto result = pull(timeout_ms);
    if (!result) {
      throw std::runtime_error("timed out waiting for PCIe result");
    }
    return std::move(*result);
  }

  void close_locked() {
    channel_.stop();
    if (state_ == PipelineState::Ready || state_ == PipelineState::Starting ||
        state_ == PipelineState::Failed || state_ == PipelineState::Stopping) {
      state_ = PipelineState::Stopping;
      remote_.stop(connection_.queue);
      state_ = PipelineState::Exited;
    }
  }

  std::string model_path_;
  ModelOptions options_;
  ConnectionOptions connection_;
  internal::RemoteRuntime remote_;
  internal::HostPcieChannel channel_;

  mutable std::mutex mu_;
  PipelineState state_ = PipelineState::Uninitialized;
  internal::PcieModelFacts facts_;
  ModelInfo model_info_;
};

Model::Model(std::string model_path, ModelOptions options, ConnectionOptions connection)
    : impl_(std::make_unique<Impl>(std::move(model_path), std::move(options),
                                   std::move(connection))) {}

Model::~Model() noexcept = default;

ModelInfo Model::info() const {
  return impl_->info();
}

std::vector<TensorInfo> Model::input_specs() const {
  return impl_->input_specs();
}

std::vector<TensorInfo> Model::output_specs() const {
  return impl_->output_specs();
}

void Model::build(const int readiness_timeout_ms) {
  impl_->build(readiness_timeout_ms);
}

bool Model::running() const {
  return impl_->running();
}

void Model::close() {
  impl_->close();
}

bool Model::push(const Tensor& tensor) {
  return impl_->push(tensor);
}

bool Model::push(const TensorList& tensors) {
  return impl_->push(tensors);
}

std::optional<TensorList> Model::pull(const int timeout_ms) {
  return impl_->pull(timeout_ms);
}

TensorList Model::run(const Tensor& tensor, const int timeout_ms) {
  return impl_->run(tensor, timeout_ms);
}

TensorList Model::run(const TensorList& tensors, const int timeout_ms) {
  return impl_->run(tensors, timeout_ms);
}

} // namespace simaai::neat::pcie
