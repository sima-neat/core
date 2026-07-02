#include "simaai/neat/pcie/SimaPCIeHost.h"

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

PipelineState state_from_remote(const std::string& state) {
  if (state == "starting")
    return PipelineState::Starting;
  if (state == "ready")
    return PipelineState::Ready;
  if (state == "failed")
    return PipelineState::Failed;
  if (state == "stopping")
    return PipelineState::Stopping;
  if (state == "exited")
    return PipelineState::Exited;
  return PipelineState::Uninitialized;
}

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

class SimaPCIeHost::Impl {
public:
  explicit Impl(ConnectionOptions connection)
      : connection_(std::move(connection)), remote_(connection_) {}

  ~Impl() noexcept {
    try {
      stop();
    } catch (...) {
    }
  }

  ModelInfo load_metadata(const std::string& model_path, const ModelOptions& options) {
    std::lock_guard<std::mutex> lock(mu_);
    if (state_ == PipelineState::Ready || channel_.is_running()) {
      throw std::runtime_error("cannot load metadata while PCIe pipeline is running");
    }
    (void)internal::write_model_options_json(options);
    facts_ = internal::read_model_facts(model_path);
    model_info_ = internal::to_public_model_info(facts_);
    return model_info_;
  }

  ModelInfo init_pipeline(const std::string& model_path, const ModelOptions& options,
                          const int readiness_timeout_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    validate_queue(connection_.queue);
    if (readiness_timeout_ms <= 0) {
      throw std::invalid_argument("readiness_timeout_ms must be positive");
    }
    if (state_ == PipelineState::Ready || channel_.is_running()) {
      stop_locked();
    }

    auto model_options = internal::write_model_options_json(options);

    facts_ = internal::read_model_facts(model_path);
    model_info_ = internal::to_public_model_info(facts_);
    const std::string remote_model_path = remote_.upload_file(model_path);

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
      channel_.configure(facts_, connection_.queue, connection_.card_id);
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
    return model_info_;
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mu_);
    stop_locked();
  }

  Status status() const {
    std::lock_guard<std::mutex> lock(mu_);
    Status out;
    out.state = state_;
    out.queue = connection_.queue;
    if (state_ == PipelineState::Ready || state_ == PipelineState::Starting ||
        state_ == PipelineState::Failed || state_ == PipelineState::Stopping) {
      const auto remote_status = remote_.read_status(connection_.queue);
      out.message = remote_status.message;
      out.error_code = remote_status.error_code;
      if (!remote_status.state.empty()) {
        out.state = state_from_remote(remote_status.state);
      }
    }
    return out;
  }

  bool push(const Tensor& tensor) {
    return push(TensorList{tensor});
  }

  bool push(const TensorList& tensors) {
    std::lock_guard<std::mutex> lock(mu_);
    ensure_ready();
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

  void ensure_ready() const {
    if (state_ != PipelineState::Ready) {
      throw std::runtime_error("PCIe host is not ready");
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

  void stop_locked() {
    channel_.stop();
    if (state_ == PipelineState::Ready || state_ == PipelineState::Starting ||
        state_ == PipelineState::Failed || state_ == PipelineState::Stopping) {
      state_ = PipelineState::Stopping;
      remote_.stop(connection_.queue);
      state_ = PipelineState::Exited;
    }
  }

  ConnectionOptions connection_;
  internal::RemoteRuntime remote_;
  internal::HostPcieChannel channel_;

  mutable std::mutex mu_;
  PipelineState state_ = PipelineState::Uninitialized;
  internal::PcieModelFacts facts_;
  ModelInfo model_info_;
};

SimaPCIeHost::SimaPCIeHost(ConnectionOptions connection)
    : impl_(std::make_unique<Impl>(std::move(connection))) {}

SimaPCIeHost::~SimaPCIeHost() noexcept = default;

ModelInfo SimaPCIeHost::load_metadata(const std::string& model_path, const ModelOptions& options) {
  return impl_->load_metadata(model_path, options);
}

ModelInfo SimaPCIeHost::init_pipeline(const std::string& model_path, const ModelOptions& options,
                                      const int readiness_timeout_ms) {
  return impl_->init_pipeline(model_path, options, readiness_timeout_ms);
}

void SimaPCIeHost::stop() {
  impl_->stop();
}

Status SimaPCIeHost::status() const {
  return impl_->status();
}

bool SimaPCIeHost::push(const Tensor& tensor) {
  return impl_->push(tensor);
}

bool SimaPCIeHost::push(const TensorList& tensors) {
  return impl_->push(tensors);
}

std::optional<TensorList> SimaPCIeHost::pull(const int timeout_ms) {
  return impl_->pull(timeout_ms);
}

TensorList SimaPCIeHost::run(const Tensor& tensor, const int timeout_ms) {
  return impl_->run(tensor, timeout_ms);
}

TensorList SimaPCIeHost::run(const TensorList& tensors, const int timeout_ms) {
  return impl_->run(tensors, timeout_ms);
}

} // namespace simaai::neat::pcie
