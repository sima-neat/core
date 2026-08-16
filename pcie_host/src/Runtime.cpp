#include "simaai/neat/pcie/Runtime.h"

#include "RuntimeModelAccess.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace simaai::neat::pcie {

class Runtime::Impl {
private:
  struct Entry {
    ModelId id = 0;
    int queue = 0;
    std::shared_ptr<Model> model;
    std::thread collector;

    std::mutex mutex;
    std::condition_variable cv;
    bool accepting = true;
    std::size_t outstanding = 0;
    std::optional<std::string> failure;
  };

public:
  explicit Impl(ConnectionOptions connection) : connection_(std::move(connection)) {
    if (connection_.queue < 0 || connection_.queue >= static_cast<int>(queues_.size())) {
      throw std::invalid_argument("queue must be in range 0..3");
    }
    if (connection_.max_inflight <= 0 || connection_.max_inflight > 256) {
      throw std::invalid_argument("Runtime max_inflight must be in range 1..256");
    }
  }

  ~Impl() noexcept {
    try {
      close();
    } catch (...) {
    }
  }

  ModelId load(std::string model_path, ModelOptions options, const int readiness_timeout_ms) {
    std::vector<ModelConfig> configs;
    configs.push_back(ModelConfig{.path = std::move(model_path), .options = std::move(options)});
    return load_models(configs, readiness_timeout_ms).front();
  }

  std::vector<ModelId> load_models(const std::vector<ModelConfig>& configs,
                                   const int readiness_timeout_ms) {
    if (configs.empty()) {
      throw std::invalid_argument("load_models requires at least one model");
    }
    if (readiness_timeout_ms <= 0) {
      throw std::invalid_argument("readiness_timeout_ms must be positive");
    }

    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    std::vector<int> assigned_queues;
    {
      std::lock_guard<std::mutex> lock(registry_mutex_);
      ensure_open_locked();
      assigned_queues = reserve_queues_locked(configs.size());
    }

    std::vector<std::shared_ptr<Entry>> new_entries;
    try {
      new_entries.reserve(configs.size());
      for (std::size_t index = 0; index < configs.size(); ++index) {
        ConnectionOptions model_connection = connection_;
        model_connection.queue = assigned_queues[index];
        auto entry = std::make_shared<Entry>();
        entry->queue = model_connection.queue;
        entry->model =
            std::make_shared<Model>(configs[index].path, configs[index].options, model_connection);
        entry->model->build(readiness_timeout_ms);
        new_entries.push_back(std::move(entry));
      }
    } catch (...) {
      close_models(new_entries);
      release_queues(assigned_queues);
      throw;
    }

    std::vector<ModelId> ids;
    try {
      {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        ensure_open_locked();
        ids.reserve(new_entries.size());
        for (const auto& entry : new_entries) {
          entry->id = next_model_id_++;
          ids.push_back(entry->id);
        }
      }
      for (const auto& entry : new_entries) {
        entry->collector = std::thread([this, entry] { collect(entry); });
      }
      {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (const auto& entry : new_entries) {
          entries_.emplace(entry->id, entry);
        }
      }
    } catch (...) {
      close_models(new_entries);
      join_collectors(new_entries);
      {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        for (const auto id : ids) {
          entries_.erase(id);
        }
      }
      release_queues(assigned_queues);
      throw;
    }
    return ids;
  }

  EnqueueResult try_enqueue(const ModelId model_id, const RequestId request_id,
                            const TensorList& tensors) {
    const auto entry = find_entry(model_id);
    std::unique_lock<std::mutex> lock(entry->mutex);
    if (!entry->accepting) {
      throw std::runtime_error("PCIe model is unloading or has failed");
    }
    if (entry->failure) {
      throw std::runtime_error(*entry->failure);
    }

    ++entry->outstanding;
    try {
      if (!internal::RuntimeModelAccess::try_push(*entry->model, request_id, tensors)) {
        --entry->outstanding;
        entry->cv.notify_all();
        return EnqueueResult::Full;
      }
    } catch (...) {
      --entry->outstanding;
      entry->cv.notify_all();
      throw;
    }
    return EnqueueResult::Accepted;
  }

  std::optional<Completion> retrieve(const int timeout_ms) {
    std::unique_lock<std::mutex> lock(completion_mutex_);
    const auto ready = [&] {
      return !completions_.empty() || !errors_.empty() || closed_notification_;
    };
    if (timeout_ms < 0) {
      completion_cv_.wait(lock, ready);
    } else if (!completion_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready)) {
      return std::nullopt;
    }

    if (!errors_.empty()) {
      std::string error = std::move(errors_.front());
      errors_.pop_front();
      throw std::runtime_error(error);
    }
    if (completions_.empty()) {
      return std::nullopt;
    }
    Completion completion = std::move(completions_.front());
    completions_.pop_front();
    return completion;
  }

  void unload(const ModelId model_id, const int drain_timeout_ms) {
    if (drain_timeout_ms < 0) {
      throw std::invalid_argument("drain_timeout_ms must be non-negative");
    }
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    const auto entry = find_entry(model_id);

    bool drained = false;
    {
      std::unique_lock<std::mutex> lock(entry->mutex);
      entry->accepting = false;
      drained = entry->cv.wait_for(lock, std::chrono::milliseconds(drain_timeout_ms), [&] {
        return entry->outstanding == 0 || entry->failure.has_value();
      });
    }

    std::exception_ptr close_error;
    try {
      entry->model->close();
    } catch (...) {
      close_error = std::current_exception();
    }
    if (entry->collector.joinable()) {
      entry->collector.join();
    }
    {
      std::lock_guard<std::mutex> lock(registry_mutex_);
      entries_.erase(model_id);
      queues_[static_cast<std::size_t>(entry->queue)] = false;
    }

    if (!drained) {
      throw std::runtime_error("timed out draining PCIe model " + std::to_string(model_id));
    }
    if (close_error) {
      std::rethrow_exception(close_error);
    }
  }

  void close() {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    std::vector<std::shared_ptr<Entry>> entries;
    {
      std::lock_guard<std::mutex> lock(registry_mutex_);
      if (closed_) {
        return;
      }
      closed_ = true;
      for (auto& [id, entry] : entries_) {
        (void)id;
        {
          std::lock_guard<std::mutex> entry_lock(entry->mutex);
          entry->accepting = false;
        }
        entries.push_back(entry);
      }
      entries_.clear();
      queues_.fill(false);
    }

    std::exception_ptr first_error;
    for (const auto& entry : entries) {
      try {
        entry->model->close();
      } catch (...) {
        if (!first_error) {
          first_error = std::current_exception();
        }
      }
    }
    join_collectors(entries);

    {
      std::lock_guard<std::mutex> lock(completion_mutex_);
      completions_.clear();
      errors_.clear();
      closed_notification_ = true;
    }
    completion_cv_.notify_all();

    if (first_error) {
      std::rethrow_exception(first_error);
    }
  }

private:
  void ensure_open_locked() const {
    if (closed_) {
      throw std::runtime_error("PCIe runtime is closed");
    }
  }

  std::vector<int> reserve_queues_locked(const std::size_t count) {
    std::vector<int> available;
    available.reserve(queues_.size());
    for (std::size_t offset = 0; offset < queues_.size(); ++offset) {
      const int queue =
          (connection_.queue + static_cast<int>(offset)) % static_cast<int>(queues_.size());
      if (!queues_[static_cast<std::size_t>(queue)]) {
        available.push_back(queue);
      }
    }
    if (count > available.size()) {
      throw std::runtime_error("insufficient PCIe queues for requested model batch");
    }
    available.resize(count);
    for (const int queue : available) {
      queues_[static_cast<std::size_t>(queue)] = true;
    }
    return available;
  }

  void release_queues(const std::vector<int>& queues) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    for (const int queue : queues) {
      queues_[static_cast<std::size_t>(queue)] = false;
    }
  }

  std::shared_ptr<Entry> find_entry(const ModelId model_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    ensure_open_locked();
    const auto it = entries_.find(model_id);
    if (it == entries_.end()) {
      throw std::invalid_argument("unknown PCIe model ID " + std::to_string(model_id));
    }
    return it->second;
  }

  void collect(const std::shared_ptr<Entry>& entry) {
    while (true) {
      std::optional<internal::RuntimeInferenceResult> result;
      try {
        result = internal::RuntimeModelAccess::pull_result(*entry->model, -1);
      } catch (const std::exception& e) {
        bool shutting_down = false;
        const std::string message =
            "PCIe model " + std::to_string(entry->id) + " receive failed: " + e.what();
        {
          std::lock_guard<std::mutex> lock(entry->mutex);
          shutting_down = !entry->accepting;
          if (!shutting_down) {
            entry->failure = message;
            entry->accepting = false;
          }
          entry->outstanding = 0;
        }
        entry->cv.notify_all();
        if (shutting_down) {
          return;
        }
        {
          std::lock_guard<std::mutex> lock(completion_mutex_);
          errors_.push_back(message);
        }
        completion_cv_.notify_all();
        return;
      }
      if (!result) {
        return;
      }

      {
        std::lock_guard<std::mutex> lock(completion_mutex_);
        completions_.push_back(Completion{
            .model_id = entry->id,
            .request_id = result->request_id,
            .outputs = std::move(result->outputs),
        });
      }
      completion_cv_.notify_one();
      {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->outstanding > 0) {
          --entry->outstanding;
        }
      }
      entry->cv.notify_all();
    }
  }

  static void close_models(const std::vector<std::shared_ptr<Entry>>& entries) noexcept {
    for (const auto& entry : entries) {
      try {
        entry->model->close();
      } catch (...) {
      }
    }
  }

  static void join_collectors(const std::vector<std::shared_ptr<Entry>>& entries) noexcept {
    for (const auto& entry : entries) {
      if (entry->collector.joinable()) {
        entry->collector.join();
      }
    }
  }

  ConnectionOptions connection_;
  std::mutex operation_mutex_;
  std::mutex registry_mutex_;
  std::unordered_map<ModelId, std::shared_ptr<Entry>> entries_;
  std::array<bool, 4> queues_{};
  ModelId next_model_id_ = 0;
  bool closed_ = false;

  std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
  std::deque<Completion> completions_;
  std::deque<std::string> errors_;
  bool closed_notification_ = false;
};

Runtime::Runtime(ConnectionOptions connection)
    : impl_(std::make_unique<Impl>(std::move(connection))) {}

Runtime::~Runtime() noexcept = default;

ModelId Runtime::load(std::string model_path, ModelOptions options,
                      const int readiness_timeout_ms) {
  return impl_->load(std::move(model_path), std::move(options), readiness_timeout_ms);
}

std::vector<ModelId> Runtime::load_models(const std::vector<ModelConfig>& models,
                                          const int readiness_timeout_ms) {
  return impl_->load_models(models, readiness_timeout_ms);
}

EnqueueResult Runtime::try_enqueue(const ModelId model_id, const RequestId request_id,
                                   const Tensor& tensor) {
  return try_enqueue(model_id, request_id, TensorList{tensor});
}

EnqueueResult Runtime::try_enqueue(const ModelId model_id, const RequestId request_id,
                                   const TensorList& tensors) {
  return impl_->try_enqueue(model_id, request_id, tensors);
}

std::optional<Completion> Runtime::retrieve(const int timeout_ms) {
  return impl_->retrieve(timeout_ms);
}

void Runtime::unload(const ModelId model_id, const int drain_timeout_ms) {
  impl_->unload(model_id, drain_timeout_ms);
}

void Runtime::close() {
  impl_->close();
}

} // namespace simaai::neat::pcie
