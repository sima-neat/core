#include "genai/GenAITypes.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/imgproc.hpp>
#endif

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

namespace simaai::neat::genai {
namespace {

#if defined(SIMA_WITH_OPENCV)
Tensor tensor_from_bgr_mat(const cv::Mat& image) {
  // Match the established NEAT/OpenCV cv::Mat convention: cv::imread and
  // Model::run(vector<cv::Mat>) use BGR unless the caller says otherwise.
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  return Tensor::from_cv_mat(rgb, ImageSpec::PixelFormat::RGB, TensorMemory::CPU);
}

std::vector<Tensor> tensors_from_cv_mats(const std::vector<cv::Mat>& images) {
  std::vector<Tensor> out;
  out.reserve(images.size());
  for (const auto& image : images) {
    out.push_back(tensor_from_bgr_mat(image));
  }
  return out;
}

std::vector<Tensor> tensors_from_cv_mats(std::initializer_list<cv::Mat> images) {
  std::vector<Tensor> out;
  out.reserve(images.size());
  for (const auto& image : images) {
    out.push_back(tensor_from_bgr_mat(image));
  }
  return out;
}
#endif

} // namespace

ImageList::ImageList(std::initializer_list<Tensor> images) : images_(images) {}

ImageList::ImageList(std::vector<Tensor> images) : images_(std::move(images)) {}

ImageList& ImageList::operator=(std::initializer_list<Tensor> images) {
  images_ = images;
  return *this;
}

ImageList& ImageList::operator=(std::vector<Tensor> images) {
  images_ = std::move(images);
  return *this;
}

#if defined(SIMA_WITH_OPENCV)
ImageList::ImageList(std::initializer_list<cv::Mat> images)
    : images_(tensors_from_cv_mats(images)) {}

ImageList::ImageList(const std::vector<cv::Mat>& images) : images_(tensors_from_cv_mats(images)) {}

ImageList& ImageList::operator=(std::initializer_list<cv::Mat> images) {
  images_ = tensors_from_cv_mats(images);
  return *this;
}

ImageList& ImageList::operator=(const std::vector<cv::Mat>& images) {
  images_ = tensors_from_cv_mats(images);
  return *this;
}
#endif

bool ImageList::empty() const {
  return images_.empty();
}

std::size_t ImageList::size() const {
  return images_.size();
}

const std::vector<Tensor>& ImageList::tensors() const {
  return images_;
}

std::vector<Tensor>& ImageList::tensors() {
  return images_;
}

struct GenerationStream::Impl {
  Impl(ProducerFn producer_in, CancelFn cancel_in)
      : producer(std::move(producer_in)), cancel_callback(std::move(cancel_in)) {
    worker = std::thread([this] { run_worker(); });
  }

  ~Impl() {
    cancel();
    if (worker.joinable()) {
      worker.join();
    }
  }

  std::optional<TokenSample> next() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [&] { return !samples.empty() || closed || error != nullptr; });
    if (!samples.empty()) {
      TokenSample sample = std::move(samples.front());
      samples.pop();
      return sample;
    }
    if (error) {
      std::rethrow_exception(error);
    }
    return std::nullopt;
  }

  void cancel() {
    cancelled = true;
    if (cancel_callback) {
      cancel_callback();
    }
  }

  void run_worker() {
    try {
      Producer controller(*this);
      producer(controller);
      controller.finish(std::string{}, std::nullopt);
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(queue_mutex);
        error = std::current_exception();
        closed = true;
      }
      queue_cv.notify_all();
    }
  }

  void push_sample(TokenSample sample) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (closed) {
        return;
      }
      samples.push(std::move(sample));
    }
    queue_cv.notify_one();
  }

  ProducerFn producer;
  CancelFn cancel_callback;
  std::thread worker;
  std::atomic<bool> cancelled = false;
  std::atomic<bool> saw_stream_end = false;

  mutable std::mutex metrics_mutex;
  GenerationMetrics metrics;
  std::string finish_reason;

  std::mutex queue_mutex;
  std::condition_variable queue_cv;
  std::queue<TokenSample> samples;
  bool closed = false;
  std::exception_ptr error;
};

GenerationStream::Producer::Producer(Impl& impl) : impl_(impl) {}

void GenerationStream::Producer::record_metric(const std::string& metric, double value) {
  std::lock_guard<std::mutex> lock(impl_.metrics_mutex);
  if (metric == "ttft") {
    impl_.metrics.time_to_first_token_s = value;
  } else if (metric == "tps") {
    impl_.metrics.tokens_per_second = value;
  } else if (metric == "FULL") {
    impl_.finish_reason = "cache_full";
  }
}

void GenerationStream::Producer::record_text(const std::string& text, bool stream_end) {
  if (!text.empty()) {
    TokenSample sample;
    sample.text = text;
    sample.metrics = current_metrics();
    push(std::move(sample));
  }
  if (stream_end) {
    impl_.saw_stream_end = true;
  }
}

void GenerationStream::Producer::push(TokenSample sample) {
  impl_.push_sample(std::move(sample));
}

void GenerationStream::Producer::finish(std::string finish_reason,
                                        std::optional<std::uint32_t> generated_tokens) {
  TokenSample sample;
  {
    std::lock_guard<std::mutex> lock(impl_.metrics_mutex);
    if (generated_tokens.has_value()) {
      impl_.metrics.generated_tokens = *generated_tokens;
    }
    if (!finish_reason.empty()) {
      impl_.finish_reason = std::move(finish_reason);
    }
    if (impl_.finish_reason.empty()) {
      impl_.finish_reason = impl_.saw_stream_end ? "stop" : "interrupted";
    }
    sample.metrics = impl_.metrics;
    sample.finish_reason = impl_.finish_reason;
  }
  sample.is_final = true;

  {
    std::lock_guard<std::mutex> lock(impl_.queue_mutex);
    if (impl_.closed) {
      return;
    }
    impl_.samples.push(std::move(sample));
    impl_.closed = true;
  }
  impl_.queue_cv.notify_all();
}

bool GenerationStream::Producer::cancelled() const {
  return impl_.cancelled.load();
}

GenerationMetrics GenerationStream::Producer::current_metrics() const {
  std::lock_guard<std::mutex> lock(impl_.metrics_mutex);
  return impl_.metrics;
}

GenerationStream::GenerationStream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GenerationStream::GenerationStream(ProducerFn producer, CancelFn cancel)
    : impl_(std::make_unique<Impl>(std::move(producer), std::move(cancel))) {}

GenerationStream::~GenerationStream() = default;

GenerationStream::GenerationStream(GenerationStream&&) noexcept = default;

GenerationStream& GenerationStream::operator=(GenerationStream&&) noexcept = default;

std::optional<TokenSample> GenerationStream::next() {
  if (!impl_) {
    return std::nullopt;
  }
  return impl_->next();
}

void GenerationStream::cancel() {
  if (impl_) {
    impl_->cancel();
  }
}

GenerationStream::iterator GenerationStream::begin() {
  return iterator(this);
}

GenerationStream::iterator GenerationStream::end() {
  return iterator();
}

GenerationStream::iterator::iterator(GenerationStream* stream) : stream_(stream) {
  advance();
}

GenerationStream::iterator::reference GenerationStream::iterator::operator*() const {
  return *current_;
}

GenerationStream::iterator::pointer GenerationStream::iterator::operator->() const {
  return &*current_;
}

GenerationStream::iterator& GenerationStream::iterator::operator++() {
  advance();
  return *this;
}

void GenerationStream::iterator::operator++(int) {
  advance();
}

void GenerationStream::iterator::advance() {
  if (!stream_) {
    current_.reset();
    return;
  }
  current_ = stream_->next();
  if (!current_.has_value()) {
    stream_ = nullptr;
  }
}

bool operator==(const GenerationStream::iterator& lhs, const GenerationStream::iterator& rhs) {
  return lhs.stream_ == rhs.stream_ && lhs.current_.has_value() == rhs.current_.has_value();
}

} // namespace simaai::neat::genai
