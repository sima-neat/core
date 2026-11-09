#include "pipeline/runtime/RunDeprecation.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>

namespace simaai::neat::runtime {

namespace {

struct DeprecatedForm {
  const char* form;
  const char* replacement;
};

constexpr std::size_t kForms = static_cast<std::size_t>(DeprecatedRunApi::Count);

// Same order as DeprecatedRunApi; the replacement text is the [[deprecated]] message of Run.h.
constexpr std::array<DeprecatedForm, kForms> kTable{{
    {"Run::push(const std::vector<cv::Mat>&)",
     "Run::push(TensorList{Tensor::from_cv_mat(mat, format, memory)})"},
    {"Run::push(std::string_view, const std::vector<cv::Mat>&)",
     "Run::push(input_name, TensorList{Tensor::from_cv_mat(mat, format, memory)})"},
    {"Run::try_push(const std::vector<cv::Mat>&)",
     "Run::try_push(TensorList{Tensor::from_cv_mat(mat, format, memory)})"},
    {"Run::try_push(std::string_view, const std::vector<cv::Mat>&)",
     "Run::try_push(input_name, TensorList{Tensor::from_cv_mat(mat, format, memory)})"},
    {"Run::pull(int)", "Run::pull(std::chrono::milliseconds) -> Result<TensorList>; for Sample "
                       "metadata, Run::pull(int, Sample&, PullError*)"},
    {"Run::pull(std::string_view, int)",
     "Run::pull(output_name, std::chrono::milliseconds) -> Result<TensorList>; for Sample "
     "metadata, Run::pull(output_name, int, Sample&, PullError*)"},
    {"Run::pull_tensors(int)", "Run::pull(std::chrono::milliseconds) -> Result<TensorList>"},
    {"Run::pull_tensors(std::string_view, int)",
     "Run::pull(output_name, std::chrono::milliseconds) -> Result<TensorList>"},
    {"Run::pull_samples(int)", "Run::pull(std::chrono::milliseconds) per item; for Sample "
                               "metadata, Run::pull(int, Sample&, PullError*)"},
    {"Run::pull_samples(std::string_view, int)",
     "Run::pull(output_name, std::chrono::milliseconds) per item; for Sample metadata, "
     "Run::pull(output_name, int, Sample&, PullError*)"},
}};

// Constant-initialised (no static-initialisation order), lock-free, one flag per form.
std::array<std::atomic<bool>, kForms> g_warned{};

} // namespace

void warn_deprecated_once(DeprecatedRunApi api) noexcept {
  const auto index = static_cast<std::size_t>(api);
  if (index >= kForms || g_warned[index].exchange(true, std::memory_order_relaxed)) {
    return;
  }
  std::fprintf(stderr, "[WARN] %s is deprecated and will be removed in the next release; use %s.\n",
               kTable[index].form, kTable[index].replacement);
}

} // namespace simaai::neat::runtime
