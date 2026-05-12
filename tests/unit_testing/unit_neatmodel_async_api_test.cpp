#include "model/Model.h"

#include <type_traits>
#include <utility>
#include <vector>
#include <iostream>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core/mat.hpp>
#endif

template <typename T, typename = void> struct has_pipeline_noarg : std::false_type {};
template <typename T>
struct has_pipeline_noarg<T, std::void_t<decltype(std::declval<T&>().pipeline())>>
    : std::true_type {};

template <typename T, typename = void> struct has_pipeline_opts : std::false_type {};
template <typename T>
struct has_pipeline_opts<T, std::void_t<decltype(std::declval<T&>().pipeline(
                                std::declval<const typename T::SessionOptions&>()))>>
    : std::true_type {};

template <typename T, typename = void> struct has_build_tensor_runoptions_only : std::false_type {};
template <typename T>
struct has_build_tensor_runoptions_only<T, std::void_t<decltype(std::declval<T&>().build(
                                               std::declval<const simaai::neat::Tensor&>(),
                                               std::declval<const simaai::neat::RunOptions&>()))>>
    : std::true_type {};

template <typename T, typename = void> struct has_build_sample_runoptions_only : std::false_type {};
template <typename T>
struct has_build_sample_runoptions_only<T, std::void_t<decltype(std::declval<T&>().build(
                                               std::declval<const simaai::neat::Sample&>(),
                                               std::declval<const simaai::neat::RunOptions&>()))>>
    : std::true_type {};

int main() {
  using simaai::neat::Model;
  using simaai::neat::NodeGroup;
  using simaai::neat::SampleList;
  using simaai::neat::Tensor;
  using simaai::neat::TensorList;

  static_assert(std::is_same_v<decltype(std::declval<Model&>().build()), Model::Runner>);
  static_assert(
      std::is_same_v<decltype(std::declval<Model&>().build(std::declval<const TensorList&>())),
                     Model::Runner>);
  static_assert(
      std::is_same_v<decltype(std::declval<Model&>().build(std::declval<const SampleList&>())),
                     Model::Runner>);
#if defined(SIMA_WITH_OPENCV)
  static_assert(std::is_same_v<decltype(std::declval<Model&>().build(
                                   std::declval<const std::vector<cv::Mat>&>())),
                               Model::Runner>);
#endif

  static_assert(
      std::is_same_v<decltype(std::declval<Model&>().run(std::declval<const TensorList&>())),
                     TensorList>);
  static_assert(
      std::is_same_v<decltype(std::declval<Model&>().run(std::declval<const SampleList&>())),
                     SampleList>);
#if defined(SIMA_WITH_OPENCV)
  static_assert(std::is_same_v<decltype(std::declval<Model&>().run(
                                   std::declval<const std::vector<cv::Mat>&>())),
                               TensorList>);
#endif
  static_assert(std::is_same_v<decltype(std::declval<Model&>().session()), NodeGroup>);
  static_assert(std::is_same_v<decltype(std::declval<Model&>().session(
                                   std::declval<const Model::SessionOptions&>())),
                               NodeGroup>);
  static_assert(!has_pipeline_noarg<Model>::value, "Model::pipeline() should not exist anymore");
  static_assert(!has_pipeline_opts<Model>::value,
                "Model::pipeline(SessionOptions) should not exist anymore");
  static_assert(!has_build_tensor_runoptions_only<Model>::value,
                "Model::build(Tensor, RunOptions) should not exist");
  static_assert(!has_build_sample_runoptions_only<Model>::value,
                "Model::build(Sample, RunOptions) should not exist");

  simaai::neat::RunOptions opt;
  opt.output_memory = simaai::neat::OutputMemory::Owned;
  opt.advanced.max_input_bytes = 0;
  (void)opt;

  std::cout << "[OK] unit_neatmodel_api_test passed\n";
  return 0;
}
