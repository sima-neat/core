#include "genai/GenAITypes.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/imgproc.hpp>
#endif

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

} // namespace simaai::neat::genai
