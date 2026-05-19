#include "genai/GenAIInternal.h"
#include "genai/GenAITypes.h"
#include "test_main.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// Verifies the lightweight GenAI image request types and validation rules
// without loading LLiMa model artifacts.
namespace {

simaai::neat::Tensor make_rgb_tensor(std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
  std::vector<std::uint8_t> data = {red, green, blue, red, green, blue,
                                    red, green, blue, red, green, blue};
  auto tensor = simaai::neat::Tensor::from_vector(data, {2, 2, 3}, simaai::neat::TensorMemory::CPU);
  tensor.layout = simaai::neat::TensorLayout::HWC;
  tensor.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::RGB, ""};
  return tensor;
}

void require_rgb_image_tensor(const simaai::neat::Tensor& tensor) {
  require(tensor.dtype == simaai::neat::TensorDType::UInt8, "image dtype mismatch");
  require(tensor.layout == simaai::neat::TensorLayout::HWC, "image layout mismatch");
  require(tensor.shape == std::vector<int64_t>({2, 2, 3}), "image shape mismatch");
  require(tensor.semantic.image.has_value(), "image semantics missing");
  require(tensor.semantic.image->format == simaai::neat::ImageSpec::PixelFormat::RGB,
          "image format mismatch");
}

void require_throws_contains(const std::function<void()>& fn, const std::string& expected) {
  try {
    fn();
  } catch (const std::exception& e) {
    require_contains(e.what(), expected, "unexpected exception text");
    return;
  }
  throw std::runtime_error("expected exception containing: " + expected);
}

} // namespace

RUN_TEST("unit_genai_image_types_test", ([] {
           using simaai::neat::Tensor;
           using simaai::neat::genai::ChatMessage;
           using simaai::neat::genai::GenerationRequest;
           using simaai::neat::genai::ImageList;
           namespace internal = simaai::neat::genai::internal;

           ImageList empty;
           require(empty.empty(), "default ImageList should be empty");
           require(empty.size() == 0U, "default ImageList size mismatch");

           const Tensor red = make_rgb_tensor(255, 0, 0);
           const Tensor green = make_rgb_tensor(0, 255, 0);

           GenerationRequest prompt_with_image;
           prompt_with_image.prompt = std::string{"What is in this image?"};
           prompt_with_image.images = {red};
           require(prompt_with_image.images.size() == 1U, "request image count mismatch");
           require_rgb_image_tensor(prompt_with_image.images.tensors().front());
           internal::validate_text_generation_request(prompt_with_image);

           ChatMessage message;
           message.role = "user";
           message.content = "Compare these images.";
           message.images = {red, green};
           require(message.images.size() == 2U, "message image count mismatch");
           require_rgb_image_tensor(message.images.tensors().at(0));
           require_rgb_image_tensor(message.images.tensors().at(1));

           GenerationRequest messages_with_images;
           messages_with_images.messages.push_back(message);
           internal::validate_text_generation_request(messages_with_images);

           GenerationRequest messages_with_top_level_images;
           messages_with_top_level_images.messages.push_back(
               ChatMessage{.role = "user", .content = "hello"});
           messages_with_top_level_images.images = {red};
           require_throws_contains(
               [&] { internal::validate_text_generation_request(messages_with_top_level_images); },
               "top-level images");

           GenerationRequest direct_and_cached;
           direct_and_cached.prompt = std::string{"hello"};
           direct_and_cached.images = {red};
           direct_and_cached.use_cached_images = true;
           require_throws_contains(
               [&] { internal::validate_text_generation_request(direct_and_cached); },
               "direct images");

           GenerationRequest message_direct_and_cached;
           message_direct_and_cached.messages.push_back(
               ChatMessage{.role = "user", .content = "hello", .images = {red}});
           message_direct_and_cached.messages.front().use_cached_images = true;
           require_throws_contains(
               [&] { internal::validate_text_generation_request(message_direct_and_cached); },
               "direct images");

#if defined(SIMA_WITH_OPENCV)
           cv::Mat bgr(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
           ImageList cv_images = {bgr};
           require(cv_images.size() == 1U, "cv ImageList size mismatch");
           require_rgb_image_tensor(cv_images.tensors().front());
           const cv::Mat roundtrip = cv_images.tensors().front().to_cv_mat_copy(
               simaai::neat::ImageSpec::PixelFormat::RGB);
           const cv::Vec3b pixel = roundtrip.at<cv::Vec3b>(0, 0);
           require(pixel[0] == 30 && pixel[1] == 20 && pixel[2] == 10,
                   "cv::Mat BGR to RGB conversion mismatch");
#endif
         }));
