// Load an image on the host and send RGB pixels with a question over PCIe.
#include "tutorial_args.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <vector>

namespace genai = simaai::neat::pcie::genai;
namespace pcie = simaai::neat::pcie;

int main(int argc, char** argv) {
  try {
    const auto args = tutorial::parse_args(argc, argv, "vlm");

    // STEP prepare-image
    const auto bgr = cv::imread(args.media, cv::IMREAD_COLOR);
    if (bgr.empty())
      throw std::invalid_argument("cannot read image: " + args.media);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous())
      rgb = rgb.clone();
    auto image = pcie::Tensor::from_vector(
        std::vector<std::uint8_t>(rgb.data, rgb.data + rgb.total() * rgb.elemSize()),
        {rgb.rows, rgb.cols, 3}, "", pcie::PixelFormat::RGB);
    // END STEP

    // STEP image-request
    genai::GenAIModel model(args.model, args.connection);
    if (!model.accepts_image())
      throw std::invalid_argument("choose a vision-language model directory");
    genai::GenerationRequest request;
    request.prompt = args.prompt;
    request.images = {image};
    request.max_new_tokens = args.max_tokens;
    std::cout << model.run(request).text << '\n';
    model.close();
    // END STEP
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "VLM tutorial: " << error.what() << '\n';
    return 1;
  }
}
