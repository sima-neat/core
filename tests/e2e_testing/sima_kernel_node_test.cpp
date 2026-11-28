// Kernel Node hardware suite on the A65: a built-in operation with stated
// outputs, provider operations loaded at runtime (with and without output
// inference), and the requests graph admission must refuse. Providers come from
// beside the Kernels registry this process loaded, so they match its build.
#include "../support/bevfusion_kernel_test_utils.h"

#include <simaai/scheduler/testing/kernel_provider.h>

#include <glib.h>

#include <random>

namespace bt = bevfusion_test;
namespace neat = bt::neat;

namespace {
constexpr const char* kArgmaxProvider = "libsima_kernel_argmax_loaded.so";
constexpr const char* kResizeLinearProvider = "libsima_kernel_resize_linear.so";

// The kernels repo's tests/provider_resize_linear.cpp reference: a seeded random
// 1600 x 900 BGR frame resized to 960 x 540. Its uniform_int_distribution<int>
// (0, 255) over mt19937(20261128) is, in libstdc++, the top byte of each raw
// draw; written out so the bytes do not depend on a library's distribution
// algorithm. The digests are those of that test's image and of the directly
// linked operation's output.
constexpr const char* kResizeInputSha256 =
    "24aaaccc8be3f6c7bfdbcb13849c8317c435d78f1b95c8ce8a587fdf1cb81c5b";
constexpr const char* kResizeOutputSha256 =
    "fc4a2df15af45efcf7affa148d974853c16f9da47bf02933fc403bcedd2a9769";
const std::vector<std::int64_t> kResizeInputShape{900, 1600, 3};
const std::vector<std::int64_t> kResizeOutputShape{540, 960, 3};

std::string provider(const char* file) {
  return simaai::scheduler::testing::kernels_provider(file);
}

std::string sha256(const std::vector<std::uint8_t>& bytes) {
  gchar* digest = g_compute_checksum_for_data(G_CHECKSUM_SHA256, bytes.data(), bytes.size());
  std::string result(digest);
  g_free(digest);
  return result;
}

// [1,4,4,8] distinct scores (37 is coprime to 128), so no tie decides.
std::vector<float> argmax_scores() {
  std::vector<float> values(128);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<float>((i * 37) % 128);
  return values;
}

neat::TensorList argmax_input() {
  return {bt::tensor(argmax_scores(), {1, 4, 4, 8}, "scores")};
}

std::vector<neat::KernelParameter> argmax_parameters() {
  return {{"axis", std::int64_t{-1}}, {"keepdims", false}};
}

neat::KernelOptions argmax_options(std::string operation) {
  neat::KernelOptions options;
  options.operation = std::move(operation);
  options.parameters = argmax_parameters();
  return options;
}

void check_argmax(const neat::KernelOptions& options, const std::string& what) {
  const auto outputs = bt::execute(neat::nodes::Kernel(options), {argmax_input()});
  bt::require(outputs.size() == 1 && outputs[0].size() == 1, what + ": one output0 per push");
  bt::expect_bytes(
      bt::values<std::int32_t>(outputs[0][0], neat::TensorDType::Int32, {1, 4, 4}, what),
      bt::argmax(argmax_scores(), 16, 8, 1), what + " output0");
}

void builtin_argmax() {
  auto options = argmax_options("argmax");
  options.outputs = std::vector<neat::KernelOutputSpec>{{{1, 4, 4}, neat::TensorDType::Int32}};
  check_argmax(options, "built-in argmax");
}

void loaded_argmax() {
  auto options = argmax_options("argmax_loaded");
  options.library = provider(kArgmaxProvider);
  check_argmax(options, "loaded argmax");
}

std::vector<std::uint8_t> resize_input() {
  std::mt19937 generator(20261128);
  std::vector<std::uint8_t> image(900 * 1600 * 3);
  for (auto& value : image)
    value = static_cast<std::uint8_t>(generator() >> 24);
  bt::require(sha256(image) == kResizeInputSha256,
              "resize_linear reference image differs from the kernels test's");
  return image;
}

void loaded_resize_linear() {
  neat::KernelOptions options;
  options.operation = "resize_linear";
  options.library = provider(kResizeLinearProvider);
  options.outputs =
      std::vector<neat::KernelOutputSpec>{{kResizeOutputShape, neat::TensorDType::UInt8}};
  const auto outputs = bt::execute(neat::nodes::Kernel(options),
                                   {{bt::tensor(resize_input(), kResizeInputShape, "image")}});
  bt::require(outputs.size() == 1 && outputs[0].size() == 1, "resize_linear: one output0");
  const auto bytes = bt::values<std::uint8_t>(outputs[0][0], neat::TensorDType::UInt8,
                                              kResizeOutputShape, "resize_linear");
  bt::require(sha256(bytes) == kResizeOutputSha256,
              "resize_linear output differs from the directly linked operation's");
}

void rejections() {
  bt::expect_rejected(neat::nodes::Kernel(argmax_options("argmax")), argmax_input(),
                      "argmax: has no output inference; pass outputs", "built-in without outputs");
  auto stated = argmax_options("argmax_loaded");
  stated.library = provider(kArgmaxProvider);
  stated.outputs = std::vector<neat::KernelOutputSpec>{{{1, 4, 4, 1}, neat::TensorDType::Int32}};
  bt::expect_rejected(neat::nodes::Kernel(stated), argmax_input(),
                      "argmax_loaded: stated outputs differ from the provider's inference",
                      "outputs that disagree with the provider's inference");
  // A file that does not exist; nothing is created.
  const auto absent = provider("libsima_kernel_absent.so");
  auto missing = argmax_options("argmax");
  missing.outputs = std::vector<neat::KernelOutputSpec>{{{1, 4, 4}, neat::TensorDType::Int32}};
  missing.library = absent;
  bt::expect_rejected(neat::nodes::Kernel(missing), argmax_input(), absent,
                      "a library that does not exist");
  bt::expect_invalid_options([] { neat::nodes::Kernel({}); }, "operation must name",
                             "no operation");
}
} // namespace

int main(int argc, char**) {
  return bt::main_test(argc, [] {
    bt::Suite suite("Kernel", {"builtin_argmax", "loaded_argmax", "loaded_resize_linear",
                               "rejects_invalid_requests"});
    suite.run("builtin_argmax", builtin_argmax);
    suite.run("loaded_argmax", loaded_argmax);
    suite.run("loaded_resize_linear", loaded_resize_linear);
    suite.run("rejects_invalid_requests", rejections);
    suite.finish();
  });
}
