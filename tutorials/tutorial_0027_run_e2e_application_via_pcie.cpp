/**
 * Run an MPK end-to-end over PCIe using explicit nodes:
 *   PCIeSrc -> Model pipeline (no appsrc/appsink) -> PCIeSink
 *
 * Single-input tensor models only.
 */

#include "neat/session.h"
#include "neat/models.h"
#include "neat/nodes.h"
#include "builder/OutputSpec.h"
#include "gst/GstHelpers.h"
#include "tutorial_common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_stop_requested{false};

void on_signal(int /*signum*/) {
  g_stop_requested.store(true, std::memory_order_relaxed);
}

bool get_arg_string(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

bool get_arg_int(int argc, char** argv, const std::string& key, int& out) {
  std::string val;
  if (!get_arg_string(argc, argv, key, val))
    return false;
  out = std::stoi(val);
  return true;
}

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == key)
      return true;
  }
  return false;
}

bool runtime_unavailable(const std::string& msg) {
  return msg.find("dispatcher unavailable") != std::string::npos ||
         msg.find("No such element") != std::string::npos ||
         msg.find("missing element") != std::string::npos ||
         msg.find("could not load") != std::string::npos ||
         msg.find("undefined symbol") != std::string::npos ||
         msg.find("libMLArt.so") != std::string::npos ||
         msg.find("build.parse_launch") != std::string::npos;
}

fs::path find_default_model(const fs::path& root) {
  const std::vector<fs::path> candidates = {
      root / "_models" / "yolov8n_real1_mpk.tar.gz",
      root / "tmp" / "yolo_v8s_mpk.tar.gz",
      root / "tmp" / "yolov8s_mpk.tar.gz",
  };
  for (const auto& p : candidates) {
    if (fs::exists(p))
      return p;
  }
  return {};
}

std::size_t mul_or_throw(std::size_t a, std::size_t b, const char* tag) {
  if (a == 0 || b == 0)
    return 0;
  if (a > (static_cast<std::size_t>(-1) / b)) {
    throw std::runtime_error(std::string("size overflow: ") + tag);
  }
  return a * b;
}

struct TensorInputShape {
  int width = -1;
  int height = -1;
  int depth = -1;
  bool used_max_limits = false;
};

TensorInputShape resolve_tensor_input_shape(const simaai::neat::InputOptions& in) {
  TensorInputShape shape;
  shape.width = (in.width > 0) ? in.width : in.max_width;
  shape.height = (in.height > 0) ? in.height : in.max_height;
  shape.depth = (in.depth > 0) ? in.depth : in.max_depth;
  shape.used_max_limits = (in.width <= 0 || in.height <= 0 || in.depth <= 0);

  if (shape.width <= 0 || shape.height <= 0 || shape.depth <= 0) {
    throw std::runtime_error(
        "invalid model tensor input shape: unresolved dynamic dimensions; "
        "set --pcie-src-buf-size explicitly or provide model max_input_* limits");
  }
  return shape;
}

std::size_t tensor_input_bytes(const TensorInputShape& shape) {
  std::size_t bytes = static_cast<std::size_t>(sizeof(float));
  bytes = mul_or_throw(bytes, static_cast<std::size_t>(shape.width), "input width");
  bytes = mul_or_throw(bytes, static_cast<std::size_t>(shape.height), "input height");
  bytes = mul_or_throw(bytes, static_cast<std::size_t>(shape.depth), "input depth");
  return bytes;
}

void print_usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--model <path-to-mpk.tar.gz>] [options]\n"
            << "Options:\n"
            << "  --model <path>             Model MPK tar.gz (default: search _models/, tmp/).\n"
            << "  --duration-ms <ms>           Run duration in milliseconds (default 3000).\n"
            << "                               Use 0 or negative to run until Ctrl+C.\n"
            << "  --pcie-src-buf-size <bytes>  Override PCIe source buffer size.\n"
            << "  --pcie-sink-buf-size <bytes> Override PCIe sink buffer size.\n"
            << "  --sink-buf-name <name>       PCIe sink buffer name (default overlay).\n"
            << "  --print-gst                  Print GST pipeline and exit.\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_usage(argv[0]);
      return 0;
    }

    const fs::path root = sima_tutorial::find_repo_root();
    std::string model_arg;
    const fs::path model_path = sima_tutorial::get_arg(argc, argv, "--model", model_arg)
                                    ? fs::path(model_arg)
                                    : find_default_model(root);
    if (model_path.empty() || !fs::exists(model_path)) {
      return sima_tutorial::skip("missing MPK (pass --model)");
    }

    if (!simaai::neat::element_exists("simaaipciesrc") ||
        !simaai::neat::element_exists("simaaipciesink")) {
      return sima_tutorial::skip("missing PCIe plugins: simaaipciesrc/simaaipciesink");
    }

    int duration_ms = 3000;
    int src_buf_override = 0;
    int sink_buf_override = 0;
    std::string sink_buf_name = "overlay";
    (void)get_arg_int(argc, argv, "--duration-ms", duration_ms);
    (void)get_arg_int(argc, argv, "--pcie-src-buf-size", src_buf_override);
    (void)get_arg_int(argc, argv, "--pcie-sink-buf-size", sink_buf_override);
    (void)get_arg_string(argc, argv, "--sink-buf-name", sink_buf_name);
    const bool print_gst = sima_tutorial::wants_print_gst(argc, argv);

    std::cout << "[STEP] Loading model: " << model_path << "\n";
    simaai::neat::Model model(model_path.string(), simaai::neat::Model::Options{});

    // Keep this tutorial to single-input tensor flow.
    const simaai::neat::InputOptions input_contract = model.input_appsrc_options(true);
    if (input_contract.media_type != "application/vnd.simaai.tensor") {
      throw std::runtime_error("model input is not tensor media type");
    }
    if (input_contract.format != "FP32") {
      throw std::runtime_error("this tutorial expects FP32 tensor model input");
    }
    const TensorInputShape input_shape = resolve_tensor_input_shape(input_contract);
    const std::size_t default_src_bytes = tensor_input_bytes(input_shape);

    simaai::neat::Model::SessionOptions popt;
    popt.include_appsrc = false;
    popt.include_appsink = false;
    const simaai::neat::NodeGroup neatmodel_group = model.session(popt);

    simaai::neat::OutputSpec in_spec;
    in_spec.media_type = input_contract.media_type;
    in_spec.format = input_contract.format;
    in_spec.width = input_shape.width;
    in_spec.height = input_shape.height;
    in_spec.depth = input_shape.depth;
    in_spec.dtype = "Float32";
    in_spec.layout = "HWC";
    in_spec.byte_size = default_src_bytes;
    const simaai::neat::OutputSpec out_spec =
        simaai::neat::derive_output_spec(neatmodel_group, in_spec);
    const std::size_t derived_sink_bytes = simaai::neat::expected_byte_size(out_spec);
    const std::size_t default_sink_bytes = std::max<std::size_t>(derived_sink_bytes, 1);

    const int src_buf_size =
        (src_buf_override > 0) ? src_buf_override : static_cast<int>(default_src_bytes);
    const int sink_buf_size =
        (sink_buf_override > 0) ? sink_buf_override : static_cast<int>(default_sink_bytes);

    simaai::neat::PCIeSrcOptions src_opt;
    src_opt.buffer_size = src_buf_size;

    simaai::neat::PCIeSinkOptions sink_opt;
    sink_opt.data_buf_name = sink_buf_name;
    sink_opt.data_buffer_size = sink_buf_size;
    sink_opt.queue = 0;
    sink_opt.sync = false;
    sink_opt.async_state = false;
    sink_opt.num_buffers = 5;

    simaai::neat::Session session;
    // Pipeline shape: pciesrc -> neatmodel -> pciesink
    session.add(simaai::neat::nodes::PCIeSrc(src_opt));
    session.add(neatmodel_group);
    session.add(simaai::neat::nodes::PCIeSink(sink_opt));

    std::cout << "[INFO] model_input=" << input_shape.width << "x" << input_shape.height << "x"
              << input_shape.depth << " " << input_contract.format;
    if (input_shape.used_max_limits) {
      std::cout << " (resolved from max_input_* limits)";
    }
    std::cout << "\n";
    std::cout << "[INFO] pcie_src_buffer_size=" << src_buf_size << "\n";
    std::cout << "[INFO] pcie_sink_buffer_size=" << sink_buf_size << "\n";
    std::cout << "[INFO] pcie_sink_buf_name=" << sink_buf_name << "\n";

    if (print_gst) {
      std::cout << session.describe_backend() << "\n";
      return 0;
    }

    if (duration_ms > 0) {
      std::cout << "[STEP] Running PCIe model runtime for " << duration_ms << " ms\n";
    } else {
      std::cout << "[STEP] Running PCIe model runtime until Ctrl+C\n";
    }

    auto run = session.build();
    if (duration_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    } else {
      g_stop_requested.store(false, std::memory_order_relaxed);
      std::signal(SIGINT, on_signal);
      std::signal(SIGTERM, on_signal);
      while (!g_stop_requested.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    }
    run.stop();

    std::cout << "[DONE]\n";
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (runtime_unavailable(msg)) {
      return sima_tutorial::skip("runtime unavailable: " + msg);
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
}
