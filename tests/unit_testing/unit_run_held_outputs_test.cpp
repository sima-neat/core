// How many pulled outputs an application may hold while it keeps pushing, on either backend
// (--backend=gstreamer|dataflow, a process each), through Model::build + Model::Runner of a YOLOv8
// detection model, as an application holds them: lockstep try_push and pull, every pulled Sample
// kept. With zero-copy outputs (OutputMemory::ZeroCopy, and Auto, which is zero-copy there) at
// least queue_depth + 2 are held before a pull finds nothing; that pull times out with the loan
// text ("zero-copy output loan credits exhausted ... release older zero-copy outputs and retry"),
// the Run goes on, and once the held outputs are released the waiting output is pulled. With
// OutputMemory::Owned (copies) 64 are held without a stall. Needs a compiled YOLOv8 detection
// package (--model-package <X_mpk.tar.gz>, SIMA_TEST_BOXDECODE_MODEL_PACKAGE or SIMA_YOLO_TAR) and
// the EV74 on the Kernels firmware; skipped without them.

#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "cli_utils.h"
#include "support/backend_selection.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;

Tensor frame(unsigned index) {
  cv::Mat mat(720, 1280, CV_8UC3);
  for (int r = 0; r < mat.rows; ++r) {
    std::uint8_t* row = mat.ptr<std::uint8_t>(r);
    for (int c = 0; c < 3 * mat.cols; ++c) {
      row[c] = static_cast<std::uint8_t>((r * 5 + c * 3 + 29 * index) & 0xffU);
    }
  }
  return Tensor::from_cv_mat(mat, ImageSpec::PixelFormat::BGR, TensorMemory::CPU);
}

Model::Options detection_options() {
  Model::Options o;
  o.preprocess.kind = InputKind::Image;
  o.preprocess.enable = AutoFlag::On;
  o.preprocess.normalize.enable = AutoFlag::On;
  o.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
  o.decode_type = BoxDecodeType::YoloV8;
  o.score_threshold = 0.49F;
  o.nms_iou_threshold = 0.5F;
  o.top_k = 100;
  return o;
}

bool ev74_runs_kernels_firmware() {
  for (int i = 0; i < 8; ++i) {
    const std::string dir = "/sys/class/remoteproc/remoteproc" + std::to_string(i) + "/";
    std::ifstream name(dir + "name");
    std::string value;
    if (name >> value && value == "ev74") {
      std::ifstream firmware(dir + "firmware");
      std::string fw;
      return firmware >> fw && fw == "sima-neat-native-cvu-fw";
    }
  }
  return false;
}

// Holds every pulled Sample until a pull finds nothing within 2 s or `cap` are held.
void holds(const std::string& archive, OutputMemory memory, int queue_depth, int cap) {
  const std::string what = std::string(memory == OutputMemory::Owned      ? "Owned"
                                       : memory == OutputMemory::ZeroCopy ? "ZeroCopy"
                                                                          : "Auto") +
                           ", queue_depth " + std::to_string(queue_depth);
  Model m(archive, detection_options());
  RunOptions ro;
  ro.output_memory = memory;
  ro.queue_depth = queue_depth;
  Model::Runner runner = m.build(TensorList{frame(0)}, Model::RouteOptions{}, ro);
  std::vector<Sample> held;
  std::string timeout_text;
  for (int i = 0; i < cap; ++i) {
    bool pushed = false;
    for (int tries = 0; tries < 300 && !pushed; ++tries) {
      pushed = runner.try_push(TensorList{frame(static_cast<unsigned>(i % 4))});
      if (!pushed) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    require(pushed, what + ": the push of frame " + std::to_string(i) + " is taken");
    Sample out = runner.pull(2000);
    if (out.empty()) {
      break;
    }
    held.push_back(std::move(out));
  }
  const int count = static_cast<int>(held.size());
  if (memory == OutputMemory::Owned) {
    require(count == cap, what + ": " + std::to_string(cap) +
                              " Owned outputs held without a stall (" + std::to_string(count) +
                              ")");
    return;
  }
  require(count >= queue_depth + 2 && count < cap,
          what + ": at least queue_depth + 2 zero-copy outputs held before a pull finds nothing (" +
              std::to_string(count) + ")");
  // The pull that found nothing, through the verb that reports why.
  Run& run = internal::ModelAccess::run(runner);
  Sample none;
  PullError err;
  require(run.pull(2000, none, &err) == PullStatus::Timeout, what + ": the next pull times out");
  require(err.message.find("zero-copy output loan credits exhausted") != std::string::npos &&
              err.message.find("release older zero-copy outputs and retry") != std::string::npos,
          what + ": the timeout says the held outputs are the reason: " + err.message);
  require(static_cast<bool>(runner) && run.running() && run.can_push() && run.can_pull(),
          what + ": the Run goes on");
  held.clear();
  require(!runner.pull(5000).empty(), what + ": once they are released the waiting output comes");
  runner.close();
}

} // namespace

int main(int argc, char** argv) {
  std::string archive;
  sima_test::get_arg(argc, argv, "--model-package", archive);
  for (const char* name : {"SIMA_TEST_BOXDECODE_MODEL_PACKAGE", "SIMA_YOLO_TAR"}) {
    if (const char* staged = std::getenv(name); archive.empty() && staged) {
      archive = staged;
    }
  }
  if (archive.empty() || !std::ifstream(archive)) {
    return skip_test("unit_run_held_outputs_test needs a YOLOv8 detection model package: "
                     "--model-package <X_mpk.tar.gz>, SIMA_TEST_BOXDECODE_MODEL_PACKAGE or "
                     "SIMA_YOLO_TAR");
  }
  try {
    sima_test::select_backend(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
  return sima_test::run_test("unit_run_held_outputs_test", [&] {
    if (!ev74_runs_kernels_firmware()) {
      std::cout << "[skip] the EV74 does not run the Kernels firmware\n";
      return;
    }
    for (const int depth : {1, 4, 8}) {
      holds(archive, OutputMemory::ZeroCopy, depth, 32);
    }
    holds(archive, OutputMemory::Auto, 4, 32);
    holds(archive, OutputMemory::Owned, 4, 64);
  });
}
