// H0 C5: Run's deprecated forms warn, at run time, exactly once per form and process, and nothing
// else does. On the backend SIMA_NEAT_BACKEND names (dataflow: RunBackend::Dataflow, requested
// explicitly; otherwise RunBackend::GStreamer); registered once per backend (tests/CMakeLists.txt),
// so each process checks "once" on its own backend.
//
// stderr is captured around the calls (as stage_chain_cache_admission_test does) and the
// "[WARN] Run::<form> is deprecated ..." lines are counted per form:
//   - the TensorList verbs, pull(milliseconds), operator(), on_output, alloc, and the forms that
//     stay (push/try_push(Sample), pull(int, Sample&, PullError*), run(...), the holder pushes)
//     print none; run(std::vector<cv::Mat>) proves Core's own call of the cv::Mat push is quiet;
//   - each of the ten deprecated forms prints exactly one line naming its replacement, however
//     often it is called: a first call that fails (closed Run) counts, a first call made by two
//     threads at once prints one line;
//   - every deprecated form still returns what it returned before C5.
// On Dataflow, the options that backend ignores (C3) warn once each too (H0 3.10).

// This file calls the deprecated forms on purpose.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/Result.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;
using namespace std::chrono_literals;

bool on_dataflow() {
  const char* env = std::getenv("SIMA_NEAT_BACKEND");
  return env != nullptr && std::string(env) == "dataflow";
}

RunOptions options() {
  RunOptions opt;
  opt.backend = on_dataflow() ? RunBackend::Dataflow : RunBackend::GStreamer;
  return opt;
}

// The forms and the replacement each line must name (RunDeprecation.cpp's table).
struct Form {
  const char* signature;
  const char* replacement;
};
const std::vector<Form>& deprecated_forms() {
  static const std::vector<Form> forms{
      {"Run::push(const std::vector<cv::Mat>&)", "use Run::push(TensorList{Tensor::from_cv_mat("},
      {"Run::push(std::string_view, const std::vector<cv::Mat>&)",
       "use Run::push(input_name, TensorList{Tensor::from_cv_mat("},
      {"Run::try_push(const std::vector<cv::Mat>&)",
       "use Run::try_push(TensorList{Tensor::from_cv_mat("},
      {"Run::try_push(std::string_view, const std::vector<cv::Mat>&)",
       "use Run::try_push(input_name, TensorList{Tensor::from_cv_mat("},
      {"Run::pull(int)", "use Run::pull(std::chrono::milliseconds) -> Result<TensorList>; for "
                         "Sample metadata, Run::pull(int, Sample&, PullError*)"},
      {"Run::pull(std::string_view, int)",
       "use Run::pull(output_name, std::chrono::milliseconds) -> Result<TensorList>"},
      {"Run::pull_tensors(int)", "use Run::pull(std::chrono::milliseconds) -> Result<TensorList>"},
      {"Run::pull_tensors(std::string_view, int)",
       "use Run::pull(output_name, std::chrono::milliseconds) -> Result<TensorList>"},
      {"Run::pull_samples(int)", "use Run::pull(std::chrono::milliseconds) per item"},
      {"Run::pull_samples(std::string_view, int)",
       "use Run::pull(output_name, std::chrono::milliseconds) per item"},
  };
  return forms;
}

// fd 2 to a temporary file while alive; text() restores it and returns what was written.
class StderrCapture {
public:
  StderrCapture() {
    file_ = std::tmpfile();
    require(file_ != nullptr, "could not open the stderr capture");
    std::fflush(stderr);
    saved_ = dup(STDERR_FILENO);
    require(saved_ >= 0, "could not preserve stderr");
    require(dup2(fileno(file_), STDERR_FILENO) >= 0, "could not capture stderr");
  }
  ~StderrCapture() {
    if (saved_ >= 0) {
      (void)text();
    }
    if (file_) {
      std::fclose(file_);
    }
  }
  StderrCapture(const StderrCapture&) = delete;
  StderrCapture& operator=(const StderrCapture&) = delete;

  std::string text() {
    std::fflush(stderr);
    if (saved_ >= 0) {
      (void)dup2(saved_, STDERR_FILENO);
      close(saved_);
      saved_ = -1;
    }
    std::rewind(file_);
    std::string out;
    char chunk[4096];
    while (const auto n = std::fread(chunk, 1U, sizeof(chunk), file_)) {
      out.append(chunk, n);
    }
    return out;
  }

private:
  std::FILE* file_ = nullptr;
  int saved_ = -1;
};

std::vector<std::string> lines_of(const std::string& text) {
  std::vector<std::string> out;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);) {
    out.push_back(line);
  }
  return out;
}

// The "[WARN] Run::... is deprecated" lines, by form signature; any other such line is an error.
std::map<std::string, std::vector<std::string>> deprecation_lines(const std::string& text) {
  std::map<std::string, std::vector<std::string>> out;
  for (const std::string& line : lines_of(text)) {
    if (line.rfind("[WARN] Run::", 0) != 0) {
      continue;
    }
    bool known = false;
    for (const Form& form : deprecated_forms()) {
      const std::string prefix = std::string("[WARN] ") + form.signature + " is deprecated";
      if (line.rfind(prefix, 0) == 0) {
        out[form.signature].push_back(line);
        known = true;
      }
    }
    require(known, "an unexpected Run warning: " + line);
  }
  return out;
}

// A 16x8 RGB tensor whose bytes encode `tag` and their position.
Tensor rgb(std::uint8_t tag) {
  Tensor t = make_color_tensor(16, 8, ImageSpec::PixelFormat::RGB, tag);
  auto map = t.storage->map(MapMode::Write);
  auto* bytes = static_cast<std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < map.size_bytes; ++i) {
    bytes[i] = static_cast<std::uint8_t>(tag + i * 7U);
  }
  return t;
}

Sample sample_of(const Tensor& t) {
  Sample s;
  s.kind = SampleKind::Tensor;
  s.tensor = t;
  return s;
}

std::vector<std::uint8_t> bytes_of(const Tensor& t) {
  return t.copy_dense_bytes_tight();
}

// An input "x" to an output "y" (bound by the first push): the Tensor verbs and the pulls.
Graph tensor_graph() {
  Graph g;
  const auto x = g.input("x");
  g.output("y", x);
  return g;
}

// A declared FP32 input "x" to an output "y": alloc.
Graph declared_graph() {
  Graph g;
  const auto x = g.input("x", {4, 8}, TensorDType::Float32);
  g.output("y", x);
  return g;
}

// A named BGR image input "image" to an output "out": the cv::Mat forms, on both backends.
Graph image_graph() {
  InputOptions in;
  in.payload_type = PayloadType::Image;
  in.format = FormatTag::BGR;
  in.width = 32;
  in.height = 24;
  Graph g;
  g.add(nodes::Input("image", in));
  g.add(nodes::Output("out"));
  return g;
}

cv::Mat bgr(std::uint8_t value) {
  return cv::Mat(24, 32, CV_8UC3, cv::Scalar(value, value + 1, value + 2));
}

// Pull one image item with the new verb; true when its pixels are `mat`'s.
bool pulled_image_is(Run& run, const cv::Mat& mat, const std::string& what) {
  const Result<TensorList> r = run.pull("out", 5000ms);
  require(r.ok() && r->size() == 1U, what + ": one item");
  const cv::Mat pixels = r->front().to_cv_mat_copy(ImageSpec::PixelFormat::BGR);
  return pixels.size() == mat.size() && pixels.type() == mat.type() &&
         cv::norm(pixels, mat, cv::NORM_INF) == 0;
}

// Phase 1: everything that must stay silent.
void silent_forms(Run& tensors, Run& images) {
  // The TensorList verbs, pull(milliseconds), operator().
  require(tensors.push({rgb(0)}) && tensors.push("x", {rgb(1)}), "push(TensorList)");
  (void)tensors.try_push({rgb(2)});
  for (int i = 0; i < 2; ++i) {
    require(tensors.pull(5000ms).ok(), "pull(milliseconds)");
  }
  (void)tensors.pull("y", 200ms);
  (void)tensors.try_push("x", {rgb(3)});
  (void)tensors.pull("y", 200ms);
  require(bytes_of(tensors({rgb(4)}).front()) == bytes_of(rgb(4)), "operator()");

  // alloc: a declared input's slot, committed by a push of it alone.
  {
    Graph dg = declared_graph();
    Run declared = dg.build(options());
    Tensor slot = declared.alloc("x");
    const std::vector<float> values(32, 5.0F);
    {
      Mapping map = slot.map_write();
      std::memcpy(map.data, values.data(), values.size() * sizeof(float));
    }
    require(declared.push("x", {slot}), "alloc commit");
    const Result<TensorList> r = declared.pull("y", 5000ms);
    require(r.ok() && bytes_of(r->front()) ==
                          bytes_of(Tensor::from_vector(values, {4, 8}, TensorMemory::CPU)),
            "alloc round trip");
    declared.close();
  }

  // The forms that stay: Sample push, the structured Sample pull, run(...).
  require(tensors.push(sample_of(rgb(6))), "push(Sample)");
  Sample out;
  PullError err;
  require(tensors.pull(5000, out, &err) == PullStatus::Ok, "pull(int, Sample&, PullError*)");
  require(bytes_of(tensors_from_sample(out).front()) == bytes_of(rgb(6)), "Sample round trip");
  (void)tensors.try_push("x", sample_of(rgb(7)));
  (void)tensors.pull("y", 200, out, &err);
  require(bytes_of(tensors.run(TensorList{rgb(8)}, 5000).front()) == bytes_of(rgb(8)),
          "run(TensorList)");
  // Core's own call of the cv::Mat push (Run::run -> enqueue_run_images) must not warn.
  require(images.run(std::vector<cv::Mat>{bgr(9)}, 5000).size() == 1U, "run(std::vector<cv::Mat>)");

  // on_output, on a Run of its own.
  Graph g = tensor_graph();
  Run delivered = g.build(options());
  std::atomic<int> seen{0};
  delivered.on_output("y", [&](TensorList) { seen.fetch_add(1); });
  require(delivered.push({rgb(10)}), "push for on_output");
  for (int i = 0; i < 100 && seen.load() == 0; ++i) {
    std::this_thread::sleep_for(20ms);
  }
  require(seen.load() == 1, "on_output delivered");
  delivered.close();
}

// Phase 2: the ten deprecated forms. Returns after every form was called at least twice.
void deprecated_calls(Run& tensors, Run& images) {
  // R18 first, from two threads at once (each on a Run of its own): one line.
  {
    Graph ga = tensor_graph();
    Graph gb = tensor_graph();
    Run ra = ga.build(options());
    Run rb = gb.build(options());
    std::atomic<bool> go{false};
    auto racer = [&](Run& run) {
      while (!go.load()) {
      }
      (void)run.pull_samples("y", 0);
    };
    std::thread a(racer, std::ref(ra));
    std::thread b(racer, std::ref(rb));
    go.store(true);
    a.join();
    b.join();
    ra.close();
    rb.close();
  }

  // R04 first on a closed Run: the failing call still warns.
  {
    Graph g = image_graph();
    Run closed = g.build(options());
    closed.close();
    try {
      (void)closed.try_push("image", std::vector<cv::Mat>{bgr(1)});
    } catch (...) {
    }
  }

  // R01-R04 on the live image Run, each twice; the pushed images round trip.
  require(images.push(std::vector<cv::Mat>{bgr(20)}), "push(std::vector<cv::Mat>)");
  require(pulled_image_is(images, bgr(20), "push(Mat)"), "push(Mat) round trip");
  require(images.push("image", std::vector<cv::Mat>{bgr(21)}), "push(name, Mat)");
  require(pulled_image_is(images, bgr(21), "push(name, Mat)"), "push(name, Mat) round trip");
  if (images.try_push(std::vector<cv::Mat>{bgr(22)})) {
    require(pulled_image_is(images, bgr(22), "try_push(Mat)"), "try_push(Mat) round trip");
  }
  if (images.try_push("image", std::vector<cv::Mat>{bgr(23)})) {
    require(pulled_image_is(images, bgr(23), "try_push(name, Mat)"),
            "try_push(name, Mat) round trip");
  }
  require(images.push(std::vector<cv::Mat>{bgr(24)}) &&
              images.push("image", std::vector<cv::Mat>{bgr(25)}),
          "the cv::Mat pushes, again");
  require(pulled_image_is(images, bgr(24), "drain 1") &&
              pulled_image_is(images, bgr(25), "drain 2"),
          "the second cv::Mat pushes round trip");
  (void)images.try_push(std::vector<cv::Mat>{bgr(26)});
  (void)images.pull("out", 500ms);

  // R13-R18 on the live tensor Run: each returns what it returned before C5.
  require(tensors.push({rgb(30)}), "push for pull(int)");
  {
    const std::optional<Sample> s = tensors.pull(5000);
    require(s.has_value() && bytes_of(tensors_from_sample(*s).front()) == bytes_of(rgb(30)),
            "pull(int) returns the Sample");
  }
  require(tensors.push({rgb(31)}), "push for pull(name, int)");
  {
    const std::optional<Sample> s = tensors.pull("y", 5000);
    require(s.has_value() && bytes_of(tensors_from_sample(*s).front()) == bytes_of(rgb(31)),
            "pull(name, int) returns the Sample");
  }
  require(tensors.push({rgb(32)}), "push for pull_tensors(int)");
  {
    const TensorList t = tensors.pull_tensors(5000);
    require(t.size() == 1U && bytes_of(t.front()) == bytes_of(rgb(32)),
            "pull_tensors(int) returns the tensors");
  }
  require(tensors.push({rgb(33)}), "push for pull_tensors(name, int)");
  {
    const TensorList t = tensors.pull_tensors("y", 5000);
    require(t.size() == 1U && bytes_of(t.front()) == bytes_of(rgb(33)),
            "pull_tensors(name, int) returns the tensors");
  }
  require(tensors.push({rgb(34)}), "push for pull_samples(int)");
  {
    const Sample s = tensors.pull_samples(5000);
    require(s.size() == 1U && bytes_of(tensors_from_sample(s).front()) == bytes_of(rgb(34)),
            "pull_samples(int) returns the item");
  }
  require(tensors.push({rgb(35)}), "push for pull_samples(name, int)");
  {
    const Sample s = tensors.pull_samples("y", 5000);
    require(s.size() == 1U && bytes_of(tensors_from_sample(s).front()) == bytes_of(rgb(35)),
            "pull_samples(name, int) returns the item");
  }
  // Second calls (timeouts: empty results, as before).
  require(!tensors.pull(0).has_value() && !tensors.pull("y", 0).has_value(), "pull timeouts");
  require(tensors.pull_tensors(0).empty() && tensors.pull_tensors("y", 0).empty(),
          "pull_tensors timeouts");
  require(tensors.pull_samples(0).empty() && tensors.pull_samples("y", 0).empty(),
          "pull_samples timeouts");
}

// On Dataflow: the options that backend ignores warn once each (C3), on every build that sets them.
void dataflow_ignored_options(std::map<std::string, int>& counts, const std::string& text) {
  for (const char* option :
       {"InputOptions::is_live", "InputOptions::do_timestamp", "InputOptions::stream_type",
        "InputOptions::caps_override", "InputOptions::max_bytes", "OutputOptions::sync"}) {
    const std::string needle =
        std::string("[WARN] Dataflow backend: ") + option + " has no meaning on this backend";
    int n = 0;
    for (const std::string& line : lines_of(text)) {
      n += line.rfind(needle, 0) == 0 ? 1 : 0;
    }
    counts[option] = n;
  }
}

void deprecation_warnings() {
  const bool dataflow = on_dataflow();
  // Initialise the backend before stderr is captured. On GStreamer the first Graph::build installs
  // Core's stdio filters (GstInit.cpp, install_stdio_filters): fd 1 and fd 2 become pipes that a
  // detached thread forwards to the original descriptors. A capture that spans that moment would
  // hand the filter the capture file, and stderr would then reach it asynchronously.
  {
    Graph g = tensor_graph();
    Run warm = g.build(options());
    require(warm.push({rgb(99)}) && warm.pull("y", 5000ms).ok(), "warm-up Run");
    warm.close();
  }
  std::string silent_text;
  std::string deprecated_text;
  std::string options_text;
  {
    StderrCapture capture;
    Graph tg = tensor_graph();
    Run tensors = tg.build(options());
    Graph ig = image_graph();
    Run images = ig.build(options());
    try {
      silent_forms(tensors, images);
    } catch (...) {
      const std::string text = capture.text();
      std::cerr << "---- captured stderr (silent forms) ----\n" << text << "----\n";
      throw;
    }
    silent_text = capture.text();
  }
  {
    StderrCapture capture;
    Graph tg = tensor_graph();
    Run tensors = tg.build(options());
    Graph ig = image_graph();
    Run images = ig.build(options());
    try {
      deprecated_calls(tensors, images);
      deprecated_calls(tensors, images); // every form again: no new line
    } catch (...) {
      const std::string text = capture.text();
      std::cerr << "---- captured stderr (deprecated forms) ----\n" << text << "----\n";
      throw;
    }
    tensors.close();
    images.close();
    deprecated_text = capture.text();
  }
  if (dataflow) {
    StderrCapture capture;
    for (int i = 0; i < 2; ++i) {
      InputOptions in;
      in.is_live = false;
      in.do_timestamp = false;
      in.stream_type = 1;
      in.caps_override = "application/vnd.simaai.tensor";
      in.max_bytes = 1U << 20U;
      OutputOptions out;
      out.sync = true;
      Graph g;
      g.add(nodes::Input("x", in));
      g.add(nodes::Output("y", out));
      Run run = g.build(options());
      require(run.push({rgb(40)}) && run.pull("y", 5000ms).ok(), "options ignored on Dataflow");
      run.close();
    }
    options_text = capture.text();
  }

  std::cout << "---- captured stderr, silent forms ----\n"
            << silent_text << "---- captured stderr, deprecated forms ----\n"
            << deprecated_text << "----" << std::endl;

  const auto silent = deprecation_lines(silent_text);
  require(silent.empty(), "the new forms, the forms that stay and Core's own calls print no "
                          "deprecation warning");

  const auto lines = deprecation_lines(deprecated_text);
  for (const Form& form : deprecated_forms()) {
    const auto it = lines.find(form.signature);
    const std::size_t n = it == lines.end() ? 0U : it->second.size();
    require(n == 1U, std::string(form.signature) + " warns exactly once (it warned " +
                         std::to_string(n) + " times)");
    require(it->second.front().find(form.replacement) != std::string::npos,
            std::string(form.signature) + " names its replacement: " + it->second.front());
    require(it->second.front().find("will be removed in the next release") != std::string::npos,
            std::string(form.signature) + " says when it goes");
  }

  if (dataflow) {
    std::cout << "---- captured stderr, ignored options (Dataflow) ----\n"
              << options_text << "----" << std::endl;
    std::map<std::string, int> counts;
    dataflow_ignored_options(counts, options_text);
    for (const auto& [option, n] : counts) {
      require(n == 1,
              option + " warns once on Dataflow (it warned " + std::to_string(n) + " times)");
    }
  }
}

} // namespace

RUN_TEST("unit_run_deprecation_test", deprecation_warnings)
