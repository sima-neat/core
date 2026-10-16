#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/Preproc.h"

#include "gst/GstHelpers.h"
#include "test_utils.h"
#include <neat.h>
#include <gstsimaaitensorbuffer.h>
#include <optional>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

bool env_bool(const char* key, bool def) {
  const char* v = std::getenv(key);
  if (!v)
    return def;
  return std::string(v) != "0";
}

int env_int(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return def;
  return std::atoi(v);
}

// The guard must establish device quiescence before terminating this process.
// An unsuccessful pull is not proof that an accepted EV request has retired.
[[noreturn]] void retain_uncertain_dynamic_frame(const char* message) {
  std::cerr << "UNRETIRED dynamic Core Preproc: " << message
            << "; retaining owners for guarded stop\n"
            << std::flush;
  for (;;)
    std::this_thread::sleep_for(std::chrono::seconds(60));
}

struct DynamicImage {
  int width, height;
  std::array<std::uint8_t, 3> rgb;
  simaai::neat::Tensor tensor;
};

void check_dynamic_output(const simaai::neat::Sample& sample, const DynamicImage& image,
                          const std::string& mode) {
  using namespace simaai::neat;
  const auto tensors = tensors_from_sample(sample);
  require(tensors.size() == 1, "dynamic Preproc output count");
  const auto& tensor = tensors.front();
  require(tensor.shape == std::vector<std::int64_t>{640, 640, 3},
          "dynamic Preproc changed its fixed output geometry");
  require(tensor.dtype == TensorDType::UInt8 || tensor.dtype == TensorDType::Int8,
          "dynamic Preproc output dtype");
  require(tensor.semantic.preprocess && tensor.semantic.preprocess->original_width == image.width &&
              tensor.semantic.preprocess->original_height == image.height,
          "dynamic Preproc published stale input geometry");
  require(tensor.storage && tensor.storage->kind == StorageKind::GstSample &&
              tensor.storage->holder,
          "dynamic Preproc lost zero-copy sample ownership");
  auto* buffer = gst_sample_get_buffer(static_cast<GstSample*>(tensor.storage->holder.get()));
  std::string issue;
  const auto publication = simaai::gst::tensor_buffer_publication(buffer, &issue);
  require(issue.empty() && bool(publication) == (mode == "compact"),
          "dynamic Preproc publication negotiation changed");
  auto cpu = tensor.clone();
  auto mapping = cpu.map(MapMode::Read);
  constexpr std::size_t bytes = 640 * 640 * 3;
  require(mapping.data && mapping.size_bytes >= bytes, "dynamic Preproc output extent");
  const auto* pixels = static_cast<const std::uint8_t*>(mapping.data);
  for (std::size_t i = 0; i < bytes; i += 3)
    require(pixels[i] == image.rgb[0] && pixels[i + 1] == image.rgb[1] &&
                pixels[i + 2] == image.rgb[2],
            "dynamic Preproc pixels differ from the constant-image oracle");
}

void dynamic_preproc(const std::string& mode, bool benchmark) {
  using namespace simaai::neat;
  using Clock = std::chrono::steady_clock;
  // Every input is created once. Distinct colors expose stale output reuse;
  // constant images give an exact resize oracle without a second resize engine.
  std::array<DynamicImage, 3> images{{{1280, 720, {64, 128, 192}, {}},
                                      {320, 240, {16, 80, 144}, {}},
                                      {640, 480, {32, 96, 160}, {}}}};
  for (auto& image : images) {
    cv::Mat pixels(image.height, image.width, CV_8UC3,
                   cv::Scalar(image.rgb[0], image.rgb[1], image.rgb[2]));
    image.tensor = Tensor::from_cv_mat(pixels, ImageSpec::PixelFormat::RGB, TensorMemory::EV74);
  }
  Graph graph;
  InputOptions input_options;
  input_options.payload_type = PayloadType::Image;
  input_options.format = FormatTag::RGB;
  input_options.memory_policy = InputMemoryPolicy::Ev74;
  graph.add(nodes::Input(input_options));

  PreprocOptions options;
  options.set_input_shape({720, 1280, 3});
  options.set_output_shape({640, 640, 3});
  options.scaled_width = options.scaled_height = 640;
  options.input_img_type = options.output_img_type = "RGB";
  options.normalize = options.aspect_ratio = options.tessellate = false;
  options.dynamic_input_dims = true;
  options.output_dtype = "EVXX_INT8";
  options.scaling_type = "BILINEAR";
  options.padding_type = "CENTER";
  options.next_cpu = "APU";
  options.upstream_name = "decoder";
  options.num_buffers = 4;
  options.q_scale = 0.25;
  options.q_zp = 0;
  graph.add(nodes::Preproc(options));
  OutputOptions output_options;
  output_options.sync = output_options.drop = false;
  output_options.max_buffers = 1;
  graph.add(nodes::Output(output_options));
  RunOptions run_options;
  // Realtime admits geometry changes after one frame; Balanced intentionally
  // debounces them for two. One outstanding frame avoids any overflow/drop.
  run_options.preset = RunPreset::Realtime;
  run_options.advanced.max_input_bytes = 1280 * 720 * 3;
  run_options.output_memory = OutputMemory::ZeroCopy;
  run_options.queue_depth = 2;
  const auto setup = Clock::now();
  auto run = graph.build(TensorList{images.front().tensor}, run_options);
  const double setup_us = std::chrono::duration<double, std::micro>(Clock::now() - setup).count();
  require(graph.last_pipeline().find("neatkernelrunner") != std::string::npos &&
              graph.last_pipeline().find("neatprocesscvu") == std::string::npos,
          "dynamic Preproc did not use the default Runtime plugin");
  const int timeout_ms = env_int("SIMA_INPUT_TIMEOUT_MS", 20000);
  const unsigned repetitions = benchmark ? 33U : 3U;
  constexpr std::array<unsigned, 7> sequence{0, 1, 2, 0, 1, 2, 0};
  struct Observation {
    unsigned image;
    double first_us;
    std::vector<double> stable_us;
  };
  std::vector<Observation> observations;
  observations.reserve(sequence.size());
  std::optional<Sample> held;
  std::int64_t frame = 0;
  for (const auto index : sequence) {
    const auto& image = images[index];
    observations.push_back({index, 0, {}});
    auto& observation = observations.back();
    observation.stable_us.reserve(repetitions - 1);
    for (unsigned repeat = 0; repeat < repetitions; ++repeat) {
      Sample input;
      input.kind = SampleKind::TensorSet;
      input.payload_type = PayloadType::Image;
      input.tensors = {image.tensor};
      input.frame_id = 1000 + frame;
      input.input_seq = 2000 + frame;
      input.orig_input_seq = 3000 + frame;
      input.stream_id = "dynamic-preproc-camera";
      input.pts_ns = ++frame * 1000000;
      Sample output;
      const auto begin = Clock::now();
      try {
        // Mark uncertainty before push: a false result does not prove that no
        // downstream work was accepted. Keep all stack-owned inputs/run alive.
        if (!run.push(input) || run.pull(timeout_ms, output) != PullStatus::Ok)
          retain_uncertain_dynamic_frame("push/pull did not establish retirement");
      } catch (const std::exception& failure) {
        retain_uncertain_dynamic_frame(failure.what());
      } catch (...) {
        retain_uncertain_dynamic_frame("push/pull threw");
      }
      const auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
      if (!repeat)
        observation.first_us = elapsed;
      else
        observation.stable_us.push_back(elapsed);
      // Checks/copies and printing are outside the measured push-to-pull span.
      require(output.frame_id == input.frame_id && output.input_seq == input.input_seq &&
                  output.orig_input_seq == input.orig_input_seq &&
                  output.stream_id == input.stream_id && output.pts_ns == input.pts_ns,
              "dynamic Preproc changed frame correlation");
      check_dynamic_output(output, image, mode);
      if (!held)
        held = output;
      check_dynamic_output(*held, images.front(), mode);
      require(held->frame_id == 1000 && held->pts_ns == 1000000,
              "retained dynamic Preproc sample changed");
    }
  }
  run.close();
  check_dynamic_output(*held, images.front(), mode);
  std::cout << "dynamic_preproc_setup_us," << setup_us << '\n';
  std::cout << "# Realtime preset, one outstanding frame. End-to-end wall time; "
               "geometry-change rows include Core caps/graph handling, "
               "not an isolated Runtime cache-miss cost or a throughput gate.\n";
  std::cout << "mode,phase,width,height,activation,first_frame_us,stable_p50_us,stable_max_us,"
               "stable_samples\n";
  for (std::size_t phase = 0; phase < observations.size(); ++phase) {
    auto& observation = observations[phase];
    const auto& image = images[observation.image];
    std::sort(observation.stable_us.begin(), observation.stable_us.end());
    std::cout << mode << ',' << phase << ',' << image.width << ',' << image.height << ','
              << (phase == 0  ? "startup"
                  : phase < 3 ? "new_geometry"
                              : "returning_geometry")
              << ',' << observation.first_us << ','
              << observation.stable_us[observation.stable_us.size() / 2] << ','
              << observation.stable_us.back() << ',' << observation.stable_us.size() << '\n';
  }
  std::cout << "[OK] dynamic preproc_standalone_test passed; " << frame << " exact frames\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string mode = "native";
    bool dynamic = false, benchmark_dynamic = false;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--dynamic")
        dynamic = true;
      else if (argument == "--benchmark-dynamic")
        dynamic = benchmark_dynamic = true;
      else {
        require(argument == "native" || argument == "compact", "invalid metadata mode or option");
        mode = argument;
      }
    }
    setenv("SIMA_CORE_COMPACT_METADATA", mode == "compact" ? "1" : "0", 1);
    setenv("SIMA_GST_RUN_INSERT_BOUNDARIES", "0", 1);
    require(simaai::neat::element_exists("neatkernelrunner"),
            "Missing SIMA preproc plugin (neatkernelrunner).");

    if (dynamic) {
      dynamic_preproc(mode, benchmark_dynamic);
      return 0;
    }

    cv::Mat img(720, 1280, CV_8UC3, cv::Scalar(64, 128, 192));
    if (!img.isContinuous())
      img = img.clone();
    const simaai::neat::Tensor tensor_rgb = simaai::neat::Tensor::from_cv_mat(
        img, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74);

    simaai::neat::GraphOptions graph_options;
    graph_options.verbose.gstreamer = env_bool("SIMA_TEST_PREPROC_GST_VERBOSE", false);
    simaai::neat::Graph p(graph_options);

    simaai::neat::InputOptions src_opt;
    src_opt.format = simaai::neat::FormatTag::RGB;
    src_opt.width = img.cols;
    src_opt.height = img.rows;
    src_opt.depth = 3;
    src_opt.is_live = true;
    src_opt.do_timestamp = true;
    src_opt.block = false;
    src_opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
    src_opt.pool_min_buffers = env_int("SIMA_PREPROC_NUM_BUFFERS", 4);
    src_opt.pool_max_buffers = src_opt.pool_min_buffers;
    src_opt.buffer_name = "decoder";
    src_opt.max_bytes = 0;

    simaai::neat::PreprocOptions pre_opt;
    pre_opt.set_input_shape({img.rows, img.cols, 3});
    pre_opt.set_output_shape({640, 640, 3});
    pre_opt.scaled_width = 640;
    pre_opt.scaled_height = 640;
    pre_opt.input_img_type = "RGB";
    pre_opt.output_img_type = "RGB";
    pre_opt.normalize = false;
    pre_opt.aspect_ratio = false;
    pre_opt.output_dtype = "EVXX_INT8";
    pre_opt.scaling_type = "BILINEAR";
    pre_opt.padding_type = "CENTER";
    pre_opt.next_cpu = "APU";
    pre_opt.upstream_name = "decoder";
    pre_opt.num_buffers = src_opt.pool_min_buffers;
    pre_opt.set_slice_shape({32, 128, 3});
    pre_opt.q_scale = 0.25;
    pre_opt.q_zp = 0;

    p.add(simaai::neat::nodes::Input(src_opt));
    p.add(simaai::neat::nodes::Preproc(pre_opt));

    simaai::neat::OutputOptions sink_opt;
    sink_opt.sync = false;
    sink_opt.drop = true;
    sink_opt.max_buffers = 1;
    p.add(simaai::neat::nodes::Output(sink_opt));

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;
    run_opt.queue_depth = 2;
    const int timeout_ms = env_int("SIMA_INPUT_TIMEOUT_MS", 20000);

    auto run = p.build(simaai::neat::TensorList{tensor_rgb}, run_opt);
    std::cerr << "[preproc pipeline] " << p.last_pipeline() << "\n";
    require(p.last_pipeline().find("neatkernelrunner") != std::string::npos &&
                p.last_pipeline().find("neatprocesscvu") == std::string::npos,
            "Preproc did not select the default Runtime execution route");
    simaai::neat::TensorList outs;
    std::optional<simaai::neat::Sample> held;
    for (int i = 0; i < 4; ++i) {
      simaai::neat::Sample input;
      input.kind = simaai::neat::SampleKind::TensorSet;
      input.tensors = {tensor_rgb};
      input.frame_id = 300 + i;
      input.input_seq = 100 + i;
      input.orig_input_seq = 200 + i;
      input.stream_id = "preproc-camera";
      input.pts_ns = (i + 1) * 1000000;
      input.dts_ns = input.pts_ns - 100;
      input.duration_ns = 33333;
      std::cerr << "[preproc correctness] submitting iteration=" << i << "\n";
      auto result = run.run(input, timeout_ms);
      std::cerr << "[preproc correctness] received iteration=" << i << "\n";
      require(result.frame_id == input.frame_id && result.input_seq == input.input_seq &&
                  result.orig_input_seq == input.orig_input_seq &&
                  result.stream_id == input.stream_id,
              "Preproc Sample correlation mismatch");
      require(result.pts_ns == input.pts_ns && result.dts_ns == input.dts_ns &&
                  result.duration_ns == input.duration_ns,
              "Preproc Sample timing mismatch");
      outs = simaai::neat::tensors_from_sample(result);
      require(outs.size() == 1, "Preproc output count");
      const auto& storage = outs.front().storage;
      require(storage && storage->kind == simaai::neat::StorageKind::GstSample && storage->holder,
              "Preproc lost zero-copy owner");
      auto* buffer = gst_sample_get_buffer(static_cast<GstSample*>(storage->holder.get()));
      std::string error;
      auto publication = simaai::gst::tensor_buffer_publication(buffer, &error);
      if (bool(publication) != (mode == "compact") || !outs.front().semantic.preprocess) {
        std::cerr << "[preproc metadata] iteration=" << i << " compact=" << bool(publication)
                  << " geometry=" << bool(outs.front().semantic.preprocess)
                  << " canonical=" << simaai::gst::tensor_buffer_has_canonical_meta(buffer)
                  << " error=" << error << "\n";
        if (auto* legacy_meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta")) {
          gchar* text = gst_structure_to_string(gst_custom_meta_get_structure(legacy_meta));
          std::cerr << "[preproc dynamic] " << (text ? text : "") << "\n";
          g_free(text);
        }
      }
      require(error.empty() && bool(publication) == (mode == "compact"),
              "Preproc publication mode");
      const auto* frame = simaai::gst::tensor_buffer_frame_metadata(buffer, &error);
      require(error.empty() && bool(frame) == (mode == "compact"),
              "Preproc frame publication mode");
      if (frame) {
        require(frame->preprocess && !gst_buffer_get_custom_meta(buffer, "GstSimaMeta"),
                "Preproc full compact facts missing");
      }
      require(outs.front().semantic.preprocess &&
                  outs.front().semantic.preprocess->original_width == img.cols &&
                  outs.front().semantic.preprocess->original_height == img.rows,
              "Preproc public geometry missing");
      if (!held)
        held = result;
      require(held->frame_id == 300 && held->pts_ns == 1000000 &&
                  held->stream_id == "preproc-camera",
              "Preproc held Sample changed");
      // The first envelope remains held while subsequent outputs are recycled.
      outs.clear();
    }
    run.close();
    outs = simaai::neat::tensors_from_sample(*held);

    require(outs.size() == 1, "Preproc output missing tensor");
    const simaai::neat::Tensor* tensor = &outs.front();
    require(tensor->shape.size() >= 2, "Preproc output missing shape");
    const int64_t h = tensor->shape[0];
    const int64_t w = tensor->shape[1];
    require(w == 640 && h == 640, "Preproc size mismatch");
    if (tensor->semantic.image.has_value()) {
      require(tensor->semantic.image->format == simaai::neat::ImageSpec::PixelFormat::RGB,
              "Preproc format mismatch");
    }
    require(tensor->dtype == simaai::neat::TensorDType::UInt8 ||
                tensor->dtype == simaai::neat::TensorDType::Int8,
            "Preproc dtype mismatch");

    simaai::neat::Tensor cpu = tensor->clone();
    simaai::neat::Mapping map = cpu.map(simaai::neat::MapMode::Read);
    const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    require(map.data != nullptr && map.size_bytes >= expected, "Preproc bytes missing");
    const uint8_t* plane = static_cast<const uint8_t*>(map.data);
    for (size_t i = 0; i < expected; i += 3) {
      if (plane[i] != 64 || plane[i + 1] != 128 || plane[i + 2] != 192) {
        throw std::runtime_error("Preproc output bytes mismatch");
      }
    }

    std::cout << "[OK] preproc_standalone_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (is_dispatcher_unavailable(msg)) {
      return fail_test("dispatcher unavailable");
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
}
