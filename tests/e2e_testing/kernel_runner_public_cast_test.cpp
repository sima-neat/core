// Real numerical qualification through public Graph/Input/Cast/Output APIs.
// Transport inspection below verifies representation; it never supplies data,
// metadata or a private prepared stage to the graph under test.
#include <neat.h>
#include "pipeline/runtime/RunCore.h"
#include <gstsimaaitensorbuffer_meta.h>
#include <simaai/scheduler/testing/probe.h>
#include <simaai/scheduler/client/tensor_buffer.h>
#include <gst/gst.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>
#include <ctime>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace simaai::neat;
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
void check(const Tensor& tensor, float value) {
  require(tensor.dtype == TensorDType::BFloat16, "Cast output dtype");
  require(tensor.shape == std::vector<int64_t>({8, 8, 3}), "Cast output shape");
  require(tensor.layout == TensorLayout::HWC, "Cast output lost explicit HWC axes");
  auto cpu = tensor.cpu().contiguous();
  auto mapping = cpu.map(MapMode::Read);
  require(mapping.data && mapping.size_bytes >= 192 * sizeof(uint16_t), "Cast output extent");
  const auto expected = static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16);
  const auto* values = static_cast<const uint16_t*>(mapping.data);
  for (std::size_t i = 0; i < 192; ++i)
    require(values[i] == expected, "Cast numerical mismatch");
}
// Read-only transport inspection: all preparation is performed by public Graph::build.
void require_prepared_runners(Run& run, unsigned expected) {
  const auto core = run_internal::core(run);
  require(bool(core), "Run lost its Core owner");
  auto* pipeline = core->pipeline.stream.pipeline_handle();
  require(pipeline && GST_IS_BIN(pipeline), "missing live Core pipeline");
  auto* iterator = gst_bin_iterate_recurse(GST_BIN(pipeline));
  GValue element = G_VALUE_INIT;
  unsigned native = 0, legacy = 0, prepared = 0;
  while (gst_iterator_next(iterator, &element) == GST_ITERATOR_OK) {
    auto* instance = GST_ELEMENT(g_value_get_object(&element));
    auto* factory = gst_element_get_factory(instance);
    const auto* name = factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : "";
    if (g_str_equal(name, "neatkernelrunner")) {
      ++native;
      const auto stage = simaai::scheduler::testing::inspect_stage(instance);
      if (stage.valid && stage.capacity > 0) {
        ++prepared;
        std::cerr << "[cast admission] element=" << GST_OBJECT_NAME(instance)
                  << " stage_capacity=" << stage.capacity << "\n";
      }
    }
    legacy += g_str_equal(name, "neatprocesscvu");
    g_value_reset(&element);
  }
  if (G_IS_VALUE(&element))
    g_value_unset(&element);
  gst_iterator_free(iterator);
  require(native == expected && prepared == expected && legacy == 0,
          "every Cast stage must have a prepared Runtime context, without a legacy executor");
}

void require_publication(const Tensor& tensor, bool compact) {
  const auto& storage = tensor.storage;
  require(storage && storage->kind == StorageKind::GstSample && storage->holder,
          "output lost zero-copy GstSample owner");
  auto* buffer = gst_sample_get_buffer(static_cast<GstSample*>(storage->holder.get()));
  std::string error;
  const auto publication = simaai::gst::tensor_buffer_publication(buffer, &error);
  require(error.empty() && bool(publication) == compact, "output publication mode mismatch");
  const auto* frame = simaai::gst::tensor_buffer_frame_metadata(buffer, &error);
  require(error.empty() && bool(frame) == compact, "output dynamic metadata authority mismatch");
  require(bool(gst_buffer_get_custom_meta(buffer, "GstSimaMeta")) != compact,
          "output legacy metadata authority mismatch");
}

void qualify_two_cast_stages(bool compact, int a65_workers) {
  GraphOptions graph_options;
  graph_options.processcvu_requested_run_target = "A65";
  graph_options.kernel_runner_a65_workers = a65_workers;
  Graph graph(graph_options);
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.memory_policy = InputMemoryPolicy::Ev74;
  graph.add(nodes::Input(input));
  CastOptions narrow;
  narrow.direction = CastDirection::Fp32ToBf16;
  narrow.element_name = "roundtrip_narrow";
  graph.add(nodes::Cast(narrow));
  CastOptions widen;
  widen.direction = CastDirection::Bf16ToFp32;
  widen.element_name = "roundtrip_widen";
  graph.add(nodes::Cast(widen));
  OutputOptions output;
  output.sync = false;
  output.drop = false;
  output.max_buffers = 2;
  graph.add(nodes::Output(output));
  RunOptions options;
  options.output_memory = OutputMemory::ZeroCopy;
  options.queue_depth = 2;
  // Exactly representable BF16 values isolate stage sequencing and retained
  // memory correctness from an unrelated rounding-mode qualification.
  constexpr std::array<float, 8> pattern{0.0F, 1.0F, -2.0F, 0.5F, 3.5F, 256.0F, -0.125F, 64.0F};
  auto values_for = [&](int frame) {
    std::vector<float> values(192);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = pattern[(i + static_cast<std::size_t>(frame)) % pattern.size()];
    return values;
  };
  auto make_input = [&](int frame) {
    auto tensor = Tensor::from_vector(values_for(frame), {8, 8, 3}, TensorMemory::EV74);
    tensor.layout = TensorLayout::HWC;
    return tensor;
  };
  auto check_output = [&](const Tensor& tensor, int frame) {
    require(tensor.dtype == TensorDType::Float32 &&
                tensor.shape == std::vector<int64_t>({8, 8, 3}) &&
                tensor.layout == TensorLayout::HWC,
            "two-stage Cast output contract");
    auto cpu = tensor.cpu().contiguous();
    auto mapping = cpu.map(MapMode::Read);
    const auto expected = values_for(frame);
    require(mapping.data && mapping.size_bytes >= expected.size() * sizeof(float),
            "two-stage Cast output extent");
    const auto* values = static_cast<const float*>(mapping.data);
    for (std::size_t i = 0; i < expected.size(); ++i)
      require(values[i] == expected[i], "two-stage Cast numerical mismatch");
  };
  auto run = graph.build(TensorList{make_input(0)}, options);
  require_prepared_runners(run, 2);
  std::optional<Sample> held;
  for (int frame = 0; frame < 4; ++frame) {
    Sample input_sample;
    input_sample.kind = SampleKind::TensorSet;
    input_sample.tensors = {make_input(frame)};
    input_sample.frame_id = 700 + frame;
    input_sample.input_seq = 800 + frame;
    input_sample.orig_input_seq = 900 + frame;
    input_sample.stream_id = "two-cast-camera";
    input_sample.pts_ns = (frame + 1) * 2000000;
    input_sample.dts_ns = input_sample.pts_ns - 100;
    input_sample.duration_ns = 33333;
    auto result = run.run(input_sample, 10000);
    require(
        result.frame_id == input_sample.frame_id && result.input_seq == input_sample.input_seq &&
            result.orig_input_seq == input_sample.orig_input_seq &&
            result.stream_id == input_sample.stream_id && result.pts_ns == input_sample.pts_ns &&
            result.dts_ns == input_sample.dts_ns && result.duration_ns == input_sample.duration_ns,
        "two-stage Cast Sample metadata");
    const auto tensors = tensors_from_sample(result);
    require(tensors.size() == 1, "two-stage Cast output count");
    check_output(tensors.front(), frame);
    require_publication(tensors.front(), compact);
    if (!held)
      held = result;
    check_output(tensors_from_sample(*held).front(), 0);
    require(held->frame_id == 700 && held->pts_ns == 2000000,
            "two-stage Cast held Sample was mutated");
  }
  run.close_input();
  Sample extra;
  require(run.pull(10000, extra) == PullStatus::Closed, "two-stage Cast EOS did not drain");
  run.close();
  require(!run.running(), "two-stage Cast did not close");
  const auto retained = tensors_from_sample(*held);
  check_output(retained.front(), 0);
  require_publication(retained.front(), compact);
}

struct MeasurementOptions {
  std::size_t samples = 0, warmup = 2000, depth = 1;
  std::string raw;
};
using Clock = std::chrono::steady_clock;
// Keep DMA owners alive on an uncertain push/pull/EOS error. The external EV
// guard must prove remoteproc offline before terminating this process group.
[[noreturn]] void uncertain_measurement() {
  std::cerr << "UNRETIRED public Core measurement: retaining owners for guarded stop\n"
            << std::flush;
  for (;;)
    std::this_thread::sleep_for(std::chrono::seconds(60));
}
void measure(Run& run, const std::string& mode, const MeasurementOptions& options) {
  struct Record {
    Clock::time_point start, submitted, observed;
  };
  std::vector<Record> records(options.samples);
  std::array<Sample, 4> inputs;
  constexpr std::array<float, 4> values{1.0F, -2.0F, 3.0F, 4.0F};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i].kind = SampleKind::TensorSet;
    inputs[i].tensors = {
        Tensor::from_vector(std::vector<float>(192, values[i]), {8, 8, 3}, TensorMemory::EV74)};
    inputs[i].tensors.front().layout = TensorLayout::HWC;
  }
  // Only main's original correctness output stays held during streaming.
  // Capture one final measured output after all submissions; even native
  // serial pool2 then has one rotating credit until no more push is needed.
  std::array<std::optional<Tensor>, 1> retained;
  const auto cpu_seconds = [] {
    timespec value{};
    require(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value) == 0, "process CPU clock");
    return double(value.tv_sec) + double(value.tv_nsec) * 1e-9;
  };
  auto batch = [&](std::size_t count, bool record) {
    std::size_t outstanding = 0;
    auto retire = [&](std::size_t frame) {
      Sample sample;
      if (run.pull(10000, sample) != PullStatus::Ok)
        uncertain_measurement();
      const auto observed = Clock::now();
      --outstanding;
      if (record)
        records[frame].observed = observed;
      // Validation is included in batch throughput, but no per-frame CPU map,
      // data scan, CSV write, or diagnostic metadata query is performed here.
      const auto& tensors = sample_tensor_list(sample);
      require(sample.frame_id == static_cast<int64_t>(frame), "measured frame identity");
      require(tensors.size() == 1, "measured output cardinality");
      const auto& tensor = tensors.front();
      require(tensor.dtype == TensorDType::BFloat16 && tensor.shape.size() == 3 &&
                  tensor.shape[0] == 8 && tensor.shape[1] == 8 && tensor.shape[2] == 3,
              "measured output dtype/shape");
      if (record && frame + 1 == count)
        retained[0] = tensor; // Final pull: no subsequent push can need this credit.
    };
    try {
      for (std::size_t frame = 0; frame < count; ++frame) {
        if (frame >= options.depth)
          retire(frame - options.depth);
        auto& input = inputs[frame % inputs.size()];
        input.frame_id = static_cast<int64_t>(frame);
        if (record)
          records[frame].start = Clock::now();
        // Conservatively retain on push failure: acceptance may be uncertain.
        ++outstanding;
        if (!run.push(input))
          uncertain_measurement();
        if (record)
          records[frame].submitted = Clock::now();
      }
      for (std::size_t frame = count > options.depth ? count - options.depth : 0; frame < count;
           ++frame)
        retire(frame);
    } catch (...) {
      if (outstanding)
        uncertain_measurement();
      throw;
    }
  };
  batch(options.warmup, false);
  const double cpu = cpu_seconds();
  const auto begin = Clock::now();
  batch(options.samples, true);
  const auto delivered = Clock::now();
  const double delivery_cpu_used = cpu_seconds() - cpu;
  // Public EOS boundary is explicit. Output availability is not mistaken for
  // callback/service retirement; close completion is reported separately.
  try {
    run.close_input();
    Sample extra;
    if (run.pull(10000, extra) != PullStatus::Closed)
      uncertain_measurement();
  } catch (...) {
    uncertain_measurement();
  }
  const auto eos = Clock::now();
  run.close();
  const auto closed = Clock::now();
  const double cpu_used = cpu_seconds() - cpu;
  for (std::size_t i = 0; i < retained.size() && i < options.samples; ++i) {
    require(bool(retained[i]), "missing retained measured output");
    check(*retained[i], values[(options.samples - 1) % values.size()]);
    const auto& storage = retained[i]->storage;
    require(storage && storage->kind == StorageKind::GstSample && storage->holder,
            "measured output lost zero-copy owner");
    auto* sample = static_cast<GstSample*>(storage->holder.get());
    std::string error;
    auto publication =
        simaai::gst::tensor_buffer_publication(gst_sample_get_buffer(sample), &error);
    require(error.empty() && bool(publication) == (mode == "compact"),
            "retained measured publication mismatch");
    const auto* frame =
        simaai::gst::tensor_buffer_frame_metadata(gst_sample_get_buffer(sample), &error);
    require(error.empty() && bool(frame) == (mode == "compact"),
            "retained measured full-frame publication mismatch");
    if (frame)
      require(!gst_buffer_get_custom_meta(gst_sample_get_buffer(sample), "GstSimaMeta"),
              "retained measured frame still has legacy metadata");
  }
  const auto us = [](auto duration) {
    return std::chrono::duration<double, std::micro>(duration).count();
  };
  std::vector<double> latency;
  latency.reserve(records.size());
  std::ofstream raw;
  if (!options.raw.empty()) {
    raw.open(options.raw);
    require(bool(raw), "open measurement CSV");
    raw << "frame,start_us,submit_return_us,application_observed_us\n" << std::setprecision(12);
  }
  for (std::size_t frame = 0; frame < records.size(); ++frame) {
    const auto& record = records[frame];
    latency.push_back(us(record.observed - record.start));
    if (raw.is_open())
      raw << frame << ',' << us(record.start - begin) << ',' << us(record.submitted - begin) << ','
          << us(record.observed - begin) << '\n';
  }
  if (raw.is_open()) {
    raw.flush();
    require(bool(raw), "write measurement CSV");
  }
  std::sort(latency.begin(), latency.end());
  const auto percentile = [&](std::size_t percent) {
    return latency[std::min(latency.size() - 1, latency.size() * percent / 100)];
  };
  std::cout << "# public_Core_boundary: immutable four-buffer input ring; one original output held "
               "during streaming; final measured output held only after last pull; exact "
               "frame/type/shape checks "
               "in batch; initial/final retained math after close; no private service fence\n"
            << "mode,depth,configured_run_queue_depth,samples,median_observed_us,p95_observed_us,"
               "p99_observed_us,max_observed_us,"
               "delivery_batch_us,eos_us,close_us,through_close_us,delivery_fps,through_close_fps,"
               "delivery_process_cpu_us_per_frame,through_close_process_cpu_us_per_frame\n"
            << std::setprecision(12) << mode << ',' << options.depth << ','
            << std::max<std::size_t>(2, options.depth) << ',' << options.samples << ','
            << percentile(50) << ',' << percentile(95) << ',' << percentile(99) << ','
            << latency.back() << ',' << us(delivered - begin) << ',' << us(eos - delivered) << ','
            << us(closed - eos) << ',' << us(closed - begin) << ','
            << options.samples * 1e6 / us(delivered - begin) << ','
            << options.samples * 1e6 / us(closed - begin) << ','
            << delivery_cpu_used * 1e6 / options.samples << ',' << cpu_used * 1e6 / options.samples
            << '\n';
}
} // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "native";
    require(mode == "native" || mode == "compact", "invalid metadata mode");
    MeasurementOptions measurement;
    int a65_workers = GraphOptions{}.kernel_runner_a65_workers;
    for (int i = 2; i < argc; ++i) {
      const std::string argument = argv[i];
      require(i + 1 < argc, "missing option value");
      const std::string value = argv[++i];
      if (argument == "--samples")
        measurement.samples = std::stoull(value);
      else if (argument == "--warmup")
        measurement.warmup = std::stoull(value);
      else if (argument == "--depth")
        measurement.depth = std::stoull(value);
      else if (argument == "--a65-workers") {
        std::size_t used = 0;
        a65_workers = std::stoi(value, &used);
        require(used == value.size() && a65_workers >= 1 && a65_workers <= 64,
                "A65 worker count must be 1..64");
      } else if (argument == "--raw")
        measurement.raw = value;
      else
        throw std::runtime_error("unknown option");
    }
    require(measurement.depth >= 1 && measurement.depth <= 8, "measurement depth must be1..8");
    require(measurement.samples <= 1000000 && measurement.warmup <= 1000000,
            "measurement frame limit");
    require(measurement.raw.empty() || measurement.samples > 0, "raw requires measurement samples");
    setenv("SIMA_CORE_COMPACT_METADATA", mode == "compact" ? "1" : "0", 1);
    setenv("SIMA_GST_RUN_INSERT_BOUNDARIES", "0", 1);
    GraphOptions graph_options;
    graph_options.processcvu_requested_run_target = "A65";
    graph_options.kernel_runner_a65_workers = a65_workers;
    Graph graph(graph_options);
    InputOptions input;
    input.payload_type = PayloadType::Tensor;
    input.memory_policy = InputMemoryPolicy::Ev74;
    graph.add(nodes::Input(input));
    CastOptions cast;
    cast.direction = CastDirection::Fp32ToBf16;
    graph.add(nodes::Cast(cast));
    OutputOptions output;
    output.sync = false;
    output.drop = false;
    output.max_buffers = static_cast<int>(std::max<std::size_t>(2, measurement.depth));
    graph.add(nodes::Output(output));
    auto make_input = [](float value) {
      auto tensor =
          Tensor::from_vector(std::vector<float>(192, value), {8, 8, 3}, TensorMemory::EV74);
      tensor.layout =
          TensorLayout::HWC; // Explicit caller-authored image axes, not shape inference.
      return tensor;
    };
    RunOptions options;
    options.output_memory = OutputMemory::ZeroCopy;
    // Queue sizing is unchanged by retained-output checks. The measured caller
    // window is depth; actual provider pool credits can constrain overlap.
    options.queue_depth = static_cast<int>(std::max<std::size_t>(2, measurement.depth));
    auto run = graph.build(TensorList{make_input(1.0F)}, options);
    require(graph.last_pipeline().find("neatkernelrunner") != std::string::npos,
            "Graph did not select the default Runtime factory");
    require_prepared_runners(run, 1);
    std::optional<Tensor> held;
    std::optional<Sample> held_sample;
    int sequence = 0;
    for (const float value : {1.0F, -2.0F, 3.0F, 4.0F}) {
      Sample input_sample;
      input_sample.kind = SampleKind::TensorSet;
      input_sample.tensors = {make_input(value)};
      input_sample.frame_id = 300 + sequence;
      input_sample.input_seq = 100 + sequence;
      input_sample.orig_input_seq = 200 + sequence;
      input_sample.stream_id = "qualification-camera";
      input_sample.pts_ns = (sequence + 1) * 1000000;
      input_sample.dts_ns = input_sample.pts_ns - 100;
      input_sample.duration_ns = 33333;
      std::cerr << "[cast correctness] submitting iteration=" << sequence << "\n";
      auto output_sample = run.run(input_sample, 10000);
      std::cerr << "[cast correctness] received iteration=" << sequence << "\n";
      require(output_sample.frame_id == input_sample.frame_id &&
                  output_sample.input_seq == input_sample.input_seq &&
                  output_sample.orig_input_seq == input_sample.orig_input_seq &&
                  output_sample.stream_id == input_sample.stream_id,
              "public Sample correlation mismatch");
      if (output_sample.pts_ns != input_sample.pts_ns ||
          output_sample.dts_ns != input_sample.dts_ns ||
          output_sample.duration_ns != input_sample.duration_ns) {
        std::cerr << "TIMING actual=" << output_sample.pts_ns << ',' << output_sample.dts_ns << ','
                  << output_sample.duration_ns << " expected=" << input_sample.pts_ns << ','
                  << input_sample.dts_ns << ',' << input_sample.duration_ns << '\n';
      }
      require(output_sample.pts_ns == input_sample.pts_ns &&
                  output_sample.dts_ns == input_sample.dts_ns &&
                  output_sample.duration_ns == input_sample.duration_ns,
              "public Sample timing mismatch");
      auto result = tensors_from_sample(output_sample);
      require(result.size() == 1, "Cast output count");
      if (result.front().dtype != TensorDType::BFloat16) {
        std::cerr << "DTYPE public=" << static_cast<int>(result.front().dtype)
                  << " expected=" << static_cast<int>(TensorDType::BFloat16) << '\n';
        const auto& owner = result.front().storage;
        if (owner && owner->kind == StorageKind::GstSample && owner->holder) {
          auto* observed = static_cast<GstSample*>(owner->holder.get());
          gchar* caps = gst_caps_to_string(gst_sample_get_caps(observed));
          std::cerr << "CAPS " << (caps ? caps : "null") << '\n';
          g_free(caps);
          simaai::scheduler::client::TensorBufferView raw;
          std::string error;
          if (simaai::scheduler::client::tensor_buffer_create_view(observed, &raw, &error)) {
            for (const auto& tensor : raw.tensors())
              std::cerr << "CANONICAL dtype=" << tensor.dtype << " bytes=" << tensor.size_bytes
                        << " logical=" << tensor.logical_index << " name=" << tensor.logical_name
                        << '\n';
          } else
            std::cerr << "CANONICAL error=" << error << '\n';
        }
      }
      check(result.front(), value);
      const auto& storage = result.front().storage;
      require(storage && storage->kind == StorageKind::GstSample && storage->holder,
              "zero-copy output lost GstSample owner");
      auto* sample = static_cast<GstSample*>(storage->holder.get());
      auto* buffer = gst_sample_get_buffer(sample);
      std::string error;
      auto publication = simaai::gst::tensor_buffer_publication(buffer, &error);
      require(error.empty(), "malformed output publication");
      require(bool(publication) == (mode == "compact"), "unexpected publication mode");
      const auto* frame = simaai::gst::tensor_buffer_frame_metadata(buffer, &error);
      require(error.empty(), "malformed current-frame publication");
      require(bool(frame) == (mode == "compact"),
              "full compact frame negotiation did not take effect");
      if (frame)
        require(!gst_buffer_get_custom_meta(buffer, "GstSimaMeta"),
                "full compact retained legacy dynamic metadata");
      if (!held) {
        held = result.front();
        held_sample = output_sample;
      }
      require(held_sample->frame_id == 300 && held_sample->input_seq == 100 &&
                  held_sample->orig_input_seq == 200 && held_sample->pts_ns == 1000000 &&
                  held_sample->dts_ns == 999900 && held_sample->duration_ns == 33333 &&
                  held_sample->stream_id == "qualification-camera",
              "held Sample metadata changed");
      check(*held, 1.0F);
      ++sequence;
    }
    if (measurement.samples)
      measure(run, mode, measurement);
    else
      run.close();
    check(*held, 1.0F);
    if (mode == "compact") {
      // Re-enter the same default executor with compact publication disabled.
      // appsrc must materialize metadata without changing execution factories.
      setenv("SIMA_CORE_COMPACT_METADATA", "0", 1);
      Graph metadata_graph(graph_options);
      metadata_graph.add(nodes::Input(input));
      cast.direction = CastDirection::Bf16ToFp32;
      metadata_graph.add(nodes::Cast(cast));
      metadata_graph.add(nodes::Output(output));
      auto metadata_run = metadata_graph.build(TensorList{*held}, options);
      require(metadata_graph.last_pipeline().find("neatkernelrunner") != std::string::npos &&
                  metadata_graph.last_pipeline().find("neatprocesscvu") == std::string::npos,
              "metadata compatibility must not select a legacy executor");
      require_prepared_runners(metadata_run, 1);
      auto result = metadata_run.run(TensorList{*held}, 10000);
      require(result.size() == 1 && result.front().dtype == TensorDType::Float32,
              "compact-to-legacy repush output type");
      require(result.front().layout == TensorLayout::HWC, "repush lost explicit HWC axes");
      auto cpu = result.front().cpu().contiguous();
      auto mapping = cpu.map(MapMode::Read);
      require(mapping.data && mapping.size_bytes >= 192 * sizeof(float), "repush output extent");
      for (std::size_t i = 0; i < 192; ++i)
        require(static_cast<const float*>(mapping.data)[i] == 1.0F, "repush numerical mismatch");
      require_publication(result.front(), false);
      metadata_run.close();
      check(*held, 1.0F);
      require_publication(*held, true);
    }
    // Restore this test's metadata mode after the compatibility repush.
    setenv("SIMA_CORE_COMPACT_METADATA", mode == "compact" ? "1" : "0", 1);
    qualify_two_cast_stages(mode == "compact", a65_workers);
    std::cout << "PASS public Core Cast: " << mode << ", a65_workers=" << a65_workers
              << ", actual math, public decode, held output\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL public Core Cast: " << error.what() << '\n';
    return 1;
  }
}
