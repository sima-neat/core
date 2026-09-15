#include "model/Model.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaDecode.h"
#include "perf/codec_perf_common.h"
#include "pipeline/Graph.h"

#include "asset_utils.h"
#include "dmabuf_test_utils.h"
#include "resnet50_test_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kFps = 30;
constexpr int kFrames = 8;
constexpr int kQueueDepth = 16;
constexpr int kPullTimeoutMs = 20000;

simaai::neat::RunOptions
e2e_run_options(simaai::neat::OutputMemory output_memory = simaai::neat::OutputMemory::Auto) {
  simaai::neat::RunOptions options;
  options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  options.queue_depth = kQueueDepth;
  // Exercise native Auto behavior; an explicit Owned case remains below.
  options.output_memory = output_memory;
  options.advanced.copy_input = false;
  options.startup_preflight = false;
  return options;
}

simaai::neat::Graph make_graph(const simaai::neat::Sample& seed, const std::string& model_path,
                               const std::string& mode, int output_buffers) {
  const std::string decoder_name = "decoder_h265_" + mode;
  simaai::neat::Graph graph("h265-decode-resnet50-" + mode);

  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Encoded;
  input.caps_override = seed.caps_string;
  input.block = true;
  input.pool_max_buffers = kQueueDepth;
  input.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  graph.add(simaai::neat::nodes::Input(input));
  graph.add(simaai::neat::nodes::Custom(sima_codec_perf::h265_parser_fragment()));

  simaai::neat::SimaDecodeOptions decoder;
  decoder.type = simaai::neat::SimaDecodeType::H265;
  decoder.decoder_name = decoder_name;
  decoder.dec_width = kWidth;
  decoder.dec_height = kHeight;
  decoder.dec_fps = kFps;
  decoder.input_buffers = 8;
  decoder.num_buffers = 64;
  graph.add(simaai::neat::nodes::SimaDecode(decoder));
  graph.add(
      simaai::neat::nodes::CapsRaw("NV12", kWidth, kHeight, kFps, simaai::neat::CapsMemory::Any));

  simaai::neat::Model::Options model_options;
  model_options.preprocess.kind = simaai::neat::InputKind::Image;
  model_options.preprocess.enable = simaai::neat::AutoFlag::On;
  model_options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  model_options.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  model_options.upstream_name = decoder_name;
  simaai::neat::Model model(model_path, model_options);

  simaai::neat::Model::RouteOptions route;
  route.include_input = false;
  route.include_output = false;
  graph.add(model.graph(route));
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(output_buffers)));
  return graph;
}

void require_zero_copy_pipeline(const simaai::neat::Graph& graph, const std::string& mode) {
  const std::string& pipeline = graph.last_pipeline();
  require(!pipeline.empty(), mode + ": built pipeline string is empty");
  require_contains(pipeline, "neatdecoder name=decoder_h265_" + mode,
                   mode + ": H.265 decoder is missing from the built pipeline");
  require_contains(pipeline, "zero-copy-output=true",
                   mode + ": decoder is not configured for zero-copy output");
}

// Test-local, read-only instrumentation. Public Graph intentionally exposes no
// GstElement handle. Observe elements while the graph is assembled, including
// request sink pads added later by ProcessCVU, and remove every hook on exit.
class DmaHandoffEvidence {
  enum class Point : std::size_t { DecoderOutput, CvuInput, CvuOutput, MlaInput };
  struct Observation {
    GstClockTime pts;
    sima_test::DmaBufSpan span;
  };
  struct Probe {
    GstPad* pad;
    gulong id;
  };
  struct Connection {
    GstElement* element;
    gulong id;
  };
  struct PadProgress {
    std::string stage;
    std::string pad;
    bool input;
    std::size_t buffers = 0;
    std::size_t memories = 0;
    std::size_t missing_pts = 0;
    GstClockTime min_pts = GST_CLOCK_TIME_NONE;
    GstClockTime max_pts = GST_CLOCK_TIME_NONE;
  };
  struct State {
    std::mutex mutex;
    std::array<std::vector<Observation>, 4> observations;
    std::vector<Probe> probes;
    std::vector<Connection> connections;
    std::vector<PadProgress> progress;
    std::atomic<bool> failed{false};
    bool closing = false;
  };
  struct ProbeContext {
    std::shared_ptr<State> state;
    std::optional<Point> point;
    std::size_t progress_index;
  };

  static void record_buffer(ProbeContext& context, GstBuffer* buffer) {
    require(buffer != nullptr, "DMA probe received no buffer");
    std::lock_guard<std::mutex> lock(context.state->mutex);
    auto& progress = context.state->progress[context.progress_index];
    ++progress.buffers;
    progress.memories += gst_buffer_n_memory(buffer);
    const GstClockTime pts = GST_BUFFER_PTS(buffer);
    if (GST_CLOCK_TIME_IS_VALID(pts)) {
      progress.min_pts = std::min(progress.min_pts, pts);
      progress.max_pts =
          GST_CLOCK_TIME_IS_VALID(progress.max_pts) ? std::max(progress.max_pts, pts) : pts;
    } else {
      ++progress.missing_pts;
    }
    // MLA output is observed only to locate a stalled handoff. Keep the
    // allocation-identity assertions limited to their existing boundaries.
    if (!context.point) {
      return;
    }
    require(gst_buffer_n_memory(buffer) > 0, "DMA probe received no payload memory");
    for (guint i = 0; i < gst_buffer_n_memory(buffer); ++i) {
      context.state->observations[static_cast<std::size_t>(*context.point)].push_back(
          {GST_BUFFER_PTS(buffer), sima_test::dmabuf_span(gst_buffer_peek_memory(buffer, i))});
    }
  }

  static GstPadProbeReturn observe(GstPad*, GstPadProbeInfo* info, gpointer data) noexcept {
    auto& context = *static_cast<ProbeContext*>(data);
    try {
      if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER_LIST) {
        GstBufferList* list = gst_pad_probe_info_get_buffer_list(info);
        require(list != nullptr, "DMA probe received no buffer list");
        for (guint i = 0; i < gst_buffer_list_length(list); ++i) {
          record_buffer(context, gst_buffer_list_get(list, i));
        }
      } else {
        record_buffer(context, gst_pad_probe_info_get_buffer(info));
      }
    } catch (...) {
      context.state->failed.store(true);
    }
    return GST_PAD_PROBE_OK;
  }

  static void attach_pad(GstElement* element, GstPad* pad, const std::shared_ptr<State>& state) {
    GstElementFactory* factory = gst_element_get_factory(element);
    const std::string name =
        factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : "";
    std::optional<Point> point;
    const bool input = GST_PAD_DIRECTION(pad) == GST_PAD_SINK;
    if (name == "neatdecoder" && !input) {
      point = Point::DecoderOutput;
    } else if (name == "neatprocesscvu") {
      point = input ? Point::CvuInput : Point::CvuOutput;
    } else if (name == "neatprocessmla" && input) {
      point = Point::MlaInput;
    }
    if (!point && name != "neatprocessmla") {
      return;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closing || std::any_of(state->probes.begin(), state->probes.end(),
                                      [pad](const Probe& probe) { return probe.pad == pad; })) {
      return;
    }
    state->probes.reserve(state->probes.size() + 1U);
    const std::size_t progress_index = state->progress.size();
    state->progress.push_back({GST_ELEMENT_NAME(element), GST_PAD_NAME(pad), input});
    auto context = std::make_unique<ProbeContext>(ProbeContext{state, point, progress_index});
    const auto types =
        static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_BUFFER | GST_PAD_PROBE_TYPE_BUFFER_LIST);
    const gulong id = gst_pad_add_probe(pad, types, observe, context.get(),
                                        [](gpointer p) { delete static_cast<ProbeContext*>(p); });
    require(id != 0, "could not attach DMA handoff pad probe");
    context.release();
    state->probes.push_back({GST_PAD(gst_object_ref(pad)), id});
  }

  static void pad_added(GstElement* element, GstPad* pad, gpointer data) noexcept {
    const auto state = *static_cast<std::shared_ptr<State>*>(data);
    try {
      attach_pad(element, pad, state);
    } catch (...) {
      state->failed.store(true);
    }
  }

  static gboolean element_added(GSignalInvocationHint*, guint count, const GValue* values,
                                gpointer data) noexcept {
    const auto state = *static_cast<std::shared_ptr<State>*>(data);
    try {
      require(count == 3, "unexpected GstBin deep-element-added signature");
      auto* element = GST_ELEMENT(g_value_get_object(&values[2]));
      GstElementFactory* factory = gst_element_get_factory(element);
      const std::string name =
          factory ? gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory)) : "";
      if (name != "neatdecoder" && name != "neatprocesscvu" && name != "neatprocessmla") {
        return TRUE;
      }
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->closing ||
            std::any_of(state->connections.begin(), state->connections.end(),
                        [element](const Connection& item) { return item.element == element; })) {
          return TRUE;
        }
        state->connections.reserve(state->connections.size() + 1U);
        auto context = std::make_unique<std::shared_ptr<State>>(state);
        const gulong id = g_signal_connect_data(
            element, "pad-added", G_CALLBACK(pad_added), context.get(),
            [](gpointer p, GClosure*) { delete static_cast<std::shared_ptr<State>*>(p); },
            GConnectFlags(0));
        require(id != 0, "could not watch DMA stage request pads");
        context.release();
        state->connections.push_back({GST_ELEMENT(gst_object_ref(element)), id});
      }
      std::unique_ptr<GstIterator, decltype(&gst_iterator_free)> iterator(
          gst_element_iterate_pads(element), gst_iterator_free);
      struct IteratorValue {
        GValue value = G_VALUE_INIT;
        ~IteratorValue() {
          if (G_VALUE_TYPE(&value) != 0) {
            g_value_unset(&value);
          }
        }
      } item;
      while (true) {
        const auto result = gst_iterator_next(iterator.get(), &item.value);
        if (result == GST_ITERATOR_OK) {
          attach_pad(element, GST_PAD(g_value_get_object(&item.value)), state);
          g_value_reset(&item.value);
        } else if (result == GST_ITERATOR_RESYNC) {
          gst_iterator_resync(iterator.get());
        } else {
          break;
        }
      }
    } catch (...) {
      state->failed.store(true);
    }
    return TRUE;
  }

public:
  DmaHandoffEvidence() : state_(std::make_shared<State>()) {
    simaai::neat::gst_init_once();
    gpointer klass = g_type_class_ref(GST_TYPE_BIN);
    signal_ = g_signal_lookup("deep-element-added", GST_TYPE_BIN);
    g_type_class_unref(klass);
    require(signal_ != 0, "GstBin element observation signal unavailable");
    hook_ = g_signal_add_emission_hook(signal_, 0, element_added, &state_, nullptr);
    require(hook_ != 0, "could not install graph DMA element observation hook");
  }

  DmaHandoffEvidence(const DmaHandoffEvidence&) = delete;
  DmaHandoffEvidence& operator=(const DmaHandoffEvidence&) = delete;

  ~DmaHandoffEvidence() {
    g_signal_remove_emission_hook(signal_, hook_);
    std::vector<Probe> probes;
    std::vector<Connection> connections;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->closing = true;
      probes.swap(state_->probes);
      connections.swap(state_->connections);
    }
    for (const auto& connection : connections) {
      g_signal_handler_disconnect(connection.element, connection.id);
      gst_object_unref(connection.element);
    }
    for (const auto& probe : probes) {
      gst_pad_remove_probe(probe.pad, probe.id);
      gst_object_unref(probe.pad);
    }
  }

  void report_async_failure(std::size_t frame, std::size_t expected) noexcept {
    try {
      std::lock_guard<std::mutex> lock(state_->mutex);
      std::fprintf(stderr,
                   "[DMA progress] async consumer_frame=%zu expected=%zu probe_failed=%d "
                   "(snapshot before stop; PTS in ns)\n",
                   frame, expected, state_->failed.load() ? 1 : 0);
      for (const auto& progress : state_->progress) {
        std::fprintf(stderr,
                     "[DMA progress] stage=%s pad=%s direction=%s buffers=%zu memories=%zu "
                     "missing_pts=%zu",
                     progress.stage.c_str(), progress.pad.c_str(), progress.input ? "sink" : "src",
                     progress.buffers, progress.memories, progress.missing_pts);
        if (GST_CLOCK_TIME_IS_VALID(progress.min_pts)) {
          std::fprintf(stderr, " pts_min=%" G_GUINT64_FORMAT " pts_max=%" G_GUINT64_FORMAT "\n",
                       progress.min_pts, progress.max_pts);
        } else {
          std::fprintf(stderr, " pts_range=none\n");
        }
      }
    } catch (...) {
      // Diagnostics must not replace the original failure or bypass teardown.
    }
  }

  void verify(const std::string& context) {
    require(!state_->failed.load(), context + ": DMA handoff probe encountered invalid storage");
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto& decoded = state_->observations[static_cast<std::size_t>(Point::DecoderOutput)];
    const auto& cvu_input = state_->observations[static_cast<std::size_t>(Point::CvuInput)];
    const auto& cvu_output = state_->observations[static_cast<std::size_t>(Point::CvuOutput)];
    const auto& mla_input = state_->observations[static_cast<std::size_t>(Point::MlaInput)];
    require(!decoded.empty() && !mla_input.empty(), context + ": decoder/MLA DMA evidence missing");
    for (const auto& frame : decoded) {
      require(std::any_of(cvu_input.begin(), cvu_input.end(),
                          [&](const Observation& input) {
                            return input.pts == frame.pts && input.span == frame.span;
                          }),
              context + ": CVU did not receive the decoder's original DMA-BUF span");
    }
    for (const auto& input : mla_input) {
      require(std::any_of(cvu_output.begin(), cvu_output.end(),
                          [&](const Observation& output) {
                            return input.pts == output.pts &&
                                   input.span.backing == output.span.backing &&
                                   input.span.offset >= output.span.offset &&
                                   input.span.offset - output.span.offset <= output.span.length &&
                                   input.span.length <= output.span.length - (input.span.offset -
                                                                              output.span.offset);
                          }),
              context + ": MLA input is not a view of a CVU-produced DMA allocation");
    }
    std::cout << "[OK] " << context
              << ": decoder->CVU and CVU-result->MLA pad DMA identities verified\n";
  }

private:
  std::shared_ptr<State> state_;
  guint signal_ = 0;
  gulong hook_ = 0;
};

void require_auto_output(const simaai::neat::Sample& sample, const std::string& context) {
  const auto tensors = simaai::neat::tensors_from_sample(sample, true);
  require(tensors.size() == 1 && tensors.front().storage && tensors.front().storage->holder,
          context + ": expected one retained model output");
  const auto& tensor = tensors.front();
  require(tensor.storage->kind == simaai::neat::StorageKind::GstSample,
          context + ": Auto copied terminal DMA-BUF output");
  auto* retained = static_cast<GstSample*>(tensor.storage->holder.get());
  require(GST_IS_SAMPLE(retained), context + ": output holder is not a GstSample");
  (void)sima_test::dmabuf_span(gst_sample_get_buffer(retained));
  require(tensor.dtype == simaai::neat::TensorDType::Float32 && tensor.is_dense() &&
              tensor.dense_bytes_tight() == sima_test::kResNet50Classes * sizeof(float),
          context + ": terminal model tensor contract mismatch");
  const auto mapping = tensor.map(simaai::neat::MapMode::Read);
  require(mapping.data && mapping.size_bytes >= sima_test::kResNet50Classes * sizeof(float),
          context + ": direct output read mapping failed");
  const auto* scores = static_cast<const float*>(mapping.data);
  require(std::all_of(scores, scores + sima_test::kResNet50Classes,
                      [](float value) { return std::isfinite(value); }),
          context + ": output scores are not finite");
}

void run_sync(const std::string& model_path, const std::vector<simaai::neat::Sample>& access_units,
              simaai::neat::OutputMemory memory = simaai::neat::OutputMemory::Auto) {
  DmaHandoffEvidence evidence;
  const std::string mode = memory == simaai::neat::OutputMemory::Owned ? "owned_sync" : "sync";
  simaai::neat::Graph graph = make_graph(access_units.front(), model_path, mode, 1);
  const simaai::neat::RunOptions options = e2e_run_options(memory);

  const simaai::neat::Sample output =
      graph.run(simaai::neat::Sample{access_units.front()}, options);
  if (memory == simaai::neat::OutputMemory::Owned) {
    sima_test::require_valid_resnet50_output(output.front(), "explicit Owned sync frame 0");
  } else {
    require_auto_output(output.front(), "Auto sync frame 0");
  }
  require_zero_copy_pipeline(graph, mode);
  evidence.verify(mode);

  std::cout << "[OK] " << mode << " Graph::run H.265 -> ResNet50 outputs=1\n";
}

void run_async(const std::string& model_path,
               const std::vector<simaai::neat::Sample>& access_units) {
  DmaHandoffEvidence evidence;
  simaai::neat::Graph graph = make_graph(access_units.front(), model_path, "async", kQueueDepth);
  simaai::neat::Run run =
      graph.build(simaai::neat::Sample{access_units.front()}, e2e_run_options());
  require_zero_copy_pipeline(graph, "async");

  std::exception_ptr producer_error;
  std::thread producer([&] {
    try {
      for (const simaai::neat::Sample& sample : access_units) {
        if (!run.push(simaai::neat::Sample{sample})) {
          throw std::runtime_error("async push failed");
        }
      }
      run.close_input();
    } catch (...) {
      producer_error = std::current_exception();
      try {
        run.close_input();
      } catch (...) {
      }
    }
  });

  std::exception_ptr consumer_error;
  std::size_t frame = 0;
  try {
    for (; frame < access_units.size(); ++frame) {
      const std::optional<simaai::neat::Sample> output = run.pull(kPullTimeoutMs);
      require(output.has_value(), "async pull timed out at frame " + std::to_string(frame));
      require_auto_output(*output, "Auto async frame " + std::to_string(frame));
    }
  } catch (...) {
    consumer_error = std::current_exception();
    evidence.report_async_failure(frame, access_units.size());
    try {
      run.stop();
    } catch (...) {
    }
  }

  producer.join();
  run.stop();
  if (producer_error) {
    std::rethrow_exception(producer_error);
  }
  if (consumer_error) {
    std::rethrow_exception(consumer_error);
  }

  evidence.verify("async");
  std::cout << "[OK] async Graph::build push/pull H.265 -> ResNet50 zero-copy outputs="
            << access_units.size() << "\n";
}

} // namespace

int main() {
  try {
    const std::string model_path = sima_test::resolve_resnet50_tar();
    require(!model_path.empty(),
            "ResNet50 model pack not found; set SIMA_MODEL_TAR or SIMA_RESNET50_TAR");

    const sima_codec_perf::CodecPerfConfig config{
        .scenario_id = "h265-decode-resnet50-e2e",
        .run_mode = "e2e",
        .decode_type = simaai::neat::SimaDecodeType::H265,
        .width = kWidth,
        .height = kHeight,
        .fps = kFps,
    };
    const std::vector<simaai::neat::Sample> access_units = sima_codec_perf::make_sample_sequence(
        sima_codec_perf::extract_h265_access_units(kFrames), config, kFrames);

    run_sync(model_path, access_units);
    run_async(model_path, access_units);
    run_sync(model_path, access_units, simaai::neat::OutputMemory::Owned);
    return 0;
  } catch (const std::exception& error) {
    return fail_test(error.what());
  }
}
