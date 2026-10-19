#include <variant>
#include <type_traits>
// Device-only production bridge + element acceptance. Run under the firmware
// guard.
#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <simaai/scheduler/testing/probe.h>
#include <simaai/scheduler/testing/reference.h>
#include <simaai/scheduler/client/tensor_buffer.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <gst/SimaPluginStaticManifestAbi.h>
#include <gstsimaaitensorbuffer_meta.h>
#include <iostream>
#include <mutex>
#include <source_location>
#include <simaai/scheduler/client/dma_memory.h>
#include <stdexcept>

#include <thread>
namespace rt = simaai::scheduler::client;
namespace gs = rt;
namespace test = simaai::scheduler::testing;
namespace tr = simaai::gst;
namespace core = simaai::neat::pipeline_internal::sima;
namespace lower = core::kernel_lowering;
namespace dma = simaai::scheduler::client::dma;
using namespace std::chrono_literals;
namespace {
using Bytes = std::vector<std::byte>;
void require(bool b, const char* message) {
  if (!b)
    throw std::runtime_error(message);
}
template <class T> T take(rt::Result<T> value) {
  if (auto* e = std::get_if<rt::Error>(&value))
    throw std::runtime_error(e->field + ": " + e->message);
  return std::get<T>(std::move(value));
}
void okay(rt::Status value) {
  if (value)
    throw std::runtime_error(value->field + ": " + value->message);
}
[[noreturn]] void uncertain(std::source_location where = std::source_location::current()) {
  std::cerr << "UNRETIRED visual trial at " << where.file_name() << ":" << where.line()
            << ": external guard must stop EV "
               "before cleanup\n";
  for (;;)
    std::this_thread::sleep_for(1s);
}
std::size_t bytes(const rt::TensorSpec& s) {
  std::size_t n = s.dtype == rt::DType::UInt8 ? 1 : 4;
  for (auto d : s.shape)
    n *= d;
  return n;
}
Bytes words(std::initializer_list<std::int32_t> values) {
  Bytes out(values.size() * 4);
  std::memcpy(out.data(), values.begin(), out.size());
  return out;
}
struct Fixture {
  const char* name;
  unsigned graph;
  std::vector<rt::TensorSpec> inputs, outputs;
  std::vector<Bytes> data;
  bool detect = false;
  int pyramid_levels = 0;
};
Fixture fixture(unsigned graph, bool detect = false) {
  // BRIEF's descriptor border excludes all candidates in a 32x32 image.
  // Give it an interior so the integration test checks real descriptor bytes.
  const std::int64_t extent = graph == 237 ? 48 : 32;
  const rt::TensorSpec image{std::vector<std::int64_t>{1, extent, extent, 1}, rt::DType::UInt8};
  Fixture f{graph == 235   ? "feature_histogram"
            : graph == 236 ? "grider_fast"
            : graph == 237 ? "track_descriptor"
                           : "track_klt",
            graph,
            {image},
            {},
            {Bytes(extent * extent)},
            detect};
  // Deterministic nondegenerate texture, not a zero-feature shortcut.
  std::uint32_t seed = 0x31415926;
  for (auto& v : f.data[0]) {
    seed = 1664525 * seed + 1013904223;
    v = std::byte(seed >> 24);
  }
  if (graph == 235)
    f.outputs = {{std::vector<std::int64_t>{1, 256}, rt::DType::Int32}};
  else if (graph == 236 || graph == 237) {
    f.outputs = {{std::vector<std::int64_t>{1, 49}, rt::DType::Int32}};
    if (graph == 237)
      f.outputs.push_back({std::vector<std::int64_t>{1, 16, 8}, rt::DType::Int32});
  } else {
    f.inputs.push_back(image);
    f.data.push_back(f.data[0]);
    f.inputs.push_back({std::vector<std::int64_t>{1, 4, 2}, rt::DType::Int32});
    f.data.push_back(words({8, 8, 12, 12, 20, 12, 20, 20}));
    f.outputs = {{std::vector<std::int64_t>{1, 4, 2}, rt::DType::Float32},
                 {std::vector<std::int64_t>{1, 4}, rt::DType::Int32}};
    if (detect)
      f.outputs.push_back({std::vector<std::int64_t>{1, 49}, rt::DType::Int32});
  }
  return f;
}
// Reference is the same qualified numerical kernel called through the public
// typed Kernels API, independently of the compatibility translator/routing.
// This proves integration equivalence, not an independent FAST/BRIEF math
// oracle.
std::vector<Bytes> direct(std::string_view name, const std::vector<rt::TensorSpec>& in,
                          const std::vector<rt::TensorSpec>& out, const std::vector<Bytes>& data,
                          test::VisualParameters parameters) {
  auto checked = take(test::reference_visual(name, rt::Target::EV74, in, out, parameters));
  std::vector<std::size_t> output_bytes;
  for (const auto& spec : out)
    output_bytes.push_back(bytes(spec));
  return test::execute_ev74(checked, data, output_bytes);
}
std::vector<Bytes> reference(const Fixture& f) {
  if (f.graph == 235) {
    std::vector<Bytes> expected{Bytes(1024)};
    std::array<std::int32_t, 256> counts{};
    for (auto value : f.data[0])
      ++counts[std::to_integer<unsigned>(value)];
    std::memcpy(expected[0].data(), counts.data(), 1024);
    return expected;
  }
  if (f.graph == 236)
    return direct("fast", f.inputs, f.outputs, f.data, test::VisualParameters{.threshold = 20});
  if (f.graph == 237)
    return direct("fastbrief", f.inputs, f.outputs, f.data,
                  test::VisualParameters{.threshold = 20});
  auto inputs = f.inputs;
  auto data = f.data;
  inputs.push_back({std::vector<std::int64_t>{1, 1}, rt::DType::Int32});
  data.push_back(words({4}));
  auto output = f.outputs;
  output.resize(2);
  auto expected =
      direct("tracker", inputs, output, data,
             test::VisualParameters{.win_half = 3, .max_iters = 12, .max_level = f.pyramid_levels});
  if (f.detect) {
    auto found = direct("fast", {f.inputs[1]}, {f.outputs[2]}, {f.data[1]},
                        test::VisualParameters{.threshold = 20});
    expected.push_back(std::move(found[0]));
  }
  return expected;
}
struct Placement {
  SimaPluginStageSpec stage{};
  std::vector<SimaPluginLogicalInput> logical;
  std::vector<SimaPluginInputBinding> bindings;
  std::vector<SimaPluginPhysicalBuffer> inputs, outputs;
  std::vector<std::size_t> input_offsets, output_offsets;
  explicit Placement(const Fixture& f, bool tight_outputs)
      : logical(f.inputs.size()), bindings(f.inputs.size()), inputs(f.inputs.size()),
        outputs(f.outputs.size()) {
    stage.frame_arena_size_bytes = 4096;
    stage.frame_arena_role = SIMA_PLUGIN_FRAME_ARENA_ALLOCATE;
    stage.frame_arena_storage_domain = SIMA_PLUGIN_FRAME_ARENA_STORAGE_CMA;
    stage.frame_arena_provenance = SIMA_PLUGIN_FRAME_ARENA_PROVENANCE_CORE_ALLOCATED;
    stage.frame_arena_required_device_access = SIMA_PLUGIN_FRAME_ARENA_ACCESS_EV74;
    std::size_t offset = 64;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      logical[i].logical_index = logical[i].backend_input_index = logical[i].physical_index = i;
      auto& binding = bindings[i];
      binding.local_logical_input_index = i;
      binding.sink_pad_index = 0;
      binding.required = TRUE;
      binding.src_stage_index = binding.src_logical_output_index =
          binding.src_physical_output_index = binding.src_output_slot = -1;
      auto& p = inputs[i];
      p.physical_index = i;
      p.size_bytes = bytes(f.inputs[i]);
      p.source_byte_offset = offset;
      p.address_source = SIMA_PLUGIN_PHYSICAL_ADDRESS_FRAME_ARENA_SPAN;
      p.required_alignment_bytes = 64;
      input_offsets.push_back(offset);
      offset += (p.size_bytes + 127) / 64 * 64;
    }
    offset = 128;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      auto& p = outputs[i];
      p.physical_index = i;
      p.size_bytes = bytes(f.outputs[i]);
      p.source_byte_offset = offset;
      p.required_alignment_bytes = tight_outputs ? 4 : 64;
      output_offsets.push_back(offset);
      offset += tight_outputs ? p.size_bytes : (p.size_bytes + 127) / 64 * 64;
    }
    require(offset < 4096, "test output arena overflow");
    stage.logical_inputs = logical.data();
    stage.logical_inputs_len = logical.size();
    stage.input_bindings = bindings.data();
    stage.input_bindings_len = bindings.size();
    stage.physical_inputs = inputs.data();
    stage.physical_inputs_len = inputs.size();
    stage.physical_outputs = outputs.data();
    stage.physical_outputs_len = outputs.size();
  }
};
gs::TensorStageRequest prepared(const Fixture& f) {
  gs::TensorStageRequest request;
  core::StageStaticSpec semantic;
  auto& c = semantic.processcvu;
  c.graph_name = f.name;
  c.graph_id = f.graph;
  c.resolved_exec_backend = "EV74";
  c.width = c.height = f.inputs[0].shape[1];
  c.batch_size = 1;
  c.num_in_tensor = f.inputs.size();
  c.threshold = c.fast_threshold = 20;
  c.max_features = 16;
  c.descriptor_words = 8;
  c.num_points = 4;
  c.win_half = 3;
  c.max_iters = 12;
  c.max_level = f.pyramid_levels;
  c.detect_new_features = f.detect;
  for (const auto& spec : f.inputs) {
    sima_ev_tensor_desc d{};
    d.dtype = spec.dtype == rt::DType::UInt8 ? SIMA_EV_DTYPE_INT8 : SIMA_EV_DTYPE_INT32;
    auto dims = spec.shape;
    d.shape.rank = dims.size();
    for (std::size_t i = 0; i < dims.size(); ++i)
      d.shape.sizes[i] = dims[i];
    c.input_tensors.emplace_back(d, spec.dtype == rt::DType::UInt8 ? "UINT8" : "INT32");
  }
  auto publication = std::make_shared<tr::TensorBufferPublishContract>();
  auto& pub = *publication;
  pub.stage_key = f.name;
  for (std::size_t i = 0; i < f.outputs.size(); ++i) {
    auto name = "result" + std::to_string(i);
    pub.physical_outputs.push_back({int(i), bytes(f.outputs[i]), name});
    tr::TensorBufferPublishLogicalOutput tensor;
    tensor.logical_index = 20 + i;
    tensor.physical_index = i;
    tensor.memory_index = i;
    tensor.backend_output_index = i;
    tensor.route_slot = i;
    tensor.logical_name = tensor.backend_name = tensor.segment_name = name;
    tensor.size_bytes = bytes(f.outputs[i]);
    tensor.dtype = f.outputs[i].dtype == rt::DType::Float32 ? SIMA_TENSOR_SET_DTYPE_FP32_V1
                                                            : SIMA_TENSOR_SET_DTYPE_INT32_V1;
    tensor.shape.assign(f.outputs[i].shape.begin(), f.outputs[i].shape.end());
    pub.logical_outputs.push_back(tensor);
    pub.output_order.push_back({int(i), tensor.logical_index, name, name});
  }
  auto recipe = take(lower::check_visual_stage(semantic));
  require(recipe->members.size() == (f.graph == 238 && f.detect ? 2U : 1U),
          "visual numerical member count changed");
  request.target = recipe->target;
  request.members = recipe->members;
  request.constant_data = recipe->constant_data;
  request.publication.contract = std::move(publication);
  request.publication.output_name = "result0";
  request.publication.input_caps = std::shared_ptr<GstCaps>(
      gst_caps_from_string("application/vnd.simaai.tensor"), gst_caps_unref);
  request.publication.output_caps =
      std::shared_ptr<GstCaps>(gst_caps_copy(request.publication.input_caps.get()), gst_caps_unref);
  return request;
}
struct Sink {
  std::mutex mutex;
  std::condition_variable changed;
  test::GstBufferOwner buffer;
  bool eos = false;
};
GstFlowReturn collect(GstPad* pad, GstObject*, GstBuffer* buffer) {
  auto& s = *static_cast<Sink*>(g_object_get_data(G_OBJECT(pad), "sink"));
  {
    std::lock_guard lock(s.mutex);
    if (s.buffer) {
      gst_buffer_unref(buffer);
      return GST_FLOW_ERROR;
    }
    s.buffer.reset(buffer);
  }
  s.changed.notify_all();
  return GST_FLOW_OK;
}
gboolean event(GstPad* pad, GstObject*, GstEvent* value) {
  auto& s = *static_cast<Sink*>(g_object_get_data(G_OBJECT(pad), "sink"));
  if (GST_EVENT_TYPE(value) == GST_EVENT_EOS) {
    {
      std::lock_guard lock(s.mutex);
      s.eos = true;
    }
    s.changed.notify_all();
  }
  gst_event_unref(value);
  return TRUE;
}
void run(const Fixture& f, bool tight_outputs = false) {
  std::cout << "START production visual plugin " << f.name << (f.detect ? " + detection" : "")
            << " pyramid=" << f.pyramid_levels << " tight_outputs=" << tight_outputs << std::endl;
  std::cout << "START visual " << f.name << " detect=" << f.detect << std::endl;
  const auto expected = reference(f);
  std::cout << "BRIDGE prepare " << f.name << std::endl;
  auto config = prepared(f);
  auto locations = std::make_shared<Placement>(f, tight_outputs);
  config.options.capacity = 2;
  config.placement = std::shared_ptr<const SimaPluginStageSpec>(locations, &locations->stage);
  dma::Error error;
  test::GstBufferOwner input(dma::allocateDmaBufBuffer(dma::HeapKind::Cma, 4096, {}, &error));
  require(bool(input), error.message().c_str());
  auto view =
      dma::DmaBufView::fromGstMemory(gst_buffer_peek_memory(input.get(), 0), 0, 4096, &error);
  auto mapped = view ? view->map(dma::CpuAccess::Write, &error) : std::nullopt;
  require(bool(mapped), error.message().c_str());
  Bytes initial(4096, std::byte{0xa5});
  for (std::size_t i = 0; i < f.data.size(); ++i)
    std::copy(f.data[i].begin(), f.data[i].end(), initial.begin() + locations->input_offsets[i]);
  std::memcpy(mapped->data(), initial.data(), initial.size());
  require(mapped->finish(&error), error.message().c_str());
  mapped.reset();
  auto* meta = gst_buffer_add_custom_meta(input.get(), "GstSimaMeta");
  require(meta, "metadata");
  gst_structure_set(gst_custom_meta_get_structure(meta), "frame-id", G_TYPE_INT64, gint64(41),
                    "timestamp", G_TYPE_UINT64, guint64(812), "input-seq", G_TYPE_INT64, gint64(9),
                    "stream-id", G_TYPE_STRING, "visual", nullptr);
  GST_BUFFER_PTS(input.get()) = 333;
  // Retain the source allocation for guard verification, not the carrier loan.
  Sink sink;
  auto* element = gst_element_factory_make("neatkernelrunner", nullptr);
  require(element, "factory");
  auto* bus = gst_bus_new();
  gst_element_set_bus(element, bus);
  auto diagnose = [&](std::source_location where = std::source_location::current()) {
    GstState state = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
    gst_element_get_state(element, &state, &pending, 0);
    std::cerr << "VISUAL PHASE FAILURE " << f.name << " state=" << gst_element_state_get_name(state)
              << " pending=" << gst_element_state_get_name(pending) << std::endl;
    while (auto* message = gst_bus_pop(bus)) {
      if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError* error = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &error, &debug);
        std::cerr << "GST ERROR " << (error ? error->message : "unknown") << " ["
                  << (debug ? debug : "") << "]" << std::endl;
        if (error)
          g_error_free(error);
        g_free(debug);
      }
      gst_message_unref(message);
    }
    uncertain(where);
  };
  auto *source = gst_pad_new("producer", GST_PAD_SRC),
       *collector = gst_pad_new("collector", GST_PAD_SINK);
  auto *in = gst_element_request_pad_simple(element, "sink_%u"),
       *out = gst_element_get_static_pad(element, "src");
  require(in && out && source && collector, "pads");
  auto* pipeline = gst_pipeline_new("visual_test");
  require(pipeline, "pipeline");
  gst_object_ref(element);
  require(gst_bin_add(GST_BIN(pipeline), element), "add numerical element");
  const std::array attachments{gs::StageAttachment{element, std::move(config)}};
  okay(gs::prepare_and_attach_tensor_stages(pipeline, attachments));
  g_object_set_data(G_OBJECT(collector), "sink", &sink);
  gst_pad_set_chain_function(collector, collect);
  gst_pad_set_event_function(collector, event);
  gst_pad_set_active(source, TRUE);
  gst_pad_set_active(collector, TRUE);
  require(gst_pad_link(source, in) == GST_PAD_LINK_OK &&
              gst_pad_link(out, collector) == GST_PAD_LINK_OK,
          "links");
  require(gst_element_set_state(element, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE, "start");
  require(gst_pad_push_event(source, gst_event_new_stream_start("visual")), "stream");
  require(gst_pad_push_event(
              source, gst_event_new_caps(attachments.front().request.publication.input_caps.get())),
          "caps");
  GstSegment segment;
  gst_segment_init(&segment, GST_FORMAT_TIME);
  require(gst_pad_push_event(source, gst_event_new_segment(&segment)), "segment");
  std::cout << "BRIDGE push " << f.name << std::endl;
  if (gst_pad_push(source, input.release()) != GST_FLOW_OK)
    diagnose();
  {
    std::unique_lock lock(sink.mutex);
    if (!sink.changed.wait_for(lock, 10s, [&] { return bool(sink.buffer); }))
      diagnose();
  }
  if (!gst_pad_push_event(source, gst_event_new_eos()))
    diagnose();
  {
    std::unique_lock lock(sink.mutex);
    if (!sink.changed.wait_for(lock, 10s, [&] { return sink.eos; }))
      diagnose();
  }
  if (gst_element_set_state(element, GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE)
    diagnose();
  auto result =
      dma::DmaBufView::fromGstMemory(gst_buffer_peek_memory(sink.buffer.get(), 0), 0, 4096, &error);
  auto result_map = result ? result->map(dma::CpuAccess::Read, &error) : std::nullopt;
  require(bool(result_map), error.message().c_str());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    auto* actual = static_cast<const std::byte*>(result_map->data()) + locations->output_offsets[i];
    std::size_t valid_bytes = expected[i].size();
    const bool features = (i == 0 && (f.graph == 236 || f.graph == 237)) || (f.detect && i == 2);
    if (features) {
      std::int32_t count = 0;
      std::memcpy(&count, expected[i].data(), 4);
      require(count > 0 && count <= 16, "FAST positive feature coverage");
      valid_bytes = 4 + std::size_t(count) * 12;
    } else if (f.graph == 237 && i == 1) {
      std::int32_t count = 0;
      std::memcpy(&count, expected[0].data(), 4);
      valid_bytes = std::size_t(count) * 32;
    }
    // Feature capacity suffixes are intentionally unspecified by the kernel.
    require(std::memcmp(actual, expected[i].data(), valid_bytes) == 0,
            "production visual output differs from independent integration "
            "reference");
  }
  require(result_map->finish(&error), error.message().c_str());
  result_map.reset();
  mapped = view->map(dma::CpuAccess::Read, &error);
  require(bool(mapped), error.message().c_str());
  require(std::memcmp(mapped->data(), initial.data(), initial.size()) == 0,
          "visual source/frame-arena guards modified");
  require(mapped->finish(&error), error.message().c_str());
  mapped.reset();
  rt::TensorBufferView tensors;
  std::string issue;
  require(rt::tensor_buffer_create_view(sink.buffer.get(), nullptr, &tensors, &issue),
          issue.c_str());
  require(tensors.tensors().size() == f.outputs.size(), "publication output count");
  for (std::size_t i = 0; i < tensors.tensors().size(); ++i) {
    const auto& t = tensors.tensors()[i];
    require(t.logical_index == int(20 + i) &&
                t.byte_offset == std::int64_t(locations->output_offsets[i]),
            "publication logical identity/physical offset");
  }
  auto* published =
      gst_custom_meta_get_structure(gst_buffer_get_custom_meta(sink.buffer.get(), "GstSimaMeta"));
  gint64 id = -1, seq = -1;
  require(gst_structure_get_int64(published, "frame-id", &id) && id == 41 &&
              gst_structure_get_int64(published, "orig-input-seq", &seq) && seq == 9 &&
              GST_BUFFER_PTS(sink.buffer.get()) == 333,
          "publication correlation/timing");
  gst_pad_unlink(source, in);
  gst_pad_unlink(out, collector);
  gst_pad_set_active(source, FALSE);
  gst_pad_set_active(collector, FALSE);
  gst_element_release_request_pad(element, in);
  for (auto* pad : {source, in, out, collector})
    gst_object_unref(pad);
  gst_element_set_bus(element, nullptr);
  gst_object_unref(bus);
  gst_object_unref(element);
  gst_object_unref(pipeline);
  std::cout << "PASS production visual plugin " << f.name << (f.detect ? " + detection" : "")
            << ": standard DMA-BUF, output bytes, physical offsets, input "
               "guards, correlation and EOS\n";
}
} // namespace
int main(int argc, char** argv) {
  std::cout << std::unitbuf;
  gst_init(&argc, &argv);
  try {
    require(argc == 1, "visual device test takes no arguments; use firmware guard");
    require(tr::tensor_buffer_register_meta(), "metadata registration");
    require(test::register_element(), "register element");
    for (unsigned graph : {235u, 236u, 237u, 238u})
      run(fixture(graph));
    run(fixture(238, true));
    // Adjacent legal outputs share cache lines: the provider must group the
    // invocation rather than self-conflict or silently repack its arena.
    run(fixture(237), true);
    run(fixture(238), true);
    auto pyramid = fixture(238);
    pyramid.pyramid_levels = 1;
    run(pyramid, true);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
