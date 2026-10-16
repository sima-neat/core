/**
 * Real Core model -> BoxDecode publication -> public SimaRender fragment.
 *
 * This deliberately tests the supported inter-component seam: public Graph/Run
 * produces the boxes and a two-input GStreamer harness joins the retained source
 * image with that exact publication. It does NOT claim that Core's monolithic
 * graph compiler supports two-input SimaRender topology. No box repacking,
 * coordinate conversion, synthetic detections, or provider drawing oracle is used.
 *
 * Usage: <test> <fixture-root> [<model.tar.gz> <people.jpg>]
 * Defaults are fixture-root/tmp/{yolo_v8s.tar.gz,yolov8s_people.jpg}. Prepare
 * fixtures before execution; missing assets are failures, never downloads/skips.
 */
#include "gst/GstInit.h"
#include "model/Model.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/SimaRender.h"
#include "pipeline/Graph.h"
#include "pipeline/TensorAdapters.h"

#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "support/core_render_retirement.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
namespace neat = simaai::neat;
namespace fs = std::filesystem;
constexpr float threshold = 0.49f;
constexpr int top_k = 100;
constexpr unsigned frame_count = 3;
constexpr GstClockTime duration = GST_SECOND / 30;
constexpr const char* stream_id = "core-render-camera";
std::atomic<unsigned> critical_count{0};

// Identical fail-containment rule to Scheduler's source-owned PipelineOwner:
// do not signal, abort, or unwind a potentially active accelerator owner. The
// guarded supervisor records the failure and owns any separate recovery action.
[[noreturn]] void retain_unknown(const char* diagnostic) noexcept {
  std::fprintf(stderr, "CORE_RENDER_RETIREMENT_UNKNOWN: %s; retaining owners and frame loans\n",
               diagnostic ? diagnostic : "non-standard exception");
  std::fflush(stderr);
  for (;;)
    ::sleep(3600);
}

void record_glib_log(const gchar* domain, GLogLevelFlags level, const gchar* message, gpointer) {
  if (level & (G_LOG_LEVEL_CRITICAL | G_LOG_LEVEL_ERROR))
    critical_count.fetch_add(1, std::memory_order_relaxed);
  g_log_default_handler(domain, level, message, nullptr);
}

// Optional TEST-only rendezvous shared with the established B2038 guarded EV
// qualification procedure. Load every selected plugin before any Model, device
// discovery, or DMA allocation. Only the source-pinned supervisor may send G
// after verifying actual mappings; EOF/refusal exits before device work.
void provider_capture_barrier() {
  const char* flag = std::getenv("SIMA_EV_CAPTURE_BARRIER");
  if (!flag)
    return;
  require(std::strcmp(flag, "1") == 0, "invalid Core fixture capture flag");
  for (const char* name :
       {"neatkernelrunner", "neatprocessmla", "neatrender", "appsrc", "appsink", "queue"}) {
    auto* factory = gst_element_factory_find(name);
    require(factory != nullptr, std::string("Core capture needs factory: ") + name);
    auto* feature = gst_plugin_feature_load(GST_PLUGIN_FEATURE(factory));
    gst_object_unref(factory);
    require(feature != nullptr, std::string("Core capture plugin load failed: ") + name);
    gst_object_unref(feature);
  }
  std::cout << "EV_PROVIDER_CAPTURE_READY: pid=" << ::getpid() << std::endl;
  require(std::getchar() == 'G', "Core provider capture not admitted before device work");
  std::cout << "EV_PROVIDER_CAPTURE_RELEASED: exact mapped providers admitted" << std::endl;
}

void combined_closure_barrier() {
  if (std::getenv("SIMA_EV_CAPTURE_BARRIER")) {
    // The same source-owned capture channel stays open until the supervisor
    // has verified positive per-stream stop/join traces and BOTH drivers.
    // Unknown private teardown must retain this process, not become EXIT0.
    std::cout << "CORE_RENDER_CLOSURE_READY: pid=" << ::getpid() << std::endl;
    if (std::getchar() != 'C')
      retain_unknown("combined source/driver closure not admitted");
    std::cout << "CORE_RENDER_CLOSURE_RELEASED: positive source and dual-driver closure admitted\n"
              << std::flush;
  }
}

struct BufferUnref {
  void operator()(GstBuffer* p) const {
    if (p)
      gst_buffer_unref(p);
  }
};
struct SampleUnref {
  void operator()(GstSample* p) const {
    if (p)
      gst_sample_unref(p);
  }
};
using Buffer = std::unique_ptr<GstBuffer, BufferUnref>;
using GstSampleOwner = std::unique_ptr<GstSample, SampleUnref>;

struct ConfigFiles {
  fs::path directory;
  ConfigFiles(int width, int height) {
    auto pattern = (fs::temp_directory_path() / "core-render-XXXXXX").string();
    require(::mkdtemp(pattern.data()) != nullptr, "create renderer config directory");
    directory = pattern;
    try {
      std::ofstream labels(directory / "labels.txt");
      for (int i = 0; i < 80; ++i)
        labels << "class-" << i << '\n';
      require(bool(labels), "write renderer labels");
      labels.close();
      const nlohmann::json config{{"original_width", width},
                                  {"original_height", height},
                                  {"image_format", "YUV420"},
                                  {"render_type", "bbox"},
                                  {"box_coordinate_type", "int32"},
                                  {"model_width", 0},
                                  {"model_height", 0},
                                  {"threshold", threshold},
                                  {"color_box", {255, 255, 255}},
                                  {"color_text", {255, 255, 255}},
                                  {"color_box_special", {0, 0, 255}},
                                  {"class_special", -1},
                                  {"label_file", (directory / "labels.txt").string()},
                                  {"debug", false}};
      std::ofstream output(directory / "render.json");
      output << config.dump();
      require(bool(output), "write renderer config");
    } catch (...) {
      std::error_code ignored;
      fs::remove_all(directory, ignored);
      throw;
    }
  }
  ~ConfigFiles() {
    std::error_code ignored;
    fs::remove_all(directory, ignored);
  }
};

std::string bus_error(GstMessage* message) {
  GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(message, &error, &debug);
  const std::string result =
      std::string(error ? error->message : "GStreamer error") + ": " + (debug ? debug : "");
  g_clear_error(&error);
  g_free(debug);
  return result;
}

class Renderer {
public:
  Renderer(const std::string& config, int width, int height) {
    neat::SimaRenderOptions options;
    options.config_path = config;
    options.sima_allocator_type = 1;
    node_ = neat::nodes::SimaRender(options);
    const auto element = node_->element_names(0).at(0);
    const std::string launch =
        "appsrc name=image format=time caps=video/x-raw,format=I420,width=" +
        std::to_string(width) + ",height=" + std::to_string(height) + ",framerate=30/1 ! queue ! " +
        element + ".sink_0 " +
        "appsrc name=boxes format=time caps=application/vnd.simaai.tensor ! queue ! " + element +
        ".sink_1 " + node_->backend_fragment(0) +
        " ! appsink name=output sync=false async=false enable-last-sample=false";
    GError* error = nullptr;
    pipeline_ = gst_parse_launch(launch.c_str(), &error);
    if (error) {
      const std::string diagnostic = error->message;
      g_clear_error(&error);
      if (pipeline_)
        gst_object_unref(pipeline_);
      pipeline_ = nullptr;
      throw std::runtime_error(diagnostic);
    }
    require(pipeline_ != nullptr, "parse public SimaRender fragment");
    image_ = gst_bin_get_by_name(GST_BIN(pipeline_), "image");
    boxes_ = gst_bin_get_by_name(GST_BIN(pipeline_), "boxes");
    sink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "output");
  }
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  ~Renderer() {
    if (!closed_) {
      try {
        close();
      } catch (...) {
        retain_unknown("renderer checked NULL teardown failed");
      }
    }
    for (auto* element : {image_, boxes_, sink_, pipeline_})
      if (element)
        gst_object_unref(element);
  }
  void start() {
    require(image_ && boxes_ && sink_, "renderer harness elements");
    require(gst_element_set_state(pipeline_, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "start public SimaRender fragment");
  }
  GstSampleOwner submit(GstBuffer* image, GstBuffer* boxes) {
    // Retain the exact owners. appsrc consumes only these additional references;
    // neither original buffer nor the Core-produced publication is modified here.
    require(gst_app_src_push_buffer(GST_APP_SRC(image_), gst_buffer_ref(image)) == GST_FLOW_OK,
            "enqueue retained source image");
    require(gst_app_src_push_buffer(GST_APP_SRC(boxes_), gst_buffer_ref(boxes)) == GST_FLOW_OK,
            "enqueue exact Core publication");
    GstSampleOwner result(gst_app_sink_try_pull_sample(GST_APP_SINK(sink_), 5 * GST_SECOND));
    if (!result) {
      auto* bus = gst_element_get_bus(pipeline_);
      auto* message = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
      gst_object_unref(bus);
      const auto diagnostic = message ? bus_error(message) : "renderer output timeout";
      if (message)
        gst_message_unref(message);
      throw std::runtime_error(diagnostic);
    }
    return result;
  }
  void finish() {
    require(gst_app_src_end_of_stream(GST_APP_SRC(image_)) == GST_FLOW_OK &&
                gst_app_src_end_of_stream(GST_APP_SRC(boxes_)) == GST_FLOW_OK,
            "renderer input EOS");
    auto* bus = gst_element_get_bus(pipeline_);
    auto* message = gst_bus_timed_pop_filtered(
        bus, 5 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    gst_object_unref(bus);
    require(message != nullptr, "renderer EOS timeout");
    const bool eos = GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS;
    const auto diagnostic = eos ? "" : bus_error(message);
    gst_message_unref(message);
    require(eos, "renderer EOS error: " + diagnostic);
    GstSampleOwner extra(gst_app_sink_try_pull_sample(GST_APP_SINK(sink_), 0));
    require(!extra, "renderer produced unexpected extra frame");
    std::cout << "CORE_RENDER_RENDERER_EOS\n" << std::flush;
    close();
  }

private:
  void close() {
    if (closed_)
      return;
    require(gst_element_set_state(pipeline_, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE,
            "renderer NULL request");
    GstState state = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
    require(gst_element_get_state(pipeline_, &state, &pending, 5 * GST_SECOND) ==
                    GST_STATE_CHANGE_SUCCESS &&
                state == GST_STATE_NULL && pending == GST_STATE_VOID_PENDING,
            "renderer NULL transition incomplete");
    closed_ = true;
    std::cout << "CORE_RENDER_RENDERER_NULL\n" << std::flush;
  }
  std::shared_ptr<neat::Node> node_;
  GstElement* pipeline_ = nullptr;
  GstElement* image_ = nullptr;
  GstElement* boxes_ = nullptr;
  GstElement* sink_ = nullptr;
  bool closed_ = false;
};

std::vector<uint8_t> bytes(GstBuffer* buffer) {
  require(buffer != nullptr, "missing GstBuffer");
  std::vector<uint8_t> result(gst_buffer_get_size(buffer));
  require(gst_buffer_extract(buffer, 0, result.data(), result.size()) == result.size(),
          "read complete observation buffer");
  return result;
}

void stamp_image(GstBuffer* image, int width, int height, const neat::Sample& input) {
  auto* meta = gst_buffer_add_custom_meta(image, "GstSimaMeta");
  require(meta != nullptr, "source image frame metadata");
  gst_structure_set(gst_custom_meta_get_structure(meta), "frame-id", G_TYPE_INT64, input.frame_id,
                    "stream-id", G_TYPE_STRING, input.stream_id.c_str(), "input-seq", G_TYPE_INT64,
                    input.input_seq, "buffer-name", G_TYPE_STRING, "original-image", nullptr);
  require(gst_buffer_add_video_meta(image, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_I420, width,
                                    height) != nullptr,
          "source image plane metadata");
  GST_BUFFER_PTS(image) = input.pts_ns;
  GST_BUFFER_DURATION(image) = input.duration_ns;
  GST_BUFFER_OFFSET(image) = input.input_seq;
}

void check_association(GstBuffer* buffer, const neat::Sample& input) {
  require(GST_BUFFER_PTS(buffer) == static_cast<GstClockTime>(input.pts_ns),
          "frame timestamp mismatch");
  auto* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  const auto* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  gint64 frame = -1, sequence = -1;
  const char* stream = structure ? gst_structure_get_string(structure, "stream-id") : nullptr;
  require(structure && gst_structure_get_int64(structure, "frame-id", &frame) &&
              frame == input.frame_id &&
              gst_structure_get_int64(structure, "input-seq", &sequence) &&
              sequence == input.input_seq && stream && std::string(stream) == input.stream_id,
          "frame/stream/sequence association mismatch");
}

const neat::Tensor& publication_tensor(const neat::Sample& output) {
  const neat::Tensor* tensor = nullptr;
  if (output.kind == neat::SampleKind::Tensor && output.tensor)
    tensor = &*output.tensor;
  else if (neat::sample_has_tensor_list(output) && output.tensors.size() == 1)
    tensor = &output.tensors.front();
  require(tensor && tensor->storage && tensor->storage->kind == neat::StorageKind::GstSample &&
              tensor->storage->holder && tensor->byte_offset == 0,
          "Core must return one zero-copy, zero-offset GstSample-backed BoxDecode publication");
  return *tensor;
}

std::vector<uint8_t> storage_bytes(const neat::Tensor& tensor, size_t expected) {
  // Use Core's public mapping authority for DMA-BUF CPU epochs. A raw Gst
  // buffer_extract/map on a standard DMA-BUF does not itself provide DMA sync.
  require(tensor.storage != nullptr, "missing observation storage");
  auto mapping = tensor.storage->map(neat::MapMode::Read);
  require(mapping.data && mapping.size_bytes >= expected, "physical observation mapping extent");
  const auto* begin = static_cast<const uint8_t*>(mapping.data);
  return {begin, begin + expected};
}

// Independent sparse pixel oracle: white rectangle straight edges have Y=255.
// Select real predicted box edges (excluding labels/corners and originally white
// pixels). Do not call the renderer/provider or reproduce its rasterizer.
std::vector<size_t> edge_probes(const std::vector<objdet::Box>& boxes,
                                const std::vector<uint8_t>& original, int width, int height) {
  std::vector<size_t> probes;
  for (const auto& box : boxes) {
    if (box.score < threshold)
      continue;
    const int x = static_cast<int>(box.x1), y = static_cast<int>(box.y1);
    const int right = static_cast<int>(box.x2), bottom = static_cast<int>(box.y2);
    require(right - x > 8 && bottom - y > 8, "box too small for independent edge probes");
    const std::array<std::array<int, 2>, 3> points{std::array<int, 2>{x, (y + bottom) / 2},
                                                   std::array<int, 2>{right, (y + bottom) / 2},
                                                   std::array<int, 2>{(x + right) / 2, bottom}};
    const auto before = probes.size();
    for (const auto& point : points) {
      if (point[0] < 0 || point[0] >= width || point[1] < 0 || point[1] >= height)
        continue;
      const size_t offset = static_cast<size_t>(point[1]) * width + point[0];
      if (original[offset] != 255)
        probes.push_back(offset);
    }
    require(probes.size() > before, "no independent non-white edge probe for detection");
  }
  require(probes.size() >= 3, "insufficient real detections for pixel oracle");
  return probes;
}

bool matches_pixels(const std::vector<uint8_t>& observed, const std::vector<uint8_t>& original,
                    const std::vector<size_t>& probes, int width, int height) {
  if (observed.size() != original.size() || probes.empty())
    return false;
  // Outside every expected person's rectangle and label region.
  const size_t corner = static_cast<size_t>(height - 1) * width;
  if (observed[0] != original[0] || observed[corner] != original[corner])
    return false;
  return std::all_of(probes.begin(), probes.end(),
                     [&](size_t offset) { return observed.at(offset) == 255; });
}

void check_layout(GstSample* sample, int width, int height) {
  GstVideoInfo info;
  gst_video_info_init(&info);
  auto* caps = gst_sample_get_caps(sample);
  require(caps && gst_video_info_from_caps(&info, caps) &&
              GST_VIDEO_INFO_FORMAT(&info) == GST_VIDEO_FORMAT_I420 &&
              GST_VIDEO_INFO_WIDTH(&info) == width && GST_VIDEO_INFO_HEIGHT(&info) == height &&
              GST_VIDEO_INFO_FPS_N(&info) == 30 && GST_VIDEO_INFO_FPS_D(&info) == 1,
          "rendered caps must retain original I420 geometry/rate");
  auto* buffer = gst_sample_get_buffer(sample);
  const auto expected = static_cast<size_t>(width) * height * 3 / 2;
  require(gst_buffer_get_size(buffer) == expected && GST_VIDEO_INFO_SIZE(&info) == expected,
          "rendered image storage extent");
  const auto* meta = gst_buffer_get_video_meta(buffer);
  require(!meta || (meta->format == GST_VIDEO_FORMAT_I420 && meta->width == unsigned(width) &&
                    meta->height == unsigned(height) && meta->n_planes == 3),
          "rendered video metadata geometry");
  for (unsigned p = 0; p < 3; ++p) {
    const size_t offset = p == 0 ? 0 : static_cast<size_t>(width) * height * (p == 1 ? 4 : 5) / 4;
    const int stride = p == 0 ? width : width / 2;
    require((meta ? meta->offset[p] : info.offset[p]) == offset &&
                (meta ? meta->stride[p] : info.stride[p]) == stride,
            "rendered video plane layout");
  }
}

void qualify(const fs::path& model_path, const fs::path& image_path) {
  require(fs::is_regular_file(model_path),
          "required local model archive missing: " + model_path.string());
  require(fs::is_regular_file(image_path), "required local image missing: " + image_path.string());
  cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  require(!bgr.empty() && bgr.cols == 1280 && bgr.rows == 720,
          "independent people-image oracle requires the canonical 1280x720 fixture");
  cv::Mat i420;
  // Explicit cold test-input preparation, not an implicit runtime staging copy.
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
  require(i420.isContinuous(), "packed I420 fixture");
  const std::vector<uint8_t> original(i420.data, i420.data + i420.total());
  ConfigFiles files(bgr.cols, bgr.rows);
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.normalize.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::BGR;
  options.decode_type = neat::BoxDecodeType::YoloV8;
  options.score_threshold = threshold;
  options.nms_iou_threshold = 0.5f;
  options.top_k = top_k;
  options.upstream_name = "decoder";
  // These owners are outside the guarded try: failure containment runs before
  // they unwind, including the current borrowed producer/renderer publications.
  std::unique_ptr<neat::Model> model_owner;
  neat::Graph graph;
  neat::Sample seed, input, outputs, extra;
  neat::Run run;
  std::unique_ptr<Renderer> renderer_owner;
  Buffer image;
  GstSampleOwner rendered;
  core_render_test::RunRetirementObservation retirement;
  try {
    model_owner = std::make_unique<neat::Model>(model_path.string(), options);
    auto& model = *model_owner;
    graph.add(neat::nodes::Input());
    graph.add(neat::nodes::groups::Preprocess(model));
    graph.add(neat::nodes::groups::Infer(model));
    graph.add(
        neat::nodes::SimaBoxDecode(model, neat::BoxDecodeType::YoloV8, threshold, 0.5f, top_k));
    graph.add(neat::nodes::Output());
    seed =
        neat::Sample::from_image(bgr, neat::ImageSpec::PixelFormat::BGR, neat::TensorMemory::EV74);
    neat::RunOptions run_options;
    run_options.output_memory = neat::OutputMemory::ZeroCopy;
    run_options.startup_preflight = false;
    renderer_owner =
        std::make_unique<Renderer>((files.directory / "render.json").string(), bgr.cols, bgr.rows);
    auto& renderer = *renderer_owner;
    run = graph.build(seed, run_options);
    renderer.start();
    for (unsigned index = 0; index < frame_count; ++index) {
      input = seed;
      input.frame_id = 1000 + index;
      input.stream_id = stream_id;
      input.input_seq = input.orig_input_seq = 2000 + index;
      input.pts_ns = GST_SECOND + index * duration;
      input.duration_ns = duration;
      image.reset(gst_buffer_new_allocate(nullptr, original.size(), nullptr));
      require(image && gst_buffer_fill(image.get(), 0, original.data(), original.size()) ==
                           original.size(),
              "prepare retained source image");
      stamp_image(image.get(), bgr.cols, bgr.rows, input);
      require(run.push(input), "Core input enqueue");
      neat::PullError error;
      const auto status = run.pull(30000, outputs, &error);
      require(status == neat::PullStatus::Ok, "Core model/BoxDecode pull failed: " + error.message);
      require(outputs.size() == 1, "Core must publish exactly one BoxDecode logical output");
      const auto& output = outputs.front();
      require(output.frame_id == input.frame_id && output.stream_id == input.stream_id &&
                  output.input_seq == input.input_seq && output.pts_ns == input.pts_ns,
              "Core lost model/BoxDecode frame association");
      std::vector<uint8_t> payload;
      std::string diagnostic;
      require(objdet::extract_bbox_payload(output, payload, diagnostic), diagnostic);
      const auto boxes = objdet::parse_boxes_strict(payload, bgr.cols, bgr.rows, top_k, false);
      const auto expected = objdet::expected_people_boxes();
      const auto matched = objdet::match_expected_boxes(boxes, expected, threshold, 0.30f);
      require(matched.ok,
              "real detections differ from independent expected boxes: " + matched.note);
      auto wrong_expected = expected;
      wrong_expected.front().class_id = 999;
      require(!objdet::match_expected_boxes(boxes, wrong_expected, threshold, 0.30f).ok,
              "wrong detection oracle escaped negative control");
      const auto& tensor = publication_tensor(output);
      auto* publication =
          gst_sample_get_buffer(static_cast<GstSample*>(tensor.storage->holder.get()));
      require(publication != nullptr && gst_buffer_n_memory(publication) == 1,
              "Core publication must expose one physical memory view");
      // Check the actual Gst publication, not only Core's logical Sample wrapper.
      check_association(publication, input);
      const auto physical_before = storage_bytes(tensor, gst_buffer_get_size(publication));
      // Physical carrier padding may legitimately exceed the logical payload. The
      // published boxes must start at offset zero; pass the original carrier,
      // including its padding/metadata, rather than manufacturing a compact one.
      require(physical_before.size() >= payload.size() &&
                  std::equal(payload.begin(), payload.end(), physical_before.begin()),
              "Core logical boxes differ from their physical publication; do not repack silently");
      const auto probes = edge_probes(boxes, original, bgr.cols, bgr.rows);
      rendered = renderer.submit(image.get(), publication);
      check_association(gst_sample_get_buffer(rendered.get()), input);
      check_layout(rendered.get(), bgr.cols, bgr.rows);
      const auto observed =
          storage_bytes(neat::from_gst_sample(rendered.get()),
                        gst_buffer_get_size(gst_sample_get_buffer(rendered.get())));
      require(matches_pixels(observed, original, probes, bgr.cols, bgr.rows),
              "renderer pixels do not match actual detected box edges");
      require(!matches_pixels(original, original, probes, bgr.cols, bgr.rows),
              "unchanged image escaped no-render negative control");
      auto corrupt = observed;
      corrupt.at(probes.front()) = original.at(probes.front());
      require(!matches_pixels(corrupt, original, probes, bgr.cols, bgr.rows),
              "corrupted edge escaped pixel negative control");
      require(bytes(image.get()) == original, "renderer mutated the retained original image");
      require(storage_bytes(tensor, gst_buffer_get_size(publication)) == physical_before,
              "renderer mutated Core detection publication");
      std::cout << "CORE_RENDER_FRAME index=" << index << " boxes=" << boxes.size()
                << " pixel_probes=" << probes.size() << " PASS\n"
                << std::flush;
      // Release only after the actual downstream output and independent checks;
      // on any exception the outer catch retains these exact owners instead.
      rendered.reset();
      outputs = neat::Sample{};
      image.reset();
      input = neat::Sample{};
    }
    retirement.capture(run);
    run.close_input();
    neat::PullError error;
    const auto status = run.pull(30000, extra, &error);
    require(status == neat::PullStatus::Closed,
            "Core EOS was not observed or an extra model output appeared: " + error.message);
    std::cout << "CORE_RENDER_MODEL_EOS\n" << std::flush;
    renderer.finish();
    retirement.stop_and_close(run);
    require(critical_count.load(std::memory_order_relaxed) == 0,
            "GStreamer/GLib emitted critical diagnostics during owned qualification");
    std::cout << "CORE_RENDER_CORE_CLOSED: joined outer workers and checked native streams owners="
              << retirement.owner_count() << " total_streams=" << retirement.stream_count()
              << " driver_streams=" << retirement.driver_stream_count()
              << "\nCORE_RENDER_MODEL_STOPPED\n"
              << std::flush;
    combined_closure_barrier();
    retirement.release();
  } catch (const std::exception& error) {
    retain_unknown(error.what());
  } catch (...) {
    retain_unknown("non-standard native-scope failure");
  }
}
} // namespace

int main(int argc, char** argv) {
  bool native_scope_started = false;
  try {
    require(argc == 2 || argc == 4, "usage: test <fixture-root> [<model.tar.gz> <people.jpg>]");
    core_render_test::require_synchronous_teardown_environment();
    g_log_set_default_handler(record_glib_log, nullptr);
    // Core owns plugin-path selection and registration. Initialize before the
    // provider rendezvous, while no Model, device or DMA owner exists yet.
    neat::gst_init_once();
    provider_capture_barrier();
    const fs::path root(argv[1]);
    const auto model = argc == 4 ? fs::path(argv[2]) : root / "tmp/yolo_v8s.tar.gz";
    const auto image = argc == 4 ? fs::path(argv[3]) : root / "tmp/yolov8s_people.jpg";
    native_scope_started = true;
    qualify(model, image);
    require(critical_count.load(std::memory_order_relaxed) == 0,
            "GStreamer/GLib emitted critical diagnostics during qualification");
    std::cout << "CORE_RENDER_INTEGRATION PASS frames=" << frame_count
              << " scope=public-Core-numerics-to-public-renderer-seam\n"
              << std::flush;
    return 0;
  } catch (const std::exception& error) {
    if (native_scope_started)
      retain_unknown(error.what());
    std::cerr << "CORE_RENDER_INTEGRATION FAIL: " << error.what() << '\n';
    return 1;
  } catch (...) {
    if (native_scope_started)
      retain_unknown("non-standard post-device failure");
    std::fputs("CORE_RENDER_INTEGRATION FAIL: non-standard pre-device failure\n", stderr);
    return 1;
  }
}
