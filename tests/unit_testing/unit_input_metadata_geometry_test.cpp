#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/NeatError.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/runtime/RunInternal.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"
#include "test_main.h"
#include "test_utils.h"
#include <opencv2/core.hpp>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <new>

namespace {
std::atomic<bool> count_allocations{false};
std::atomic<unsigned> allocations{0};
} // namespace
void* operator new(std::size_t size) {
  if (count_allocations.load(std::memory_order_relaxed))
    ++allocations;
  if (void* p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
  return ::operator new(size);
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
  std::free(p);
}

namespace {
using namespace simaai::neat;
template <class Fn> void rejects(Fn fn, const char* code) {
  try {
    fn();
  } catch (const NeatError& e) {
    require(e.report().error_code == code, "wrong structured rejection code");
    require(!e.report().repro_note.empty(), "rejection must carry diagnostic");
    return;
  }
  throw std::runtime_error("invalid input metadata was accepted");
}
void mat_conversion_contract() {
  for (const auto format : {FormatTag::Auto, FormatTag::RGB, FormatTag::BGR, FormatTag::GRAY8,
                            FormatTag::FP32, FormatTag::EVXX_FLOAT32}) {
    const bool tensor = format == FormatTag::FP32 || format == FormatTag::EVXX_FLOAT32;
    const int type = tensor ? CV_32FC3 : format == FormatTag::GRAY8 ? CV_8UC1 : CV_8UC3;
    cv::Mat parent(5, 7, type, cv::Scalar::all(7));
    const cv::Mat roi = parent(cv::Rect(1, 1, 4, 3));
    InputOptions options;
    options.format = format;
    options.memory_policy = InputMemoryPolicy::SystemMemory;
    const auto geometry = describe_mat_geometry_or_throw(roi, options, "Mat normalization");
    const auto converted = tensor_from_cv_mat(roi, options, "Mat construction");
    require(converted.dtype == geometry.dtype && converted.layout == geometry.layout &&
                converted.shape == (roi.channels() == 1 ? std::vector<std::int64_t>{3, 4}
                                                        : std::vector<std::int64_t>{3, 4, 3}),
            "Mat construction disagrees with validated geometry");
    require(converted.read_only && converted.storage &&
                converted.storage->kind == StorageKind::CpuExternal,
            "host Mat policy must retain borrowed storage");
    const auto mapping = converted.map_read();
    require(mapping.data == roi.data &&
                converted.strides_bytes.front() == std::int64_t(roi.step[0]),
            "host Mat normalization must preserve source storage/strides");
  }
  InputOptions mismatch;
  mismatch.format = FormatTag::RGB;
  mismatch.memory_policy = InputMemoryPolicy::SystemMemory;
  const cv::Mat gray(3, 4, CV_8UC1);
  std::string validation_error, conversion_error;
  try {
    (void)describe_mat_geometry_or_throw(gray, mismatch, "same context");
  } catch (const std::invalid_argument& error) {
    validation_error = error.what();
  }
  try {
    (void)tensor_from_cv_mat(gray, mismatch, "same context");
  } catch (const std::invalid_argument& error) {
    conversion_error = error.what();
  }
  require(!validation_error.empty() && validation_error == conversion_error,
          "Mat construction must preserve the geometry validator's original diagnostic");
}
void metadata_contract() {
  InputOptions opt;
  opt.payload_type = PayloadType::Image;
  opt.format = FormatTag::RGB;
  opt.max_width = 96;
  opt.max_height = 96;
  opt.max_depth = 3;
  InputStreamOptions stream;
  stream.max_input_bytes = 96 * 96 * 3;
  cv::Mat image(48, 64, CV_8UC3, cv::Scalar(4, 7, 9));
  Tensor tensor = Tensor::from_cv_mat_view(image, ImageSpec::PixelFormat::RGB);
  Sample sample = sample_from_tensors(TensorList{tensor});
  // The pure common normalizer must not map, clone, allocate, or author caps.
  allocations = 0;
  count_allocations = true;
  for (int i = 0; i < 100; ++i) {
    validate_public_input(image, opt, stream, "test");
    validate_public_input(tensor, opt, stream, "test");
    validate_public_input(sample, opt, stream, "test");
  }
  count_allocations = false;
  require(allocations == 0, "successful metadata preflight allocated");
  const auto geometry = describe_tensor_geometry_or_throw(tensor, opt, "test");
  const auto spec = describe_tensor_spec_or_throw(tensor, opt, "test");
  require(geometry.required_bytes_actual == spec.required_bytes_actual &&
              geometry.width == spec.width && geometry.height == spec.height,
          "preflight and materialized builder disagree");
  cv::Mat oversized(48, 97, CV_8UC3);
  rejects([&] { validate_public_input(oversized, opt, stream, "test"); },
          error_codes::kInputCapacity);
  auto short_tensor = tensor;
  short_tensor.storage = make_cpu_owned_storage(1);
  rejects([&] { validate_public_input(short_tensor, opt, stream, "test"); },
          error_codes::kInputShape);
  auto overflow = tensor;
  overflow.shape = {std::numeric_limits<int64_t>::max(), 64, 3};
  rejects([&] { validate_public_input(overflow, opt, stream, "test"); }, error_codes::kInputShape);
  auto bad_stride = tensor;
  bad_stride.strides_bytes = {std::numeric_limits<int64_t>::max(), 3, 1};
  rejects([&] { validate_public_input(bad_stride, opt, stream, "test"); },
          error_codes::kInputShape);
  // Non-contiguous matrices remain accepted: bounds are the original view,
  // whereas the later explicit copy transports its tight logical pixels.
  cv::Mat roi = image(cv::Rect(2, 2, 20, 20));
  validate_public_input(roi, opt, stream, "test");
  InputOptions fp32;
  fp32.payload_type = PayloadType::Tensor;
  fp32.max_width = 8;
  bool exact_fp32_rejection = false;
  try {
    (void)describe_mat_geometry_or_throw(image, fp32, "FP32 Mat contract");
  } catch (const std::invalid_argument& error) {
    exact_fp32_rejection =
        std::string(error.what()).find("must already be CV_32F") != std::string::npos;
  }
  require(exact_fp32_rejection, "established FP32 Mat diagnostic changed");
  cv::Mat float_image(4, 9, CV_32F);
  rejects([&] { validate_public_input(float_image, fp32, InputStreamOptions{}, "test"); },
          error_codes::kInputCapacity);
  // Match canonical rank/batch, not lossy W/H/D projections. Single-channel
  // matrices retain the public HW <-> unit-channel HWC/CHW convenience alias.
  const auto mat_geometry = describe_mat_geometry_or_throw(float_image, fp32, "test");
  const std::vector<std::int64_t> single_channel{1, 4, 9, 1}, batched{2, 4, 9, 1};
  require(input_geometry_matches_logical_shape(mat_geometry, single_channel, TensorLayout::HWC),
          "unit-channel HW convenience alias lost");
  require(!input_geometry_matches_logical_shape(mat_geometry, batched, TensorLayout::HWC),
          "non-unit leading batch silently dropped");
  const std::vector<std::int64_t> high_rank{2, 3, 4, 5, 6}, different_batch{3, 3, 4, 5, 6};
  InputGeometry rank_geometry;
  rank_geometry.shape = high_rank;
  require(input_geometry_matches_logical_shape(rank_geometry, high_rank, TensorLayout::Unknown) &&
              !input_geometry_matches_logical_shape(rank_geometry, different_batch,
                                                    TensorLayout::Unknown),
          "higher-rank canonical shape was reduced to trailing dimensions");
  // TensorSet is a transport container, not permission to skip logical bounds.
  Tensor wide = Tensor::from_vector(std::vector<float>(36, 1.0f), {4, 9}, TensorMemory::CPU);
  wide.layout = TensorLayout::HW;
  rejects(
      [&] {
        validate_public_input(sample_from_tensors(TensorList{wide, wide}), fp32,
                              InputStreamOptions{}, "test");
      },
      error_codes::kInputCapacity);
  auto opaque = tensor;
  opaque.storage = std::make_shared<TensorBuffer>(*tensor.storage);
  unsigned maps = 0;
  opaque.storage->map_fn = [&](MapMode) {
    ++maps;
    return Mapping{};
  };
  validate_public_input(opaque, opt, stream, "test");
  require(maps == 0, "preflight accessed provider mapping");
  // Planar metadata, including a late-plane overflow, uses the same extent
  // normalization as ordinary contiguous image admission.
  for (auto format :
       {ImageSpec::PixelFormat::NV12, ImageSpec::PixelFormat::NV16, ImageSpec::PixelFormat::I420}) {
    auto planar = make_nv12_tensor(8, 6);
    planar.semantic.image->format = format;
    InputOptions plane_opt;
    plane_opt.payload_type = PayloadType::Image;
    if (format == ImageSpec::PixelFormat::NV16) {
      planar.storage = make_cpu_owned_storage(96);
      planar.planes[1].shape = {6, 8};
    } else if (format == ImageSpec::PixelFormat::I420) {
      planar.planes[1].role = PlaneRole::U;
      planar.planes[1].shape = {3, 4};
      planar.planes[1].strides_bytes = {4, 1};
      Plane v = planar.planes[1];
      v.role = PlaneRole::V;
      v.byte_offset = 60;
      planar.planes.push_back(v);
    }
    validate_public_input(planar, plane_opt, InputStreamOptions{}, "test");
    auto short_plane = planar;
    short_plane.storage = make_cpu_owned_storage(1);
    rejects([&] { validate_public_input(short_plane, plane_opt, InputStreamOptions{}, "test"); },
            error_codes::kInputShape);
  }
  validate_public_input(image, opt, stream, "test");
}

Sample make_mixed_sample() {
  Tensor image =
      Tensor::from_vector(std::vector<std::uint8_t>(48 * 64, 7), {48, 64}, TensorMemory::CPU);
  image.layout = TensorLayout::HW;
  // Independently typed features deliberately exceed the representative's width
  // AND per-view byte limit. Two image views also alias the same allocation.
  Tensor features =
      Tensor::from_vector(std::vector<std::int32_t>(1024, 3), {1024}, TensorMemory::CPU);
  return sample_from_tensors(TensorList{image, image, features});
}

InputOptions mixed_input_options() {
  InputOptions options;
  options.payload_type = PayloadType::Tensor;
  options.format = FormatTag::UINT8;
  options.max_width = 64;
  options.max_height = 48;
  options.max_depth = 1;
  return options;
}

void aggregate_metadata_contract() {
  const auto options = mixed_input_options();
  InputStreamOptions stream;
  stream.max_input_bytes = 48 * 64;
  auto mixed = make_mixed_sample();
  validate_public_input(mixed, options, stream, "mixed input");

  Sample first;
  first.kind = SampleKind::Tensor;
  first.tensor = mixed.tensors.front();
  Sample later = sample_from_tensors(TensorList{mixed.tensors[1], mixed.tensors[2]});
  Sample nested;
  nested.kind = SampleKind::Bundle;
  nested.fields = {later};
  Sample bundle;
  bundle.kind = SampleKind::Bundle;
  bundle.fields = {first, nested};
  validate_public_input(bundle, options, stream, "nested mixed input");

  // Intrinsic storage validation must accept a valid non-dense later view;
  // ordinary representative tensor transport has a different dense contract.
  auto strided = mixed;
  strided.tensors.back().shape = {2, 2};
  strided.tensors.back().strides_bytes = {16, 4};
  strided.tensors.back().storage = make_cpu_owned_storage(24);
  validate_public_input(strided, options, stream, "strided later member");

  auto invalid = mixed;
  invalid.tensors.front() = mixed.tensors.back();
  rejects([&] { validate_public_input(invalid, options, stream, "wrong representative"); },
          error_codes::kInputShape);
  invalid = mixed;
  invalid.tensors.back().storage = make_cpu_owned_storage(1);
  rejects([&] { validate_public_input(invalid, options, stream, "short later member"); },
          error_codes::kInputShape);
  invalid.tensors.back().storage.reset();
  rejects([&] { validate_public_input(invalid, options, stream, "missing later backing"); },
          error_codes::kInputShape);
  invalid = mixed;
  invalid.tensors.back().byte_offset = -1;
  rejects([&] { validate_public_input(invalid, options, stream, "negative later offset"); },
          error_codes::kInputShape);
  invalid = mixed;
  invalid.tensors.back().shape = {0};
  rejects([&] { validate_public_input(invalid, options, stream, "empty later extent"); },
          error_codes::kInputShape);
  invalid = mixed;
  invalid.tensors.back().strides_bytes = {std::numeric_limits<std::int64_t>::max()};
  rejects([&] { validate_public_input(invalid, options, stream, "later stride overflow"); },
          error_codes::kInputShape);
  invalid = mixed;
  invalid.tensors.back().strides_bytes = {4, 4};
  rejects([&] { validate_public_input(invalid, options, stream, "later stride rank"); },
          error_codes::kInputShape);
  invalid = mixed;
  invalid.tensors.back() = make_nv12_tensor(8, 6);
  invalid.tensors.back().planes.back().byte_offset = -1;
  rejects([&] { validate_public_input(invalid, options, stream, "later plane offset"); },
          error_codes::kInputShape);
  auto invalid_nested = bundle;
  invalid_nested.fields.back().fields.front().tensors.back().storage = make_cpu_owned_storage(1);
  rejects([&] { validate_public_input(invalid_nested, options, stream, "short nested member"); },
          error_codes::kInputShape);
  Sample empty;
  empty.kind = SampleKind::Bundle;
  rejects([&] { validate_public_input(empty, options, stream, "empty bundle"); },
          error_codes::kInputShape);

  unsigned maps = 0;
  mixed.tensors.back().storage = std::make_shared<TensorBuffer>(*mixed.tensors.back().storage);
  mixed.tensors.back().storage->map_fn = [&](MapMode) {
    ++maps;
    return Mapping{};
  };
  allocations = 0;
  count_allocations = true;
  for (int i = 0; i < 100; ++i) {
    validate_public_input(mixed, options, stream, "mixed input");
    validate_public_input(bundle, options, stream, "nested mixed input");
    validate_public_input(strided, options, stream, "strided later member");
  }
  count_allocations = false;
  require(allocations == 0, "aggregate metadata admission allocated");
  require(maps == 0, "aggregate metadata admission mapped payload storage");
}

void aggregate_queue_contract() {
  auto core = runtime::RunCore::create_graph_compat();
  core->graph_execution_.reset();
  core->mode = RunMode::Async;
  core->pipeline.supports_push = true;
  core->opt.queue_depth = 1;
  core->opt.overflow_policy = OverflowPolicy::KeepLatest;
  core->push_sample_policy = runtime::PushSamplePolicy::PreserveSample;
  core->pipeline.tensor_input_opt_for_cv = mixed_input_options();
  core->pipeline.stream_opt.max_input_bytes = 48 * 64;
  auto valid = make_mixed_sample();
  valid.frame_id = 17;
  require(core->push_samples(Sample{valid}, false), "mixed input admission refused");
  auto invalid = valid;
  invalid.tensors.back().storage = make_cpu_owned_storage(1);
  const auto submitted = core->inputs_enqueued.load();
  rejects([&] { core->push_samples(Sample{invalid}, false); }, error_codes::kInputShape);
  require(core->pipeline.in_queue.size() == 1 && core->pipeline.in_queue.front().msg.frame_id == 17,
          "bad later member evicted valid pending input");
  require(core->inputs_enqueued.load() == submitted && core->inputs_dropped.load() == 0 &&
              !core->stop_requested.load(),
          "bad later member changed queue or terminal state");
  valid.frame_id = 18;
  require(core->push_samples(Sample{valid}, false), "mixed recovery admission refused");
  const auto& queued = core->pipeline.in_queue.front().msg;
  require(queued.frame_id == 18 && queued.tensors.size() == 3 &&
              queued.tensors.back().storage == valid.tensors.back().storage &&
              core->inputs_dropped.load() == 1,
          "mixed recovery changed identity, copy policy or replacement accounting");
  core->close();
}

// Only the producer is a test double. Admission, validation and replacement use
// the real RunCore path with no worker/device owner and deterministic occupancy.
struct RoutedCarrier final : pipeline_internal::InputRouteProcessor {
  Sample output;
  void validate_images(const std::vector<cv::Mat>&, const char*) const override {}
  Sample process_tensors(const TensorList&, const char*) const override {
    return output;
  }
  Sample process_samples(const Sample&, const char*) const override {
    return output;
  }
};

void routed_queue_contract() {
  for (const bool block : {false, true}) {
    auto core = runtime::RunCore::create_graph_compat();
    core->graph_execution_.reset();
    core->mode = RunMode::Async;
    core->pipeline.supports_push = true;
    core->opt.queue_depth = 1;
    core->opt.overflow_policy = OverflowPolicy::KeepLatest;
    core->push_sample_policy = runtime::PushSamplePolicy::PreserveSample;
    InputOptions collapsed;
    collapsed.payload_type = PayloadType::Tensor;
    collapsed.format = FormatTag::FP32;
    collapsed.max_width = 2;
    collapsed.max_height = 1;
    collapsed.max_depth = 4;
    core->pipeline.tensor_input_opt_for_cv = collapsed;
    core->pipeline.stream_opt.max_input_bytes = 2 * 256 * sizeof(float);
    auto route = std::make_shared<RoutedCarrier>();
    core->pipeline.input_route_processor = route;
    Tensor feature =
        Tensor::from_vector(std::vector<float>(2 * 256, 1), {1, 2, 256}, TensorMemory::CPU);
    feature.layout = TensorLayout::HWC;
    Tensor proposal =
        Tensor::from_vector(std::vector<float>(2 * 4, 2), {1, 2, 4}, TensorMemory::CPU);
    proposal.layout = TensorLayout::HWC;
    auto valid = sample_from_tensors({feature, proposal});
    valid.frame_id = 17;
    route->output = valid;
    require(core->push_samples(valid, block), "heterogeneous routed carrier refused");
    require(core->pipeline.in_queue.front().msg.tensors.front().storage == feature.storage,
            "routed admission copied the carrier");
    const auto submitted = core->inputs_enqueued.load();
    const auto rejected = [&](Sample invalid, const char* code = error_codes::kInputShape) {
      route->output = std::move(invalid);
      rejects([&] { core->push_samples(valid, block); }, code);
      require(core->pipeline.in_queue.size() == 1 &&
                  core->pipeline.in_queue.front().msg.frame_id == 17 &&
                  core->inputs_enqueued.load() == submitted && core->inputs_dropped.load() == 0 &&
                  !core->stop_requested.load(),
              "invalid routed output changed pending work or poisoned the run");
    };
    for (std::size_t member : {0U, 1U}) {
      auto invalid = valid;
      invalid.tensors[member].storage = make_cpu_owned_storage(1);
      rejected(invalid);
      invalid = valid;
      invalid.tensors[member].byte_offset = -1;
      rejected(invalid);
      invalid = valid;
      invalid.tensors[member].strides_bytes = {4};
      rejected(invalid);
    }
    core->pipeline.stream_opt.max_input_bytes = 2 * 256 * sizeof(float) - 1;
    rejected(valid, error_codes::kInputCapacity);
    core->pipeline.stream_opt.max_input_bytes = 2 * 256 * sizeof(float);
    route->output = valid;
    std::swap(route->output.tensors[0], route->output.tensors[1]);
    route->output.frame_id = 18;
    if (block) {
      // Blocking push intentionally waits for capacity even with KeepLatest.
      // This no-worker fixture must consume the retained item before recovery.
      std::lock_guard lock(core->pipeline.in_mu);
      core->pipeline.in_queue.pop_front();
    }
    require(core->push_samples(valid, block), "valid routed input failed after rejection");
    require(core->pipeline.in_queue.front().msg.frame_id == 18 &&
                core->pipeline.in_queue.front().msg.tensors.back().storage == feature.storage &&
                core->inputs_dropped.load() == (block ? 0U : 1U),
            "routed replacement lost member identity or queue accounting");
    // The source geometry is still authoritative when there is no router.
    core->pipeline.input_route_processor.reset();
    rejects([&] { core->push_samples(valid, block); }, error_codes::kInputCapacity);
    core->close();
  }
}

void queue_metadata_contract() {
  // Exercise the real RunCore queue admission with no worker/device owner.
  // Occupancy is deterministic; this is not a fake replacement queue algorithm.
  auto core = runtime::RunCore::create_graph_compat();
  core->graph_execution_.reset();
  core->mode = RunMode::Async;
  core->pipeline.supports_push = true;
  core->opt.queue_depth = 1;
  core->opt.overflow_policy = OverflowPolicy::KeepLatest;
  core->push_sample_policy = runtime::PushSamplePolicy::PreserveSample;
  InputOptions options;
  options.payload_type = PayloadType::Image;
  options.format = FormatTag::RGB;
  options.max_width = 96;
  options.max_height = 96;
  options.max_depth = 3;
  core->pipeline.tensor_input_opt_for_cv = options;
  const Tensor valid = make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x4A);
  Sample sample = sample_from_tensors(TensorList{valid});
  sample.frame_id = 17;
  require(core->push_samples(Sample{sample}, false), "initial queue admission refused");
  auto short_tensor = valid;
  short_tensor.storage = make_cpu_owned_storage(1);
  const auto submitted = core->inputs_enqueued.load();
  rejects([&] { core->push_samples(Sample{sample_from_tensors(TensorList{short_tensor})}, false); },
          error_codes::kInputShape);
  require(core->pipeline.in_queue.size() == 1 && core->pipeline.in_queue.front().msg.frame_id == 17,
          "rejected short backing evicted pending work");
  require(core->inputs_enqueued.load() == submitted && core->inputs_dropped.load() == 0 &&
              !core->stop_requested.load(),
          "rejection changed queue counters or terminal state");
  sample.frame_id = 18;
  require(core->push_samples(Sample{sample}, false), "valid input refused after rejection");
  require(core->pipeline.in_queue.front().msg.frame_id == 18 && core->inputs_dropped.load() == 1,
          "ordinary KeepLatest replacement changed");
  core->pipeline.input_closed = true;
  require(!core->push_samples(Sample{sample}, false), "closed queue accepted input");
  core->close();
}

} // namespace
RUN_TEST("unit_input_metadata_geometry_test", ([] {
           mat_conversion_contract();
           metadata_contract();
           aggregate_metadata_contract();
           queue_metadata_contract();
           aggregate_queue_contract();
           routed_queue_contract();
         }));
