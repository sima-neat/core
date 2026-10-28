// Public EV74 regressions for Internals #329-333 and Core #1110-1111.
// Numerical truth is authored geometry/residual arithmetic, never native LK.
#include "../support/visual_frontend_test_utils.h"
#include "../support/visual_tracker_sequence.h"

#include <optional>

namespace vt = visual_test;
namespace neat = vt::neat;
namespace {
constexpr int kWidth = 640;
constexpr int kHeight = 480;

struct Frame {
  int width = kWidth;
  int height = kHeight;
  vt::Image previous;
  vt::Image current;
  std::vector<std::int32_t> points;
  int dx = 0;
  int dy = 0;
  neat::TensorList tensors;
  std::optional<neat::Tensor> parent;
  vt::Image parent_bytes;
};

std::vector<std::int32_t> interior_points(int generation = 0) {
  std::vector<std::int32_t> result;
  for (int i = 0; i < 64; ++i) {
    result.push_back(160 + 32 * (i % 8) + 3 * generation);
    result.push_back(160 + 16 * (i / 8) + 2 * generation);
  }
  return result;
}
std::vector<std::int32_t> blob_points() {
  std::vector<std::int32_t> result;
  for (int y : {96, 192, 288, 384})
    for (int x : {96, 224, 352, 480}) {
      result.push_back(x);
      result.push_back(y);
    }
  return result;
}
vt::Image blobs() {
  // Fixture qualified independently against unconstrained OpenCV before any
  // candidate EV measurement. Broad asymmetric features avoid the ambiguous
  // short-correlation random texture that also defeats reference LK.
  const auto points = blob_points();
  vt::Image result(kWidth * kHeight);
  for (int y = 0; y < kHeight; ++y)
    for (int x = 0; x < kWidth; ++x) {
      double value = 25;
      for (std::size_t i = 0; i < points.size() / 2; ++i) {
        const double dx = x - points[2 * i], dy = y - points[2 * i + 1];
        value += (100 + 3 * i) * std::exp(-(dx * dx + 1.3 * dy * dy) / (2 * 24.0 * 24.0));
        value += 35 * std::exp(-((dx - 11) * (dx - 11) + (dy + 7) * (dy + 7)) / (2 * 11.0 * 11.0));
      }
      result[y * kWidth + x] = static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
    }
  return result;
}
Frame frame(vt::Image previous, std::vector<std::int32_t> points, int dx = 0, int dy = 0,
            int width = kWidth, int height = kHeight) {
  Frame result;
  result.width = width;
  result.height = height;
  result.previous = std::move(previous);
  result.current = vt::shifted(result.previous, width, height, dx, dy);
  result.points = std::move(points);
  result.dx = dx;
  result.dy = dy;
  return result;
}
void allocate(Frame& frame, bool shared = false, int prefix = 64) {
  const int count = static_cast<int>(frame.points.size() / 2);
  if (!shared) {
    frame.tensors = {vt::image(frame.previous, frame.width, frame.height, 1, "prev_image"),
                     vt::image(frame.current, frame.width, frame.height, 1, "cur_image"),
                     vt::seeds(frame.points, count)};
    return;
  }
  // Public Tensor storage sharing and byte_offset views. Payload bytes are
  // authored at their actual types; no conversion or physical-route IDs.
  const std::size_t prev = prefix;
  const std::size_t cur = prev + frame.previous.size() + 64;
  const std::size_t seeds = cur + frame.current.size() + 64;
  frame.parent_bytes.assign(seeds + frame.points.size() * sizeof(std::int32_t) + 64, 0xa5);
  std::copy(frame.previous.begin(), frame.previous.end(), frame.parent_bytes.begin() + prev);
  std::copy(frame.current.begin(), frame.current.end(), frame.parent_bytes.begin() + cur);
  std::memcpy(frame.parent_bytes.data() + seeds, frame.points.data(),
              frame.points.size() * sizeof(std::int32_t));
  frame.parent = neat::Tensor::from_vector(frame.parent_bytes,
                                           {static_cast<std::int64_t>(frame.parent_bytes.size())},
                                           neat::TensorMemory::EV74);
  auto image_view = [&](std::size_t offset, const char* name) {
    auto result = *frame.parent;
    result.shape = {1, frame.height, frame.width};
    result.strides_bytes.clear();
    result.byte_offset = static_cast<std::int64_t>(offset);
    result.layout = neat::TensorLayout::HW;
    result.axis_semantics = {neat::TensorAxisSemantic::N, neat::TensorAxisSemantic::H,
                             neat::TensorAxisSemantic::W};
    vt::name(result, name);
    return result;
  };
  auto point_view = *frame.parent;
  point_view.dtype = neat::TensorDType::Int32;
  point_view.shape = {1, count, 2};
  point_view.strides_bytes.clear();
  point_view.byte_offset = static_cast<std::int64_t>(seeds);
  point_view.axis_semantics.clear();
  vt::name(point_view, "input_points");
  frame.tensors = {image_view(prev, "prev_image"), image_view(cur, "cur_image"), point_view};
}
neat::TrackKLTOptions options(const Frame& frame, int level = 2) {
  neat::TrackKLTOptions result;
  result.width = frame.width;
  result.height = frame.height;
  result.num_points = static_cast<int>(frame.points.size() / 2);
  result.win_half = 7;
  result.max_iters = 20;
  result.max_level = level;
  result.num_buffers = 4;
  result.grid_x = result.grid_y = 1;
  result.min_px_dist = 0;
  result.max_features = 16;
  return result;
}
neat::Graph graph(const neat::TrackKLTOptions& options) {
  neat::Graph result;
  result.add(neat::nodes::Input(
      vt::input_options(options.width, options.height, options.batch_size, true)));
  result.add(neat::nodes::TrackKLT(options));
  result.add(neat::nodes::Output());
  return result;
}
std::vector<neat::TensorList> execute(std::vector<Frame>& frames, const neat::TrackKLTOptions& opts,
                                      bool shared = false, bool unnamed = false) {
  std::vector<neat::TensorList> inputs;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    allocate(frames[i], shared, 64 + int(i) * 64);
    if (unnamed)
      for (auto& tensor : frames[i].tensors)
        tensor.route.name.clear();
    inputs.push_back(frames[i].tensors);
  }
  auto pipeline = graph(opts);
  auto result = vt::execute(pipeline, inputs);
  for (const auto& frame : frames)
    if (frame.parent)
      vt::require(frame.parent->copy_dense_bytes_tight() == frame.parent_bytes,
                  "read-only shared carrier or guard bytes changed");
  return result;
}
struct Tracks {
  std::vector<float> xy;
  std::vector<std::int32_t> status;
  std::vector<float> error;
};
Tracks read(const neat::TensorList& output, const neat::TrackKLTOptions& options) {
  const std::size_t error_slot = options.detect_new_features ? 3 : 2;
  vt::require(output.size() == error_slot + (options.publish_error ? 1 : 0),
              "tracker public tuple arity/order mismatch");
  Tracks result;
  result.xy = vt::values<float>(output[0], neat::TensorDType::Float32,
                                {options.batch_size, options.num_points, 2});
  result.status = vt::values<std::int32_t>(output[1], neat::TensorDType::Int32,
                                           {options.batch_size, options.num_points, 1});
  if (options.publish_error)
    result.error = vt::values<float>(output[error_slot], neat::TensorDType::Float32,
                                     {options.batch_size, options.num_points, 1});
  for (auto status : result.status)
    vt::require(status == 0 || status == 1, "status must be binary");
  return result;
}
void correct(const Frame& frame, const Tracks& result) {
  std::size_t accepted = 0, accurate = 0;
  double maximum = 0;
  for (std::size_t i = 0; i < result.status.size(); ++i) {
    const double error = std::hypot(result.xy[2 * i] - frame.points[2 * i] - frame.dx,
                                    result.xy[2 * i + 1] - frame.points[2 * i + 1] - frame.dy);
    accepted += result.status[i] == 1;
    accurate += result.status[i] == 1 && std::isfinite(error) && error <= 0.3;
    maximum =
        std::isfinite(error) ? std::max(maximum, error) : std::numeric_limits<double>::infinity();
  }
  std::cout << "KLT_RECALL accepted=" << accepted << " accurate_0.3px=" << accurate
            << " total=" << result.status.size() << " max_error_px=" << maximum << '\n';
  vt::require(accurate == result.status.size(), "valid-track recall/endpoint accuracy failed");
}
void motion_controls(int level) {
  std::vector<Frame> frames;
  for (auto shift : {std::array{0, 0}, std::array{3, 2}, std::array{8, 0}, std::array{-3, -2}})
    frames.push_back(frame(vt::texture(kWidth, kHeight, 7), interior_points(), shift[0], shift[1]));
  const auto opts = options(frames.front(), level);
  const auto results = execute(frames, opts);
  for (std::size_t i = 0; i < frames.size(); ++i)
    correct(frames[i], read(results[i], opts));
}
void allocations(bool shared, bool unnamed) {
  std::vector<Frame> frames;
  const std::array<std::array<int, 2>, 3> shifts{{{3, 2}, {-2, 3}, {3, -2}}};
  for (int i = 0; i < 3; ++i)
    frames.push_back(frame(vt::texture(kWidth, kHeight, 7 + 23 * i), interior_points(i),
                           shifts[i][0], shifts[i][1]));
  const auto opts = options(frames.front());
  const auto results = execute(frames, opts, shared, unnamed);
  for (std::size_t i = 0; i < frames.size(); ++i)
    correct(frames[i], read(results[i], opts));
}
void reload(int level) {
  const auto previous = blobs();
  std::vector<Frame> frames;
  if (level == 0) {
    frames.push_back(frame(previous, blob_points(), 6, 0));
    frames.push_back(frame(previous, blob_points(), 0, -6));
  } else {
    frames.push_back(frame(previous, blob_points(), 24, 0));
    frames.push_back(frame(previous, blob_points(), -24, 0));
    frames.push_back(frame(previous, blob_points(), 0, 24));
    frames.push_back(frame(previous, blob_points(), 32, 0));
  }
  const auto opts = options(frames.front(), level);
  const auto results = execute(frames, opts);
  for (std::size_t i = 0; i < frames.size(); ++i)
    correct(frames[i], read(results[i], opts));
}
void borders(int level, bool tiny) {
  const int width = tiny ? 32 : kWidth, height = tiny ? 24 : kHeight;
  std::vector<std::int32_t> points;
  const std::vector<int> distances =
      tiny ? std::vector<int>{0, 1, 3, 7} : std::vector<int>{0, 1, 8, 16, 43, 44, 87, 88};
  for (int d : distances) {
    for (auto xy : {std::array{d, height / 2}, std::array{width - 1 - d, height / 2},
                    std::array{width / 2, d}, std::array{width / 2, height - 1 - d}}) {
      points.push_back(xy[0]);
      points.push_back(xy[1]);
    }
  }
  for (auto xy : {std::array{0, 0}, std::array{width - 1, 0}, std::array{0, height - 1},
                  std::array{width - 1, height - 1}}) {
    points.push_back(xy[0]);
    points.push_back(xy[1]);
  }
  std::vector<Frame> frames{frame(vt::texture(width, height, 7), points, 0, 0, width, height)};
  auto opts = options(frames.front(), level);
  opts.publish_error = true;
  const auto results = execute(frames, opts);
  const auto tracks = read(results.front(), opts);
  correct(frames.front(), tracks);
  for (float error : tracks.error)
    vt::require(std::isfinite(error) && error < 0.01F, "identity edge residual must be near zero");
}
void border_motion() {
  std::vector<std::int32_t> points;
  for (int d : {16, 31, 43, 44, 63, 87, 88})
    for (auto xy : {std::array{d, 128}, std::array{kWidth - 1 - d, 256}, std::array{128, d},
                    std::array{384, kHeight - 1 - d}}) {
      points.push_back(xy[0]);
      points.push_back(xy[1]);
    }
  // Independent OpenCV qualification: all28 points for both signs and levels
  // agree with authored translation within0.001px; retain the0.3px EV bound.
  for (int level : {2, 3}) {
    std::vector<Frame> frames;
    for (int sign : {-1, 1})
      frames.push_back(frame(vt::texture(kWidth, kHeight, 7), points, 3 * sign, 2 * sign));
    const auto opts = options(frames.front(), level);
    const auto outputs = execute(frames, opts);
    for (std::size_t f = 0; f < frames.size(); ++f)
      correct(frames[f], read(outputs[f], opts));
  }
}
float sample(const vt::Image& image, int width, int height, float x, float y) {
  const int ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
  const float fx = x - ix, fy = y - iy;
  auto p = [&](int sx, int sy) {
    return float(image[std::clamp(sy, 0, height - 1) * width + std::clamp(sx, 0, width - 1)]);
  };
  return p(ix, iy) * (1 - fx) * (1 - fy) + p(ix + 1, iy) * fx * (1 - fy) +
         p(ix, iy + 1) * (1 - fx) * fy + p(ix + 1, iy + 1) * fx * fy;
}
float residual(const Frame& frame, const Tracks& tracks, std::size_t point, int half) {
  double sum = 0;
  for (int y = -half; y <= half; ++y)
    for (int x = -half; x <= half; ++x)
      sum += std::abs(sample(frame.previous, frame.width, frame.height,
                             float(frame.points[2 * point] + x),
                             float(frame.points[2 * point + 1] + y)) -
                      sample(frame.current, frame.width, frame.height, tracks.xy[2 * point] + x,
                             tracks.xy[2 * point + 1] + y));
  return float(sum / ((2 * half + 1) * (2 * half + 1)));
}
void quality() {
  auto identity = frame(blobs(), blob_points());
  auto mismatch = identity;
  for (auto& value : mismatch.current)
    value = static_cast<std::uint8_t>(std::min(255, int(value) + 17));
  for (float threshold : {-1.F, 0.F, 255.F}) {
    std::vector<Frame> frames{identity, mismatch};
    auto opts = options(identity, 0);
    opts.publish_error = true;
    opts.error_threshold = threshold;
    const auto result = execute(frames, opts);
    for (std::size_t f = 0; f < frames.size(); ++f) {
      const auto tracks = read(result[f], opts);
      for (std::size_t p = 0; p < tracks.status.size(); ++p) {
        const float expected = residual(frames[f], tracks, p, opts.win_half);
        vt::require(std::isfinite(tracks.error[p]) &&
                        std::abs(tracks.error[p] - expected) <= 0.002F,
                    "error is not the independent final-coordinate MAE");
        vt::require(tracks.status[p] == ((threshold >= 0 && tracks.error[p] > threshold) ? 0 : 1),
                    "status does not implement configured residual threshold");
        if (f == 0)
          vt::require(tracks.error[p] == 0.F && tracks.status[p] == 1,
                      "zero-error equality must pass threshold zero");
        else
          vt::require(tracks.error[p] > 1.F, "photometric mismatch fixture lost sensitivity");
      }
    }
  }
  // Filtering is a solver policy, not a side effect of publishing another port.
  std::vector<Frame> unpublished{identity, mismatch};
  auto opts = options(identity, 0);
  opts.error_threshold = 0.F;
  const auto result = execute(unpublished, opts);
  const auto good = read(result[0], opts);
  const auto bad = read(result[1], opts);
  correct(identity, good);
  for (auto status : bad.status)
    vt::require(status == 0, "residual filtering must work without error publication");
  // Symmetric integer quadratic: sum(gx)==sum(gy)==0, so adding 17 intensity
  // units gives an exactly zero update and an independently known MAE of 17.
  vt::Image quadratic(kWidth * kHeight);
  for (int y = 0; y < kHeight; ++y)
    for (int x = 0; x < kWidth; ++x) {
      const int dx = x - kWidth / 2, dy = y - kHeight / 2;
      quadratic[y * kWidth + x] = std::uint8_t(64 + std::min(150, dx * dx + dy * dy));
    }
  auto boundary = frame(quadratic, {kWidth / 2, kHeight / 2});
  for (auto& value : boundary.current)
    value = std::uint8_t(value + 17);
  for (float threshold : {16.F, 17.F, 18.F}) {
    std::vector<Frame> frames{boundary};
    auto policy = options(boundary, 0);
    policy.publish_error = true;
    policy.error_threshold = threshold;
    const auto output = execute(frames, policy);
    const auto track = read(output[0], policy);
    vt::require(track.xy[0] == kWidth / 2 && track.xy[1] == kHeight / 2 && track.error[0] == 17.F &&
                    track.status[0] == (threshold < 17.F ? 0 : 1),
                "positive independent residual below/equal/above threshold boundary");
  }
  auto occluded = identity;
  occluded.current.assign(kWidth * kHeight, 128);
  auto unrelated = identity;
  unrelated.current = vt::texture(kWidth, kHeight, 93);
  std::vector<Frame> rejected{occluded, unrelated};
  auto reject = options(identity, 0);
  reject.publish_error = true;
  reject.error_threshold = 0.F;
  const auto rejected_output = execute(rejected, reject);
  for (std::size_t f = 0; f < rejected.size(); ++f) {
    const auto tracks = read(rejected_output[f], reject);
    for (std::size_t p = 0; p < tracks.status.size(); ++p) {
      vt::require(tracks.status[p] == 0 && tracks.error[p] > 0.F,
                  "fully occluded/unrelated image passed zero-residual quality policy");
      if (std::isfinite(tracks.error[p]))
        vt::require(std::abs(tracks.error[p] - residual(rejected[f], tracks, p, reject.win_half)) <=
                        0.002F,
                    "rejected finite candidate residual mismatch");
    }
  }
}
void texture_quality() {
  vt::Image weak(kWidth * kHeight);
  for (int y = 0; y < kHeight; ++y)
    for (int x = 0; x < kWidth; ++x)
      weak[y * kWidth + x] = static_cast<std::uint8_t>(64 + ((x / 2 + y / 2) % 2));
  vt::Image aperture(kWidth * kHeight);
  for (int y = 0; y < kHeight; ++y)
    for (int x = 0; x < kWidth; ++x)
      aperture[y * kWidth + x] = std::uint8_t(64 + (x % 64));
  std::vector<Frame> frames{frame(weak, interior_points()),
                            frame(vt::texture(kWidth, kHeight, 7), interior_points()),
                            frame(vt::Image(kWidth * kHeight, 128), interior_points()),
                            frame(aperture, interior_points())};
  auto opts = options(frames.front(), 0);
  opts.publish_error = true;
  opts.min_eig_threshold = 1e-5F;
  const auto results = execute(frames, opts);
  for (std::size_t f = 0; f < frames.size(); ++f) {
    const auto tracks = read(results[f], opts);
    if (f == 1) {
      correct(frames[f], tracks);
    } else {
      for (std::size_t i = 0; i < tracks.status.size(); ++i)
        vt::require(tracks.status[i] == 0 && std::isinf(tracks.error[i]) && tracks.error[i] > 0,
                    "singular/low-texture template was not rejected with unevaluated error");
    }
  }
}
void convergence() {
  std::vector<Frame> frames{frame(blobs(), blob_points()), frame(blobs(), blob_points(), 6)};
  for (bool strict : {false, true}) {
    auto opts = options(frames.front(), 0);
    opts.max_iters = 1;
    opts.publish_error = true;
    opts.require_convergence = strict;
    const auto result = execute(frames, opts);
    correct(frames[0], read(result[0], opts));
    const auto tracks = read(result[1], opts);
    for (std::size_t i = 0; i < tracks.status.size(); ++i) {
      vt::require(tracks.status[i] == (strict ? 0 : 1), "iteration-limit convergence policy");
      vt::require(std::isfinite(tracks.error[i]),
                  "finite unconverged candidate needs evaluated error");
    }
  }
}
void inactive_sentinel() {
  auto points = interior_points();
  points[0] = points[1] = -1;
  std::vector<Frame> frames{frame(vt::texture(kWidth, kHeight, 7), points)};
  auto opts = options(frames.front(), 2);
  opts.publish_error = true;
  const auto output = execute(frames, opts);
  const auto tracks = read(output.front(), opts);
  vt::require(tracks.status[0] == 0 && tracks.xy[0] == -1 && tracks.xy[1] == -1 &&
                  std::isinf(tracks.error[0]) && tracks.error[0] > 0,
              "invalid seed must preserve coordinates and report unevaluated error");
  for (std::size_t i = 1; i < tracks.status.size(); ++i)
    vt::require(tracks.status[i] == 1 && tracks.xy[2 * i] == points[2 * i] &&
                    tracks.xy[2 * i + 1] == points[2 * i + 1],
                "invalid slot must not poison valid subsequent points");
}

bool expected_rejection(const std::exception& error) {
  const std::string message = error.what();
  return message.find("misconfig.input_shape") != std::string::npos &&
         message.find("ordered tensor input contract") != std::string::npos;
}
void input_contract() {
  auto fixture = frame(vt::texture(kWidth, kHeight, 7), interior_points(), 3, 2);
  allocate(fixture);
  const auto opts = options(fixture);
  const std::array<std::array<int, 3>, 5> permutations{
      {{0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
  std::vector<neat::TensorList> invalid;
  for (const auto& p : permutations)
    invalid.push_back({fixture.tensors[p[0]], fixture.tensors[p[1]], fixture.tensors[p[2]]});
  auto duplicate = fixture.tensors;
  duplicate[1].route.name = "prev_image";
  invalid.push_back(duplicate);
  auto wrong_dtype = fixture.tensors;
  wrong_dtype[2].dtype = neat::TensorDType::Float32;
  invalid.push_back(wrong_dtype);
  auto missing = fixture.tensors;
  missing.pop_back();
  invalid.push_back(missing);
  std::size_t cold_rejected = 0;
  for (const auto& input : invalid) {
    auto pipeline = graph(opts);
    try {
      auto unexpected = pipeline.build(input, vt::run_options());
      unexpected.close();
    } catch (const std::exception& error) {
      if (!expected_rejection(error))
        throw;
      ++cold_rejected;
    }
  }
  auto pipeline = graph(opts);
  auto run = pipeline.build(fixture.tensors, vt::run_options());
  std::vector<neat::TensorList> recovered;
  std::size_t warm_rejected = 0;
  try {
    for (const auto& input : invalid) {
      try {
        (void)run.run(input, vt::kTimeoutMs);
      } catch (const std::exception& error) {
        if (!expected_rejection(error))
          throw;
        ++warm_rejected;
      }
      recovered.push_back(run.run(fixture.tensors, vt::kTimeoutMs));
    }
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  std::cout << "EV_RUN_CLOSED expected_rejections=" << warm_rejected << '\n';
  vt::require(cold_rejected == invalid.size() && warm_rejected == invalid.size(),
              "not every malformed/permuted input was rejected before dispatch");
  for (const auto& output : recovered)
    correct(fixture, read(output, opts));
}
void detector(const vt::DetectorCase& test, bool publish_error = false) {
  const auto& first = test.frames.front();
  neat::TrackKLTOptions opts;
  opts.width = first.width;
  opts.height = first.height;
  opts.batch_size = test.batch;
  opts.num_points = 1;
  opts.win_half = 7;
  opts.max_iters = 20;
  opts.max_level = 0;
  opts.num_buffers = 4;
  opts.detect_new_features = 1;
  opts.fast_threshold = 30;
  opts.max_features = test.config.capacity;
  opts.grid_x = test.config.columns;
  opts.grid_y = test.config.rows;
  opts.min_px_dist = test.config.distance;
  opts.publish_error = publish_error;
  auto images = vt::detector_frames(test);
  std::vector<neat::TensorList> inputs;
  for (const auto& current : images) {
    std::vector<std::int32_t> points(2 * test.batch, -1);
    inputs.push_back({vt::image(vt::Image(first.width * first.height * test.batch, 64), first.width,
                                first.height, test.batch, "prev_image"),
                      current[0], vt::seeds(points, 1, test.batch)});
    vt::name(inputs.back()[1], "cur_image");
  }
  auto pipeline = graph(opts);
  const auto results = vt::execute(pipeline, inputs);
  vt::check_detector_output(test, results, 2, publish_error ? 4 : 3);
  for (const auto& result : results) {
    const auto tracks = read(result, opts);
    for (auto status : tracks.status)
      vt::require(status == 0, "inactive detector-only seed became a valid track");
    for (auto error : tracks.error)
      vt::require(std::isinf(error) && error > 0, "detector/error output slots collided");
  }
}
void tuple_combinations() {
  auto input = frame(vt::texture(kWidth, kHeight, 7), interior_points());
  for (bool publish : {false, true}) {
    std::vector<Frame> frames{input};
    auto opts = options(input);
    opts.publish_error = publish;
    auto results = execute(frames, opts);
    correct(input, read(results.front(), opts));
  }
  const auto cases = vt::detector_cases(false);
  detector(cases.front(), false);
  detector(cases.front(), true);
}
int ambiguous_motion_diagnostic() {
  std::vector<Frame> frames;
  for (int displacement : {12, 16, 24})
    frames.push_back(frame(vt::texture(kWidth, kHeight, 7), interior_points(), displacement));
  const auto opts = options(frames.front(), 2);
  const auto outputs = execute(frames, opts);
  for (std::size_t f = 0; f < frames.size(); ++f) {
    const auto tracks = read(outputs[f], opts);
    std::size_t accepted = 0, accurate = 0;
    for (std::size_t p = 0; p < tracks.status.size(); ++p) {
      accepted += tracks.status[p] == 1;
      accurate += tracks.status[p] == 1 &&
                  std::hypot(tracks.xy[2 * p] - frames[f].points[2 * p] - frames[f].dx,
                             tracks.xy[2 * p + 1] - frames[f].points[2 * p + 1]) <= 0.3;
    }
    std::cout << "KLT_AMBIGUOUS_STRESS dx=" << frames[f].dx << " accepted=" << accepted
              << " accurate_0.3px=" << accurate << " supplied=" << tracks.status.size() << '\n';
  }
  std::cout
      << "KLT_DIAGNOSTIC_COMPLETE accuracy_qualification=NOT_ASSERTED canonical_suite=NOT_RUN\n";
  return 0;
}
} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--diagnostic-random-motion")
    return vt::main_test(1, [] { ambiguous_motion_diagnostic(); });
  if (argc == 3 && (std::string_view(argv[1]) == "--sequence-manifest" ||
                    std::string_view(argv[1]) == "--sequence-quality-manifest")) {
    std::cout << std::unitbuf;
    try {
      return vt::sequence_main(argv[2], std::string_view(argv[1]) == "--sequence-quality-manifest");
    } catch (const vt::Failure& error) {
      std::cerr << "[SEQUENCE_FAILURE] " << error.what() << '\n';
      return 1;
    } catch (const std::exception& error) {
      std::cerr << "[SEQUENCE_RUNTIME_FAILURE] " << error.what() << '\n';
      return 2;
    }
  }
  return vt::main_test(argc, [] {
    std::vector<std::string> names{
        "allocation_generations", "shared_offset_views", "unnamed_positional",
        "input_contract",         "small_motion_level2", "small_motion_level3",
        "tile_reload_level0",     "tile_reload_level2",  "borders_level0",
        "borders_level2",         "borders_level3",      "tiny_coarse_image",
        "border_motion",          "quality_residual",    "normalized_texture",
        "convergence_policy",     "inactive_sentinel",   "optional_output_tuples"};
    for (const auto& name : vt::detector_names(false))
      names.push_back("detector_" + name);
    vt::Suite suite("TrackKLT", names);
    suite.run("allocation_generations", [] { allocations(false, false); });
    suite.run("shared_offset_views", [] { allocations(true, false); });
    suite.run("unnamed_positional", [] { allocations(false, true); });
    suite.run("input_contract", input_contract);
    suite.run("small_motion_level2", [] { motion_controls(2); });
    suite.run("small_motion_level3", [] { motion_controls(3); });
    suite.run("tile_reload_level0", [] { reload(0); });
    suite.run("tile_reload_level2", [] { reload(2); });
    suite.run("borders_level0", [] { borders(0, false); });
    suite.run("borders_level2", [] { borders(2, false); });
    suite.run("borders_level3", [] { borders(3, false); });
    suite.run("tiny_coarse_image", [] { borders(3, true); });
    suite.run("border_motion", border_motion);
    suite.run("quality_residual", quality);
    suite.run("normalized_texture", texture_quality);
    suite.run("convergence_policy", convergence);
    suite.run("inactive_sentinel", inactive_sentinel);
    suite.run("optional_output_tuples", tuple_combinations);
    for (const auto& test : vt::detector_cases(false))
      suite.run("detector_" + test.name, [&] { detector(test); });
    suite.finish();
  });
}
