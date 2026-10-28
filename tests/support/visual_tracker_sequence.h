#pragma once

#include "visual_frontend_test_utils.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace visual_test {
// Diagnostic-only mode of the same public kernel executable. Input artifacts
// are staged and hash-verified by the caller; no network/download/hidden assets.
// Manifest:
// NEAT_KLT_SEQUENCE_V1
// W H POINTS LEVEL HALF_WINDOW ITERATIONS ERROR_THRESHOLD MIN_EIG STRICT_CONVERGENCE
// pair_id previous.gray current.gray seeds.txt
// ... paths are relative to the manifest directory; .gray is exact raw UInt8,
// seeds.txt contains exactly 2*POINTS integer coordinates.
inline Image sequence_image(const std::filesystem::path& path, std::size_t bytes) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "sequence image unavailable: " + path.string());
  Image result(bytes);
  input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
  require(input.gcount() == static_cast<std::streamsize>(bytes) && input.peek() == EOF,
          "sequence image byte count mismatch: " + path.string());
  return result;
}
template <class Options>
bool sequence_quality(Options& options, bool requested, float error, float eigen, bool strict) {
  if constexpr (requires(Options o) {
                  o.publish_error;
                  o.error_threshold;
                  o.min_eig_threshold;
                  o.require_convergence;
                }) {
    options.publish_error = requested;
    options.error_threshold = error;
    options.min_eig_threshold = eigen;
    options.require_convergence = strict;
    return requested;
  } else {
    require(!requested && error == -1.F && eigen == 0.F && !strict,
            "selected baseline API does not support quality mode");
    return false;
  }
}
inline int sequence_main(const std::filesystem::path& manifest, bool quality) {
  std::ifstream input(manifest);
  require(bool(input), "sequence manifest unavailable");
  std::string magic;
  std::getline(input, magic);
  require(magic == "NEAT_KLT_SEQUENCE_V1", "unsupported sequence manifest version");
  int width = 0, height = 0, count = 0, level = 0, half = 0, iterations = 0, strict = 0;
  float threshold = -1, eigen = 0;
  input >> width >> height >> count >> level >> half >> iterations >> threshold >> eigen >> strict;
  require(bool(input) && width > 0 && width <= 4096 && height > 0 && height <= 4096 && count > 0 &&
              count <= 65536 && (strict == 0 || strict == 1),
          "invalid sequence geometry/configuration");
  require(quality || (threshold == -1.F && eigen == 0.F && strict == 0),
          "quality thresholds require explicit sequence-quality mode");
  struct Pair {
    std::string id;
    Image previous;
    Image current;
    std::vector<std::int32_t> points;
  };
  std::vector<Pair> pairs;
  std::string line;
  std::getline(input, line);
  require(line.find_first_not_of(" \t\r") == std::string::npos,
          "extra sequence configuration fields");
  while (std::getline(input, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    std::istringstream record(line);
    std::string id, previous, current, seed_path, extra;
    require(bool(record >> id >> previous >> current >> seed_path) && !(record >> extra),
            "malformed sequence pair record");
    require(id.find(',') == std::string::npos, "sequence pair ID contains CSV delimiter");
    Pair pair{id,
              sequence_image(manifest.parent_path() / previous, std::size_t(width) * height),
              sequence_image(manifest.parent_path() / current, std::size_t(width) * height),
              {}};
    std::ifstream seeds_file(manifest.parent_path() / seed_path);
    require(bool(seeds_file), "sequence seeds unavailable");
    std::int64_t value = 0;
    while (seeds_file >> value) {
      require(value >= std::numeric_limits<std::int32_t>::min() &&
                  value <= std::numeric_limits<std::int32_t>::max(),
              "sequence seed outside Int32");
      pair.points.push_back(static_cast<std::int32_t>(value));
    }
    require(seeds_file.eof() && pair.points.size() == std::size_t(2 * count),
            "sequence seed count/format mismatch");
    pairs.push_back(std::move(pair));
  }
  require(input.eof() && !pairs.empty(), "empty or malformed sequence manifest");
  neat::TrackKLTOptions options;
  options.width = width;
  options.height = height;
  options.num_points = count;
  options.win_half = half;
  options.max_iters = iterations;
  options.max_level = level;
  options.detect_new_features = 0;
  options.num_buffers = 4;
  const bool has_error = sequence_quality(options, quality, threshold, eigen, strict != 0);
  neat::Graph graph;
  graph.add(neat::nodes::Input(input_options(width, height, 1, true)));
  graph.add(neat::nodes::TrackKLT(options));
  graph.add(neat::nodes::Output());
  auto tensors = [&](const Pair& pair) {
    return neat::TensorList{image(pair.previous, width, height, 1, "prev_image"),
                            image(pair.current, width, height, 1, "cur_image"),
                            seeds(pair.points, count)};
  };
  const auto seed = tensors(pairs.front());
  auto run = graph.build(seed, run_options());
  struct Row {
    std::vector<float> xy, error;
    std::vector<std::int32_t> status;
    double ms;
  };
  std::vector<Row> rows;
  try {
    for (const auto& pair : pairs) {
      const auto inputs = tensors(pair);
      const auto start = std::chrono::steady_clock::now();
      auto output = run.run(inputs, kTimeoutMs);
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
      require(output.size() == (has_error ? 3U : 2U), "sequence output arity");
      Row row{values<float>(output[0], neat::TensorDType::Float32, {1, count, 2}),
              {},
              values<std::int32_t>(output[1], neat::TensorDType::Int32, {1, count, 1}),
              ms};
      if (has_error)
        row.error = values<float>(output[2], neat::TensorDType::Float32, {1, count, 1});
      rows.push_back(std::move(row));
    }
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  std::cout << "EV_RUN_CLOSED sequence_pairs=" << rows.size() << '\n';
  std::cout << "KLT_SEQUENCE_HEADER,pair,point,x0,y0,x1,y1,status,error,run_ms\n";
  std::cout << std::setprecision(9);
  for (std::size_t frame = 0; frame < rows.size(); ++frame) {
    const auto& row = rows[frame];
    for (int p = 0; p < count; ++p) {
      require(row.status[p] == 0 || row.status[p] == 1, "sequence status must be binary");
      std::cout << "KLT_SEQUENCE_POINT," << pairs[frame].id << ',' << p << ','
                << pairs[frame].points[2 * p] << ',' << pairs[frame].points[2 * p + 1] << ','
                << row.xy[2 * p] << ',' << row.xy[2 * p + 1] << ',' << row.status[p] << ',';
      if (has_error)
        std::cout << row.error[p];
      else
        std::cout << "NA";
      std::cout << ',' << row.ms << '\n';
    }
  }
  std::cout << "KLT_SEQUENCE_COMPLETE pairs=" << rows.size() << " quality_mode=" << quality
            << " accuracy_qualification=NOT_ASSERTED canonical_suite=NOT_RUN\n";
  return 0;
}
} // namespace visual_test
