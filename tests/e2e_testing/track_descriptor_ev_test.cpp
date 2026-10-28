// Public EV74 regression for Internals #332 and Core #1111. The independent
// BRIEF oracle checks selected-point/descriptor alignment, not just shape.
#include "../support/visual_frontend_test_utils.h"

namespace vt = visual_test;
namespace neat = vt::neat;

int main(int argc, char**) {
  return vt::main_test(argc, [] {
    auto names = vt::detector_names(true);
    names.insert(names.begin(), "brief_reference_pattern");
    vt::Suite suite("TrackDescriptor", names);
    suite.run("brief_reference_pattern", [] {
      vt::MarkerImage ramp(64, 64);
      for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
          ramp.pixels[y * 64 + x] = std::uint8_t((3 * x + 5 * y) & 255);
      constexpr std::array<std::uint32_t, 8> words{0x62cbb37dU, 0x251f182cU, 0x5a9ae013U,
                                                   0xec45a131U, 0x94a70091U, 0xef443215U,
                                                   0x4f9d5315U, 0x32f8955cU};
      vt::require(vt::brief(ramp, {32, 32, 0}) == words,
                  "independent BRIEF pattern/bit-order literal self-check");
    });
    for (const auto& test : vt::detector_cases(true)) {
      suite.run(test.name, [&] {
        const auto& first = test.frames.front();
        neat::Graph graph;
        graph.add(neat::nodes::Input(vt::input_options(first.width, first.height, test.batch)));
        neat::TrackDescriptorOptions options;
        options.width = first.width;
        options.height = first.height;
        options.batch_size = test.batch;
        options.max_features = test.config.capacity;
        options.grid_x = test.config.columns;
        options.grid_y = test.config.rows;
        options.min_px_dist = test.config.distance;
        options.threshold = 30;
        options.num_buffers = 4;
        graph.add(neat::nodes::TrackDescriptor(options));
        graph.add(neat::nodes::Output());
        const auto results = vt::execute(graph, vt::detector_frames(test));
        vt::check_detector_output(test, results, 0, 2, 15);
        for (std::size_t frame = 0; frame < results.size(); ++frame) {
          const auto descriptors = vt::values<std::uint32_t>(
              results[frame][1], neat::TensorDType::Int32, {test.batch, test.config.capacity, 8});
          for (int batch = 0; batch < test.batch; ++batch) {
            const auto& fixture = test.frames[test.batch == 2 ? batch : frame];
            const auto expected = vt::selected_markers(fixture, test.config, 15);
            for (std::size_t i = 0; i < expected.size(); ++i) {
              const auto golden = vt::brief(fixture, expected[i]);
              for (std::size_t word = 0; word < golden.size(); ++word)
                vt::require(
                    descriptors[(std::size_t(batch) * test.config.capacity + i) * 8 + word] ==
                        golden[word],
                    "BRIEF selected-row mismatch row=" + std::to_string(i) +
                        " word=" + std::to_string(word));
            }
          }
        }
      });
    }
    suite.finish();
  });
}
