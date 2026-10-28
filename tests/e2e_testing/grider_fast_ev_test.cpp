// Public EV74 regression for Internals #332/#333 and Core #1111.
#include "../support/visual_frontend_test_utils.h"

namespace vt = visual_test;
namespace neat = vt::neat;

int main(int argc, char**) {
  return vt::main_test(argc, [] {
    auto cases = vt::detector_names(false);
    cases.push_back("shared_offset_view");
    vt::Suite suite("GriderFast", cases);
    for (const auto& test : vt::detector_cases(false)) {
      suite.run(test.name, [&] {
        const auto& frame = test.frames.front();
        neat::Graph graph;
        graph.add(neat::nodes::Input(vt::input_options(frame.width, frame.height, test.batch)));
        neat::GriderFastOptions options;
        options.width = frame.width;
        options.height = frame.height;
        options.batch_size = test.batch;
        options.max_features = test.config.capacity;
        options.grid_x = test.config.columns;
        options.grid_y = test.config.rows;
        options.min_px_dist = test.config.distance;
        options.threshold = 30;
        options.num_buffers = 4;
        graph.add(neat::nodes::GriderFast(options));
        graph.add(neat::nodes::Output());
        const auto results = vt::execute(graph, vt::detector_frames(test));
        vt::check_detector_output(test, results, 0, 1);
      });
    }
    suite.run("shared_offset_view", [] {
      const auto fixture = vt::grid_markers();
      neat::Graph graph;
      graph.add(neat::nodes::Input(vt::input_options(fixture.width, fixture.height)));
      neat::GriderFastOptions options;
      options.width = fixture.width;
      options.height = fixture.height;
      options.max_features = 96;
      options.grid_x = 8;
      options.grid_y = 6;
      options.min_px_dist = 0;
      options.threshold = 30;
      options.num_buffers = 4;
      graph.add(neat::nodes::GriderFast(options));
      graph.add(neat::nodes::Output());
      std::vector<neat::Tensor> owners;
      std::vector<vt::Image> bytes;
      std::vector<neat::TensorList> frames;
      for (int generation = 1; generation <= 3; ++generation) {
        const std::size_t offset = generation * 64;
        bytes.emplace_back(offset + fixture.pixels.size() + 64, 0xa5);
        std::copy(fixture.pixels.begin(), fixture.pixels.end(), bytes.back().begin() + offset);
        owners.push_back(neat::Tensor::from_vector(bytes.back(),
                                                   {static_cast<std::int64_t>(bytes.back().size())},
                                                   neat::TensorMemory::EV74));
        auto view = owners.back();
        view.shape = {1, fixture.height, fixture.width};
        view.strides_bytes.clear();
        view.byte_offset = static_cast<std::int64_t>(offset);
        view.layout = neat::TensorLayout::HW;
        view.axis_semantics = {neat::TensorAxisSemantic::N, neat::TensorAxisSemantic::H,
                               neat::TensorAxisSemantic::W};
        vt::name(view, "input_image");
        frames.push_back({std::move(view)});
      }
      const auto outputs = vt::execute(graph, frames);
      const auto expected = vt::selected_markers(fixture, {96, 8, 6, 0});
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        vt::require(outputs[i].size() == 1, "shared-view output arity");
        vt::compare_features(vt::read_features(outputs[i][0], 96, 1)[0], expected);
        vt::require(owners[i].copy_dense_bytes_tight() == bytes[i],
                    "read-only grayscale subview changed carrier/guards");
      }
    });
    suite.finish();
  });
}
