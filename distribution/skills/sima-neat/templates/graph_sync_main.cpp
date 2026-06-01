#include <neat.h>

#include <stdexcept>

int main() {
  simaai::neat::Graph graph;
  graph.add(simaai::neat::nodes::InputAppSrc());
  graph.add(simaai::neat::nodes::OutputAppSink());

  // TODO: construct a representative input tensor for your model/pipeline.
  // simaai::neat::Tensor input = ...;
  // auto run = graph.build(input, simaai::neat::PipelineRunMode::Sync);
  // simaai::neat::Sample out = run.push_and_pull(input);

  return 0;
}
