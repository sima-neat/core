#include <neat.h>

int main() {
  simaai::neat::Session session;
  session.add(simaai::neat::nodes::InputAppSrc());
  session.add(simaai::neat::nodes::OutputAppSink());

  // TODO: construct a representative input tensor.
  // simaai::neat::Tensor input = ...;
  // auto run = session.build(input, simaai::neat::PipelineRunMode::Async);
  //
  // if (!run.push(input)) { /* handle push failure */ }
  //
  // simaai::neat::Sample out;
  // simaai::neat::PullError err;
  // auto st = run.pull(1000, out, &err);
  // switch (st) {
  //   case simaai::neat::PullStatus::Ok: break;
  //   case simaai::neat::PullStatus::Timeout: break;
  //   case simaai::neat::PullStatus::Closed: break;
  //   case simaai::neat::PullStatus::Error: break;
  // }

  return 0;
}
