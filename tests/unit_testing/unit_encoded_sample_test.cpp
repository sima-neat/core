#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Session.h"
#include "nodes/io/Input.h"
#include "test_utils.h"

#include <chrono>
#include <thread>
#include <vector>

int main() {
  using namespace simaai::neat;

  Sample sample =
      make_encoded_sample(std::vector<uint8_t>(10, 0xAB), "video/x-h264", 100, 200, 300);
  require(sample.tensors.size() == 1U, "encoded Sample should expose one tensor in the TensorList");
  const auto& enc_tensor = sample.tensors.front();
  require(enc_tensor.semantic.encoded.has_value(), "encoded Sample missing semantic");
  require(enc_tensor.semantic.encoded->codec == simaai::neat::EncodedSpec::Codec::H264,
          "encoded Sample codec mismatch");
  require(enc_tensor.dtype == TensorDType::UInt8, "encoded Sample dtype mismatch");
  require(enc_tensor.shape.size() == 1 && enc_tensor.shape[0] == 10,
          "encoded Sample shape mismatch");
  require(sample.pts_ns == 100 && sample.dts_ns == 200 && sample.duration_ns == 300,
          "encoded Sample timestamps mismatch");

  Session p;
  InputOptions src_opt;
  src_opt.media_type.clear();
  src_opt.use_simaai_pool = false;
  p.add(nodes::Input(src_opt));
  p.custom("fakesink name=encoded_sink sync=false");

  RunOptions run_opt;
  Run run = p.build(SampleList{sample}, RunMode::Async, run_opt);

  require(run.push(SampleList{sample}), "encoded Sample push failed");
  Sample sample2 = make_encoded_sample(std::vector<uint8_t>(7, 0xCD), "video/x-h264");
  require(run.push(SampleList{sample2}), "encoded Sample push2 failed");

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  require(run.last_error().empty(), "unexpected error after encoded pushes");

  Sample bad = make_encoded_sample(std::vector<uint8_t>(5, 0xEF), "video/x-h265");
  require(run.push(SampleList{bad}), "encoded Sample push3 failed");

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!run.last_error().empty())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(!run.last_error().empty(), "expected caps change error");
  require_contains(run.last_error(), "caps change", "caps change error missing detail");
  return 0;
}
