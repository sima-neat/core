#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#include "model_archive_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/RunInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/core.hpp>

#include <cstring>
#include <string>

namespace {

cv::Mat make_bgr(int w, int h, uint8_t value = 0x11) {
  cv::Mat img(h, w, CV_8UC3, cv::Scalar(value, value, value));
  if (!img.isContinuous()) {
    img = img.clone();
  }
  return img;
}

simaai::neat::Graph make_basic_video_graph(const simaai::neat::InputOptions& src_opt) {
  simaai::neat::Graph graph;
  graph.add(simaai::neat::nodes::Input(src_opt));
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::Latest()));
  return graph;
}

simaai::neat::RunOptions make_async_realtime_run_options() {
  simaai::neat::RunOptions opt;
  opt.preset = simaai::neat::RunPreset::Realtime;
  opt.queue_depth = 2;
  opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  return opt;
}

simaai::neat::Tensor make_ambiguous_layout_tensor_chw(int w, int h, int c) {
  const std::size_t elems =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * static_cast<std::size_t>(c);
  auto storage = simaai::neat::make_cpu_owned_storage(elems * sizeof(float));
  auto map = storage->map(simaai::neat::MapMode::Write);
  if (map.data && map.size_bytes > 0) {
    std::memset(map.data, 0, map.size_bytes);
  }

  simaai::neat::Tensor t;
  t.storage = storage;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::Unknown;
  t.shape = {c, h, w};
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  return t;
}

bool has_expected_oversize_error(const std::string& s) {
  return (s.find("exceeds effective max") != std::string::npos) ||
         (s.find("input_exceeds_effective_max") != std::string::npos) ||
         (s.find("shape_change_requires_rebuild") != std::string::npos);
}

} // namespace

RUN_TEST("unit_inputstream_dynamic_policy_matrix_test", ([] {
           using namespace simaai::neat;

           // 1) Bounded dynamic: seed < max may grow up to max, then oversize fails.
           {
             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Image;
             src_opt.format = simaai::neat::FormatTag::BGR;
             src_opt.width = 16;
             src_opt.height = 16;
             src_opt.depth = 3;
             src_opt.max_width = 64;
             src_opt.max_height = 64;
             src_opt.max_depth = 3;
             src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;

             Graph graph = make_basic_video_graph(src_opt);
             RunOptions run_opt = make_async_realtime_run_options();
             // Keep byte-guard out of this case so oversize failure is from effective dims.
             run_opt.advanced.max_input_bytes = 1u << 20;

             Run run = graph.build(std::vector<cv::Mat>{make_bgr(16, 16)}, run_opt);
             (void)run.run(std::vector<cv::Mat>{make_bgr(16, 16)}, 1000);
             (void)run.run(std::vector<cv::Mat>{make_bgr(48, 48)}, 1000);
             require(wait_for_reneg(run, 1, 1000),
                     "bounded dynamic: expected renegotiation up to max bounds");

             bool threw = false;
             try {
               (void)run.run(std::vector<cv::Mat>{make_bgr(80, 80)}, 1000);
             } catch (const std::exception& e) {
               threw = true;
               const std::string emsg = e.what();
               const std::string lerr = run.last_error();
               require(has_expected_oversize_error(emsg) || has_expected_oversize_error(lerr),
                       "bounded dynamic: unexpected oversize error text: e.what='" + emsg +
                           "' last_error='" + lerr + "'");
             }
             require(threw, "bounded dynamic: expected oversize input to fail");
           }

           // 2) Elastic mode: default guard applied, growth allowed under guard and blocked above
           // guard.
           {
             sima_test::ScopedEnvVar elastic_mb("SIMA_INPUTSTREAM_ELASTIC_MAX_MB", "1");

             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Image;
             src_opt.format = simaai::neat::FormatTag::BGR;
             src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;

             Graph graph = make_basic_video_graph(src_opt);
             RunOptions run_opt = make_async_realtime_run_options();
             run_opt.advanced.max_input_bytes = 0;

             Run run = graph.build(std::vector<cv::Mat>{make_bgr(64, 64)}, run_opt);
             (void)run.run(std::vector<cv::Mat>{make_bgr(64, 64)}, 1000);
             (void)run.run(std::vector<cv::Mat>{make_bgr(256, 256)}, 1000);
             require(run_internal::input_stats(run).alloc_grows > 0,
                     "elastic mode: expected allocation growth for larger in-bound frame");

             bool threw_guard = false;
             try {
               (void)run.run(std::vector<cv::Mat>{make_bgr(1024, 1024)}, 1000);
             } catch (const std::exception& e) {
               threw_guard = true;
               const std::string emsg = e.what();
               const std::string lerr = run.last_error();
               const bool matched = (emsg.find("max_input_bytes") != std::string::npos) ||
                                    (lerr.find("max_input_bytes") != std::string::npos) ||
                                    has_expected_oversize_error(emsg) ||
                                    has_expected_oversize_error(lerr);
               require(matched, "elastic mode: unexpected guard error text: e.what='" + emsg +
                                    "' last_error='" + lerr + "'");
             }
             require(threw_guard, "elastic mode: expected byte-guard failure for oversized frame");
           }

           // 3) caps_override path: deterministic renegotiation block reason.
           {
             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Image;
             src_opt.format = simaai::neat::FormatTag::BGR;
             src_opt.caps_override = "video/x-raw,format=BGR,width=16,height=16";
             src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;

             Graph graph = make_basic_video_graph(src_opt);
             RunOptions run_opt = make_async_realtime_run_options();
             run_opt.advanced.max_input_bytes = 0;

             Run run = graph.build(std::vector<cv::Mat>{make_bgr(16, 16)}, run_opt);
             (void)run.run(std::vector<cv::Mat>{make_bgr(16, 16)}, 1000);

             bool threw_caps_override = false;
             try {
               (void)run.run(std::vector<cv::Mat>{make_bgr(32, 32)}, 1000);
             } catch (const std::exception& e) {
               threw_caps_override = true;
               const std::string emsg = e.what();
               const std::string lerr = run.last_error();
               const bool matched =
                   (emsg.find("caps_override_blocks_renegotiation") != std::string::npos) ||
                   (lerr.find("caps_override_blocks_renegotiation") != std::string::npos) ||
                   (emsg.find("shape_change_requires_rebuild") != std::string::npos) ||
                   (lerr.find("shape_change_requires_rebuild") != std::string::npos);
               require(matched, "caps_override: unexpected block reason text: e.what='" + emsg +
                                    "' last_error='" + lerr + "'");
             }
             require(threw_caps_override,
                     "caps_override: expected renegotiation block when geometry changes");
           }

           // 4) Validation parity: build-time and push-time checks use effective-max semantics.
           {
             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Image;
             src_opt.format = simaai::neat::FormatTag::BGR;
             src_opt.width = 16;
             src_opt.height = 16;
             src_opt.depth = 3;
             src_opt.max_width = 16;
             src_opt.max_height = 16;
             src_opt.max_depth = 3;
             src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;

             Graph graph = make_basic_video_graph(src_opt);
             RunOptions run_opt = make_async_realtime_run_options();
             // Keep byte-guard out of this parity case; we want effective-max checks in both paths.
             run_opt.advanced.max_input_bytes = 1u << 20;
             cv::Mat oversized = make_bgr(24, 16);

             bool build_threw = false;
             std::string build_err;
             try {
               (void)graph.build(std::vector<cv::Mat>{oversized}, run_opt);
             } catch (const std::exception& e) {
               build_threw = true;
               build_err = e.what();
             }
             require(build_threw, "parity: expected build(input) oversize validation failure");
             require(has_expected_oversize_error(build_err),
                     "parity: build(input) unexpected oversize validation message: " + build_err);

             Run run = graph.build(std::vector<cv::Mat>{make_bgr(16, 16)}, run_opt);
             (void)run.run(std::vector<cv::Mat>{make_bgr(16, 16)}, 1000);

             bool push_threw = false;
             std::string push_err;
             try {
               (void)run.run(std::vector<cv::Mat>{oversized}, 1000);
             } catch (const std::exception& e) {
               push_threw = true;
               push_err = e.what();
             }
             require(push_threw, "parity: expected push-path oversize validation failure");
             const std::string push_last_err = run.last_error();
             require(has_expected_oversize_error(push_err) ||
                         has_expected_oversize_error(push_last_err),
                     "parity: push path unexpected oversize validation message: e.what='" +
                         push_err + "' last_error='" + push_last_err + "'");
           }

           // 5) Generic tensors without an explicit HWC/CHW/HW hint should still build.
           {
             InputOptions src_opt;
             src_opt.payload_type = simaai::neat::PayloadType::Tensor;
             src_opt.format = simaai::neat::FormatTag::FP32;
             src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;

             Graph graph;
             graph.add(nodes::Input(src_opt));
             graph.add(nodes::Output(OutputOptions::Latest()));

             RunOptions run_opt = make_async_realtime_run_options();
             Tensor ambiguous = make_ambiguous_layout_tensor_chw(16, 16, 3);

             try {
               (void)graph.build(TensorList{ambiguous}, run_opt);
             } catch (const std::exception& e) {
               require(false, std::string("generic tensor build should not fail: ") + e.what());
             }
           }
         }));
