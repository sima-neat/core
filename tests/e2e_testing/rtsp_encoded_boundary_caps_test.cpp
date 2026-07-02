#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string trim_copy(const std::string& value) {
  const std::size_t start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(start, end - start + 1);
}

std::vector<std::string> split_rtsp_urls(const std::string& value) {
  std::vector<std::string> out;
  std::string current;
  for (char c : value) {
    if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      const std::string trimmed = trim_copy(current);
      if (!trimmed.empty()) {
        out.push_back(trimmed);
      }
      current.clear();
      continue;
    }
    current.push_back(c);
  }
  const std::string trimmed = trim_copy(current);
  if (!trimmed.empty()) {
    out.push_back(trimmed);
  }
  return out;
}

std::string first_rtsp_url_from_env() {
  if (const char* single = std::getenv("SIMANEAT_TEST_RTSP_H264_URL"); single && *single) {
    return trim_copy(single);
  }
  if (const char* many = std::getenv("SIMANEAT_TEST_RTSP_H264_URLS"); many && *many) {
    std::vector<std::string> urls = split_rtsp_urls(many);
    if (!urls.empty()) {
      return urls.front();
    }
  }
  if (const char* single = std::getenv("SIMANEAT_APPS_TEST_RTSP_URL"); single && *single) {
    return trim_copy(single);
  }
  if (const char* many = std::getenv("SIMANEAT_APPS_TEST_RTSP_URLS"); many && *many) {
    std::vector<std::string> urls = split_rtsp_urls(many);
    if (!urls.empty()) {
      return urls.front();
    }
  }
  return {};
}

simaai::neat::Sample pull_one_encoded_sample(simaai::neat::Run& run, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  simaai::neat::PullError err;
  while (std::chrono::steady_clock::now() < deadline) {
    simaai::neat::Sample sample;
    const simaai::neat::PullStatus status = run.pull(250, sample, &err);
    if (status == simaai::neat::PullStatus::Ok) {
      return sample;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      throw std::runtime_error("RTSP encoded source closed before producing a sample");
    }
    std::string message = err.message.empty() ? "RTSP encoded source pull failed" : err.message;
    if (!err.code.empty()) {
      message += " (code=" + err.code + ")";
    }
    throw std::runtime_error(message);
  }
  throw std::runtime_error("RTSP encoded source did not produce a sample before timeout");
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();
    const std::string rtsp_url = first_rtsp_url_from_env();
    if (rtsp_url.empty()) {
      std::cout << "[SKIP] set SIMANEAT_TEST_RTSP_H264_URL or "
                   "SIMANEAT_TEST_RTSP_H264_URLS\n";
      return 77;
    }

    require(simaai::neat::element_exists("rtspsrc"), "missing GStreamer element: rtspsrc");
    require(simaai::neat::element_exists("rtph264depay"),
            "missing GStreamer element: rtph264depay");
    require(simaai::neat::element_exists("h264parse"), "missing GStreamer element: h264parse");
    require(simaai::neat::element_exists("appsrc"), "missing GStreamer element: appsrc");
    require(simaai::neat::element_exists("appsink"), "missing GStreamer element: appsink");

    simaai::neat::Graph source;
    source.add(simaai::neat::nodes::RTSPInput(rtsp_url, /*latency_ms=*/200, /*tcp=*/true));
    source.add(simaai::neat::nodes::H264Depacketize(/*payload_type=*/96,
                                                    /*h264_parse_config_interval=*/1));
    source.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(4)));

    simaai::neat::RunOptions source_opt;
    source_opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;
    simaai::neat::Run source_run = source.build(source_opt);
    simaai::neat::Sample encoded = pull_one_encoded_sample(source_run, /*timeout_ms=*/8000);

    require(encoded.payload_type == simaai::neat::PayloadType::Encoded,
            "RTSP parsed output should be an encoded Sample");
    require_contains(encoded.caps_string, "video/x-h264",
                     "RTSP parsed output should carry H264 caps");
    require(encoded.caps_string.find("video/x-raw,format=ENCODED") == std::string::npos,
            "RTSP parsed output should not carry raw ENCODED caps");
    require(encoded.tensors.size() == 1U, "RTSP parsed output should carry one tensor");
    require(encoded.tensors.front().storage != nullptr, "RTSP parsed tensor missing storage");
    require(encoded.tensors.front().storage->kind == simaai::neat::StorageKind::GstSample,
            "zero-copy RTSP parsed output should preserve the GstSample holder");

    simaai::neat::Graph downstream;
    simaai::neat::InputOptions input_opt;
    input_opt.payload_type = simaai::neat::PayloadType::Auto;
    input_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
    downstream.add(simaai::neat::nodes::Input(input_opt));
    downstream.custom("identity name=encoded_boundary_probe silent=true ! "
                      "fakesink name=encoded_boundary_sink sync=false");

    simaai::neat::RunOptions downstream_opt;
    simaai::neat::Run downstream_run =
        downstream.build(simaai::neat::Sample{encoded}, downstream_opt);
    const std::string pipeline = downstream.last_pipeline();
    require_contains(pipeline, "video/x-h264",
                     "encoded boundary appsrc should preserve upstream H264 caps");
    require(pipeline.find("video/x-raw,format=ENCODED") == std::string::npos,
            "encoded boundary appsrc must not infer raw ENCODED caps");
    require(downstream_run.push(simaai::neat::Sample{encoded}),
            "encoded boundary downstream push failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    require(downstream_run.last_error().empty(),
            "encoded boundary downstream run reported an error");

    std::cout << "[OK] rtsp_encoded_boundary_caps_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
