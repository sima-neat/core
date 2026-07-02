#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/groups/HttpMjpegDecodedInput.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/Run.h"
#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using simaai::neat::Graph;
using simaai::neat::InputMemoryPolicy;
using simaai::neat::InputOptions;
using simaai::neat::OutputMemory;
using simaai::neat::OutputOptions;
using simaai::neat::PayloadType;
using simaai::neat::PullError;
using simaai::neat::PullStatus;
using simaai::neat::Run;
using simaai::neat::RunOptions;
using simaai::neat::Sample;
using simaai::neat::StorageKind;
using simaai::neat::nodes::groups::HttpMjpegDecodedInput;
using simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions;
using simaai::neat::nodes::groups::RtspCodec;
using simaai::neat::nodes::groups::RtspDecodedInput;
using simaai::neat::nodes::groups::RtspDecodedInputOptions;
using simaai::neat::nodes::groups::RtspEncodedInput;
using simaai::neat::nodes::groups::RtspEncodedInputOptions;

constexpr int kDefaultFrames = 30;
constexpr int kDefaultTimeoutMs = 20000;
constexpr int kMjpegDecodeFps = 30;

enum class CaseKind {
  RtspH264Decoded,
  RtspMjpegEncodedBoundary,
  RtspMjpegDecoded,
  HttpMjpegDecoded,
};

struct TestCase {
  CaseKind kind;
  std::string name;
  std::string singular_env;
  std::string plural_env;
  std::string expected_caps;
};

struct Args {
  CaseKind kind = CaseKind::RtspH264Decoded;
  bool has_case = false;
  int frames = kDefaultFrames;
  int timeout_ms = kDefaultTimeoutMs;
};

std::string trim_copy(const std::string& value) {
  const std::size_t start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(start, end - start + 1);
}

std::vector<std::string> split_urls(const std::string& value) {
  std::vector<std::string> urls;
  std::string current;
  for (const char c : value) {
    if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      const std::string trimmed = trim_copy(current);
      if (!trimmed.empty()) {
        urls.push_back(trimmed);
      }
      current.clear();
      continue;
    }
    current.push_back(c);
  }
  const std::string trimmed = trim_copy(current);
  if (!trimmed.empty()) {
    urls.push_back(trimmed);
  }
  return urls;
}

std::string first_url_from_env(const TestCase& test_case) {
  if (const char* single = std::getenv(test_case.singular_env.c_str()); single && *single) {
    return trim_copy(single);
  }
  if (const char* many = std::getenv(test_case.plural_env.c_str()); many && *many) {
    const std::vector<std::string> urls = split_urls(many);
    if (!urls.empty()) {
      return urls.front();
    }
  }
  return {};
}

bool starts_with(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

TestCase test_case_for(CaseKind kind) {
  switch (kind) {
  case CaseKind::RtspH264Decoded:
    return {kind, "rtsp-h264-decoded", "SIMANEAT_TEST_RTSP_H264_URL",
            "SIMANEAT_TEST_RTSP_H264_URLS", "video/x-raw"};
  case CaseKind::RtspMjpegEncodedBoundary:
    return {kind, "rtsp-mjpeg-encoded-boundary", "SIMANEAT_TEST_RTSP_MJPEG_URL",
            "SIMANEAT_TEST_RTSP_MJPEG_URLS", "image/jpeg"};
  case CaseKind::RtspMjpegDecoded:
    return {kind, "rtsp-mjpeg-decoded", "SIMANEAT_TEST_RTSP_MJPEG_URL",
            "SIMANEAT_TEST_RTSP_MJPEG_URLS", "video/x-raw"};
  case CaseKind::HttpMjpegDecoded:
    return {kind, "http-mjpeg-decoded", "SIMANEAT_TEST_HTTP_MJPEG_URL",
            "SIMANEAT_TEST_HTTP_MJPEG_URLS", "video/x-raw"};
  }
  throw std::runtime_error("unknown test case");
}

std::vector<TestCase> all_test_cases() {
  return {test_case_for(CaseKind::RtspH264Decoded),
          test_case_for(CaseKind::RtspMjpegEncodedBoundary),
          test_case_for(CaseKind::RtspMjpegDecoded), test_case_for(CaseKind::HttpMjpegDecoded)};
}

void replace_all(std::string& value, const std::string& needle, const std::string& replacement) {
  if (needle.empty()) {
    return;
  }
  std::size_t pos = 0;
  while ((pos = value.find(needle, pos)) != std::string::npos) {
    value.replace(pos, needle.size(), replacement);
    pos += replacement.size();
  }
}

std::vector<std::string> configured_urls_from_env(const TestCase& test_case) {
  std::vector<std::string> urls;
  if (const char* single = std::getenv(test_case.singular_env.c_str()); single && *single) {
    const std::string trimmed = trim_copy(single);
    if (!trimmed.empty()) {
      urls.push_back(trimmed);
    }
  }
  if (const char* many = std::getenv(test_case.plural_env.c_str()); many && *many) {
    const std::vector<std::string> split = split_urls(many);
    urls.insert(urls.end(), split.begin(), split.end());
  }
  return urls;
}

std::string redact_configured_stream_urls(std::string value) {
  for (const auto& test_case : all_test_cases()) {
    for (const auto& url : configured_urls_from_env(test_case)) {
      replace_all(value, url, "<redacted-stream-url>");
    }
  }
  return value;
}

void suppress_url_bearing_runtime_diagnostics() {
  ::setenv("SIMA_PULL_TIMEOUT_DIAG", "0", 1);
  ::setenv("SIMA_ASYNC_TPUT_DIAG", "0", 1);
  ::setenv("SIMA_GRAPH_DIAG_ON_STOP", "0", 1);
  ::setenv("SIMA_GRAPH_PIPELINE_DIAG_SUMMARY", "0", 1);
  ::setenv("SIMA_INPUTSTREAM_DEBUG", "0", 1);
  ::setenv("SIMA_GRAPH_DEBUG", "0", 1);
  ::setenv("SIMA_PIPELINE_STRING_DEBUG", "0", 1);
}

CaseKind parse_case_kind(const std::string& value) {
  if (value == "rtsp-h264-decoded") {
    return CaseKind::RtspH264Decoded;
  }
  if (value == "rtsp-mjpeg-encoded-boundary") {
    return CaseKind::RtspMjpegEncodedBoundary;
  }
  if (value == "rtsp-mjpeg-decoded") {
    return CaseKind::RtspMjpegDecoded;
  }
  if (value == "http-mjpeg-decoded") {
    return CaseKind::HttpMjpegDecoded;
  }
  throw std::runtime_error("unknown --case: " + value);
}

void print_help(const char* exe) {
  std::cout << "Usage: " << exe << " [--case NAME] [--frames N] [--timeout-ms N]\n"
            << "Cases:\n"
            << "  rtsp-h264-decoded\n"
            << "  rtsp-mjpeg-encoded-boundary\n"
            << "  rtsp-mjpeg-decoded\n"
            << "  http-mjpeg-decoded\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      std::exit(0);
    }
    if (arg == "--case") {
      args.kind = parse_case_kind(require_value("--case"));
      args.has_case = true;
    } else if (arg == "--frames") {
      args.frames = std::stoi(require_value("--frames"));
    } else if (arg == "--timeout-ms") {
      args.timeout_ms = std::stoi(require_value("--timeout-ms"));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (args.frames <= 0) {
    throw std::runtime_error("--frames must be positive");
  }
  if (args.timeout_ms <= 0) {
    throw std::runtime_error("--timeout-ms must be positive");
  }
  return args;
}

RunOptions make_run_options(OutputMemory output_memory) {
  RunOptions options;
  options.output_memory = output_memory;
  return options;
}

Sample pull_or_throw(Run& run, const std::string& output_name, int timeout_ms,
                     const std::string& context) {
  Sample sample;
  PullError error;
  const PullStatus status = run.pull(output_name, timeout_ms, sample, &error);
  if (status == PullStatus::Ok) {
    return sample;
  }
  if (status == PullStatus::Timeout) {
    throw std::runtime_error(context + ": timed out");
  }
  if (status == PullStatus::Closed) {
    throw std::runtime_error(context + ": closed before producing output");
  }

  std::string message = error.message.empty() ? context + ": pull failed" : error.message;
  if (!error.code.empty()) {
    message += " (code=" + error.code + ")";
  }
  throw std::runtime_error(redact_configured_stream_urls(message));
}

Graph make_source_graph(const TestCase& test_case, const std::string& url) {
  Graph graph(test_case.name);
  switch (test_case.kind) {
  case CaseKind::RtspH264Decoded: {
    RtspDecodedInputOptions options;
    options.url = url;
    options.codec = RtspCodec::H264;
    graph.add(RtspDecodedInput(options));
    break;
  }
  case CaseKind::RtspMjpegEncodedBoundary: {
    RtspEncodedInputOptions options;
    options.url = url;
    options.codec = RtspCodec::MJPEG;
    graph.add(RtspEncodedInput(options));
    break;
  }
  case CaseKind::RtspMjpegDecoded: {
    RtspDecodedInputOptions options;
    options.url = url;
    options.codec = RtspCodec::MJPEG;
    options.dec_fps = kMjpegDecodeFps;
    graph.add(RtspDecodedInput(options));
    break;
  }
  case CaseKind::HttpMjpegDecoded: {
    require(starts_with(url, "http://") || starts_with(url, "https://"),
            "HTTP MJPEG runtime test requires an http:// or https:// URL");
    HttpMjpegDecodedInputOptions options;
    options.url = url;
    options.dec_fps = kMjpegDecodeFps;
    if (starts_with(url, "https://")) {
      options.ssl_strict = false;
    }
    graph.add(HttpMjpegDecodedInput(options));
    break;
  }
  }

  graph.add(simaai::neat::nodes::Output("source", OutputOptions::EveryFrame(8)));
  return graph;
}

void require_sample_contract(const TestCase& test_case, const Sample& sample) {
  const std::string metadata =
      sample.caps_string + " " + sample.media_type + " " + sample.payload_tag + " " + sample.format;
  require_contains(metadata, test_case.expected_caps, test_case.name + ": sample caps mismatch");

  switch (test_case.kind) {
  case CaseKind::RtspMjpegEncodedBoundary:
    require(simaai::neat::sample_payload_type(sample) == PayloadType::Encoded,
            test_case.name + ": expected encoded payload");
    require(sample.caps_string.find("video/x-raw,format=ENCODED") == std::string::npos,
            test_case.name + ": encoded output must not use raw ENCODED caps");
    require(sample.tensors.size() == 1U, test_case.name + ": expected one encoded tensor");
    require(sample.tensors.front().storage != nullptr,
            test_case.name + ": encoded tensor missing storage");
    require(sample.tensors.front().storage->kind == StorageKind::GstSample,
            test_case.name + ": zero-copy encoded output should preserve GstSample storage");
    break;

  case CaseKind::RtspH264Decoded:
  case CaseKind::RtspMjpegDecoded:
  case CaseKind::HttpMjpegDecoded: {
    require(simaai::neat::sample_payload_type(sample) == PayloadType::Image,
            test_case.name + ": expected raw image payload");
    const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
    require(tensors.size() == 1U, test_case.name + ": expected one decoded tensor");
    const auto& tensor = tensors.front();
    require(!tensor.shape.empty(), test_case.name + ": decoded tensor shape is empty");
    require(tensor.storage != nullptr, test_case.name + ": decoded tensor missing storage");
    break;
  }
  }
}

void run_encoded_boundary(const TestCase& test_case, const std::string& url, int frames,
                          int timeout_ms) {
  Graph source_graph = make_source_graph(test_case, url);
  Run source_run = source_graph.build(make_run_options(OutputMemory::ZeroCopy));
  Sample first_sample =
      pull_or_throw(source_run, "source", timeout_ms, test_case.name + ": source pull");
  require_sample_contract(test_case, first_sample);

  InputOptions input_options;
  input_options.payload_type = simaai::neat::sample_payload_type(first_sample);
  input_options.caps_override = first_sample.caps_string;
  input_options.is_live = true;
  input_options.block = true;
  input_options.pool_max_buffers = 8;
  input_options.memory_policy = InputMemoryPolicy::SystemMemory;

  Graph boundary_graph("rtsp-mjpeg-encoded-boundary-pass-through");
  boundary_graph.add(simaai::neat::nodes::Input("encoded", input_options));
  boundary_graph.add(simaai::neat::nodes::Output("encoded", OutputOptions::EveryFrame(8)));
  Run boundary_run =
      boundary_graph.build(Sample{first_sample}, make_run_options(OutputMemory::ZeroCopy));

  int source_pull = 1;
  int boundary_push = 0;
  int boundary_pull = 0;
  auto push_and_pull = [&](const Sample& sample) {
    require(boundary_run.push("encoded", sample), test_case.name + ": boundary push failed");
    ++boundary_push;
    Sample out =
        pull_or_throw(boundary_run, "encoded", timeout_ms, test_case.name + ": boundary pull");
    ++boundary_pull;
    require_sample_contract(test_case, out);
  };

  push_and_pull(first_sample);
  while (source_pull < frames) {
    Sample sample =
        pull_or_throw(source_run, "source", timeout_ms, test_case.name + ": source pull");
    ++source_pull;
    require_sample_contract(test_case, sample);
    push_and_pull(sample);
  }

  boundary_run.close();
  source_run.close();
  std::cout << "[OK] " << test_case.name << " frames=" << frames << " source_pull=" << source_pull
            << " boundary_push=" << boundary_push << " boundary_pull=" << boundary_pull << "\n";
}

void run_decoded_source(const TestCase& test_case, const std::string& url, int frames,
                        int timeout_ms) {
  Graph graph = make_source_graph(test_case, url);
  Run run = graph.build(make_run_options(OutputMemory::Owned));
  int pulled = 0;
  while (pulled < frames) {
    Sample sample = pull_or_throw(run, "source", timeout_ms, test_case.name + ": source pull");
    require_sample_contract(test_case, sample);
    ++pulled;
  }
  run.close();
  std::cout << "[OK] " << test_case.name << " frames=" << pulled << "\n";
}

int skip_missing_env(const TestCase& test_case) {
  std::cout << "[SKIP] set " << test_case.singular_env << " or " << test_case.plural_env
            << " to run " << test_case.name << "\n";
  return 77;
}

void run_test_case(const TestCase& test_case, const std::string& url, const Args& args) {
  if (test_case.kind == CaseKind::RtspMjpegEncodedBoundary) {
    run_encoded_boundary(test_case, url, args.frames, args.timeout_ms);
  } else {
    run_decoded_source(test_case, url, args.frames, args.timeout_ms);
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    suppress_url_bearing_runtime_diagnostics();
    simaai::neat::gst_init_once();
    const Args args = parse_args(argc, argv);

    if (args.has_case) {
      const TestCase test_case = test_case_for(args.kind);
      const std::string url = first_url_from_env(test_case);
      if (url.empty()) {
        return skip_missing_env(test_case);
      }
      run_test_case(test_case, url, args);
      return 0;
    }

    const std::vector<TestCase> test_cases = all_test_cases();
    bool ran_any_case = false;
    for (const auto& test_case : test_cases) {
      const std::string url = first_url_from_env(test_case);
      if (url.empty()) {
        std::cout << "[SKIP] set " << test_case.singular_env << " or " << test_case.plural_env
                  << " to run " << test_case.name << "\n";
        continue;
      }
      run_test_case(test_case, url, args);
      ran_any_case = true;
    }
    return ran_any_case ? 0 : 77;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << redact_configured_stream_urls(e.what()) << "\n";
    return 1;
  }
}
