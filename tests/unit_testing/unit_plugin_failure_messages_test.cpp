#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/EncodedSampleUtil.h"
#include "nodes/common/Output.h"
#include "nodes/common/Caps.h"
#include "nodes/io/Input.h"
#include "nodes/sima/H264EncodeSima.h"
#include "nodes/sima/SimaDecode.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "test_utils.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gst/gst.h>

#include <unistd.h>

static void maybe_dump_error(const std::string& label, const std::string& err) {
  const char* env = std::getenv("SIMA_TEST_DUMP_ERRORS");
  if (!env || !*env || std::string(env) == "0")
    return;
  std::cout << "[ERR] " << label << ": " << err << "\n";
}

using simaai::neat::Graph;
using simaai::neat::NeatError;
using simaai::neat::Run;
using simaai::neat::RunOptions;
using simaai::neat::Sample;
using simaai::neat::nodes::Custom;
using simaai::neat::nodes::H264EncodeSima;
using simaai::neat::nodes::Input;
using simaai::neat::nodes::Output;

struct TempFile {
  std::string path;
  explicit TempFile(std::string p) : path(std::move(p)) {}
  ~TempFile() {
    if (!path.empty()) {
      std::remove(path.c_str());
    }
  }
};

static TempFile write_temp(const std::string& prefix, const std::string& contents) {
  const std::string path = "/tmp/" + prefix + "_" + std::to_string(::getpid()) + ".json";
  std::ofstream out(path);
  require(out.is_open(), "failed to create temp config: " + path);
  out << contents;
  out.close();
  return TempFile(path);
}

static std::string expect_pipeline_error(const std::string& label,
                                         const std::function<void()>& fn) {
  try {
    fn();
  } catch (const NeatError& e) {
    return std::string(e.what());
  } catch (const std::exception& e) {
    throw std::runtime_error(label + ": expected NeatError, got: " + e.what());
  }
  throw std::runtime_error(label + ": expected NeatError, got success");
}

static void require_configuration_error(const std::string& err, const std::string& stage) {
  require_contains(err, "[io.parse]", "expected the configuration error code");
  require_contains(err, "The reported pipeline stage configuration is invalid.",
                   "expected a plain-language configuration failure title");
  require_contains(err, "Stage: " + stage, "expected the failing stage");
  require_contains(err, "How to fix:", "expected actionable user guidance");
  require_contains(err, "Diagnostic ID: gstreamer.configuration_invalid",
                   "expected a stable diagnostic id");
  require(err.find("GST ERROR") == std::string::npos,
          "production error should not expose a raw GST ERROR dump");
  require(err.find("config_path=") == std::string::npos,
          "production error should not expose raw plugin key/value context");
  require(err.find("hint=") == std::string::npos,
          "production error should not expose raw plugin hint fields");
}

static void require_caps_production_error(const std::string& err, const std::string& stage) {
  require_contains(err, "[misconfig.media_caps]",
                   "expected the media-capability configuration code");
  require_contains(err, "Two connected pipeline stages require incompatible media caps.",
                   "expected a plain-language caps failure title");
  require_contains(err, "Stage: " + stage, "expected the failing stage");
  require_contains(err, "How to fix:", "expected actionable user guidance");
  require_contains(err, "Diagnostic ID: gstreamer.caps_incompatible",
                   "expected a stable diagnostic id");
  require(err.find("GST ERROR") == std::string::npos,
          "production error should not expose a raw GST ERROR dump");
}

static void test_processmla_failure() {
  TempFile cfg = write_temp("sima_mla_bad", "{\n"
                                            "  \"caps\": {\n"
                                            "    \"sink_pads\": [],\n"
                                            "    \"src_pads\": []\n"
                                            "  },\n"
                                            "  \"simaai__params\": {\n"
                                            "    \"batch_size\": 1,\n"
                                            "    \"outputs\": [\n"
                                            "      {\"name\": \"out0\", \"size\": 4}\n"
                                            "    ]\n"
                                            "  }\n"
                                            "}\n");
  Graph p;
  const std::string frag =
      "fakesrc ! neatprocessmla name=mla_fail config=" + cfg.path + " num-buffers=4 ! fakesink";
  p.add(Custom(frag, simaai::neat::InputRole::Source));

  RunOptions opt;
  const std::string err = expect_pipeline_error("processmla", [&] {
    Run run = p.build(opt);
    (void)run;
  });

  maybe_dump_error("processmla", err);
  require_configuration_error(err, "mla_fail");
}

static void test_boxdecode_failure() {
  TempFile cfg = write_temp("sima_box_bad", "{\n"
                                            "  \"caps\": {\n"
                                            "    \"sink_pads\": [],\n"
                                            "    \"src_pads\": []\n"
                                            "  },\n"
                                            "  \"memory\": {\n"
                                            "    \"next_cpu\": 0\n"
                                            "  },\n"
                                            "  \"system\": {\n"
                                            "    \"out_buf_queue\": 1\n"
                                            "  },\n"
                                            "  \"buffers\": {\n"
                                            "    \"input\": [\n"
                                            "      {\"name\": \"input0\", \"size\": 1}\n"
                                            "    ],\n"
                                            "    \"output\": {\n"
                                            "      \"size\": 1\n"
                                            "    }\n"
                                            "  }\n"
                                            "}\n");
  Graph p;
  const std::string frag =
      "fakesrc ! neatobjectdecode name=box_fail config=" + cfg.path + " ! fakesink";
  p.add(Custom(frag, simaai::neat::InputRole::Source));

  RunOptions opt;
  const std::string err = expect_pipeline_error("boxdecode", [&] {
    Run run = p.build(opt);
    (void)run;
  });

  maybe_dump_error("boxdecode", err);
  require_configuration_error(err, "box_fail");
}

// Kernel identity and parameters are admitted into a retained Runtime stage,
// not reparsed by an operation-specific element from a legacy manifest.

static void test_neatdecoder_failure() {
  const std::string caps = "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au,"
                           "parsed=(boolean)true,width=64,height=64,framerate=30/1";

  simaai::neat::InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Encoded;
  src_opt.format = simaai::neat::FormatTag::H264;
  src_opt.caps_override = caps;
  src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;

  Graph p;
  p.add(Input(src_opt));
  simaai::neat::SimaDecodeOptions dec;
  dec.type = simaai::neat::SimaDecodeType::H264;
  dec.sima_allocator_type = 2;
  dec.out_format = simaai::neat::FormatTag::NV12;
  dec.decoder_name = "decoder_fail";
  dec.raw_output = false;
  dec.dec_width = 128;
  p.add(simaai::neat::nodes::SimaDecode(dec));
  p.add(Output(simaai::neat::OutputOptions::Latest()));

  Sample sample =
      simaai::neat::make_encoded_sample(std::vector<uint8_t>{0x00, 0x00, 0x00, 0x01}, caps);

  RunOptions opt;
  opt.queue_depth = 1;

  const std::string err = expect_pipeline_error("neatdecoder", [&] {
    Run run = p.build(simaai::neat::Sample{sample}, opt);
    (void)run;
  });

  maybe_dump_error("neatdecoder", err);
  require_caps_production_error(err, "decoder_fail");
}

static void test_neatencoder_dynamic_caps() {
  setenv("SIMA_NEATENCODER_ALLOW_CAPS_CHANGE", "1", 1);
  const int in_w = 32;
  const int in_h = 32;

  simaai::neat::InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Image;
  src_opt.format = simaai::neat::FormatTag::NV12;
  src_opt.width = in_w;
  src_opt.height = in_h;
  src_opt.fps_n = 30;
  src_opt.fps_d = 1;
  src_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;

  Graph p;
  p.add(Input(src_opt));
  p.add(H264EncodeSima(/*w=*/64, /*h=*/64, /*fps=*/30));
  p.add(Output(simaai::neat::OutputOptions::Latest()));

  simaai::neat::Tensor input = make_nv12_tensor(in_w, in_h);

  RunOptions opt;
  opt.queue_depth = 1;

  Run run = p.build(simaai::neat::TensorList{input}, opt);
  (void)run;
}

int main() {
  try {
    simaai::neat::gst_init_once();

    bool ran_any = false;

    if (simaai::neat::element_exists("neatprocessmla")) {
      test_processmla_failure();
      ran_any = true;
    } else {
      std::cout << "[SKIP] neatprocessmla element missing\n";
    }

    require(simaai::neat::element_exists("neatkernelrunner"),
            "Core startup must register the Runtime kernel element");
    ran_any = true;

    if (simaai::neat::element_exists("neatobjectdecode")) {
      test_boxdecode_failure();
      ran_any = true;
    } else {
      std::cout << "[SKIP] neatobjectdecode element missing\n";
    }

    if (simaai::neat::element_exists("neatdecoder")) {
      test_neatdecoder_failure();
      ran_any = true;
    } else {
      std::cout << "[SKIP] neatdecoder element missing\n";
    }

    if (simaai::neat::element_exists("neatencoder")) {
      test_neatencoder_dynamic_caps();
      ran_any = true;
    } else {
      std::cout << "[SKIP] neatencoder element missing\n";
    }

    if (!ran_any) {
      return fail_test("no target plugins available for failure-message tests");
    }

    std::cout << "[OK] unit_plugin_failure_messages_test passed\n";
    return 0;
  } catch (const std::runtime_error& e) {
    return fail_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
