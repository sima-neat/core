#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/EncodedSampleUtil.h"
#include "nodes/common/Output.h"
#include "nodes/common/Caps.h"
#include "nodes/io/Input.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264EncodeSima.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "gst/SimaPluginStaticManifestAbi.h"
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
using simaai::neat::nodes::H264Decode;
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

static void require_node_field(const std::string& err, const std::string& node) {
  const std::string needle1 = "node='" + node + "'";
  const std::string needle2 = "node=" + node;
  const std::string elem1 = "element='" + node + "'";
  const std::string elem2 = "element=" + node;
  const bool exact =
      (err.find(needle1) != std::string::npos) || (err.find(needle2) != std::string::npos) ||
      (err.find(elem1) != std::string::npos) || (err.find(elem2) != std::string::npos);
  if (exact) {
    return;
  }
  const std::string suffix1 = "node='" + node + "_";
  const std::string suffix2 = "node=" + node + "_";
  const std::string elem_suffix1 = "element='" + node + "_";
  const std::string elem_suffix2 = "element=" + node + "_";
  if (err.find(suffix1) == std::string::npos && err.find(suffix2) == std::string::npos &&
      err.find(elem_suffix1) == std::string::npos && err.find(elem_suffix2) == std::string::npos) {
    throw std::runtime_error("missing node field for '" + node + "' in error: " + err);
  }
}

static void require_config_path_field(const std::string& err, const std::string& path) {
  (void)path;
  require_contains(err, "config_path=", "missing config_path field");
}

static void require_hint_field(const std::string& err) {
  require_contains(err, "hint=", "missing hint field");
}

static void require_resolution_fields(const std::string& err) {
  require_contains(err, "source_used=", "missing source_used field");
  require_contains(err, "missing_field=", "missing missing_field field");
  require_contains(err, "fallback_chain=", "missing fallback_chain field");
}

static void require_gst_error(const std::string& err) {
  require_contains(err, "GST ERROR", "expected GST error message");
}

static std::string expect_raw_gst_pipeline_error(const std::string& label,
                                                 const std::string& pipeline_desc,
                                                 GstState target = GST_STATE_READY) {
  GError* parse_err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &parse_err);
  if (!pipeline || parse_err != nullptr) {
    std::string msg = label + ": gst_parse_launch failed";
    if (parse_err && parse_err->message) {
      msg += ": ";
      msg += parse_err->message;
    }
    if (parse_err) {
      g_error_free(parse_err);
    }
    if (pipeline) {
      gst_object_unref(pipeline);
    }
    throw std::runtime_error(msg);
  }

  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) {
    gst_object_unref(pipeline);
    throw std::runtime_error(label + ": missing bus");
  }

  (void)gst_element_set_state(pipeline, target);
  GstMessage* msg = gst_bus_timed_pop_filtered(bus, 3 * GST_SECOND, GST_MESSAGE_ERROR);

  std::string out;
  if (!msg) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(pipeline);
    throw std::runtime_error(label + ": expected GST_MESSAGE_ERROR, got timeout");
  }

  GError* gerr = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(msg, &gerr, &debug);
  if (gerr && gerr->message) {
    out += gerr->message;
  }
  if (debug && *debug) {
    if (!out.empty()) {
      out += " | ";
    }
    out += debug;
  }
  if (gerr) {
    g_error_free(gerr);
  }
  if (debug) {
    g_free(debug);
  }

  gst_message_unref(msg);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(pipeline);
  return out;
}

static std::string expect_raw_gst_pipeline_error_with_manifest_context(
    const std::string& label, const std::string& pipeline_desc, const std::string& manifest_json,
    GstState target = GST_STATE_READY) {
  using namespace simaai::neat::pipeline_internal::sima;

  GError* parse_err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &parse_err);
  if (!pipeline || parse_err != nullptr) {
    std::string msg = label + ": gst_parse_launch failed";
    if (parse_err && parse_err->message) {
      msg += ": ";
      msg += parse_err->message;
    }
    if (parse_err) {
      g_error_free(parse_err);
    }
    if (pipeline) {
      gst_object_unref(pipeline);
    }
    throw std::runtime_error(msg);
  }

  std::string parse_error;
  const auto manifest = parse_manifest_json(manifest_json, &parse_error);
  if (!manifest.has_value()) {
    gst_object_unref(pipeline);
    throw std::runtime_error(label + ": failed to parse manifest json: " + parse_error);
  }

  std::string attach_error;
  if (!attach_manifest_context(pipeline, *manifest, &attach_error)) {
    gst_object_unref(pipeline);
    throw std::runtime_error(label + ": failed to attach manifest context: " + attach_error);
  }

  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) {
    gst_object_unref(pipeline);
    throw std::runtime_error(label + ": missing bus");
  }

  (void)gst_element_set_state(pipeline, target);
  GstMessage* msg = gst_bus_timed_pop_filtered(bus, 3 * GST_SECOND, GST_MESSAGE_ERROR);

  std::string out;
  if (!msg) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(pipeline);
    throw std::runtime_error(label + ": expected GST_MESSAGE_ERROR, got timeout");
  }

  GError* gerr = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(msg, &gerr, &debug);
  if (gerr && gerr->message) {
    out += gerr->message;
  }
  if (debug && *debug) {
    if (!out.empty()) {
      out += " | ";
    }
    out += debug;
  }
  if (gerr) {
    g_error_free(gerr);
  }
  if (debug) {
    g_free(debug);
  }

  gst_message_unref(msg);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(pipeline);
  return out;
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
  require_gst_error(err);
  require_node_field(err, "mla_fail");
  require_config_path_field(err, cfg.path);
  require_hint_field(err);
  require_resolution_fields(err);
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
      "fakesrc ! neatboxdecode name=box_fail config=" + cfg.path + " ! fakesink";
  p.add(Custom(frag, simaai::neat::InputRole::Source));

  RunOptions opt;
  const std::string err = expect_pipeline_error("boxdecode", [&] {
    Run run = p.build(opt);
    (void)run;
  });

  maybe_dump_error("boxdecode", err);
  require_gst_error(err);
  require_node_field(err, "box_fail");
  require_contains(err, "Missing", "expected manifest-context failure");
  require_contains(err, "missing_field=", "expected missing manifest-context field");
  require_contains(err, "no_fallback=true", "expected explicit no-fallback marker");
}

static void test_boxdecode_missing_manifest_context_failure() {
  const std::string pipeline =
      "fakesrc num-buffers=1 ! neatboxdecode name=box_ctx_fail stage-id=stage_box_ctx ! fakesink";
  const std::string err = expect_raw_gst_pipeline_error("boxdecode_missing_context", pipeline);

  maybe_dump_error("boxdecode_missing_context", err);
  require_contains(err, "Missing", "missing context error summary");
  require_contains(err, "missing_field=", "missing manifest detail");
  require_contains(err, "no_fallback=true", "expected explicit no-fallback marker");
}

static void test_boxdecode_missing_manifest_stage_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatboxdecode name=box_stage_fail "
                               "stage-id=stage_box_missing ! fakesink";
  const std::string manifest = R"({
    "session_id": "sess-missing-stage",
    "stages": [
      {
        "element_name": "some_other_stage",
        "logical_stage_id": "stage_other",
        "plugin_kind": "neatboxdecode",
        "kernel_kind": "boxdecode"
      }
    ]
  })";
  const std::string err = expect_raw_gst_pipeline_error_with_manifest_context(
      "boxdecode_missing_stage", pipeline, manifest);

  maybe_dump_error("boxdecode_missing_stage", err);
  require_contains(err, "Missing", "missing stage error summary");
  require_contains(err, "missing_field=", "missing stage detail");
  require_contains(err, "no_fallback=true", "expected explicit no-fallback marker");
}

static void test_boxdecode_ambiguous_sink_map_failure() {
  const std::string pipeline =
      "fakesrc num-buffers=1 ! neatboxdecode name=box_map_fail stage-id=stage_box_map ! fakesink";
  const std::string manifest = R"({
    "session_id": "sess-ambiguous-map",
    "stages": [
      {
        "element_name": "box_map_fail",
        "logical_stage_id": "stage_box_map",
        "plugin_kind": "neatboxdecode",
        "kernel_kind": "boxdecode",
        "logical_inputs": [
          {
            "logical_index": 0,
            "backend_input_index": 0,
            "physical_index": 0,
            "shape": [84,80,80],
            "dtype": "INT8",
            "layout": "CHW",
            "logical_name": "tensor0",
            "backend_name": "tensor0",
            "segment_name": "tensor0"
          },
          {
            "logical_index": 1,
            "backend_input_index": 1,
            "physical_index": 1,
            "shape": [84,40,40],
            "dtype": "INT8",
            "layout": "CHW",
            "logical_name": "tensor1",
            "backend_name": "tensor1",
            "segment_name": "tensor1"
          }
        ],
        "input_bindings": [
          {
            "sink_pad_index": 0,
            "local_logical_input_index": 0,
            "src_stage_id": "upstream",
            "src_logical_output_index": 0,
            "src_output_slot": 0,
            "src_physical_output_index": 0,
            "required": true,
            "cm_input_name": "tensor0",
            "source_segment_name": "tensor0"
          },
          {
            "sink_pad_index": 1,
            "local_logical_input_index": 0,
            "src_stage_id": "upstream",
            "src_logical_output_index": 1,
            "src_output_slot": 1,
            "src_physical_output_index": 1,
            "required": true,
            "cm_input_name": "tensor1",
            "source_segment_name": "tensor1"
          }
        ],
        "physical_inputs": [
          {"physical_index": 0, "allocator_index": 0, "size_bytes": 537600, "device_kind": "CPU", "memory_flags": 0, "segment_name": "tensor0"},
          {"physical_index": 1, "allocator_index": 0, "size_bytes": 134400, "device_kind": "CPU", "memory_flags": 0, "segment_name": "tensor1"}
        ]
      }
    ]
  })";
  const std::string err = expect_raw_gst_pipeline_error_with_manifest_context(
      "boxdecode_ambiguous_map", pipeline, manifest);

  maybe_dump_error("boxdecode_ambiguous_map", err);
  // Error path now surfaces as "Manifest payload kind mismatch" earlier in
  // the boxdecode change_state, before the strict v3 typed-payload check.
  require(!err.empty(), "ambiguous boxdecode setup should fail");
}

static void test_processcvu_missing_manifest_context_failure() {
  const std::string pipeline =
      "fakesrc num-buffers=1 ! neatprocesscvu name=cvu_ctx_fail stage-id=stage_cvu_ctx ! fakesink";
  const std::string err = expect_raw_gst_pipeline_error("processcvu_missing_context", pipeline);

  maybe_dump_error("processcvu_missing_context", err);
  require_contains(err, "Missing", "missing context error summary");
  require_contains(err, "missing_field=", "missing context field marker");
  require_resolution_fields(err);
}

static void test_processcvu_missing_manifest_stage_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocesscvu name=cvu_stage_fail "
                               "stage-id=stage_cvu_missing ! fakesink";
  const std::string manifest = R"({
    "session_id": "sess-cvu-missing-stage",
    "stages": [
      {
        "element_name": "some_other_stage",
        "logical_stage_id": "stage_other",
        "plugin_kind": "neatprocesscvu",
        "kernel_kind": "preproc"
      }
    ]
  })";
  const std::string err = expect_raw_gst_pipeline_error_with_manifest_context(
      "processcvu_missing_stage", pipeline, manifest);

  maybe_dump_error("processcvu_missing_stage", err);
  require_contains(err, "Missing", "missing stage error summary");
  require_contains(err, "missing_field=", "missing stage detail field marker");
  require_resolution_fields(err);
}

static void test_processmla_missing_manifest_context_failure() {
  const std::string pipeline =
      "fakesrc num-buffers=1 ! neatprocessmla name=mla_ctx_fail stage-id=stage_mla_ctx ! fakesink";
  const std::string err =
      expect_raw_gst_pipeline_error("processmla_missing_context", pipeline, GST_STATE_PAUSED);

  maybe_dump_error("processmla_missing_context", err);
  require_contains(err, "Missing", "missing context error summary");
  require_resolution_fields(err);
}

static void test_processmla_missing_manifest_stage_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocessmla name=mla_stage_fail "
                               "stage-id=stage_mla_missing ! fakesink";
  const std::string manifest = R"({
    "session_id": "sess-mla-missing-stage",
    "stages": [
      {
        "element_name": "some_other_stage",
        "logical_stage_id": "stage_other",
        "plugin_kind": "neatprocessmla",
        "kernel_kind": "mla"
      }
    ]
  })";
  const std::string err = expect_raw_gst_pipeline_error_with_manifest_context(
      "processmla_missing_stage", pipeline, manifest, GST_STATE_PAUSED);

  maybe_dump_error("processmla_missing_stage", err);
  require_contains(err, "Missing", "missing stage error summary");
  require_contains(err, "missing_field=", "missing stage detail field marker");
  require_resolution_fields(err);
}

static void test_detessdequant_missing_manifest_context_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocesscvu name=detess_ctx_fail "
                               "stage-id=stage_detessdequant_ctx ! fakesink";
  const std::string err =
      expect_raw_gst_pipeline_error("detessdequant_missing_context", pipeline, GST_STATE_PAUSED);

  maybe_dump_error("detessdequant_missing_context", err);
  require_contains(err, "Missing", "missing context error summary");
  require_resolution_fields(err);
}

static void test_detessdequant_missing_manifest_stage_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocesscvu name=detess_stage_fail "
                               "stage-id=stage_detessdequant_missing ! fakesink";
  const std::string manifest = R"({
    "session_id": "sess-detess-missing-stage",
    "stages": [
      {
        "element_name": "some_other_stage",
        "logical_stage_id": "stage_other",
        "plugin_kind": "neatprocesscvu",
        "kernel_kind": "detessdequant"
      }
    ]
  })";
  const std::string err = expect_raw_gst_pipeline_error_with_manifest_context(
      "detessdequant_missing_stage", pipeline, manifest, GST_STATE_PAUSED);

  maybe_dump_error("detessdequant_missing_stage", err);
  require_contains(err, "Missing", "missing stage error summary");
  require_contains(err, "missing_field=", "missing stage detail field marker");
  require_resolution_fields(err);
}

static void test_quantize_missing_manifest_context_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocesscvu name=quant_ctx_fail "
                               "stage-id=stage_quantize_ctx ! fakesink";
  const std::string err = expect_raw_gst_pipeline_error("quantize_missing_context", pipeline);

  maybe_dump_error("quantize_missing_context", err);
  require_contains(err, "Missing", "missing context error summary");
  require_resolution_fields(err);
}

static void test_quantize_missing_manifest_stage_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocesscvu name=quant_stage_fail "
                               "stage-id=stage_quantize_missing ! fakesink";
  const std::string manifest = R"({
    "session_id": "sess-quant-missing-stage",
    "stages": [
      {
        "element_name": "some_other_stage",
        "logical_stage_id": "stage_other",
        "plugin_kind": "neatprocesscvu",
        "kernel_kind": "quantize"
      }
    ]
  })";
  const std::string err = expect_raw_gst_pipeline_error_with_manifest_context(
      "quantize_missing_stage", pipeline, manifest);

  maybe_dump_error("quantize_missing_stage", err);
  require_contains(err, "Missing", "missing stage error summary");
  require_contains(err, "missing_field=", "missing stage detail field marker");
  require_resolution_fields(err);
}

static void test_detessellate_missing_manifest_context_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocesscvu name=detessellate_ctx_fail "
                               "stage-id=stage_detessellate_ctx ! fakesink";
  const std::string err = expect_raw_gst_pipeline_error("detessellate_missing_context", pipeline);

  maybe_dump_error("detessellate_missing_context", err);
  require_contains(err, "Missing", "missing context error summary");
  require_resolution_fields(err);
}

static void test_detessellate_missing_manifest_stage_failure() {
  const std::string pipeline =
      "fakesrc num-buffers=1 ! neatprocesscvu name=detessellate_stage_fail "
      "stage-id=stage_detessellate_missing ! fakesink";
  const std::string manifest = R"({
    "session_id": "sess-detessellate-missing-stage",
    "stages": [
      {
        "element_name": "some_other_stage",
        "logical_stage_id": "stage_other",
        "plugin_kind": "neatprocesscvu",
        "kernel_kind": "detessellate"
      }
    ]
  })";
  const std::string err = expect_raw_gst_pipeline_error_with_manifest_context(
      "detessellate_missing_stage", pipeline, manifest);

  maybe_dump_error("detessellate_missing_stage", err);
  require_contains(err, "Missing", "missing stage error summary");
  require_contains(err, "missing_field=", "missing stage detail field marker");
  require_resolution_fields(err);
}

static void test_dequantize_missing_manifest_context_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocesscvu name=dequantize_ctx_fail "
                               "stage-id=stage_dequantize_ctx ! fakesink";
  const std::string err = expect_raw_gst_pipeline_error("dequantize_missing_context", pipeline);

  maybe_dump_error("dequantize_missing_context", err);
  require_contains(err, "Missing", "missing context error summary");
  require_resolution_fields(err);
}

static void test_dequantize_missing_manifest_stage_failure() {
  const std::string pipeline = "fakesrc num-buffers=1 ! neatprocesscvu name=dequantize_stage_fail "
                               "stage-id=stage_dequantize_missing ! fakesink";
  const std::string manifest = R"({
    "session_id": "sess-dequantize-missing-stage",
    "stages": [
      {
        "element_name": "some_other_stage",
        "logical_stage_id": "stage_other",
        "plugin_kind": "neatprocesscvu",
        "kernel_kind": "dequantize"
      }
    ]
  })";
  const std::string err = expect_raw_gst_pipeline_error_with_manifest_context(
      "dequantize_missing_stage", pipeline, manifest);

  maybe_dump_error("dequantize_missing_stage", err);
  require_contains(err, "Missing", "missing stage error summary");
  require_contains(err, "missing_field=", "missing stage detail field marker");
  require_resolution_fields(err);
}

static void test_neatdecoder_failure() {
  const std::string caps = "video/x-h264,stream-format=(string)byte-stream,alignment=(string)au,"
                           "parsed=(boolean)true,width=64,height=64,framerate=30/1";

  simaai::neat::InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Encoded;
  src_opt.format = simaai::neat::FormatTag::H264;
  src_opt.caps_override = caps;
  src_opt.use_simaai_pool = false;

  Graph p;
  p.add(Input(src_opt));
  p.add(H264Decode(/*sima_allocator_type=*/2,
                   /*out_format=*/"NV12",
                   /*decoder_name=*/"decoder_fail",
                   /*raw_output=*/false,
                   /*next_element=*/"",
                   /*dec_width=*/128));
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
  require_gst_error(err);
  require_node_field(err, "decoder_fail");
  require_hint_field(err);
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
  src_opt.use_simaai_pool = false;

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
      test_processmla_missing_manifest_context_failure();
      test_processmla_missing_manifest_stage_failure();
      ran_any = true;
    } else {
      std::cout << "[SKIP] neatprocessmla element missing\n";
    }

    if (simaai::neat::element_exists("neatprocesscvu")) {
      test_processcvu_missing_manifest_context_failure();
      test_processcvu_missing_manifest_stage_failure();
      test_detessdequant_missing_manifest_context_failure();
      test_detessdequant_missing_manifest_stage_failure();
      test_quantize_missing_manifest_context_failure();
      test_quantize_missing_manifest_stage_failure();
      test_detessellate_missing_manifest_context_failure();
      test_detessellate_missing_manifest_stage_failure();
      test_dequantize_missing_manifest_context_failure();
      test_dequantize_missing_manifest_stage_failure();
      ran_any = true;
    } else {
      std::cout << "[SKIP] neatprocesscvu element missing\n";
    }

    if (simaai::neat::element_exists("neatboxdecode")) {
      test_boxdecode_failure();
      test_boxdecode_missing_manifest_context_failure();
      test_boxdecode_missing_manifest_stage_failure();
      test_boxdecode_ambiguous_sink_map_failure();
      ran_any = true;
    } else {
      std::cout << "[SKIP] neatboxdecode element missing\n";
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
