#pragma once

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <string>

namespace sima_test {

struct GstPipelineRunResult {
  GstStateChangeReturn set_state_result = GST_STATE_CHANGE_FAILURE;
  GstStateChangeReturn wait_state_result = GST_STATE_CHANGE_FAILURE;
  GstState current_state = GST_STATE_NULL;
  GstState pending_state = GST_STATE_VOID_PENDING;
  std::string error;
  std::string debug;
};

inline void require_plugin_or_skip(const char* factory) {
  simaai::neat::gst_init_once();
  if (!simaai::neat::element_exists(factory)) {
    skip_test_exception(std::string("missing plugin: ") + factory);
  }
}

inline void require_not_contains(const std::string& haystack,
                                 const std::string& needle,
                                 const std::string& msg) {
  if (haystack.find(needle) != std::string::npos) {
    throw std::runtime_error(msg + " (found unexpected: " + needle + ")");
  }
}

inline GstPipelineRunResult run_raw_gst_pipeline(
    const std::string& label,
    const std::string& pipeline_desc,
    const simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest* manifest = nullptr,
    GstState target = GST_STATE_READY,
    GstClockTime timeout = 3 * GST_SECOND) {
  using simaai::neat::pipeline_internal::sima::attach_manifest_context;

  simaai::neat::gst_init_once();

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

  if (manifest != nullptr) {
    std::string attach_error;
    if (!attach_manifest_context(pipeline, *manifest, &attach_error)) {
      gst_object_unref(pipeline);
      throw std::runtime_error(label + ": failed to attach manifest context: " + attach_error);
    }
  }

  GstBus* bus = gst_element_get_bus(pipeline);
  if (!bus) {
    gst_object_unref(pipeline);
    throw std::runtime_error(label + ": missing bus");
  }

  GstPipelineRunResult result;
  result.set_state_result = gst_element_set_state(pipeline, target);
  result.wait_state_result =
      gst_element_get_state(pipeline, &result.current_state, &result.pending_state, timeout);

  GstClockTime error_wait = 100 * GST_MSECOND;
  if (result.wait_state_result == GST_STATE_CHANGE_FAILURE ||
      result.set_state_result == GST_STATE_CHANGE_FAILURE) {
    error_wait = timeout;
  }
  GstMessage* msg =
      gst_bus_timed_pop_filtered(bus, error_wait, static_cast<GstMessageType>(GST_MESSAGE_ERROR));
  if (msg) {
    GError* gerr = nullptr;
    gchar* debug = nullptr;
    gst_message_parse_error(msg, &gerr, &debug);
    if (gerr && gerr->message) {
      result.error = gerr->message;
    }
    if (debug && *debug) {
      result.debug = debug;
      if (!result.error.empty()) {
        result.error += " | ";
        result.error += debug;
      } else {
        result.error = debug;
      }
    }
    if (gerr) {
      g_error_free(gerr);
    }
    if (debug) {
      g_free(debug);
    }
    gst_message_unref(msg);
  }

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(pipeline);
  return result;
}

} // namespace sima_test
