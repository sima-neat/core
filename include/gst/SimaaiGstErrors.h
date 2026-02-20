#pragma once

#include <gst/gst.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

struct SimaaiGstErrorContext {
  const char* plugin = nullptr;
  const char* node = nullptr;
  const char* config_path = nullptr;
  const char* model_path = nullptr;
  int graph_id = -1;
  std::int64_t frame_id = -1;
  const char* stream_id = nullptr;
  const char* input_caps = nullptr;
  const char* output_caps = nullptr;
  const char* input_dims = nullptr;
  const char* output_dims = nullptr;
  const char* allocator = nullptr;
  const char* dispatcher_err = nullptr;
  const char* hint = nullptr;
  const char* manifest_version = nullptr;
  const char* stage_key = nullptr;
  const char* source_used = nullptr;
  const char* missing_field = nullptr;
  const char* fallback_chain = nullptr;
};

inline const char* simaai_gst_safe(const char* s) {
  return s ? s : "";
}

inline void simaai_gst_append_kv(std::ostringstream& ss, const char* key, const char* value) {
  if (!value || !*value)
    return;
  if (ss.tellp() > 0)
    ss << ' ';
  ss << key << "='" << value << "'";
}

inline void simaai_gst_append_kv_required(std::ostringstream& ss, const char* key,
                                          const char* value) {
  if (ss.tellp() > 0)
    ss << ' ';
  ss << key << "='" << simaai_gst_safe(value) << "'";
}

inline void simaai_gst_append_kv_int(std::ostringstream& ss, const char* key, std::int64_t value) {
  if (ss.tellp() > 0)
    ss << ' ';
  ss << key << '=' << value;
}

inline std::string simaai_gst_format_context(const SimaaiGstErrorContext& ctx) {
  std::ostringstream ss;
  simaai_gst_append_kv(ss, "plugin", ctx.plugin);
  simaai_gst_append_kv(ss, "node", ctx.node);
  simaai_gst_append_kv(ss, "config_path", ctx.config_path);
  simaai_gst_append_kv(ss, "model_path", ctx.model_path);
  if (ctx.graph_id >= 0)
    simaai_gst_append_kv_int(ss, "graph_id", ctx.graph_id);
  if (ctx.frame_id >= 0)
    simaai_gst_append_kv_int(ss, "frame_id", ctx.frame_id);
  simaai_gst_append_kv(ss, "stream_id", ctx.stream_id);
  simaai_gst_append_kv(ss, "input_caps", ctx.input_caps);
  simaai_gst_append_kv(ss, "output_caps", ctx.output_caps);
  simaai_gst_append_kv(ss, "input_dims", ctx.input_dims);
  simaai_gst_append_kv(ss, "output_dims", ctx.output_dims);
  simaai_gst_append_kv(ss, "allocator", ctx.allocator);
  simaai_gst_append_kv(ss, "dispatcher_err", ctx.dispatcher_err);
  simaai_gst_append_kv(ss, "hint", ctx.hint);
  simaai_gst_append_kv(ss, "manifest_version", ctx.manifest_version);
  simaai_gst_append_kv(ss, "stage_key", ctx.stage_key);
  simaai_gst_append_kv_required(
      ss, "source_used",
      (ctx.source_used && *ctx.source_used) ? ctx.source_used : "infer->caps/properties/context");
  simaai_gst_append_kv_required(ss, "missing_field",
                                (ctx.missing_field && *ctx.missing_field) ? ctx.missing_field
                                                                           : "none");
  simaai_gst_append_kv_required(
      ss, "fallback_chain",
      (ctx.fallback_chain && *ctx.fallback_chain) ? ctx.fallback_chain
                                                  : "infer->caps/properties/context->json");
  return ss.str();
}

inline std::string simaai_gst_format_detail(const SimaaiGstErrorContext& ctx, const char* detail) {
  std::ostringstream ss;
  std::string base = simaai_gst_format_context(ctx);
  if (!base.empty())
    ss << base;
  if (detail && *detail) {
    simaai_gst_append_kv(ss, "detail", detail);
  }
  return ss.str();
}

inline std::string simaai_gst_vformat(const char* fmt, va_list ap) {
  if (!fmt || !*fmt)
    return {};
  va_list ap_copy;
  va_copy(ap_copy, ap);
  const int needed = std::vsnprintf(nullptr, 0, fmt, ap_copy);
  va_end(ap_copy);
  if (needed <= 0)
    return {};
  std::vector<char> buf(static_cast<size_t>(needed) + 1, '\0');
  std::vsnprintf(buf.data(), buf.size(), fmt, ap);
  return std::string(buf.data());
}

inline std::string simaai_gst_format_detailf(const SimaaiGstErrorContext& ctx, const char* fmt,
                                             ...) {
  va_list ap;
  va_start(ap, fmt);
  std::string detail = simaai_gst_vformat(fmt, ap);
  va_end(ap);
  return simaai_gst_format_detail(ctx, detail.c_str());
}

#define SIMAAI_GST_FATAL(self, domain, code, summary, detail)                                      \
  do {                                                                                             \
    GST_ERROR_OBJECT((self), "%s", (summary));                                                     \
    gst_element_message_full(GST_ELEMENT(self), GST_MESSAGE_ERROR, (domain), (code),               \
                             g_strdup(summary), g_strdup((detail) ? (detail) : ""), __FILE__,      \
                             G_STRFUNC, __LINE__);                                                 \
  } while (0)

#define SIMAAI_GST_WARN(self, summary, detail)                                                     \
  do {                                                                                             \
    if ((detail) && *(detail)) {                                                                   \
      GST_WARNING_OBJECT((self), "%s | %s", (summary), (detail));                                  \
    } else {                                                                                       \
      GST_WARNING_OBJECT((self), "%s", (summary));                                                 \
    }                                                                                              \
  } while (0)
