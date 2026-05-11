/**
 * @file
 * @ingroup gst
 * @brief Structured error/warning helpers for SiMa's GStreamer plugins.
 *
 * Provides a `SimaaiGstErrorContext` struct gathering the diagnostic fields plugins
 * carry around (plugin/node names, graph and frame ids, stream id, caps, dims, allocator,
 * etc.) plus a small set of inline formatters that turn that context into a stable,
 * key-value-formatted line. The `SIMAAI_GST_FATAL` and `SIMAAI_GST_WARN` macros are the
 * standard call sites that route a formatted message into both the GStreamer logging
 * subsystem and an element-level error message.
 *
 * This header is consumed by SiMa's GStreamer plugin sources directly; framework code
 * sees the resulting bus messages via @ref GstBusWatch.h.
 */
#pragma once

#include <gst/gst.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Bag of diagnostic fields used to build SiMa GStreamer error/warning messages.
 *
 * Every field is optional; formatters skip null/empty values. Integer fields use `-1`
 * as their "unset" sentinel and are skipped when negative. Populate as many fields as
 * the plugin has on hand at the failure site — more context means easier triage.
 */
struct SimaaiGstErrorContext {
  const char* plugin = nullptr;          ///< Owning plugin name, e.g., `"simaai-process-mla"`.
  const char* node = nullptr;            ///< Logical node identifier within the pipeline.
  const char* config_path = nullptr;     ///< Path to the plugin's config JSON, if any.
  const char* model_path = nullptr;      ///< Path to the model artifact, if any.
  int graph_id = -1;                     ///< Graph identifier; -1 if unset.
  std::int64_t frame_id = -1;            ///< Frame counter where the error occurred; -1 if unset.
  const char* stream_id = nullptr;       ///< GStreamer stream id.
  const char* input_caps = nullptr;      ///< Input pad caps as a string.
  const char* output_caps = nullptr;     ///< Output pad caps as a string.
  const char* input_dims = nullptr;      ///< Input tensor dims summary.
  const char* output_dims = nullptr;     ///< Output tensor dims summary.
  const char* allocator = nullptr;       ///< Allocator name in use.
  const char* dispatcher_err = nullptr;  ///< Dispatcher-layer error string, if any.
  const char* hint = nullptr;            ///< Free-form actionable hint for the user.
  const char* stage_key = nullptr;       ///< Stage key that failed.
  const char* source_used = nullptr;     ///< Origin of the config decision (e.g., `"framework"`).
  const char* missing_field = nullptr;   ///< Missing required field name, if a config-validation failure.
  const char* fallback_chain = nullptr;  ///< Comma-separated fallback chain that was tried.
};

/// Returns `s` if non-null, or the empty string if null. Convenience for printf-style sites.
inline const char* simaai_gst_safe(const char* s) {
  return s ? s : "";
}

/// Append `key='value'` (space-prefixed if not the first entry) when `value` is non-empty.
inline void simaai_gst_append_kv(std::ostringstream& ss, const char* key, const char* value) {
  if (!value || !*value)
    return;
  if (ss.tellp() > 0)
    ss << ' ';
  ss << key << "='" << value << "'";
}

/// Append `key='value'` always, substituting empty string when `value` is null.
inline void simaai_gst_append_kv_required(std::ostringstream& ss, const char* key,
                                          const char* value) {
  if (ss.tellp() > 0)
    ss << ' ';
  ss << key << "='" << simaai_gst_safe(value) << "'";
}

/// Append `key=<value>` (unquoted, integer).
inline void simaai_gst_append_kv_int(std::ostringstream& ss, const char* key, std::int64_t value) {
  if (ss.tellp() > 0)
    ss << ' ';
  ss << key << '=' << value;
}

/**
 * @brief Format an entire `SimaaiGstErrorContext` as a single key-value line.
 * @return Space-separated `key='value'` (or `key=N`) pairs; null/empty values are skipped.
 */
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
  simaai_gst_append_kv(ss, "stage_key", ctx.stage_key);
  simaai_gst_append_kv_required(
      ss, "source_used",
      (ctx.source_used && *ctx.source_used) ? ctx.source_used : "framework");
  simaai_gst_append_kv_required(ss, "missing_field",
                                (ctx.missing_field && *ctx.missing_field) ? ctx.missing_field
                                                                           : "none");
  simaai_gst_append_kv_required(
      ss, "fallback_chain",
      (ctx.fallback_chain && *ctx.fallback_chain) ? ctx.fallback_chain
                                                  : "none");
  return ss.str();
}

/**
 * @brief Format `ctx` and append a `detail='...'` field if `detail` is non-empty.
 * @param ctx Diagnostic context to format.
 * @param detail Free-form detail string (may be null/empty).
 */
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

/// vsnprintf-into-string helper used by the `*detailf` formatters; returns empty on null/empty fmt.
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

/**
 * @brief printf-style variant of `simaai_gst_format_detail`.
 *
 * Formats `fmt` and the variadic args into a detail string, then appends it to the
 * formatted context line.
 */
inline std::string simaai_gst_format_detailf(const SimaaiGstErrorContext& ctx, const char* fmt,
                                             ...) {
  va_list ap;
  va_start(ap, fmt);
  std::string detail = simaai_gst_vformat(fmt, ap);
  va_end(ap);
  return simaai_gst_format_detail(ctx, detail.c_str());
}

/**
 * @brief Post a fatal error on the GStreamer bus and log it via `GST_ERROR_OBJECT`.
 * @param self GstElement-like pointer the message is attributed to.
 * @param domain GError domain for the message.
 * @param code GError code within @p domain.
 * @param summary Short, user-facing summary string.
 * @param detail Detail string (may be empty); attached as the message's debug field.
 */
#define SIMAAI_GST_FATAL(self, domain, code, summary, detail)                                      \
  do {                                                                                             \
    GST_ERROR_OBJECT((self), "%s", (summary));                                                     \
    gst_element_message_full(GST_ELEMENT(self), GST_MESSAGE_ERROR, (domain), (code),               \
                             g_strdup(summary), g_strdup((detail) ? (detail) : ""), __FILE__,      \
                             G_STRFUNC, __LINE__);                                                 \
  } while (0)

/**
 * @brief Log a non-fatal warning via `GST_WARNING_OBJECT`, optionally with a detail suffix.
 * @param self GstElement-like pointer the warning is attributed to.
 * @param summary Short summary string.
 * @param detail Detail string; if non-empty it's appended after a `|` separator.
 */
#define SIMAAI_GST_WARN(self, summary, detail)                                                     \
  do {                                                                                             \
    if ((detail) && *(detail)) {                                                                   \
      GST_WARNING_OBJECT((self), "%s | %s", (summary), (detail));                                  \
    } else {                                                                                       \
      GST_WARNING_OBJECT((self), "%s", (summary));                                                 \
    }                                                                                              \
  } while (0)
