#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/SessionError.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"

#include <gst/gst.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
using pipeline_internal::env_bool;
using pipeline_internal::lower_copy;
using pipeline_internal::trim_copy;

namespace {

std::string decorate_with_error_code(const std::string& code, const std::string& message) {
  return pipeline_internal::error_util::decorate_error(code, message);
}

std::string with_hint(const std::string& message, const std::string& hint) {
  return pipeline_internal::error_util::append_hint(message, hint);
}

[[noreturn]] void throw_push_returned_false(const char* where, const std::string& last_err) {
  const char* tag = where ? where : "Run";
  const std::string code = error_codes::kRuntimePull;
  const std::string hint = "queue may be full, input is closed, or stream is stopping.";
  if (!last_err.empty()) {
    throw std::runtime_error(decorate_with_error_code(
        code, with_hint(std::string(tag) + ": push returned false: " + last_err, hint)));
  }
  throw std::runtime_error(
      decorate_with_error_code(code, with_hint(std::string(tag) + ": push returned false", hint)));
}

[[noreturn]] void throw_pull_failure_with_context(const char* where, PullStatus status,
                                                  PullError&& err, const std::string& last_err) {
  const char* tag = where ? where : "Run";
  const std::string code = err.code.empty() ? std::string(error_codes::kRuntimePull) : err.code;
  const std::string hint =
      "inspect Run::report()/SessionReport bus diagnostics for the first terminal error.";
  if (status == PullStatus::Timeout) {
    throw std::runtime_error(decorate_with_error_code(
        code, with_hint(std::string(tag) + ": timeout waiting for output (status=Timeout)", hint)));
  }
  if (status == PullStatus::Closed) {
    if (!last_err.empty()) {
      throw std::runtime_error(decorate_with_error_code(
          code,
          with_hint(std::string(tag) + ": stream closed (status=Closed): " + last_err, hint)));
    }
    throw std::runtime_error(decorate_with_error_code(
        code, with_hint(std::string(tag) + ": stream closed before output (status=Closed)", hint)));
  }
  if (err.report.has_value()) {
    if (err.report->error_code.empty()) {
      err.report->error_code = code;
    }
    if (err.report->repro_note.empty()) {
      err.report->repro_note =
          with_hint(std::string(tag) + ": pull failed without report details (status=Error)", hint);
    }
    const std::string msg =
        err.message.empty() ? decorate_with_error_code(code, err.report->repro_note) : err.message;
    throw SessionError(msg, std::move(*err.report));
  }
  if (!err.message.empty()) {
    throw std::runtime_error(decorate_with_error_code(code, err.message));
  }
  if (!last_err.empty()) {
    throw std::runtime_error(decorate_with_error_code(
        code, with_hint(std::string(tag) + ": pull failed (status=Error): " + last_err, hint)));
  }
  throw std::runtime_error(decorate_with_error_code(
      code, with_hint(std::string(tag) + ": pull failed without details (status=Error)", hint)));
}

struct SegmentInfo {
  std::string plugin;
  std::string name;
  std::string config;
};

std::string parse_first_token(const std::string& seg) {
  size_t i = 0;
  while (i < seg.size() && std::isspace(static_cast<unsigned char>(seg[i])))
    ++i;
  size_t j = i;
  while (j < seg.size() && !std::isspace(static_cast<unsigned char>(seg[j])))
    ++j;
  if (j <= i)
    return {};
  return seg.substr(i, j - i);
}

std::string parse_key_value(const std::string& seg, const std::string& key) {
  const size_t pos = seg.find(key);
  if (pos == std::string::npos)
    return {};
  size_t i = pos + key.size();
  while (i < seg.size() && std::isspace(static_cast<unsigned char>(seg[i])))
    ++i;
  if (i >= seg.size())
    return {};
  if (seg[i] == '"') {
    size_t j = i + 1;
    while (j < seg.size() && seg[j] != '"')
      ++j;
    if (j <= i + 1)
      return {};
    return seg.substr(i + 1, j - i - 1);
  }
  size_t j = i;
  while (j < seg.size() && !std::isspace(static_cast<unsigned char>(seg[j])) && seg[j] != '!') {
    ++j;
  }
  if (j <= i)
    return {};
  return seg.substr(i, j - i);
}

std::vector<SegmentInfo> parse_pipeline_segments(const std::string& pipeline) {
  std::vector<SegmentInfo> out;
  if (pipeline.empty())
    return out;
  size_t start = 0;
  while (start < pipeline.size()) {
    const size_t bang = pipeline.find('!', start);
    const std::string part =
        (bang == std::string::npos) ? pipeline.substr(start) : pipeline.substr(start, bang - start);
    start = (bang == std::string::npos) ? pipeline.size() : bang + 1;
    const std::string seg = trim_copy(part);
    if (seg.empty())
      continue;
    SegmentInfo info;
    info.plugin = parse_first_token(seg);
    info.name = parse_key_value(seg, "name=");
    info.config = parse_key_value(seg, "config=");
    out.push_back(std::move(info));
  }
  return out;
}

bool read_json_file(const std::string& path, nlohmann::json& out) {
  if (path.empty())
    return false;
  std::ifstream in(path);
  if (!in.is_open())
    return false;
  try {
    in >> out;
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

bool is_detess_segment(const SegmentInfo& seg) {
  const std::string pl = lower_copy(seg.plugin);
  const std::string nl = lower_copy(seg.name);
  const std::string cl = lower_copy(seg.config);
  return (pl.find("detess") != std::string::npos) || (nl.find("detess") != std::string::npos) ||
         (cl.find("detess") != std::string::npos);
}

void dump_detess_output_pool(GstElement* pipeline, const std::string& elem_name) {
  if (!pipeline || elem_name.empty())
    return;
  GstElement* elem = gst_bin_get_by_name(GST_BIN(pipeline), elem_name.c_str());
  if (!elem) {
    std::fprintf(stderr, "[DIAG] detess_pool: element %s not found\n", elem_name.c_str());
    return;
  }

  GstPad* pad = gst_element_get_static_pad(elem, "src");
  if (!pad) {
    gst_object_unref(elem);
    std::fprintf(stderr, "[DIAG] detess_pool: element %s missing src pad\n", elem_name.c_str());
    return;
  }

  GstCaps* caps = gst_pad_get_current_caps(pad);
  GstQuery* query = gst_query_new_allocation(caps, TRUE);
  gboolean ok = gst_pad_peer_query(pad, query);
  if (!ok) {
    ok = gst_pad_query(pad, query);
  }
  if (ok) {
    const guint pools = gst_query_get_n_allocation_pools(query);
    std::fprintf(stderr, "[DIAG] detess_pool element=%s pools=%u\n", elem_name.c_str(), pools);
    for (guint i = 0; i < pools; ++i) {
      GstBufferPool* pool = nullptr;
      guint size = 0;
      guint min_buffers = 0;
      guint max_buffers = 0;
      gst_query_parse_nth_allocation_pool(query, i, &pool, &size, &min_buffers, &max_buffers);
      std::fprintf(stderr, "[DIAG] detess_pool[%u] size=%u min=%u max=%u\n", i, size, min_buffers,
                   max_buffers);
      if (pool)
        gst_object_unref(pool);
    }
  } else {
    std::fprintf(stderr, "[DIAG] detess_pool: allocation query failed for %s\n", elem_name.c_str());
  }

  if (query)
    gst_query_unref(query);
  if (caps)
    gst_caps_unref(caps);
  gst_object_unref(pad);
  gst_object_unref(elem);
}

void warn_no_warmup_once() {
  static std::atomic<bool> warned{false};
  if (warned.exchange(true))
    return;
  std::printf("[WARN] Run::warmup: warm=0; throughput stability may vary without warmup.\n");
}

} // namespace

PullStatus Run::pull(int timeout_ms, Sample& out, PullError* err) {
  pipeline_internal::error_util::set_pull_error(err, "", "");
  const auto set_terminal_error = [&](const std::string& code, const std::string& message) {
    pipeline_internal::error_util::set_pull_error(err, code, message);
  };

  if (!state_) {
    set_terminal_error(error_codes::kRuntimePull, "Run::pull: stream is closed");
    return PullStatus::Closed;
  }
  auto st = state_;
  if (!st->supports_pull) {
    set_terminal_error(error_codes::kRuntimePull,
                       "Run::pull: pipeline has no Output (pull not supported)");
    return PullStatus::Error;
  }
  auto diag = st->stream.diag_ctx();
  const auto handle_stream_error = [&](const std::string& msg) {
    SessionReport rep = diag ? diag->snapshot_basic() : SessionReport{};
    std::string code = rep.error_code;
    if (code.empty()) {
      code = error_codes::kRuntimePull;
    }
    std::string note = "Run::pull: " + msg;
    rep.error_code = code;
    rep.repro_note = note;
    pipeline_internal::error_util::set_pull_error(err, std::move(code), std::move(note),
                                                  std::move(rep));
    return PullStatus::Error;
  };

  const std::string early_err = last_error();
  if (!early_err.empty()) {
    return handle_stream_error(early_err);
  }

  auto done = [&]() {
    if (st->supports_push) {
      const auto outputs_done = st->outputs_pulled.load() + st->outputs_dropped.load();
      return st->input_closed && st->input_thread_done.load() &&
             outputs_done >= st->inputs_pushed.load() && st->out_queue.empty();
    }
    return (st->stop_requested.load() || !st->stream.running()) && st->out_queue.empty();
  };

  std::unique_lock<std::mutex> lock(st->out_mu);
  if (timeout_ms < 0) {
    st->out_cv.wait(
        lock, [&]() { return st->stop_requested.load() || !st->out_queue.empty() || done(); });
  } else {
    const auto deadline = std::chrono::milliseconds(timeout_ms);
    if (!st->out_cv.wait_for(lock, deadline, [&]() {
          return st->stop_requested.load() || !st->out_queue.empty() || done();
        })) {
      lock.unlock();
      if (env_bool("SIMA_PULL_TIMEOUT_DIAG", true) && !st->pull_timeout_logged.exchange(true)) {
        const auto diag = st->stream.diag_ctx();
        if (diag) {
          std::ostringstream oss;
          oss << "[DIAG] pull_timeout Run::pull\n";
          if (!diag->pipeline_string.empty()) {
            oss << "Pipeline:\n" << diag->pipeline_string << "\n";
          }
          const std::string boundary = pipeline_internal::boundary_summary(diag);
          if (!boundary.empty())
            oss << boundary;
          const std::string flow = pipeline_internal::element_flow_summary(diag);
          if (!flow.empty())
            oss << flow;
          std::fprintf(stderr, "%s", oss.str().c_str());

          if (!diag->pipeline_string.empty()) {
            const auto segs = parse_pipeline_segments(diag->pipeline_string);
            for (const auto& seg : segs) {
              if (!is_detess_segment(seg))
                continue;
              if (!seg.config.empty()) {
                nlohmann::json j;
                if (read_json_file(seg.config, j)) {
                  std::fprintf(stderr, "[DIAG] detess_config plugin=%s name=%s config=%s\n%s\n",
                               seg.plugin.c_str(), seg.name.empty() ? "<none>" : seg.name.c_str(),
                               seg.config.c_str(), j.dump(2).c_str());
                } else {
                  std::fprintf(stderr,
                               "[DIAG] detess_config plugin=%s name=%s config=%s read_failed\n",
                               seg.plugin.c_str(), seg.name.empty() ? "<none>" : seg.name.c_str(),
                               seg.config.c_str());
                }
              }
              // This query path can block inside GStreamer pad locks when the
              // pipeline is already stalled. Keep it opt-in for deep debugging.
              if (env_bool("SIMA_PULL_TIMEOUT_POOL_DIAG", false)) {
                GstElement* pipeline = st->stream.pipeline_handle();
                if (pipeline && !seg.name.empty()) {
                  dump_detess_output_pool(pipeline, seg.name);
                }
              }
            }
          }
        }
      }
      set_terminal_error(error_codes::kRuntimePull, "Run::pull: timeout waiting for output");
      return PullStatus::Timeout;
    }
  }

  if (!st->out_queue.empty()) {
    out = std::move(st->out_queue.front());
    st->out_queue.pop_front();
    st->outputs_pulled.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    st->out_cv.notify_one();
    return PullStatus::Ok;
  }

  const bool is_done = done();
  lock.unlock();

  const std::string late_err = last_error();
  if (!late_err.empty()) {
    return handle_stream_error(late_err);
  }

  if (is_done) {
    set_terminal_error(error_codes::kRuntimePull, "Run::pull: stream closed");
    stop();
    return PullStatus::Closed;
  }

  set_terminal_error(error_codes::kRuntimePull, "Run::pull: stream closed before output");
  return PullStatus::Closed;
}

std::optional<Sample> Run::pull(int timeout_ms) {
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return out;
  }
  if (status == PullStatus::Error) {
    if (err.report.has_value()) {
      SessionReport rep = std::move(*err.report);
      if (rep.error_code.empty()) {
        rep.error_code = err.code.empty() ? error_codes::kRuntimePull : err.code;
      }
      if (rep.repro_note.empty()) {
        rep.repro_note = pipeline_internal::error_util::append_hint(
            "Run::pull: pull returned Error without report details",
            "inspect Run::report()/SessionReport bus diagnostics for root cause.");
      }
      const std::string msg = err.message.empty()
                                  ? decorate_with_error_code(rep.error_code, rep.repro_note)
                                  : err.message;
      throw SessionError(msg, std::move(rep));
    }
    if (!err.message.empty()) {
      throw std::runtime_error(decorate_with_error_code(
          err.code.empty() ? std::string(error_codes::kRuntimePull) : err.code, err.message));
    }
    const std::string last_err = last_error();
    const std::string msg =
        last_err.empty()
            ? pipeline_internal::error_util::append_hint(
                  "Run::pull: pull returned Error without detail",
                  "inspect Run::report()/SessionReport bus diagnostics for root cause.")
            : pipeline_internal::error_util::append_hint(
                  "Run::pull: " + last_err,
                  "inspect Run::report()/SessionReport bus diagnostics for root cause.");
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull, msg));
  }
  return std::nullopt;
}

std::optional<simaai::neat::Tensor> Run::pull_tensor(int timeout_ms) {
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status != PullStatus::Ok)
    return std::nullopt;
  if (out.kind != SampleKind::Tensor || !out.tensor.has_value()) {
    return std::nullopt;
  }
  return out.tensor;
}

simaai::neat::Tensor Run::pull_tensor_or_throw(int timeout_ms) {
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    if (out.kind == SampleKind::Tensor && out.tensor.has_value()) {
      return *out.tensor;
    }
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::pull_tensor_or_throw: output is not tensor"));
  }
  if (status == PullStatus::Timeout) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::pull_tensor_or_throw: timeout (status=Timeout)"));
  }
  if (status == PullStatus::Closed) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::pull_tensor_or_throw: closed (status=Closed)"));
  }
  if (err.report.has_value()) {
    SessionReport rep = std::move(*err.report);
    if (rep.error_code.empty()) {
      rep.error_code = err.code.empty() ? error_codes::kRuntimePull : err.code;
    }
    if (rep.repro_note.empty()) {
      rep.repro_note = pipeline_internal::error_util::append_hint(
          "Run::pull_tensor_or_throw: pull returned Error without report details",
          "inspect Run::report()/SessionReport bus diagnostics for root cause.");
    }
    const std::string msg = err.message.empty()
                                ? decorate_with_error_code(rep.error_code, rep.repro_note)
                                : err.message;
    throw SessionError(msg, std::move(rep));
  }
  if (!err.message.empty()) {
    throw std::runtime_error(decorate_with_error_code(
        err.code.empty() ? std::string(error_codes::kRuntimePull) : err.code, err.message));
  }
  const std::string last_err = last_error();
  throw std::runtime_error(decorate_with_error_code(
      error_codes::kRuntimePull,
      last_err.empty()
          ? pipeline_internal::error_util::append_hint(
                "Run::pull_tensor_or_throw: pull returned Error without detail",
                "inspect Run::report()/SessionReport bus diagnostics for root cause.")
          : pipeline_internal::error_util::append_hint(
                "Run::pull_tensor_or_throw: " + last_err,
                "inspect Run::report()/SessionReport bus diagnostics for root cause.")));
}

std::optional<simaai::neat::Tensor> Run::pull_tensor_matching(const std::string& payload_tag,
                                                              int timeout_ms) {
  const bool filter = !payload_tag.empty();
  auto start = std::chrono::steady_clock::now();
  while (true) {
    Sample out;
    PullError err;
    int remaining = timeout_ms;
    if (timeout_ms >= 0) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
      remaining = static_cast<int>(timeout_ms - elapsed);
      if (remaining < 0)
        remaining = 0;
    }
    const PullStatus status = pull(remaining, out, &err);
    if (status != PullStatus::Ok)
      return std::nullopt;
    if (out.kind != SampleKind::Tensor || !out.tensor.has_value()) {
      continue;
    }
    if (!filter || out.payload_tag == payload_tag) {
      return out.tensor;
    }
  }
}

Sample Run::push_and_pull(const cv::Mat& input, int timeout_ms) {
  if (!push(input)) {
    throw_push_returned_false("Run::push_and_pull", last_error());
  }
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return out;
  }
  throw_pull_failure_with_context("Run::push_and_pull", status, std::move(err), last_error());
}

Sample Run::push_and_pull(const simaai::neat::Tensor& input, int timeout_ms) {
  if (!push(input)) {
    throw_push_returned_false("Run::push_and_pull", last_error());
  }
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return out;
  }
  throw_pull_failure_with_context("Run::push_and_pull", status, std::move(err), last_error());
}

Sample Run::push_and_pull_holder(const std::shared_ptr<void>& holder, int timeout_ms) {
  if (!push_holder(holder)) {
    throw_push_returned_false("Run::push_and_pull_holder", last_error());
  }
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return out;
  }
  throw_pull_failure_with_context("Run::push_and_pull_holder", status, std::move(err),
                                  last_error());
}

Sample Run::run(const cv::Mat& input, int timeout_ms) {
  return push_and_pull(input, timeout_ms);
}

Sample Run::run(const simaai::neat::Tensor& input, int timeout_ms) {
  return push_and_pull(input, timeout_ms);
}

Sample Run::run(const Sample& input, int timeout_ms) {
  if (!push(input)) {
    throw_push_returned_false("Run::run", last_error());
  }
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return out;
  }
  throw_pull_failure_with_context("Run::run", status, std::move(err), last_error());
}

int Run::warmup(const cv::Mat& input, int warm, int timeout_ms) {
  if (warm < 0) {
    warm = pipeline_internal::env_int("SIMA_ASYNC_WARMUP", 0);
  }
  if (warm <= 0) {
    warn_no_warmup_once();
    return 0;
  }
  for (int i = 0; i < warm; ++i) {
    (void)push_and_pull(input, timeout_ms);
  }
  return warm;
}

} // namespace simaai::neat
