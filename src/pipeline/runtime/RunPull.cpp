#include "pipeline/Run.h"
#include "RunInternal.h"

#include "pipeline/DetectionTypes.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/GstDiagnosticsUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleUtil.h"

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
#include <string_view>
#include <utility>
#include <vector>

namespace simaai::neat {
using pipeline_internal::env_bool;
using pipeline_internal::lower_copy;
using pipeline_internal::trim_copy;

namespace {

bool attach_public_holder_loan_or_error(runtime::RunCore& core, Sample& sample,
                                        std::string* message) {
  auto gate = core.holder_loan_gate;
  if (!gate || !gate->enabled() || !pipeline_internal::sample_has_device_gstsample_holder(sample)) {
    return true;
  }
  const int required = pipeline_internal::count_distinct_device_gstsample_holders(sample);
  if (core.pipeline.stream_opt.holder_loan_credits_auto &&
      required > core.pipeline.stream_opt.holder_loan_per_sample_arity) {
    core.pipeline.stream_opt.holder_loan_per_sample_arity = required;
    const int sample_window = std::max(1, core.pipeline.stream_opt.holder_loan_sample_window);
    core.pipeline.stream_opt.holder_loan_credits = sample_window * required;
    gate->configure(core.pipeline.stream_opt.holder_loan_credits);
  }
  std::string err;
  if (!pipeline_internal::attach_zero_copy_loan_to_sample(sample, gate, &err)) {
    if (message) {
      std::ostringstream oss;
      oss << "zero-copy output loan credits exhausted";
      if (!err.empty()) {
        oss << ": " << err;
      }
      oss << " (sample_requires=" << required << " holder" << (required == 1 ? "" : "s")
          << ", credit_limit=" << gate->credit_limit() << ", in_flight=" << gate->inflight() << ")";
      *message = oss.str();
    }
    return false;
  }
  return true;
}

constexpr auto kPullWaitPollQuantum = std::chrono::milliseconds(50);

std::string decorate_with_error_code(const std::string& code, const std::string& message) {
  return pipeline_internal::error_util::decorate_error(code, message);
}

std::string with_hint(const std::string& message, const std::string& hint) {
  return pipeline_internal::error_util::append_hint(message, hint);
}

void validate_run_image_inputs(const std::vector<cv::Mat>& inputs, const char* where) {
  const char* tag = where ? where : "Run::run";
  if (inputs.empty()) {
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull,
                                                      std::string(tag) + ": empty image list"));
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].empty()) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          std::string(tag) + ": empty image input at index " + std::to_string(i)));
    }
  }
}

bool input_options_expect_tensor_media_local(const std::optional<InputOptions>& opt) {
  if (!opt.has_value()) {
    return false;
  }
  const std::string media = pipeline_internal::lower_copy(resolve_input_media_type(*opt));
  return media == "application/vnd.simaai.tensor";
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
      "inspect the attached GraphReport diagnostics for the first terminal error.";
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
    throw NeatError(msg, std::move(*err.report));
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

bool is_preproc_segment(const SegmentInfo& seg) {
  const std::string pl = lower_copy(seg.plugin);
  const std::string nl = lower_copy(seg.name);
  const std::string cl = lower_copy(seg.config);
  return (pl.find("processcvu") != std::string::npos) &&
         ((nl.find("preproc") != std::string::npos) || (cl.find("preproc") != std::string::npos) ||
          (cl.find("quant") != std::string::npos) || (cl.find("tess") != std::string::npos));
}

bool is_mla_segment(const SegmentInfo& seg) {
  const std::string pl = lower_copy(seg.plugin);
  const std::string nl = lower_copy(seg.name);
  const std::string cl = lower_copy(seg.config);
  return (pl.find("processmla") != std::string::npos) ||
         (nl.find("processmla") != std::string::npos) ||
         (cl.find("process_mla") != std::string::npos);
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

std::string available_output_names(const runtime::RunCore& core) {
  if (!core.graph_execution_) {
    return "[]";
  }
  std::vector<std::string> names;
  names.reserve(core.graph_execution_->plan.named_outputs.size());
  for (const auto& kv : core.graph_execution_->plan.named_outputs) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << names[i];
  }
  oss << "]";
  return oss.str();
}

std::optional<runtime::Endpoint> named_output_endpoint(const runtime::RunCore& core,
                                                       std::string_view output_name) {
  if (!core.graph_execution_) {
    return std::nullopt;
  }
  const auto& named = core.graph_execution_->plan.named_outputs;
  auto it = named.find(std::string(output_name));
  if (it == named.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::string default_output_name(const runtime::RunCore& core) {
  if (!core.graph_execution_ || !core.graph_execution_->plan.default_output.has_value()) {
    return "default";
  }
  const auto& def = *core.graph_execution_->plan.default_output;
  for (const auto& [name, ep] : core.graph_execution_->plan.named_outputs) {
    if (ep.node == def.node && ep.port == def.port && ep.kind == def.kind) {
      return name;
    }
  }
  return "default";
}

} // namespace

PullStatus Run::pull(int timeout_ms, Sample& out, PullError* err) {
  if (!core_) {
    pipeline_internal::error_util::set_pull_error(err, error_codes::kRuntimePull,
                                                  "Run::pull: stream is closed");
    return PullStatus::Closed;
  }
  return core_->pull(timeout_ms, out, err);
}

PullStatus Run::pull(std::string_view output_name, int timeout_ms, Sample& out, PullError* err) {
  if (output_name.empty()) {
    pipeline_internal::error_util::set_pull_error(err, error_codes::kRuntimePull,
                                                  "Run::pull(name): output name is empty");
    return PullStatus::Error;
  }
  if (!core_) {
    pipeline_internal::error_util::set_pull_error(err, error_codes::kRuntimePull,
                                                  "Run::pull(name): stream is closed");
    return PullStatus::Closed;
  }
  return core_->pull_named_output(output_name, timeout_ms, out, err);
}

bool runtime::RunCore::attach_public_output_loan(Sample& sample, std::string* err) {
  return attach_public_holder_loan_or_error(*this, sample, err);
}

PullStatus runtime::RunCore::pull(int timeout_ms, Sample& out, PullError* err) {
  pipeline_internal::error_util::set_pull_error(err, "", "");
  const auto set_terminal_error = [&](const std::string& code, const std::string& message) {
    pipeline_internal::error_util::set_pull_error(err, code, message);
  };

  auto* st = this;
  if (st->graph_execution_) {
    const auto& endpoint = st->graph_execution_->plan.default_output;
    if (!endpoint.has_value()) {
      set_terminal_error(error_codes::kRuntimePull,
                         "Run::pull: graph has no unambiguous default output; use pull(name, "
                         "...). Available outputs: " +
                             available_output_names(*this));
      return PullStatus::Error;
    }
    auto sample = st->graph_pull(endpoint->node, timeout_ms);
    if (sample.has_value()) {
      Sample value = std::move(*sample);
      std::string loan_error;
      if (!attach_public_holder_loan_or_error(*st, value, &loan_error)) {
        set_terminal_error(error_codes::kRuntimePull,
                           "Run::pull: " + (loan_error.empty()
                                                ? std::string("zero-copy output loan failed")
                                                : loan_error));
        return PullStatus::Error;
      }
      const auto now = std::chrono::steady_clock::now();
      st->record_graph_sample_output(default_output_name(*st), value, now);
      runtime::trace_graph_message_event(runtime::TraceGraphMessageEventType::GraphOutputPull,
                                         st->graph_execution_.get(), runtime::invalid_edge_index(),
                                         value, default_output_name(*st));
      out = std::move(value);
      st->outputs_pulled.fetch_add(1, std::memory_order_relaxed);
      std::lock_guard<std::mutex> timing_lock(st->latency_mu);
      if (!st->pull_timing_init) {
        st->first_pull_at = now;
        st->pull_timing_init = true;
      }
      st->last_pull_at = now;
      return PullStatus::Ok;
    }
    const std::string graph_err = st->last_error();
    if (!graph_err.empty()) {
      set_terminal_error(error_codes::kRuntimePull, graph_err);
      return PullStatus::Error;
    }
    set_terminal_error(error_codes::kRuntimePull, "Run::pull: timeout waiting for graph output");
    return PullStatus::Timeout;
  }
  if (!st->pipeline.supports_pull) {
    set_terminal_error(error_codes::kRuntimePull,
                       "Run::pull: pipeline has no Output (pull not supported)");
    return PullStatus::Error;
  }
  auto diag = st->pipeline.stream.diag_ctx();
  const auto handle_stream_error = [&](const std::string& msg) {
    GraphReport rep = diag ? diag->snapshot_basic() : GraphReport{};
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
    if (st->pipeline.supports_push) {
      const auto outputs_done = st->outputs_pulled.load() + st->outputs_dropped.load();
      return st->pipeline.input_closed && st->pipeline.input_thread_done.load() &&
             outputs_done >= st->inputs_pushed.load() && st->pipeline.out_queue.empty();
    }
    return (st->stop_requested.load() || !st->pipeline.stream.running()) &&
           st->pipeline.out_queue.empty();
  };
  const auto wait_ready = [&]() {
    return st->stop_requested.load() || !st->pipeline.out_queue.empty() || done() ||
           (st->pipeline.supports_push && !st->pipeline.stream.running());
  };

  std::optional<std::chrono::steady_clock::time_point> deadline;
  if (timeout_ms >= 0) {
    deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  }

  std::unique_lock<std::mutex> lock(st->pipeline.out_mu);
  if (timeout_ms < 0) {
    while (!wait_ready()) {
      const std::string loop_err = last_error();
      if (!loop_err.empty()) {
        lock.unlock();
        return handle_stream_error(loop_err);
      }
      st->pipeline.out_cv.wait_for(lock, kPullWaitPollQuantum);
    }
  } else {
    while (!wait_ready()) {
      const std::string loop_err = last_error();
      if (!loop_err.empty()) {
        lock.unlock();
        return handle_stream_error(loop_err);
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= *deadline) {
        lock.unlock();
        if (timeout_ms == 0) {
          // A zero-timeout pull is a non-blocking poll. This is used by pull_samples()
          // to drain already-queued outputs after it has received the first sample, so
          // "no extra sample available right now" must not look like a real stall.
          return PullStatus::Timeout;
        }
        if (env_bool("SIMA_PULL_TIMEOUT_DIAG", true) &&
            !st->pipeline.pull_timeout_logged.exchange(true)) {
          const auto diag = st->pipeline.stream.diag_ctx();
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
                const bool detess = is_detess_segment(seg);
                const bool preproc = is_preproc_segment(seg);
                const bool mla = is_mla_segment(seg);
                if (!detess && !preproc && !mla)
                  continue;
                if (!seg.config.empty()) {
                  nlohmann::json j;
                  if (read_json_file(seg.config, j)) {
                    const char* label =
                        detess ? "detess_config" : (preproc ? "preproc_config" : "mla_config");
                    std::fprintf(stderr, "[DIAG] %s plugin=%s name=%s config=%s\n%s\n", label,
                                 seg.plugin.c_str(), seg.name.empty() ? "<none>" : seg.name.c_str(),
                                 seg.config.c_str(), j.dump(2).c_str());
                  } else {
                    const char* label =
                        detess ? "detess_config" : (preproc ? "preproc_config" : "mla_config");
                    std::fprintf(stderr, "[DIAG] %s plugin=%s name=%s config=%s read_failed\n",
                                 label, seg.plugin.c_str(),
                                 seg.name.empty() ? "<none>" : seg.name.c_str(),
                                 seg.config.c_str());
                  }
                }
                // This query path can block inside GStreamer pad locks when the
                // pipeline is already stalled. Keep it opt-in for deep debugging.
                if (detess && env_bool("SIMA_PULL_TIMEOUT_POOL_DIAG", false)) {
                  GstElement* pipeline = st->pipeline.stream.pipeline_handle();
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
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now);
      st->pipeline.out_cv.wait_for(lock, std::min(kPullWaitPollQuantum, remaining));
    }
  }

  while (!st->pipeline.out_queue.empty()) {
    std::string loan_error;
    if (!attach_public_holder_loan_or_error(*st, st->pipeline.out_queue.front(), &loan_error)) {
      const std::string message =
          "Run::pull: " +
          (loan_error.empty() ? std::string("zero-copy output loan unavailable") : loan_error) +
          "; release older zero-copy outputs and retry";

      const std::string loop_err = last_error();
      if (!loop_err.empty()) {
        lock.unlock();
        return handle_stream_error(loop_err);
      }

      if (timeout_ms == 0) {
        lock.unlock();
        set_terminal_error(error_codes::kRuntimePull, message);
        return PullStatus::Timeout;
      }
      if (timeout_ms > 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= *deadline) {
          lock.unlock();
          set_terminal_error(error_codes::kRuntimePull, message);
          return PullStatus::Timeout;
        }
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now);
        st->pipeline.out_cv.wait_for(lock, std::min(kPullWaitPollQuantum, remaining));
      } else {
        st->pipeline.out_cv.wait_for(lock, kPullWaitPollQuantum);
      }
      continue;
    }
    out = std::move(st->pipeline.out_queue.front());
    st->pipeline.out_queue.pop_front();
    st->outputs_pulled.fetch_add(1, std::memory_order_relaxed);
    {
      const auto now = std::chrono::steady_clock::now();
      std::lock_guard<std::mutex> timing_lock(st->latency_mu);
      if (!st->pull_timing_init) {
        st->first_pull_at = now;
        st->pull_timing_init = true;
      }
      st->last_pull_at = now;
    }
    lock.unlock();
    st->pipeline.out_cv.notify_one();
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

PullStatus runtime::RunCore::pull_named_output(std::string_view output_name, int timeout_ms,
                                               Sample& out, PullError* err) {
  pipeline_internal::error_util::set_pull_error(err, "", "");
  const auto set_terminal_error = [&](const std::string& code, const std::string& message) {
    pipeline_internal::error_util::set_pull_error(err, code, message);
  };

  if (!graph_execution_) {
    set_terminal_error(error_codes::kRuntimePull,
                       "Run::pull(name): named outputs require a graph-backed Run");
    return PullStatus::Error;
  }
  if (output_name.empty()) {
    set_terminal_error(error_codes::kRuntimePull, "Run::pull(name): output name is empty");
    return PullStatus::Error;
  }
  const auto endpoint = named_output_endpoint(*this, output_name);
  if (!endpoint.has_value()) {
    set_terminal_error(error_codes::kRuntimePull, "Run::pull(\"" + std::string(output_name) +
                                                      "\"): unknown output. Available outputs: " +
                                                      available_output_names(*this));
    return PullStatus::Error;
  }

  auto sample = graph_pull(endpoint->node, timeout_ms);
  if (sample.has_value()) {
    Sample value = std::move(*sample);
    std::string loan_error;
    if (!attach_public_holder_loan_or_error(*this, value, &loan_error)) {
      set_terminal_error(
          error_codes::kRuntimePull,
          "Run::pull(\"" + std::string(output_name) + "\"): " +
              (loan_error.empty() ? std::string("zero-copy output loan failed") : loan_error));
      return PullStatus::Error;
    }
    const auto now = std::chrono::steady_clock::now();
    record_graph_sample_output(output_name, value, now);
    runtime::trace_graph_message_event(runtime::TraceGraphMessageEventType::GraphOutputPull,
                                       graph_execution_.get(), runtime::invalid_edge_index(), value,
                                       output_name);
    out = std::move(value);
    outputs_pulled.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> timing_lock(latency_mu);
    if (!pull_timing_init) {
      first_pull_at = now;
      pull_timing_init = true;
    }
    last_pull_at = now;
    return PullStatus::Ok;
  }

  const std::string graph_err = last_error();
  if (!graph_err.empty()) {
    set_terminal_error(error_codes::kRuntimePull, graph_err);
    return PullStatus::Error;
  }
  set_terminal_error(error_codes::kRuntimePull, "Run::pull(\"" + std::string(output_name) +
                                                    "\"): timeout waiting for graph output");
  return PullStatus::Timeout;
}

std::optional<Sample> runtime::RunCore::pull_optional(int timeout_ms) {
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return out;
  }
  if (status == PullStatus::Error) {
    if (err.report.has_value()) {
      GraphReport rep = std::move(*err.report);
      if (rep.error_code.empty()) {
        rep.error_code = err.code.empty() ? error_codes::kRuntimePull : err.code;
      }
      if (rep.repro_note.empty()) {
        rep.repro_note = pipeline_internal::error_util::append_hint(
            "Run::pull: pull returned Error without report details",
            "inspect the attached GraphReport diagnostics for root cause.");
      }
      const std::string msg = err.message.empty()
                                  ? decorate_with_error_code(rep.error_code, rep.repro_note)
                                  : err.message;
      throw NeatError(msg, std::move(rep));
    }
    if (!err.message.empty()) {
      throw std::runtime_error(decorate_with_error_code(
          err.code.empty() ? std::string(error_codes::kRuntimePull) : err.code, err.message));
    }
    const std::string last_err = last_error();
    const std::string msg =
        last_err.empty() ? pipeline_internal::error_util::append_hint(
                               "Run::pull: pull returned Error without detail",
                               "inspect the attached GraphReport diagnostics for root cause.")
                         : pipeline_internal::error_util::append_hint(
                               "Run::pull: " + last_err,
                               "inspect the attached GraphReport diagnostics for root cause.");
    throw std::runtime_error(decorate_with_error_code(error_codes::kRuntimePull, msg));
  }
  return std::nullopt;
}

void Run::require_async_pull_mode(const char* where) const {
  if (core_ && core_->mode == RunMode::Sync) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, std::string(where ? where : "Run") +
                                       ": pull is not allowed in sync mode; use run(...) instead"));
  }
}

std::optional<Sample> Run::pull(int timeout_ms) {
  require_async_pull_mode("Run::pull");
  return core_ ? core_->pull_optional(timeout_ms) : std::nullopt;
}

std::optional<Sample> Run::pull(std::string_view output_name, int timeout_ms) {
  require_async_pull_mode("Run::pull(name)");
  Sample out;
  PullError err;
  const PullStatus status = pull(output_name, timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return out;
  }
  if (status == PullStatus::Error) {
    throw_pull_failure_with_context("Run::pull(name)", status, std::move(err), last_error());
  }
  return std::nullopt;
}

TensorList Run::pull_tensors(int timeout_ms) {
  require_async_pull_mode("Run::pull_tensors");
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return tensors_from_sample(out, true);
  }
  if (status == PullStatus::Timeout || status == PullStatus::Closed) {
    return {};
  }
  throw_pull_failure_with_context("Run::pull_tensors", status, std::move(err), last_error());
}

TensorList Run::pull_tensors(std::string_view output_name, int timeout_ms) {
  require_async_pull_mode("Run::pull_tensors(name)");
  Sample out;
  PullError err;
  const PullStatus status = pull(output_name, timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    return tensors_from_sample(out, true);
  }
  if (status == PullStatus::Timeout || status == PullStatus::Closed) {
    return {};
  }
  throw_pull_failure_with_context("Run::pull_tensors(name)", status, std::move(err), last_error());
}

Sample Run::pull_samples(int timeout_ms) {
  require_async_pull_mode("Run::pull_samples");
  Sample out;
  auto start = std::chrono::steady_clock::now();
  while (true) {
    PullError err;
    Sample sample;
    int remaining = timeout_ms;
    if (timeout_ms >= 0) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
      remaining = static_cast<int>(timeout_ms - elapsed);
      if (remaining < 0) {
        remaining = 0;
      }
    }
    const PullStatus status = pull(out.empty() ? remaining : 0, sample, &err);
    if (status == PullStatus::Ok) {
      out.push_back(std::move(sample));
      continue;
    }
    if ((status == PullStatus::Timeout || status == PullStatus::Closed) && !out.empty()) {
      return out;
    }
    if (status == PullStatus::Timeout || status == PullStatus::Closed) {
      return out;
    }
    throw_pull_failure_with_context("Run::pull_samples", status, std::move(err), last_error());
  }
}

Sample Run::pull_samples(std::string_view output_name, int timeout_ms) {
  require_async_pull_mode("Run::pull_samples(name)");
  Sample out;
  auto start = std::chrono::steady_clock::now();
  while (true) {
    PullError err;
    Sample sample;
    int remaining = timeout_ms;
    if (timeout_ms >= 0) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
      remaining = static_cast<int>(timeout_ms - elapsed);
      if (remaining < 0) {
        remaining = 0;
      }
    }
    const PullStatus status = pull(output_name, out.empty() ? remaining : 0, sample, &err);
    if (status == PullStatus::Ok) {
      out.push_back(std::move(sample));
      continue;
    }
    if ((status == PullStatus::Timeout || status == PullStatus::Closed) && !out.empty()) {
      return out;
    }
    if (status == PullStatus::Timeout || status == PullStatus::Closed) {
      return out;
    }
    throw_pull_failure_with_context("Run::pull_samples(name)", status, std::move(err),
                                    last_error());
  }
}

TensorList Run::pull_tensors_strict(int timeout_ms) {
  Sample out;
  PullError err;
  const PullStatus status = pull(timeout_ms, out, &err);
  if (status == PullStatus::Ok) {
    // Normalise detection-stage output tags before unwrapping to TensorList so the
    // type-honest DetectionSpec slot is the source of truth even when the producing
    // stage only signalled via Sample-level payload_tag / format.
    tag_detection_format_in_sample(out);
    return tensors_from_sample(out, true);
  }
  throw_pull_failure_with_context("Run::run", status, std::move(err), last_error());
}

Sample Run::pull_samples_strict(int timeout_ms) {
  Sample sample;
  PullError err;
  const PullStatus status = pull(timeout_ms, sample, &err);
  if (status == PullStatus::Ok) {
    tag_detection_format_in_sample(sample);
    return Sample{std::move(sample)};
  }
  throw_pull_failure_with_context("Run::run", status, std::move(err), last_error());
}

void Run::enqueue_run_images(const std::vector<cv::Mat>& inputs) {
  auto st = core_;
  validate_run_image_inputs(inputs, "Run::run");
  if (!st) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::run: stream is closed"));
  }
  if (st->graph_execution_) {
    if (!push(inputs)) {
      throw_push_returned_false("Run::run", last_error());
    }
    return;
  }
  if (!st->pipeline.supports_push) {
    throw std::runtime_error(decorate_with_error_code(
        error_codes::kRuntimePull, "Run::run: pipeline has no Input (push not supported)"));
  }
  if (!input_options_expect_tensor_media_local(st->pipeline.tensor_input_opt_for_cv)) {
    if (inputs.size() != 1U) {
      throw std::runtime_error(decorate_with_error_code(
          error_codes::kRuntimePull,
          "Run::run: raw-image ingress supports exactly one cv::Mat per inference item"));
    }
    if (!push_impl(inputs.front(), true)) {
      throw_push_returned_false("Run::run", last_error());
    }
    return;
  }
  TensorList tensors;
  tensors.reserve(inputs.size());
  for (const auto& input : inputs) {
    tensors.emplace_back(
        tensor_from_cv_mat(input, *st->pipeline.tensor_input_opt_for_cv, "Run::run(inputs)"));
  }
  if (!push_message_impl(pipeline_internal::sample_from_tensors_for_input(
                             tensors, *st->pipeline.tensor_input_opt_for_cv),
                         true)) {
    throw_push_returned_false("Run::run", last_error());
  }
}

void Run::enqueue_run_tensors(const TensorList& inputs) {
  if (inputs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::run: empty tensor list"));
  }
  if (core_ && core_->graph_execution_) {
    if (!push(inputs)) {
      throw_push_returned_false("Run::run", last_error());
    }
    return;
  }
  if (core_ && core_->pipeline.input_route_processor) {
    if (!push_message_impl(
            core_->pipeline.input_route_processor->process_tensors(inputs, "Run::run"), true)) {
      throw_push_returned_false("Run::run", last_error());
    }
    return;
  }
  const Sample sample = (core_ && core_->pipeline.tensor_input_opt_for_cv.has_value())
                            ? pipeline_internal::sample_from_tensors_for_input(
                                  inputs, *core_->pipeline.tensor_input_opt_for_cv)
                            : sample_from_tensors(inputs);
  if (!push_message_impl(sample, true)) {
    throw_push_returned_false("Run::run", last_error());
  }
}

void Run::enqueue_run_samples(const Sample& inputs) {
  if (inputs.empty()) {
    throw std::runtime_error(
        decorate_with_error_code(error_codes::kRuntimePull, "Run::run: empty sample list"));
  }
  if (core_ && core_->graph_execution_) {
    if (!push(inputs)) {
      throw_push_returned_false("Run::run", last_error());
    }
    return;
  }
  if (core_ && core_->pipeline.input_route_processor) {
    if (!push_message_impl(
            core_->pipeline.input_route_processor->process_samples(inputs, "Run::run"), true)) {
      throw_push_returned_false("Run::run", last_error());
    }
    return;
  }
  for (const auto& msg : inputs) {
    if (!push_sample_impl(msg, true)) {
      throw_push_returned_false("Run::run", last_error());
    }
  }
}

TensorList Run::run(const std::vector<cv::Mat>& inputs, int timeout_ms) {
  enqueue_run_images(inputs);
  return pull_tensors_strict(timeout_ms);
}

TensorList Run::run(const TensorList& inputs, int timeout_ms) {
  enqueue_run_tensors(inputs);
  return pull_tensors_strict(timeout_ms);
}

Sample Run::run(const Sample& inputs, int timeout_ms) {
  enqueue_run_samples(inputs);
  if (core_ && core_->pipeline.input_route_processor) {
    return pull_samples_strict(timeout_ms);
  }
  if (inputs.size() <= 1U) {
    return pull_samples_strict(timeout_ms);
  }
  Sample out;
  out.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    Sample batch = pull_samples_strict(timeout_ms);
    if (batch.kind == SampleKind::Bundle) {
      for (auto& field : batch.fields) {
        out.push_back(std::move(field));
      }
    } else if (!batch.empty()) {
      out.push_back(std::move(batch));
    }
  }
  return out;
}

} // namespace simaai::neat
