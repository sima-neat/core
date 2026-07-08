#include "InputStreamInternal.h"

#include "pipeline/internal/CpuVisibleSample.h"

namespace simaai::neat {
InputStream InputStream::create(GstElement* pipeline, GstElement* appsrc, GstElement* appsink,
                                const SampleSpec& spec, const InputOptions& src_opt,
                                const InputStreamOptions& opt, std::shared_ptr<DiagCtx> diag,
                                std::shared_ptr<void> guard) {
  auto state = std::make_shared<State>();
  state->pipeline = pipeline;
  state->appsrc = appsrc;
  state->appsink = appsink;
  state->src_opt = src_opt;
  state->opt = opt;
  state->dynamic_capability = opt.dynamic_capability;
  state->shape_policy = opt.shape_policy;
  state->shape_limits = opt.shape_limits;
  state->byte_guard_origin = opt.byte_guard_origin;
  state->allow_ingress_cvu_format_renegotiation = opt.allow_ingress_cvu_format_renegotiation;
  state->allow_dynamic_growth =
      (state->dynamic_capability != InputStreamOptions::DynamicCapability::StaticOnly) &&
      (state->shape_policy != InputStreamOptions::ShapePolicy::LockedByCapsOverride);
  state->max_input_bytes_guard = opt.max_input_bytes;
  state->diag = std::move(diag);
  state->guard = std::move(guard);
  state->lifetime_token = std::make_shared<int>(0);
  if (opt.holder_loan_credits > 0) {
    state->holder_loan_gate =
        std::make_shared<pipeline_internal::HolderLoanGate>(opt.holder_loan_credits);
  }
  state->timing_enabled = opt.enable_timings;
  state->current_key = spec.caps_key;
  state->last_spec = spec;
  GstCaps* caps = caps_from_spec(spec);
  state->current_caps = gst_caps_ref(caps);
  if (spec.kind == SampleMediaKind::RawVideo) {
    gst_video_info_from_caps(&state->current_vinfo, caps);
  } else if (spec.kind == SampleMediaKind::Tensor) {
    state->current_tensor_spec = spec;
  }
  gst_caps_unref(caps);
  if (state->max_input_bytes_guard > 0 &&
      spec.required_bytes_actual > state->max_input_bytes_guard) {
    std::ostringstream msg;
    msg << "InputStream::create: initial input exceeds max_input_bytes"
        << " (required=" << spec.required_bytes_actual
        << ", max_input_bytes=" << state->max_input_bytes_guard << ")";
    throw std::runtime_error(msg.str());
  }
  state->alloc_bytes = spec.required_bytes_actual;
  return InputStream(std::move(state));
}

InputStream::InputStream(std::shared_ptr<State> state) : state_(std::move(state)) {}

InputStream::InputStream(InputStream&& other) noexcept : state_(std::move(other.state_)) {}

InputStream& InputStream::operator=(InputStream&& other) noexcept {
  if (this != &other) {
    close();
    state_ = std::move(other.state_);
  }
  return *this;
}

InputStream::~InputStream() {
  close();
}

InputStream::operator bool() const noexcept {
  return state_ && state_->pipeline;
}

bool InputStream::can_push() const noexcept {
  return state_ && state_->appsrc;
}

bool InputStream::can_pull() const noexcept {
  return state_ && state_->appsink;
}

bool InputStream::running() const {
  return state_ && state_->running.load();
}

std::string InputStream::last_error() const {
  if (!state_)
    return {};
  std::lock_guard<std::mutex> lock(state_->error_mu);
  return state_->error;
}

InputStreamStats InputStream::stats() const {
  InputStreamStats out;
  if (!state_)
    return out;
  out.push_count = state_->push_count.load();
  out.push_failures = state_->push_failures.load();
  out.pull_count = state_->pull_count.load();
  out.poll_count = state_->poll_count.load();
  out.dropped_frames = state_->dropped_frames.load();
  out.renegotiations = state_->renegotiations.load();
  out.alloc_grows = state_->alloc_grows.load();
  out.growth_blocked = state_->growth_blocked.load();
  out.renegotiation_blocked = state_->renegotiation_blocked.load();
  const auto avg_us = [](std::uint64_t total_ns, std::uint64_t count) -> double {
    if (count == 0)
      return 0.0;
    return static_cast<double>(total_ns) / static_cast<double>(count) / 1000.0;
  };
  out.avg_alloc_us = avg_us(state_->alloc_ns.load(), out.push_count);
  out.avg_map_us = avg_us(state_->map_ns.load(), out.push_count);
  out.avg_copy_us = avg_us(state_->copy_ns.load(), out.push_count);
  out.avg_push_us = avg_us(state_->push_ns.load(), out.push_count);
  out.avg_pull_wait_us = avg_us(state_->pull_wait_ns.load(), out.pull_count);
  out.avg_decode_us = avg_us(state_->decode_ns.load(), out.pull_count);
  return out;
}

std::string InputStream::diagnostics_summary() const {
  if (!state_ || !state_->diag)
    return {};
  std::ostringstream oss;
  if (!state_->diag->pipeline_string.empty()) {
    oss << "Pipeline:\n" << state_->diag->pipeline_string << "\n";
  }
  const std::string boundary = pipeline_internal::boundary_summary(state_->diag);
  if (!boundary.empty())
    oss << boundary;
  const std::string stages = pipeline_internal::stage_timing_summary(state_->diag);
  if (!stages.empty())
    oss << stages;
  const std::string elements = pipeline_internal::element_timing_summary(state_->diag);
  if (!elements.empty())
    oss << elements;
  if (inputstream_debug_enabled()) {
    GraphReport rep = state_->diag->snapshot_basic();
    if (!rep.bus.empty()) {
      oss << "Bus:\n";
      const size_t max_lines = std::min<size_t>(rep.bus.size(), 10);
      for (size_t i = 0; i < max_lines; ++i) {
        const auto& msg = rep.bus[i];
        oss << "  - [" << msg.type << "] " << msg.src << ": " << msg.detail << "\n";
      }
    }
  }
  return oss.str();
}

std::shared_ptr<DiagCtx> InputStream::diag_ctx() const {
  if (!state_)
    return {};
  return state_->diag;
}

GstElement* InputStream::pipeline_handle() const {
  if (!state_)
    return nullptr;
  return state_->pipeline;
}

void InputStream::set_on_caps_change(std::function<void(const SampleSpec&, const SampleSpec&)> cb) {
  if (!state_)
    return;
  state_->on_caps_change = std::move(cb);
}

void InputStream::start(std::function<void(Sample)> on_output) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::start: stream is closed");
  }
  {
    std::lock_guard<std::mutex> lock(state_->stop_mu);
    state_->stop_started.store(false);
  }
  if (!state_->appsink) {
    throw std::runtime_error("InputStream::start: appsink not available (no Output)");
  }
  if (state_->running.load()) {
    throw std::runtime_error("InputStream::start: stream already running");
  }
  if (state_->opt.reuse_input_buffer) {
    std::fprintf(stderr,
                 "[WARN] InputStream::start: reuse_input_buffer is unsafe for async streams; "
                 "disabling to avoid data races.\n");
    state_->opt.reuse_input_buffer = false;
  }
  state_->callback = std::move(on_output);
  state_->stop_requested.store(false);
  state_->worker_done.store(false);
  state_->teardown_on_exit.store(false);
  state_->use_callbacks = inputstream_use_appsink_callbacks_enabled();
  state_->cb_eos.store(false);
  state_->cb_queue_max = std::max<std::size_t>(1, state_->opt.appsink_max_buffers);
  state_->running.store(true);

  auto st = state_;
  st->worker = std::thread([st]() {
    const int worker_poll_ms = (st->opt.worker_poll_ms > 0) ? st->opt.worker_poll_ms
                                                            : inputstream_worker_poll_ms_default();
    const int timeout_ms = st->opt.timeout_ms;
    const bool timings = st->timing_enabled;
    auto cb = st->callback;
    std::int64_t last_output_ns = 0;
    try {
      if (st->use_callbacks && st->appsink) {
        st->cb_handlers = GstAppSinkCallbacks{
            appsink_eos,
            appsink_new_preroll,
            appsink_new_sample,
        };
        st->cb_ctx = std::make_unique<InputStream::State::CallbackCtx>();
        st->cb_ctx->st = st;
        gst_app_sink_set_callbacks(GST_APP_SINK(st->appsink), &st->cb_handlers, st->cb_ctx.get(),
                                   nullptr);
      }
      while (!st->stop_requested.load()) {
        if (st->teardown_started.load() || !st->pipeline || !st->appsink) {
          break;
        }
        std::chrono::steady_clock::time_point t_wait_start{};
        if (timings)
          t_wait_start = std::chrono::steady_clock::now();
        bool eos_seen = false;
        GstSample* sample = nullptr;
        if (st->use_callbacks) {
          {
            std::unique_lock<std::mutex> lock(st->cb_mu);
            if (st->cb_queue.empty() && !st->stop_requested.load() && !st->cb_eos.load()) {
              st->cb_cv.wait_for(lock, std::chrono::milliseconds(worker_poll_ms));
            }
            if (!st->cb_queue.empty()) {
              sample = st->cb_queue.front();
              st->cb_queue.pop_front();
            } else if (st->cb_eos.load()) {
              eos_seen = true;
            }
          }
          if (!sample) {
            if (!st->teardown_started.load() && st->pipeline) {
              GstCallGuard guard(*st);
              pipeline_internal::throw_if_bus_error(st->pipeline, st->diag, "InputStream::start");
              pipeline_internal::drain_bus(st->pipeline, st->diag, "InputStream::start", &eos_seen);
            }
          }
        } else {
          if (!st->teardown_started.load() && st->pipeline && st->appsink) {
            GstCallGuard guard(*st);
            auto sample_opt = pipeline_internal::try_pull_sample_sliced(
                st->pipeline, st->appsink, worker_poll_ms, st->diag, "InputStream::start",
                &eos_seen);
            if (sample_opt.has_value()) {
              sample = sample_opt.value();
            }
          }
        }
        std::chrono::steady_clock::time_point t_wait_end{};
        if (timings)
          t_wait_end = std::chrono::steady_clock::now();
        if (timings) {
          st->poll_count.fetch_add(1, std::memory_order_relaxed);
        }
        if (!sample) {
          bool sink_eos = false;
          if (!st->teardown_started.load() && st->appsink) {
            GstCallGuard guard(*st);
            sink_eos = gst_app_sink_is_eos(GST_APP_SINK(st->appsink));
          }
          if (eos_seen || sink_eos) {
            if (eos_debug_enabled()) {
              const char* pipeline_name =
                  st->pipeline ? gst_element_get_name(st->pipeline) : "<null>";
              std::fprintf(stderr, "[INPUTSTREAM] eos_seen=%d sink_eos=%d pipeline=%s\n",
                           static_cast<int>(eos_seen), static_cast<int>(sink_eos), pipeline_name);
            }
            st->stop_requested.store(true);
            break;
          }
          if (timeout_ms > 0) {
            const std::int64_t last_push_ns = st->last_push_ns.load(std::memory_order_relaxed);
            if (last_push_ns > 0) {
              const auto now_tp = std::chrono::steady_clock::now();
              const std::int64_t now_ns = static_cast<std::int64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(now_tp.time_since_epoch())
                      .count());
              const std::int64_t last_activity_ns =
                  (last_output_ns > last_push_ns) ? last_output_ns : last_push_ns;
              const std::int64_t timeout_ns = static_cast<std::int64_t>(timeout_ms) * 1000000;
              if (now_ns - last_activity_ns > timeout_ns) {
                if (inputstream_dot_on_timeout_enabled()) {
                  pipeline_internal::maybe_dump_dot(st->pipeline, "inputstream_timeout");
                }
                std::lock_guard<std::mutex> lock(st->error_mu);
                st->error = "InputStream::start: timeout waiting for output";
                st->stop_requested.store(true);
                break;
              }
            }
          }
          continue;
        }
        if (timings) {
          st->pull_wait_ns.fetch_add(
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(t_wait_end - t_wait_start)
                      .count()),
              std::memory_order_relaxed);
        }
        const std::int64_t inflight = st->inflight.load(std::memory_order_relaxed);
        if (inflight > 0) {
          st->inflight.fetch_sub(1, std::memory_order_relaxed);
        }
        std::chrono::steady_clock::time_point t_decode_start{};
        if (timings)
          t_decode_start = std::chrono::steady_clock::now();
        Sample out = decode_sample_from_inputstream_state(*st, sample, "InputStream::start");
        if (st->opt.prepare_output_cpu_visible) {
          (void)pipeline_internal::prepare_sample_for_cpu_read(out);
        }
        if (pipeline_or_graph_debug_enabled()) {
          const char* kind = "Unknown";
          if (out.kind == SampleKind::TensorSet)
            kind = "TensorSet";
          else if (out.kind == SampleKind::Bundle)
            kind = "Bundle";
          std::fprintf(
              stderr,
              "[INPUTSTREAM] output kind=%s frame_id=%lld stream_id=%s media=%s format=%s tag=%s\n",
              kind, static_cast<long long>(out.frame_id), out.stream_id.c_str(),
              out.media_type.c_str(), out.format.c_str(), out.payload_tag.c_str());
        }
        std::chrono::steady_clock::time_point t_decode_end{};
        if (timings)
          t_decode_end = std::chrono::steady_clock::now();
        const bool pool_dbg = inputstream_pool_debug_enabled();
        if (pool_dbg) {
          if (out.kind == SampleKind::TensorSet && !out.tensors.empty()) {
            std::size_t gst_tensors = 0;
            for (const auto& tensor : out.tensors) {
              if (tensor.storage && tensor.storage->kind == simaai::neat::StorageKind::GstSample) {
                gst_tensors++;
              }
            }
            const auto& first = out.tensors.front();
            const bool has_holder = first.storage && static_cast<bool>(first.storage->holder);
            std::fprintf(stderr,
                         "[INPUTSTREAM] out tensors=%zu gstsample_tensors=%zu "
                         "first_storage_kind=%d holder=%s\n",
                         out.tensors.size(), gst_tensors,
                         first.storage ? static_cast<int>(first.storage->kind) : -1,
                         has_holder ? "true" : "false");
          }
          GstBuffer* buf = gst_sample_get_buffer(sample);
          const guint before = buf ? GST_MINI_OBJECT_REFCOUNT_VALUE(buf) : 0u;
          if (buf) {
            GstSample* sample_ref = gst_sample_ref(sample);
            const guint after_sample_ref = GST_MINI_OBJECT_REFCOUNT_VALUE(buf);
            gst_sample_unref(sample_ref);
            const guint after_sample_unref = GST_MINI_OBJECT_REFCOUNT_VALUE(buf);
            std::fprintf(stderr,
                         "[INPUTSTREAM] sample_ref buffer=%p refcnt(before=%u after_ref=%u "
                         "after_unref=%u)\n",
                         static_cast<void*>(buf), before, after_sample_ref, after_sample_unref);
          }
          GstBuffer* buf_ref = buf ? gst_buffer_ref(buf) : nullptr;
          std::fprintf(stderr, "[INPUTSTREAM] before sample_unref buffer=%p refcnt=%u pool=%p\n",
                       static_cast<void*>(buf), before,
                       buf ? static_cast<void*>(buf->pool) : nullptr);
          gst_sample_unref(sample);
          if (buf_ref) {
            const guint after = GST_MINI_OBJECT_REFCOUNT_VALUE(buf_ref);
            std::fprintf(stderr, "[INPUTSTREAM] after sample_unref buffer=%p refcnt=%u pool=%p\n",
                         static_cast<void*>(buf_ref), after,
                         buf_ref ? static_cast<void*>(buf_ref->pool) : nullptr);
            gst_buffer_unref(buf_ref);
          }
        } else {
          gst_sample_unref(sample);
        }
        if (st->stop_requested.load()) {
          break;
        }
        if (timings) {
          st->decode_ns.fetch_add(
              static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             t_decode_end - t_decode_start)
                                             .count()),
              std::memory_order_relaxed);
          st->pull_count.fetch_add(1, std::memory_order_relaxed);
          last_output_ns = static_cast<std::int64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(t_decode_end.time_since_epoch())
                  .count());
        } else {
          const auto now_tp = std::chrono::steady_clock::now();
          last_output_ns = static_cast<std::int64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(now_tp.time_since_epoch())
                  .count());
        }
        if (cb)
          cb(std::move(out));
      }
    } catch (const std::exception& e) {
      if (pipeline_or_graph_debug_enabled()) {
        std::fprintf(stderr, "[INPUTSTREAM] worker_error: %s\n", e.what());
      }
      std::lock_guard<std::mutex> lock(st->error_mu);
      st->error = e.what();
    }
    st->running.store(false);
    if (st->teardown_on_exit.load() && st->pipeline) {
      std::lock_guard<std::mutex> lock(st->pipeline_mu);
      if (st->pipeline && !st->teardown_started.exchange(true)) {
        GstElement* pipeline = st->pipeline;
        st->pipeline = nullptr;
        pipeline_internal::stop_and_unref_no_flush(pipeline, st->opt.prefer_synchronous_teardown);
      }
    }
    st->worker_done.store(true);
  });
}

void InputStream::stop() {
  if (!state_)
    return;
  if (stop_trace_enabled()) {
    std::fprintf(
        stderr, "[STOP] InputStream::stop begin state=%p tid=%zu\n",
        static_cast<void*>(state_.get()),
        static_cast<std::size_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())));
  }
  std::unique_lock<std::mutex> stop_lock(state_->stop_mu);
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] InputStream::stop acquired state=%p\n",
                 static_cast<void*>(state_.get()));
  }
  if (state_->stop_started.load()) {
    if (stop_trace_enabled()) {
      std::fprintf(stderr, "[STOP] InputStream::stop already_started state=%p\n",
                   static_cast<void*>(state_.get()));
    }
    return;
  }
  state_->stop_started.store(true);
  state_->stop_requested.store(true);
  state_->teardown_on_exit.store(false);
  state_->running.store(false);
  GstElement* pipeline_ref = nullptr;
  GstElement* appsrc_ref = nullptr;
  std::unique_ptr<InputStream::State::CallbackCtx> cb_ctx_local;
  bool callbacks_active = false;
  {
    std::lock_guard<std::mutex> lock(state_->pipeline_mu);
    if (state_->use_callbacks && state_->pipeline && !state_->teardown_started.load() &&
        state_->appsink) {
      GstAppSinkCallbacks empty{};
      gst_app_sink_set_callbacks(GST_APP_SINK(state_->appsink), &empty, nullptr, nullptr);
      {
        std::lock_guard<std::mutex> qlock(state_->cb_mu);
        for (auto* sample : state_->cb_queue) {
          gst_sample_unref(sample);
        }
        state_->cb_queue.clear();
      }
      state_->cb_eos.store(true);
      cb_ctx_local = std::move(state_->cb_ctx);
      callbacks_active = true;
      state_->cb_cv.notify_all();
    }
    if (state_->pipeline && !state_->teardown_started.load()) {
      pipeline_ref = state_->pipeline;
    }
    if (state_->appsrc) {
      appsrc_ref = state_->appsrc;
    }
  }
  if (callbacks_active) {
    const int timeout_ms = inputstream_cb_stop_timeout_ms();
    if (timeout_ms > 0) {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
      std::unique_lock<std::mutex> lock(state_->cb_mu);
      while (state_->cb_inflight.load(std::memory_order_acquire) > 0) {
        if (std::chrono::steady_clock::now() >= deadline)
          break;
        state_->cb_cv.wait_for(lock, std::chrono::milliseconds(10));
      }
    }
    if (state_->cb_inflight.load(std::memory_order_acquire) > 0) {
      std::fprintf(stderr, "[INPUTSTREAM] stop: appsink callbacks still in-flight after %dms\n",
                   inputstream_cb_stop_timeout_ms());
      std::terminate();
    }
  }
  // Avoid synchronous flush/EOS events here; they can block while upstream
  // threads hold GStreamer locks. Pipeline teardown will handle flushing.
  if (appsrc_ref) {
    if (stop_trace_enabled()) {
      std::fprintf(stderr, "[STOP] InputStream::stop eos state=%p appsrc=%p\n",
                   static_cast<void*>(state_.get()), static_cast<void*>(appsrc_ref));
    }
    bool expected = false;
    if (state_->eos_sent.compare_exchange_strong(expected, true)) {
      GstFlowReturn ret = gst_app_src_end_of_stream(GST_APP_SRC(appsrc_ref));
      (void)ret;
    }
  }
  const bool stop_unblock = inputstream_stop_unblock_enabled();
  if (stop_unblock && appsrc_ref) {
    GstCallGuard guard(*state_);
    g_object_set(G_OBJECT(appsrc_ref), "block", FALSE, nullptr);
    if (inputstream_debug_enabled() || graph_debug_enabled()) {
      std::fprintf(stderr, "[INPUTSTREAM] stop: appsrc block=false\n");
    }
  }
  const bool stop_flush = inputstream_stop_flush_enabled();
  if (stop_flush && pipeline_ref) {
    if (stop_trace_enabled()) {
      std::fprintf(stderr, "[STOP] InputStream::stop flush state=%p pipeline=%p\n",
                   static_cast<void*>(state_.get()), static_cast<void*>(pipeline_ref));
    }
    const int flush_timeout_ms = inputstream_stop_flush_timeout_ms();
    if (flush_timeout_ms <= 0) {
      GstCallGuard guard(*state_);
      gst_element_send_event(pipeline_ref, gst_event_new_flush_start());
      gst_element_send_event(pipeline_ref, gst_event_new_flush_stop(TRUE));
      if (inputstream_debug_enabled() || graph_debug_enabled()) {
        std::fprintf(stderr, "[INPUTSTREAM] stop: flush_start/stop sent\n");
      }
    } else {
      auto done = std::make_shared<std::atomic<bool>>(false);
      GstElement* flush_pipeline = GST_ELEMENT(gst_object_ref(pipeline_ref));
      auto st = state_;
      std::thread flush_thread([flush_pipeline, st, done]() {
        GstCallGuard guard(*st);
        gst_element_send_event(flush_pipeline, gst_event_new_flush_start());
        gst_element_send_event(flush_pipeline, gst_event_new_flush_stop(TRUE));
        gst_object_unref(flush_pipeline);
        done->store(true, std::memory_order_relaxed);
      });
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(flush_timeout_ms);
      while (!done->load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() >= deadline)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      if (done->load(std::memory_order_relaxed)) {
        flush_thread.join();
        if (inputstream_debug_enabled() || graph_debug_enabled()) {
          std::fprintf(stderr, "[INPUTSTREAM] stop: async flush sent\n");
        }
      } else {
        if (stop_trace_enabled()) {
          std::fprintf(stderr, "[STOP] InputStream::stop flush timeout=%dms; detaching\n",
                       flush_timeout_ms);
        }
        flush_thread.detach();
      }
    }
  }
  const int stop_timeout_ms = inputstream_stop_timeout_ms();
  if (pipeline_ref) {
    if (stop_trace_enabled()) {
      std::fprintf(stderr, "[STOP] InputStream::stop wait_gst_calls state=%p\n",
                   static_cast<void*>(state_.get()));
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(stop_timeout_ms);
    std::unique_lock<std::mutex> lock(state_->gst_mu);
    while (state_->gst_calls.load(std::memory_order_relaxed) > 0) {
      if (std::chrono::steady_clock::now() >= deadline)
        break;
      state_->gst_cv.wait_for(lock, std::chrono::milliseconds(10));
    }
    if (stop_trace_enabled()) {
      std::fprintf(stderr, "[STOP] InputStream::stop wait_gst_calls_done state=%p calls=%d\n",
                   static_cast<void*>(state_.get()),
                   state_->gst_calls.load(std::memory_order_relaxed));
    }
  }
  {
    if (stop_trace_enabled()) {
      std::fprintf(stderr, "[STOP] InputStream::stop teardown state=%p pipeline=%p\n",
                   static_cast<void*>(state_.get()), static_cast<void*>(pipeline_ref));
    }
    std::lock_guard<std::mutex> lock(state_->pipeline_mu);
    if (state_->pipeline && !state_->teardown_started.exchange(true)) {
      GstElement* pipeline = state_->pipeline;
      state_->pipeline = nullptr;
      pipeline_internal::stop_and_unref_no_flush(pipeline, state_->opt.prefer_synchronous_teardown);
    }
  }
  if (state_->worker.joinable()) {
    if (stop_trace_enabled()) {
      std::fprintf(stderr, "[STOP] InputStream::stop join state=%p\n",
                   static_cast<void*>(state_.get()));
    }
    if (stop_timeout_ms > 0) {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(stop_timeout_ms);
      while (!state_->worker_done.load()) {
        if (std::chrono::steady_clock::now() >= deadline)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    if (state_->worker_done.load()) {
      state_->worker.join();
    } else {
      if (pipeline_or_graph_debug_enabled()) {
        std::fprintf(stderr, "[INPUTSTREAM] stop: worker did not exit within %dms; detaching\n",
                     inputstream_stop_timeout_ms());
      }
      state_->worker.detach();
      return;
    }
  }
  state_->running.store(false);
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] InputStream::stop end state=%p\n",
                 static_cast<void*>(state_.get()));
  }
}

void InputStream::stop_async() {
  if (!state_)
    return;
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] InputStream::stop_async\n");
  }
  state_->stop_requested.store(true);
  state_->running.store(false);
  if (state_->use_callbacks) {
    state_->cb_eos.store(true);
    state_->cb_cv.notify_all();
  }
  if (pipeline_or_graph_debug_enabled()) {
    std::fprintf(stderr, "[INPUTSTREAM] stop_async\n");
  }
}

void InputStream::close() {
  if (!state_)
    return;
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] InputStream::close begin\n");
  }
  stop();
  state_->lifetime_token.reset();
  if (state_->pending_buffer) {
    release_input_buffer(state_->pending_buffer, "InputStream::close:pending_buffer");
    state_->pending_buffer = nullptr;
    state_->pending_spec.reset();
    state_->pending_alloc_ns = 0;
    state_->pending_map_ns = 0;
    state_->pending_copy_ns = 0;
  }
  if (state_->reusable_buffer) {
    release_input_buffer(state_->reusable_buffer, "InputStream::close:reusable_buffer");
    state_->reusable_buffer = nullptr;
    state_->reusable_bytes = 0;
  }
  if (state_->current_caps) {
    gst_caps_unref(state_->current_caps);
    state_->current_caps = nullptr;
  }
  if (state_->appsrc) {
    gst_object_unref(state_->appsrc);
    state_->appsrc = nullptr;
  }
  if (state_->appsink) {
    gst_object_unref(state_->appsink);
    state_->appsink = nullptr;
  }
  if (state_->pipeline) {
    if (!state_->teardown_started.exchange(true)) {
      GstElement* pipeline = state_->pipeline;
      state_->pipeline = nullptr;
      pipeline_internal::stop_and_unref_no_flush(pipeline, state_->opt.prefer_synchronous_teardown);
    } else {
      state_->pipeline = nullptr;
    }
  }
  state_.reset();
  if (stop_trace_enabled()) {
    std::fprintf(stderr, "[STOP] InputStream::close end\n");
  }
}

void InputStream::signal_eos() {
  if (!state_ || !state_->appsrc)
    return;
  bool expected = false;
  if (!state_->eos_sent.compare_exchange_strong(expected, true))
    return;
  GstFlowReturn ret = gst_app_src_end_of_stream(GST_APP_SRC(state_->appsrc));
  if (ret != GST_FLOW_OK) {
    const char* flow_name = gst_flow_get_name(ret);
    std::ostringstream oss;
    oss << "InputStream::signal_eos: gst_app_src_end_of_stream failed"
        << " (flow=" << static_cast<int>(ret) << ":" << (flow_name ? flow_name : "<unknown>")
        << "). " << "Hint: pipeline may already be flushing or stopped.";
    throw std::runtime_error(oss.str());
  }
}

void InputStream::drain_before_teardown(int timeout_ms) {
  if (!state_ || !state_->pipeline || !state_->appsink)
    return;
  if (timeout_ms <= 0)
    return;

  // Best-effort EOS to let downstream elements flush gracefully.
  if (state_->appsrc) {
    try {
      signal_eos();
    } catch (const std::exception&) {
      // Ignore EOS errors during teardown drain.
    }
  } else {
    gst_element_send_event(state_->pipeline, gst_event_new_eos());
  }

  const int worker_poll_ms = std::max(5, std::min(50, timeout_ms / 10));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    bool eos_seen = false;
    auto sample_opt = pipeline_internal::try_pull_sample_sliced(
        state_->pipeline, state_->appsink, worker_poll_ms, state_->diag,
        "InputStream::drain_before_teardown", &eos_seen);
    if (sample_opt.has_value()) {
      gst_sample_unref(sample_opt.value());
    }
    if (eos_seen || gst_app_sink_is_eos(GST_APP_SINK(state_->appsink))) {
      break;
    }
  }
}
} // namespace simaai::neat
