#include "pipeline/LatencyProfiler.h"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

namespace simaai::neat {

namespace {

std::string fmt_bytes(std::uint64_t bytes) {
  char buf[64];
  if (bytes >= (1ULL << 30)) {
    std::snprintf(buf, sizeof(buf), "%.2f GiB",
                  static_cast<double>(bytes) / static_cast<double>(1ULL << 30));
  } else if (bytes >= (1ULL << 20)) {
    std::snprintf(buf, sizeof(buf), "%.2f MiB",
                  static_cast<double>(bytes) / static_cast<double>(1ULL << 20));
  } else if (bytes >= (1ULL << 10)) {
    std::snprintf(buf, sizeof(buf), "%.2f KiB",
                  static_cast<double>(bytes) / static_cast<double>(1ULL << 10));
  } else {
    std::snprintf(buf, sizeof(buf), "%llu B",
                  static_cast<unsigned long long>(bytes));
  }
  return std::string(buf);
}

std::string fmt_ms(double ms) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f", ms);
  return std::string(buf);
}

std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 4);
  for (char c : in) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char esc[8];
          std::snprintf(esc, sizeof(esc), "\\u%04x",
                        static_cast<unsigned int>(c) & 0xFFu);
          out += esc;
        } else {
          out += c;
        }
    }
  }
  return out;
}

int backend_tid(const std::string& backend) {
  if (backend == "MLA")       return 100;
  if (backend == "A65")       return 200;
  if (backend == "EV74")      return 201;
  if (backend == "BoxDecode") return 300;
  if (backend == "Memcpy")    return 400;
  return 999;
}

}  // namespace

std::string LatencyProfiler::to_text(const ProfilerReport& r) {
  std::ostringstream o;
  o << "== SiMa NEAT Latency Profiler ==\n";
  if (!r.mpk_path.empty()) {
    o << "mpk:        " << r.mpk_path << "\n";
  }
  if (!r.description.empty()) {
    o << "label:      " << r.description << "\n";
  }
  o << "frames:     " << r.frames_total
    << "   warmup:  " << r.warmup_frames << "\n";
  o << "ring_emits: " << r.profiler_emits
    << "   dropped: " << r.profiler_dropped << "\n";
  o << "\n";

  o << "[end-to-end]\n"
    << "  pushed=" << r.end_to_end.inputs_pushed
    << " pulled=" << r.end_to_end.outputs_pulled
    << " dropped_in=" << r.end_to_end.inputs_dropped
    << " dropped_out=" << r.end_to_end.outputs_dropped << "\n"
    << "  avg=" << fmt_ms(r.end_to_end.avg_latency_ms)
    << " min=" << fmt_ms(r.end_to_end.min_latency_ms)
    << " max=" << fmt_ms(r.end_to_end.max_latency_ms) << " ms\n\n";

  o << "[input stream]\n"
    << "  push_count=" << r.input_stream.push_count
    << " pull_count=" << r.input_stream.pull_count
    << " push_failures=" << r.input_stream.push_failures << "\n"
    << "  avg_alloc=" << fmt_ms(r.input_stream.avg_alloc_us / 1000.0)
    << " avg_map=" << fmt_ms(r.input_stream.avg_map_us / 1000.0)
    << " avg_copy=" << fmt_ms(r.input_stream.avg_copy_us / 1000.0)
    << " avg_push=" << fmt_ms(r.input_stream.avg_push_us / 1000.0)
    << " avg_pull_wait=" << fmt_ms(r.input_stream.avg_pull_wait_us / 1000.0)
    << " ms (per-frame)\n\n";

  if (!r.kernel_aggregates.empty()) {
    o << "[kernel invocations]\n"
      << "  backend  phys_in  out  count   avg_ms   min_ms   max_ms   total_ms  kernel  (stage)\n";
    for (const auto& a : r.kernel_aggregates) {
      o << "  " << std::left << std::setw(8) << a.backend << " "
        << std::right << std::setw(7) << a.physical_input_index << " "
        << std::setw(4) << a.output_slot << " "
        << std::setw(6) << a.count << "  "
        << std::setw(7) << fmt_ms(a.avg_ms()) << "  "
        << std::setw(7) << fmt_ms(a.min_ms) << "  "
        << std::setw(7) << fmt_ms(a.max_ms) << "  "
        << std::setw(8) << fmt_ms(a.total_ms) << "  "
        << a.kernel_name;
      if (!a.stage_name.empty() && a.stage_name != a.kernel_name) {
        o << "  (" << a.stage_name << ")";
      }
      o << "\n";
    }
    o << "\n";
  }

  if (!r.diag.element_timings.empty()) {
    o << "[per-element aggregate]   (legacy reuse)\n";
    for (const auto& e : r.diag.element_timings) {
      const double avg_us = e.samples > 0
                                ? (static_cast<double>(e.total_us) /
                                   static_cast<double>(e.samples))
                                : 0.0;
      o << "  " << std::left << std::setw(28) << e.element_name
        << " samples=" << e.samples
        << " avg=" << fmt_ms(avg_us / 1000.0)
        << " max=" << fmt_ms(static_cast<double>(e.max_us) / 1000.0)
        << " ms\n";
    }
    o << "\n";
  }

  if (!r.diag.element_pad_timings.empty()) {
    // Phase A: per-pad rows.  inter_arrival_avg = how often buffers hit this
    // pad (lower = busy stage).  queue_wait = time the buffer spent on the
    // upstream queue (sink pads only); high queue_wait = upstream is faster
    // than this stage / queue is the gate.  bytes = total throughput.
    o << "[per-pad timing]   (Phase A)\n"
      << "  element                       pad        dir   samples  "
         "avg_inter_ms  max_inter_ms  avg_queue_wait_ms  max_queue_wait_ms  bytes\n";
    auto sorted = r.diag.element_pad_timings;
    std::sort(sorted.begin(), sorted.end(),
              [](const RunElementPadTimingStats& a,
                 const RunElementPadTimingStats& b) {
                if (a.element_name != b.element_name)
                  return a.element_name < b.element_name;
                if (a.is_sink != b.is_sink) return a.is_sink && !b.is_sink;
                return a.pad_name < b.pad_name;
              });
    for (const auto& p : sorted) {
      if (p.samples == 0) continue;
      const double avg_inter_us =
          p.samples > 1
              ? (static_cast<double>(p.inter_arrival_total_us) /
                 static_cast<double>(p.samples - 1))
              : 0.0;
      const double avg_qw_us =
          p.queue_wait_samples > 0
              ? (static_cast<double>(p.queue_wait_total_us) /
                 static_cast<double>(p.queue_wait_samples))
              : 0.0;
      o << "  " << std::left << std::setw(30) << p.element_name << " "
        << std::setw(10) << p.pad_name << " "
        << std::setw(5) << (p.is_sink ? "sink" : "src") << " "
        << std::right << std::setw(7) << p.samples << "  "
        << std::setw(12) << fmt_ms(avg_inter_us / 1000.0) << "  "
        << std::setw(12)
        << fmt_ms(static_cast<double>(p.inter_arrival_max_us) / 1000.0) << "  "
        << std::setw(17) << fmt_ms(avg_qw_us / 1000.0) << "  "
        << std::setw(17)
        << fmt_ms(static_cast<double>(p.queue_wait_max_us) / 1000.0) << "  "
        << fmt_bytes(p.bytes) << "\n";
    }
    o << "\n";
  }

  bool any_memcpy = false;
  for (const auto& m : r.memcpy_sites) {
    if (m.calls > 0) { any_memcpy = true; break; }
  }
  if (any_memcpy) {
    o << "[memcpy sites]\n"
      << "  site                       calls   total_ms   max_ms   bytes\n";
    for (const auto& m : r.memcpy_sites) {
      if (m.calls == 0 && m.total_ns == 0) continue;
      o << "  " << std::left << std::setw(26) << m.site_name << " "
        << std::right << std::setw(5) << m.calls << "   "
        << std::setw(8) << fmt_ms(m.total_ms()) << "   "
        << std::setw(6) << fmt_ms(static_cast<double>(m.max_ns) / 1.0e6)
        << "  " << fmt_bytes(m.total_bytes) << "\n";
    }
  }

  return o.str();
}

std::string LatencyProfiler::to_chrome_trace(const ProfilerReport& r) {
  std::ostringstream o;
  o << "{\n  \"traceEvents\": [\n";

  bool first = true;
  for (const auto& inv : r.kernel_invocations) {
    if (!first) o << ",\n";
    first = false;
    const std::uint64_t ts_us  = inv.start_ns / 1000ULL;
    const std::uint64_t dur_us = (inv.end_ns - inv.start_ns) / 1000ULL;
    o << "    {\"name\":\"" << json_escape(inv.kernel_name) << "\","
      << "\"cat\":\"" << json_escape(inv.backend) << "\","
      << "\"ph\":\"X\","
      << "\"pid\":1,\"tid\":" << backend_tid(inv.backend) << ","
      << "\"ts\":" << ts_us << ",\"dur\":" << dur_us << ","
      << "\"args\":{"
      << "\"frame_id\":" << inv.frame_id << ","
      << "\"physical_input_index\":" << inv.physical_input_index << ","
      << "\"output_slot\":" << inv.output_slot << ","
      << "\"request_id\":" << inv.request_id << ","
      << "\"phase\":\"" << json_escape(inv.phase) << "\","
      << "\"stage\":\"" << json_escape(inv.stage_name) << "\","
      << "\"in_segment\":\"" << json_escape(inv.in_segment) << "\","
      << "\"out_segment\":\"" << json_escape(inv.out_segment) << "\","
      << "\"bytes\":" << inv.bytes
      << "}}";
  }

  // Memcpy sites get one X per (site, total) so the visualizer shows them on
  // a separate swimlane.  Use ts=0 since per-call timestamps were aggregated.
  for (const auto& m : r.memcpy_sites) {
    if (m.calls == 0) continue;
    if (!first) o << ",\n";
    first = false;
    o << "    {\"name\":\"" << json_escape(m.site_name) << " (aggregate)\","
      << "\"cat\":\"Memcpy\",\"ph\":\"X\","
      << "\"pid\":1,\"tid\":" << backend_tid("Memcpy") << ","
      << "\"ts\":0,\"dur\":" << (m.total_ns / 1000ULL) << ","
      << "\"args\":{"
      << "\"calls\":" << m.calls << ","
      << "\"total_bytes\":" << m.total_bytes << ","
      << "\"max_ns\":" << m.max_ns
      << "}}";
  }

  o << "\n  ],\n";
  o << "  \"otherData\": {\n"
    << "    \"mpk\":\"" << json_escape(r.mpk_path) << "\",\n"
    << "    \"description\":\"" << json_escape(r.description) << "\",\n"
    << "    \"frames_total\":" << r.frames_total << ",\n"
    << "    \"warmup_frames\":" << r.warmup_frames << ",\n"
    << "    \"profiler_emits\":" << r.profiler_emits << ",\n"
    << "    \"profiler_dropped\":" << r.profiler_dropped << ",\n"
    << "    \"avg_latency_ms\":" << r.end_to_end.avg_latency_ms << ",\n"
    << "    \"min_latency_ms\":" << r.end_to_end.min_latency_ms << ",\n"
    << "    \"max_latency_ms\":" << r.end_to_end.max_latency_ms << "\n"
    << "  }\n"
    << "}\n";
  return o.str();
}

}  // namespace simaai::neat
