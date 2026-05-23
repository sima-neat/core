#include "asset_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace {

namespace fs = std::filesystem;

constexpr int kW1 = 320;
constexpr int kH1 = 240;
constexpr int kW2 = 640;
constexpr int kH2 = 360;
constexpr int kExpectedFrames = 60;

int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

fs::path find_repo_root() {
  std::error_code ec;
  fs::path path = fs::current_path(ec);
  if (ec) {
    return fs::current_path();
  }
  while (!path.empty()) {
    if (fs::exists(path / "tests" / "assets" / "decoder" / "dynamic_caps.h264", ec) && !ec) {
      return path;
    }
    const fs::path parent = path.parent_path();
    if (parent == path)
      break;
    path = parent;
  }
  return fs::current_path();
}

std::string find_named_element(const std::string& gst, const std::string& element) {
  const std::string needle = element + " name=";
  const std::size_t start = gst.find(needle);
  if (start == std::string::npos)
    return "";
  const std::size_t name_start = start + needle.size();
  const std::size_t name_end = gst.find_first_of(" \t\n", name_start);
  if (name_end == std::string::npos)
    return gst.substr(name_start);
  return gst.substr(name_start, name_end - name_start);
}

struct CapsEvents {
  std::mutex mu;
  std::set<std::pair<int, int>> dims;
};

GstPadProbeReturn on_caps_event(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer user_data) {
  if (!info || !user_data)
    return GST_PAD_PROBE_OK;
  if (!(GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM)) {
    return GST_PAD_PROBE_OK;
  }

  GstEvent* ev = GST_PAD_PROBE_INFO_EVENT(info);
  if (!ev || GST_EVENT_TYPE(ev) != GST_EVENT_CAPS)
    return GST_PAD_PROBE_OK;

  GstCaps* caps = nullptr;
  gst_event_parse_caps(ev, &caps);
  if (!caps)
    return GST_PAD_PROBE_OK;

  GstStructure* s = gst_caps_get_structure(caps, 0);
  int out_w = 0;
  int out_h = 0;
  if (gst_structure_get_int(s, "width", &out_w) && gst_structure_get_int(s, "height", &out_h)) {
    auto* state = static_cast<CapsEvents*>(user_data);
    std::lock_guard<std::mutex> lk(state->mu);
    state->dims.insert({out_w, out_h});
  }
  return GST_PAD_PROBE_OK;
}

std::set<std::pair<int, int>> probe_caps_with_gst(const fs::path& combined, int64_t& frames_out) {
  std::ostringstream pipeline_desc;
  pipeline_desc
      << "filesrc location=" << combined.string() << " ! h264parse "
      << "! video/x-h264,parsed=true,stream-format=(string)byte-stream,"
      << "alignment=(string)au "
      << "! neatdecoder name=decoder drop-on-resolution-change=false sima-allocator-type=2 "
      << "! appsink name=mysink emit-signals=false sync=false max-buffers=0 drop=false";

  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_desc.str().c_str(), &err);
  if (!pipeline) {
    std::string msg = "gst_parse_launch failed";
    if (err && err->message)
      msg = err->message;
    if (err)
      g_error_free(err);
    throw std::runtime_error(msg);
  }

  GstElement* appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  require(appsink != nullptr, "appsink not found");
  GstElement* decoder = gst_bin_get_by_name(GST_BIN(pipeline), "decoder");
  require(decoder != nullptr, "decoder not found");

  GstBus* bus = gst_element_get_bus(pipeline);
  require(bus != nullptr, "pipeline bus not found");

  CapsEvents caps_events;
  GstPad* dec_src = gst_element_get_static_pad(decoder, "src");
  require(dec_src != nullptr, "decoder src pad not found");
  gst_pad_add_probe(dec_src, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, on_caps_event, &caps_events,
                    nullptr);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  const int64_t start_ms = now_ms();
  const int64_t timeout_ms = 20000;
  bool eos = false;
  std::string bus_error;
  frames_out = 0;

  while (!eos) {
    GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 200 * GST_MSECOND);
    if (sample) {
      frames_out++;
      gst_sample_unref(sample);
    }

    while (GstMessage* msg = gst_bus_pop(bus)) {
      if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
        GError* gerr = nullptr;
        gchar* dbg = nullptr;
        gst_message_parse_error(msg, &gerr, &dbg);
        if (gerr && gerr->message) {
          bus_error = gerr->message;
        } else {
          bus_error = "unknown gst error";
        }
        if (gerr)
          g_error_free(gerr);
        if (dbg)
          g_free(dbg);
        gst_message_unref(msg);
        eos = true;
        break;
      }
      if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
        gst_message_unref(msg);
        eos = true;
        break;
      }
      gst_message_unref(msg);
    }

    if ((now_ms() - start_ms) > timeout_ms) {
      break;
    }
  }

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(dec_src);
  gst_object_unref(decoder);
  gst_object_unref(bus);
  gst_object_unref(appsink);
  gst_object_unref(pipeline);

  if (!bus_error.empty()) {
    throw std::runtime_error("pipeline error: " + bus_error);
  }

  std::set<std::pair<int, int>> caps_seen;
  {
    std::lock_guard<std::mutex> lk(caps_events.mu);
    caps_seen = caps_events.dims;
  }
  return caps_seen;
}

} // namespace

int main() {
  try {
    simaai::neat::gst_init_once();

    if (!simaai::neat::element_exists("filesrc") || !simaai::neat::element_exists("h264parse") ||
        !simaai::neat::element_exists("appsink") || !simaai::neat::element_exists("neatdecoder")) {
      throw std::runtime_error(
          "Missing required GStreamer elements (filesrc/h264parse/appsink/neatdecoder)");
    }

    const fs::path combined =
        find_repo_root() / "tests" / "assets" / "decoder" / "dynamic_caps.h264";
    require(fs::exists(combined),
            "Missing decoder dynamic fixture tests/assets/decoder/dynamic_caps.h264. "
            "Run tests/tools/make_decoder_dynamic_fixture.py");

    int64_t gst_frames = 0;
    std::set<std::pair<int, int>> caps_seen = probe_caps_with_gst(combined, gst_frames);
    if (gst_frames == 0) {
      throw std::runtime_error("gst probe: no frames decoded");
    }
    if (caps_seen.count({kW1, kH1}) == 0 || caps_seen.count({kW2, kH2}) == 0) {
      std::ostringstream oss;
      oss << "gst probe expected caps " << kW1 << "x" << kH1 << " and " << kW2 << "x" << kH2
          << " but saw:";
      for (const auto& d : caps_seen) {
        oss << " " << d.first << "x" << d.second;
      }
      throw std::runtime_error(oss.str());
    }

    simaai::neat::Graph p;
    p.add(simaai::neat::nodes::FileInput(combined.string()));
    simaai::neat::H264ParseOptions parse_opt;
    parse_opt.config_interval = 1;
    parse_opt.enforce_caps = true;
    parse_opt.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
    parse_opt.stream_format = simaai::neat::H264ParseOptions::StreamFormat::ByteStream;
    p.add(simaai::neat::nodes::H264Parse(parse_opt));
    p.add(simaai::neat::nodes::H264Decode(/*sima_allocator_type=*/2, /*out_format=*/"NV12"));
    p.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(kExpectedFrames)));

    const std::string gst_desc = p.describe_backend();
    const std::string decoder_name = find_named_element(gst_desc, "neatdecoder");

    simaai::neat::RunOptions run_opt;
    run_opt.output_memory = simaai::neat::OutputMemory::Owned;

    simaai::neat::Run runner = p.build(run_opt);
    struct RunGuard {
      simaai::neat::Run* run = nullptr;
      ~RunGuard() {
        if (run)
          run->stop();
      }
    } guard{&runner};

    const int64_t start_ms = now_ms();
    const int64_t timeout_ms = 20000;
    std::set<std::pair<int, int>> dims_seen;
    int64_t total_frames = 0;

    while ((now_ms() - start_ms) < timeout_ms) {
      simaai::neat::Sample out;
      simaai::neat::PullError err;
      const simaai::neat::PullStatus status = runner.pull(200, out, &err);
      if (status == simaai::neat::PullStatus::Timeout) {
        continue;
      }
      if (status == simaai::neat::PullStatus::Closed) {
        break;
      }
      if (status == simaai::neat::PullStatus::Error) {
        std::string msg = err.message.empty() ? "pull failed" : err.message;
        if (!err.code.empty())
          msg += " (code=" + err.code + ")";
        throw std::runtime_error(msg);
      }

      const auto tensors = simaai::neat::tensors_from_sample(out, true);
      if (tensors.size() != 1U) {
        throw std::runtime_error("expected one tensor output");
      }
      const simaai::neat::Tensor& t = tensors.front();
      if (!t.is_nv12()) {
        throw std::runtime_error("expected NV12 tensor output");
      }
      const int out_w = t.width();
      const int out_h = t.height();
      if (out_w <= 0 || out_h <= 0) {
        throw std::runtime_error("invalid tensor dimensions");
      }
      dims_seen.insert({out_w, out_h});
      total_frames++;
      if (dims_seen.count({kW1, kH1}) && dims_seen.count({kW2, kH2})) {
        break;
      }
    }

    if (total_frames == 0) {
      throw std::runtime_error("no frames decoded");
    }
    const std::uint64_t caps_changes =
        decoder_name.empty() ? 0 : caps_changes_for(runner, decoder_name);
    for (const auto& d : dims_seen) {
      if ((d.first == kW1 && d.second == kH1) || (d.first == kW2 && d.second == kH2)) {
        continue;
      }
      std::ostringstream oss;
      oss << "unexpected tensor dims " << d.first << "x" << d.second;
      throw std::runtime_error(oss.str());
    }

    std::cout << "[OK] decoder_dynamic_caps_test passed (" << total_frames
              << " frames, tensor dims seen:";
    for (const auto& d : dims_seen) {
      std::cout << " " << d.first << "x" << d.second;
    }
    if (!decoder_name.empty()) {
      std::cout << ", caps_changes=" << caps_changes;
    }
    std::cout << ", gst caps seen:";
    for (const auto& d : caps_seen) {
      std::cout << " " << d.first << "x" << d.second;
    }
    std::cout << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
