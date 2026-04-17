// tutorial_0024_send_and_receive_via_pcie.cpp
// Story: loopback data between host and SoC over PCIe.
// What you learn:
// - PCIeSrc receives video frames from the host via PCIe.
// - PCIeSink sends processed data back to the host via PCIe.
// - Together they form a round-trip pipeline useful for validation and testing.
//
// Prereqs:
// - SiMa MLSoC target with PCIe link to host.
// - GStreamer plugins: simaaipciesrc, simaaipciesink.
//
// Configure and build (from the project root on the SoC)
//    cmake -S . -B build
//    cmake --build build --target tutorial_0024_send_and_receive_via_pcie
//
// Print the GStreamer pipeline without running
//    ./build/tutorial_0024_send_and_receive_via_pcie --print-gst
//
// Run with defaults (640x480 RGB, 3 seconds)
//    ./build/tutorial_0024_send_and_receive_via_pcie
//
// Run with custom settings
//    ./build/tutorial_0024_send_and_receive_via_pcie --width 1920 --height 1080 --duration-ms 10000
//
// Try:
//   ./build/tutorial_0024_send_and_receive_via_pcie --print-gst
//   ./build/tutorial_0024_send_and_receive_via_pcie --duration-ms 5000
//   ./build/tutorial_0024_send_and_receive_via_pcie --width 640 --height 480
//
// --------------------------------------------------------------------------
// HOST-SIDE COMPANION SCRIPT
// --------------------------------------------------------------------------
// Run this script on host *before* starting the SoC tutorial binary:
//
//   python3 _host_side/send_receive_via_pcie.py --width 640 --height 480 --frames 100
//
// Complete script (reference copy):
//
//   #!/usr/bin/env python3
//   """
//   send_receive_via_pcie.py – send test frames to SoC and read them back via PCIe.
//
//   Uses the GStreamer 'pciehost' element for host↔SoC communication.
//   No SiMa SDK required — only GStreamer with the pciehost plugin.
//
//   Pipeline:  appsrc → capsfilter → pciehost → appsink
//
//   Usage:
//       python3 send_receive_via_pcie.py --width 640 --height 480 --frames 100
//   """
//
//   import argparse
//   import numpy as np
//   import threading
//   import time
//   import sys
//
//   import gi
//   gi.require_version("Gst", "1.0")
//   from gi.repository import Gst
//
//   Gst.init(None)
//
//
//   class LoopbackTest:
//       """Push test frames through pciehost and validate the round-trip."""
//
//       def __init__(self, width, height, num_frames):
//           self.width = width
//           self.height = height
//           self.num_frames = num_frames
//           self.frame_size = width * height * 3  # RGB24
//
//           self.received = 0
//           self.mismatches = 0
//           self.lock = threading.Lock()
//           self.done = threading.Event()
//
//           self._build_pipeline()
//
//       # ── GStreamer pipeline ──────────────────────────────────────────────
//       def _build_pipeline(self):
//           self.pipeline = Gst.Pipeline.new("loopback")
//
//           self.appsrc = Gst.ElementFactory.make("appsrc", "src")
//           capsfilter = Gst.ElementFactory.make("capsfilter", "caps")
//           self.pciehost = Gst.ElementFactory.make("pciehost", "pcie")
//           self.appsink = Gst.ElementFactory.make("appsink", "sink")
//
//           if not self.pciehost:
//               sys.exit("ERROR: 'pciehost' GStreamer element not found. "
//                        "Make sure the pciehost plugin is installed.")
//
//           caps = Gst.caps_from_string(
//               f"video/x-raw,format=RGB,width={self.width},height={self.height}"
//           )
//           capsfilter.set_property("caps", caps)
//
//           self.pciehost.set_property("buffersize", self.frame_size)
//
//           self.appsink.set_property("emit-signals", True)
//           self.appsink.connect("new-sample", self._on_new_sample)
//
//           for el in (self.appsrc, capsfilter, self.pciehost, self.appsink):
//               self.pipeline.add(el)
//
//           self.appsrc.link(capsfilter)
//           capsfilter.link(self.pciehost)
//           self.pciehost.link(self.appsink)
//
//       # ── Receive callback ───────────────────────────────────────────────
//       def _on_new_sample(self, sink):
//           sample = sink.emit("pull-sample")
//           if not sample:
//               return Gst.FlowReturn.ERROR
//
//           buf = sample.get_buffer()
//           ok, mapinfo = buf.map(Gst.MapFlags.READ)
//           if not ok:
//               return Gst.FlowReturn.ERROR
//
//           result = np.frombuffer(bytes(mapinfo.data), dtype=np.uint8)
//           buf.unmap(mapinfo)
//
//           with self.lock:
//               idx = self.received
//               self.received += 1
//
//           # Expected pattern: each frame was filled with (idx % 256).
//           expected = np.full(self.frame_size, idx % 256, dtype=np.uint8)
//           if not np.array_equal(expected, result):
//               print(f"MISMATCH at frame {idx}")
//               with self.lock:
//                   self.mismatches += 1
//           else:
//               print(f"frame {idx}: OK")
//
//           if self.received >= self.num_frames:
//               self.done.set()
//
//           return Gst.FlowReturn.OK
//
//       # ── Run ────────────────────────────────────────────────────────────
//       def run(self):
//           self.pipeline.set_state(Gst.State.PLAYING)
//
//           for i in range(self.num_frames):
//               fill = i % 256
//               frame = np.full(self.frame_size, fill, dtype=np.uint8)
//               buf = Gst.Buffer.new_wrapped(frame.tobytes())
//               self.appsrc.emit("push-buffer", buf)
//               time.sleep(0.033)  # ~30 fps pacing
//
//           # Wait for all responses (timeout after 10 s).
//           self.done.wait(timeout=10.0)
//
//           self.appsrc.emit("end-of-stream")
//           self.pipeline.set_state(Gst.State.NULL)
//
//           ok = self.num_frames - self.mismatches
//           print(f"\nhost loopback complete: {ok}/{self.num_frames} frames OK")
//           return 0 if self.mismatches == 0 else 1
//
//
//   def main():
//       parser = argparse.ArgumentParser(
//           description="PCIe loopback test using GStreamer pciehost element")
//       parser.add_argument("--width",  type=int, default=640)
//       parser.add_argument("--height", type=int, default=480)
//       parser.add_argument("--frames", type=int, default=100)
//       args = parser.parse_args()
//
//       test = LoopbackTest(args.width, args.height, args.frames)
//       sys.exit(test.run())
//
//
//   if __name__ == "__main__":
//       main()
// --------------------------------------------------------------------------

#include "neat/session.h"
#include "neat/nodes.h"
#include "gst/GstHelpers.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

bool wants_help(int argc, char** argv) {
  return has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h");
}

bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

int skip(const std::string& reason) {
  std::cout << "SKIP: " << reason << "\n";
  return 0;
}

} // namespace

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--width <w>] [--height <h>] [--duration-ms <ms>]\n";
  print_common_flags(std::cout);
  std::cout << "  --width <w>          Frame width  (default 640)\n";
  std::cout << "  --height <h>         Frame height (default 480)\n";
  std::cout << "  --duration-ms <ms>   Pipeline run time in ms (default 3000)\n";
  std::cout << "  --sink-buf <name>    PCIeSink data buffer name (default overlay)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    // 0) Check that the required SiMa PCIe plugins are available.
    if (!simaai::neat::element_exists("simaaipciesrc") ||
        !simaai::neat::element_exists("simaaipciesink")) {
      return skip("missing PCIe plugins (simaaipciesrc/simaaipciesink)");
    }

    // 1) Parse CLI arguments.
    const int width = parse_int_arg(argc, argv, "--width", 640);
    const int height = parse_int_arg(argc, argv, "--height", 480);
    const int duration_ms = parse_int_arg(argc, argv, "--duration-ms", 3000);

    std::string sink_buf = "overlay";
    get_arg(argc, argv, "--sink-buf", sink_buf);

    // 2) Configure the source: receive RGB frames from the host.
    simaai::neat::PCIeSrcOptions src_opt;
    src_opt.buffer_size = width * height * 3; // RGB24
    src_opt.format = "RGB";
    src_opt.width = width;
    src_opt.height = height;
    src_opt.fps_n = 30;
    src_opt.fps_d = 1;

    // 3) Configure the sink: send data back to the host.
    simaai::neat::PCIeSinkOptions sink_opt;
    sink_opt.data_buf_name = sink_buf;
    sink_opt.data_buffer_size = width * height * 3;
    sink_opt.num_buffers = 4; // keep memory usage low on SoC
    sink_opt.sync = false;    // run as fast as possible for loopback

    // 4) Build the pipeline: PCIeSrc → PCIeSink.
    simaai::neat::Session p;
    p.add(simaai::neat::nodes::PCIeSrc(src_opt));
    p.add(simaai::neat::nodes::PCIeSink(sink_opt));

    if (wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    std::cout << "Pipeline graph:\n" << p.describe() << "\n";
    std::cout << "Running PCIe loopback for " << duration_ms << " ms ...\n";
    std::cout << "(Start the host-side script now if not already running.)\n";

    // 5) Build and run the pipeline
    auto run = p.build();

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

    run.stop();

    std::cout << "[OK] tutorial_0024 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
