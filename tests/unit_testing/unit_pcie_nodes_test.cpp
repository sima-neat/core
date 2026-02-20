#include "gst/GstHelpers.h"
#include "nodes/sima/PCIeSink.h"
#include "nodes/sima/PCIeSrc.h"

#include "test_main.h"

#include <string>

RUN_TEST("unit_pcie_nodes_test", [] {
  const bool has_pcie_plugins = simaai::neat::element_exists("neatpciesrc") &&
                                simaai::neat::element_exists("neatpciesink");
  if (!has_pcie_plugins) {
    throw std::runtime_error("pcie plugins missing (neatpciesrc/neatpciesink)");
  }

  {
    simaai::neat::PCIeSrcOptions opt;
    opt.buffer_size = 1000000;
    opt.format = "BGR";
    opt.width = 640;
    opt.height = 480;
    opt.fps_n = 30;
    opt.fps_d = 1;

    auto node = simaai::neat::nodes::PCIeSrc(opt);
    const std::string frag = node->backend_fragment(0);
    require_contains(frag, "neatpciesrc name=n0_pciesrc", "pciesrc fragment name mismatch");
    require_contains(frag, "buffer-size=1000000", "pciesrc buffer-size mismatch");
    require_contains(frag, "capsfilter name=n0_pciesrc_caps", "pciesrc capsfilter missing");

    const auto names = node->element_names(0);
    require(names.size() == 2, "pciesrc element_names size mismatch");
    require(names[0] == "n0_pciesrc", "pciesrc primary element name mismatch");
    require(names[1] == "n0_pciesrc_caps", "pciesrc caps element name mismatch");

    auto* provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(node.get());
    require(provider != nullptr, "pciesrc OutputSpecProvider missing");
    const simaai::neat::OutputSpec spec = provider->output_spec({});
    require(spec.media_type == "video/x-raw", "pciesrc output media_type mismatch");
    require(spec.format == "BGR", "pciesrc output format mismatch");
    require(spec.width == 640 && spec.height == 480, "pciesrc output dimensions mismatch");
    require(spec.fps_num == 30 && spec.fps_den == 1, "pciesrc output fps mismatch");
  }

  {
    simaai::neat::PCIeSinkOptions opt;
    opt.data_buf_name = "processed-output";
    opt.data_buffer_size = 786432;
    opt.num_buffers = 6;
    opt.queue = 1;
    opt.use_multi_buffers = true;
    opt.param_buf_name = "camera_params";
    opt.param_buffer_size = 96;
    opt.sync = false;
    opt.async_state = false;
    opt.max_lateness_ns = 1000;
    opt.processing_deadline_ns = 5000000;
    opt.transmit_kpi = true;
    opt.qos = true;

    auto node = simaai::neat::nodes::PCIeSink(opt);
    const std::string frag = node->backend_fragment(1);
    require_contains(frag, "neatpciesink name=n1_pciesink", "pciesink fragment name mismatch");
    require_contains(frag, "data-buf-name=\"processed-output\"", "pciesink data-buf-name mismatch");
    require_contains(frag, "data-buffer-size=786432", "pciesink data-buffer-size mismatch");
    require_contains(frag, "num-buffers=6", "pciesink num-buffers mismatch");
    require_contains(frag, "queue=1", "pciesink queue mismatch");
    require_contains(frag, "sync=false", "pciesink sync mismatch");
    require_contains(frag, "async=false", "pciesink async mismatch");
    require_contains(frag, "use-multi-buffers=true", "pciesink multi-buffer flag missing");
    require_contains(frag, "param-buffer-size=96", "pciesink param-buffer-size mismatch");
    require_contains(frag, "processing-deadline=5000000", "pciesink processing-deadline mismatch");
    require_contains(frag, "transmit=true", "pciesink transmit mismatch");
    require_contains(frag, "qos=true", "pciesink qos mismatch");

    const auto names = node->element_names(1);
    require(names.size() == 1, "pciesink element_names size mismatch");
    require(names[0] == "n1_pciesink", "pciesink element name mismatch");
  }

  {
    // Default PCIeSink: transmit and qos emitted as false
    auto node = simaai::neat::nodes::PCIeSink({});
    const std::string frag = node->backend_fragment(2);
    require_contains(frag, "transmit=false", "pciesink default transmit mismatch");
    require_contains(frag, "qos=false", "pciesink default qos mismatch");
  }

  {
    // PCIeSrc with unrecognized format sets a note
    simaai::neat::PCIeSrcOptions opt;
    opt.format = "YUYV";
    opt.width = 320;
    opt.height = 240;
    auto node = simaai::neat::nodes::PCIeSrc(opt);
    auto* provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(node.get());
    require(provider != nullptr, "pciesrc OutputSpecProvider missing for unknown format");
    const simaai::neat::OutputSpec spec = provider->output_spec({});
    require(spec.note.find("unrecognized format") != std::string::npos,
            "pciesrc should note unrecognized format");
  }
});
