#include "gst/GstHelpers.h"
#include "nodes/sima/PCIeSink.h"
#include "nodes/sima/PCIeSrc.h"

#include "test_main.h"

#include <stdexcept>
#include <string>

RUN_TEST("unit_pcie_nodes_test", [] {
  const bool has_pcie_plugins =
      simaai::neat::element_exists("neatpciesrc") && simaai::neat::element_exists("neatpciesink");
  if (!has_pcie_plugins) {
    throw std::runtime_error("pcie plugins missing (neatpciesrc/neatpciesink)");
  }

  {
    simaai::neat::PCIeSrcOptions opt;
    opt.queue = 2;
    opt.buffer_size = 1000000;

    auto node = simaai::neat::nodes::PCIeSrc(opt);
    const std::string frag = node->backend_fragment(0);
    require_contains(frag, "neatpciesrc name=n0_pciesrc", "pciesrc fragment name mismatch");
    require_contains(frag, "queue=2", "pciesrc queue mismatch");
    require_contains(frag, "buffer-size=1000000", "pciesrc buffer-size mismatch");
    require(frag.find("capsfilter") == std::string::npos,
            "pciesrc must use caps received from the host");

    const auto names = node->element_names(0);
    require(names.size() == 1, "pciesrc element_names size mismatch");
    require(names[0] == "n0_pciesrc", "pciesrc primary element name mismatch");
  }

  {
    simaai::neat::PCIeSinkOptions opt;
    opt.config_file = "/tmp/neat-pcie-sink.json";
    opt.queue = 1;
    opt.transmit_kpi = true;

    auto node = simaai::neat::nodes::PCIeSink(opt);
    const std::string frag = node->backend_fragment(1);
    require_contains(frag, "neatpciesink name=n1_pciesink", "pciesink fragment name mismatch");
    require_contains(frag, "queue=1", "pciesink queue mismatch");
    require_contains(frag, "config=\"/tmp/neat-pcie-sink.json\"", "pciesink config mismatch");
    require_contains(frag, "transmit=true", "pciesink transmit mismatch");
    require(frag.find("data-buf-name=") == std::string::npos,
            "pciesink must not emit legacy buffer properties");
    require(frag.find("sync=") == std::string::npos,
            "pciesink must not emit GstBaseSink timing properties");
    require(frag.find("qos=") == std::string::npos,
            "pciesink must not emit GstBaseSink QoS properties");

    const auto names = node->element_names(1);
    require(names.size() == 1, "pciesink element_names size mismatch");
    require(names[0] == "n1_pciesink", "pciesink element name mismatch");
  }

  {
    // Default PCIeSink emits only properties supported by neatpciesink.
    auto node = simaai::neat::nodes::PCIeSink({});
    const std::string frag = node->backend_fragment(2);
    require_contains(frag, "queue=0", "pciesink default queue mismatch");
    require_contains(frag, "transmit=false", "pciesink default transmit mismatch");
    require(frag.find("qos=") == std::string::npos,
            "pciesink default fragment must not emit unsupported qos");
  }
});
