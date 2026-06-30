#include "builder/Node.h"
#include "builder/OutputSpec.h"
#include "nodes/common/Caps.h"
#include "nodes/io/Input.h"
#include "nodes/groups/GroupOutputSpec.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>

namespace simaai::neat::graph {
simaai::neat::InputOptions input_opts_from_spec(const simaai::neat::OutputSpec& spec,
                                                bool complete);
}

int main() {
  try {
    simaai::neat::OutputSpec nv12;
    nv12.media_type = "video/x-raw";
    nv12.format = "NV12";
    nv12.width = 4;
    nv12.height = 2;
    const std::size_t nv12_bytes = simaai::neat::expected_byte_size(nv12);
    require(nv12_bytes == 12, "NV12 byte size mismatch");

    simaai::neat::InputOptions appsrc_opt;
    appsrc_opt.payload_type = simaai::neat::PayloadType::Image;
    appsrc_opt.format = simaai::neat::FormatTag::RGB;
    appsrc_opt.width = 10;
    appsrc_opt.height = 8;
    appsrc_opt.fps_n = 24;
    appsrc_opt.fps_d = 1;
    appsrc_opt.use_simaai_pool = false;

    auto appsrc = simaai::neat::nodes::Input(appsrc_opt);
    const auto* appsrc_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(appsrc.get());
    require(appsrc_provider != nullptr, "Input should provide output spec");
    const simaai::neat::OutputSpec input_spec = appsrc_provider->output_spec({});
    require(input_spec.fps_num == 24 && input_spec.fps_den == 1,
            "Input output_spec should preserve fps options");
    require(input_spec.memory == "SystemMemory",
            "Auto policy should preserve use_simaai_pool=false memory reporting");

    simaai::neat::InputOptions ev74_opt = appsrc_opt;
    ev74_opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
    ev74_opt.use_simaai_pool = false;
    auto ev74_input = simaai::neat::nodes::Input(ev74_opt);
    const auto* ev74_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(ev74_input.get());
    require(ev74_provider != nullptr, "EV74 Input should provide output spec");
    require(ev74_provider->output_spec({}).memory == "SimaAI",
            "Ev74 memory policy should report SimaAI memory");

    simaai::neat::InputOptions system_opt = appsrc_opt;
    system_opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
    system_opt.use_simaai_pool = true;
    auto system_input = simaai::neat::nodes::Input(system_opt);
    const auto* system_provider =
        dynamic_cast<const simaai::neat::OutputSpecProvider*>(system_input.get());
    require(system_provider != nullptr, "SystemMemory Input should provide output spec");
    require(system_provider->output_spec({}).memory == "SystemMemory",
            "SystemMemory policy should report SystemMemory");

    auto caps =
        simaai::neat::nodes::CapsRaw("GRAY8", 10, 8, 30, simaai::neat::CapsMemory::SystemMemory);
    std::vector<std::shared_ptr<simaai::neat::Node>> nodes{appsrc, caps};

    simaai::neat::OutputSpec derived = simaai::neat::derive_output_spec(nodes);
    std::cerr << "[DBG] derived media=" << derived.media_type << " format=" << derived.format
              << " w=" << derived.width << " h=" << derived.height << " d=" << derived.depth
              << " layout=" << derived.layout << " dtype=" << derived.dtype
              << " mem=" << derived.memory << " bytes=" << derived.byte_size << "\n";
    require(derived.media_type == "video/x-raw", "derived media_type mismatch");
    require(derived.format == "GRAY8", "derived format mismatch");
    require(derived.width == 10 && derived.height == 8, "derived shape mismatch");
    require(derived.layout == "HW", "derived layout mismatch");
    require(derived.byte_size == 80, "derived byte_size mismatch");
    require(derived.fps_num == 30 && derived.fps_den == 1, "derived fps mismatch");

    simaai::neat::InputOptions boundary_opt =
        simaai::neat::graph::input_opts_from_spec(derived, true);
    require(boundary_opt.fps_n == 30 && boundary_opt.fps_d == 1,
            "boundary input options should preserve derived fps");

    simaai::neat::nodes::groups::ImageInputGroupOptions img_opt;
    img_opt.output_caps.enable = true;
    img_opt.output_caps.format = simaai::neat::FormatTag::NV12;
    img_opt.output_caps.width = 32;
    img_opt.output_caps.height = 16;
    img_opt.output_caps.fps = 15;
    simaai::neat::OutputSpec img_spec =
        simaai::neat::nodes::groups::ImageInputGroupOutputSpec(img_opt);
    require(img_spec.format == "NV12", "image group spec format mismatch");
    require(img_spec.width == 32 && img_spec.height == 16, "image group spec shape mismatch");
    require(img_spec.byte_size == 32 * 16 * 3 / 2, "image group spec byte size mismatch");

    std::cout << "[OK] unit_outputspec_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
