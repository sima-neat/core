#include "gst/GstHelpers.h"
#include "nodes/sima/SimaArgMax.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "nodes/sima/SimaRender.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  try {
    const bool has_neatargmax = simaai::neat::element_exists("neatargmax");
    const bool has_simaairender = simaai::neat::element_exists("simaairender");
    require(simaai::neat::element_exists("neatboxdecode"),
            "required plugin missing (neatboxdecode)");

    simaai::neat::require_element("neatboxdecode", "unit_sima_plugins_test");
    if (has_neatargmax) {
      simaai::neat::require_element("neatargmax", "unit_sima_plugins_test");
    }
    if (has_simaairender) {
      simaai::neat::require_element("simaairender", "unit_sima_plugins_test");
    }

    {
      simaai::neat::SimaArgMaxOptions opt;
      opt.config_path = "/tmp/argmax.json";
      opt.sima_allocator_type = 1;
      opt.silent = true;
      opt.emit_signals = false;
      opt.transmit = false;

      auto node = simaai::neat::nodes::SimaArgMax(opt);
      if (has_neatargmax) {
        const std::string frag = node->backend_fragment(2);
        require_contains(frag, "neatargmax name=n2_argmax", "argmax fragment name mismatch");
        require_contains(frag, "config=\"/tmp/argmax.json\"", "argmax config missing");
        require_contains(frag, "sima-allocator-type=1", "argmax allocator missing");
        require_contains(frag, "silent=true", "argmax silent missing");
        require_contains(frag, "emit-signals=false", "argmax emit-signals missing");
        require_contains(frag, "transmit=false", "argmax transmit missing");
      }

      auto* provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(node.get());
      require(provider != nullptr, "argmax OutputSpecProvider missing");
      simaai::neat::OutputSpec spec = provider->output_spec({});
      require(spec.media_type == "application/vnd.simaai.tensor",
              "argmax output media_type mismatch");
      require(spec.format == "ARGMAX", "argmax output format mismatch");
    }

    {
      simaai::neat::SimaRenderOptions opt;
      opt.config_path = "/tmp/render.json";
      opt.sima_allocator_type = 2;
      opt.silent = false;
      opt.emit_signals = true;
      opt.transmit = false;

      auto node = simaai::neat::nodes::SimaRender(opt);
      if (has_simaairender) {
        const std::string frag = node->backend_fragment(3);
        require_contains(frag, "simaairender name=n3_render", "render fragment name mismatch");
        require_contains(frag, "config=\"/tmp/render.json\"", "render config missing");
        require_contains(frag, "sima-allocator-type=2", "render allocator missing");
        require_contains(frag, "silent=false", "render silent missing");
        require_contains(frag, "emit-signals=true", "render emit-signals missing");
        require_contains(frag, "transmit=false", "render transmit missing");
      }

      auto* provider = dynamic_cast<simaai::neat::OutputSpecProvider*>(node.get());
      require(provider != nullptr, "render OutputSpecProvider missing");
      simaai::neat::OutputSpec spec = provider->output_spec({});
      require(spec.note == "simaairender", "render output note mismatch");
    }

    std::cout << "[OK] unit_sima_plugins_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
