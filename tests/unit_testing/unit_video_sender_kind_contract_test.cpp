#include "builder/internal/InputSpecSpecialization.h"
#include "nodes/groups/internal/VideoSenderRawIngress.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  using namespace simaai::neat;
  using namespace simaai::neat::nodes::groups::internal;
  static_assert(kVideoSenderRawIngressDirectKind == "VideoSenderRawIngress[direct_nv12]");
  static_assert(kVideoSenderRawIngressMaterializeKind == "VideoSenderRawIngress[convert_to_nv12]");
  try {
    const auto node = VideoSenderRawIngress(64, 64, 30);
    if (node->kind() != kVideoSenderRawIngressMaterializeKind)
      throw std::runtime_error("default ingress disagrees with input allocation policy");
    const auto* specializer = dynamic_cast<const internal::InputSpecSpecializer*>(node.get());
    if (!specializer)
      throw std::runtime_error("raw ingress has no specialization contract");
    OutputSpec input;
    input.media_type = "video/x-raw";
    input.format = "NV12";
    input.memory = "SimaAI";
    input.certainty = SpecCertainty::Authoritative;
    internal::InputSpecSpecializationContext context;
    for (const bool layout_aware : {false, true}) {
      context.set_capability(kNeatEncoderInputLayoutAwareCapability, layout_aware);
      const auto selected = specializer->specialize_for_input(input, context);
      const auto expected =
          layout_aware ? kVideoSenderRawIngressDirectKind : kVideoSenderRawIngressMaterializeKind;
      if (selected->kind() != expected)
        throw std::runtime_error("selected ingress disagrees with input allocation policy");
      if (selected->backend_fragment(0).find("neatencoderinput") == std::string::npos)
        throw std::runtime_error(
            "ingress must retain actual DMA/layout validation in either variant");
    }
    // Strict ingress keeps require_direct through specialization and rejects other input.
    const auto strict = VideoSenderRawIngress({.width = 64,
                                               .height = 64,
                                               .fps = 30,
                                               .fallback_element_names = {},
                                               .require_direct = true});
    const auto* strict_specializer =
        dynamic_cast<const internal::InputSpecSpecializer*>(strict.get());
    const auto selected = strict_specializer->specialize_for_input(input, context);
    const auto config = video_sender_raw_ingress_config(*selected);
    if (!config || !config->require_direct ||
        selected->backend_fragment(0).find("require-direct=true") == std::string::npos)
      throw std::runtime_error("strict ingress lost require_direct when specialized");
    const auto rejects = [&](OutputSpec bad, const std::string& text) {
      try {
        (void)strict_specializer->specialize_for_input(bad, context);
      } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(text) != std::string::npos)
          return;
      }
      throw std::runtime_error("strict ingress accepted input it must reject: " + text);
    };
    auto rgb = input;
    rgb.format = "RGB";
    rejects(rgb, "EncoderNv12");
    auto fractional = input;
    fractional.fps_num = 30000;
    fractional.fps_den = 1001;
    rejects(fractional, "exact integer frame rate");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
