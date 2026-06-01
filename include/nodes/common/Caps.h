/**
 * @file
 * @ingroup nodes_common
 * @brief Caps helpers (typed `capsfilter` shortcuts) and the raw-fragment escape hatch.
 *
 * These free functions return Node instances that emit `capsfilter` GStreamer fragments
 * with type-safe arguments — you don't have to hand-write `caps=video/x-raw,...`. The
 * `Custom()` factory is the escape hatch for emitting an arbitrary GStreamer fragment.
 */
#pragma once

#include "builder/Node.h"
#include "contracts/ContractTypes.h"

#include <memory>
#include <string>

namespace simaai::neat::nodes {

/**
 * @brief Escape hatch — emit an arbitrary GStreamer launch-string fragment.
 *
 * Use only when no typed Node fits. Element names inside `fragment` should follow the
 * `n<idx>_<role>` convention if you want them to show up cleanly in `Graph::describe()`.
 *
 * @param fragment Raw GStreamer fragment text (no leading `!`).
 * @param role     The Node's `InputRole`. Set to `Source` if your fragment generates
 *                 input; `Push` if it accepts external samples; default `None` for an
 *                 internal stage.
 */
std::shared_ptr<simaai::neat::Node>
Custom(std::string fragment, simaai::neat::InputRole role = simaai::neat::InputRole::None);

/**
 * @brief Generic caps filter — pin a format / dimensions / fps / memory at this point.
 * @param format Caps `format` string (e.g., `"NV12"`, `"RGB"`).
 * @param width  Width in pixels, or `-1` to leave unconstrained.
 * @param height Height in pixels, or `-1` to leave unconstrained.
 * @param fps    Frame rate, or `-1` to leave unconstrained.
 * @param memory Memory tag (e.g., `Any`, `SysMem`, `MlaShm`).
 */
std::shared_ptr<simaai::neat::Node>
CapsRaw(std::string format, int width = -1, int height = -1, int fps = -1,
        simaai::neat::CapsMemory memory = simaai::neat::CapsMemory::Any);

/// Convenience: pin NV12 in system memory at the given resolution and fps.
std::shared_ptr<simaai::neat::Node> CapsNV12SysMem(int w, int h, int fps);
/// Convenience: pin I420 at the given resolution / fps / memory tag.
std::shared_ptr<simaai::neat::Node>
CapsI420(int w, int h, int fps, simaai::neat::CapsMemory memory = simaai::neat::CapsMemory::Any);

} // namespace simaai::neat::nodes
