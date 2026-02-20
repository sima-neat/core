/**
 * @file
 * @ingroup nodes_groups
 * @brief OutputSpec helpers for input groups.
 */
#pragma once

#include "builder/OutputSpec.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/RtspDecodedInput.h"
#include "nodes/groups/VideoInputGroup.h"

namespace simaai::neat::nodes::groups {

OutputSpec ImageInputGroupOutputSpec(const ImageInputGroupOptions& opt);
OutputSpec RtspDecodedInputOutputSpec(const RtspDecodedInputOptions& opt);
OutputSpec VideoInputGroupOutputSpec(const VideoInputGroupOptions& opt);

} // namespace simaai::neat::nodes::groups
