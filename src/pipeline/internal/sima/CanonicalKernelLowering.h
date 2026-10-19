#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include <simaai/scheduler/client/kernel.h>

namespace simaai::neat::pipeline_internal::sima::kernel_lowering {
namespace rt = ::simaai::scheduler::client;
namespace gs = ::simaai::scheduler::client;
// These lower Core's own semantic/model contracts. No prepared numerical ABI
// object crosses into Scheduler, and checked executable ownership is retained.
rt::Result<rt::TensorSpec> tensor_spec_from_model_descriptor(const sima_ev_tensor_desc&,
                                                             std::string_view meaning);
rt::Result<rt::TensorSpec> tensor_spec_from_model_descriptor(const AuthoredTensorDescriptor&);
rt::Result<std::shared_ptr<KernelStageRecipe>> check_transform_stage(const StageStaticSpec&);
rt::Result<std::shared_ptr<KernelStageRecipe>> check_visual_stage(const StageStaticSpec&);
bool visual_family(std::string_view);
} // namespace simaai::neat::pipeline_internal::sima::kernel_lowering
