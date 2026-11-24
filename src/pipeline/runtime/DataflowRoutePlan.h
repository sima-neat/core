#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The planner of a model route on the Dataflow backend as a pure function of the stage manifest
// and of what the build says of the input (DataflowModelRoute.h). These two functions have
// libsima_neat's default visibility, as Core's other internal functions do, so that a unit test
// feeds the planner manifests of its own: stated on each, since the route the planner returns is
// the hidden type of DataflowModelRoute.h, which would otherwise hide the function. Nothing of the
// Dataflow runtime is named here.

#include "builder/Node.h"
#include "nodes/io/Input.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

struct __attribute__((visibility("hidden"))) DataflowModelRoute; // DataflowModelRoute.h

// The stage manifest of `route` (the nodes of GStreamer's segment: the Input, the model's stage
// nodes, the Output) as GStreamer's build compiles, renders and lowers it under `options`: each
// stage's contract and, for a kernel stage, its checked kernel recipe. nullopt with `why` (the
// model its subject) when the contracts do not compile or the kernel stages do not lower. Cold; no
// device is touched.
__attribute__((visibility("default")))
std::optional<pipeline_internal::sima::SimaPluginStaticManifest>
dataflow_model_manifest(const std::vector<std::shared_ptr<Node>>& route,
                        const GraphOptions& options, pipeline_internal::DataflowNotCarried* why);

// The route's stages as this backend runs them, read from `manifest`: one operator per stage, the
// image stage's frames as `declared` (the Input's options) and `seed` (the build's seed, or none)
// say them, an EV74 kernel stage only where `ev74_obstacle` is empty (otherwise it says what the
// EV74 runs instead, soc::ev74_runs_kernels_firmware). Null with `why` otherwise: its subject the
// stage (or the route, for a fact that is no one stage's), its fact what of it this backend does
// not carry; a stage placed on an EV74 that does not run the Kernels firmware is the board's fact
// (infra.dispatcher_unavailable, DataflowRemedy::BackendOrFirmware). A pure function: no device, no
// file, no system state is read, and nothing is loaded (the kernel requests are the manifest's,
// checked by the Kernels package).
__attribute__((visibility("default"))) std::shared_ptr<const DataflowModelRoute>
plan_dataflow_route_stages(const pipeline_internal::sima::SimaPluginStaticManifest& manifest,
                           const InputOptions& declared, std::span<const Tensor> seed,
                           const std::string& ev74_obstacle,
                           pipeline_internal::DataflowNotCarried* why);

} // namespace simaai::neat::runtime
