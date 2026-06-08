#pragma once

#include "pipeline/Run.h"

#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

struct PluginAttributionResult {
  const GraphNodeMetrics* node = nullptr;
  std::string attribution_source;
  std::string mapping_error;
};

PluginAttributionResult attribute_plugin_latency(const MeasurePluginLatency& plugin,
                                                 const std::vector<GraphNodeMetrics>& nodes);

void inherit_plugin_node_identity(MeasurePluginLatency* plugin, const GraphNodeMetrics& node);

void attribute_plugin_latency_to_nodes(const std::vector<GraphNodeMetrics>& nodes,
                                       std::vector<MeasurePluginLatency>* attributed,
                                       std::vector<MeasurePluginLatency>* unattributed);

} // namespace simaai::neat::pipeline_internal
