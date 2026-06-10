#include "TraceAttribution.h"

#include <algorithm>

namespace simaai::neat::pipeline_internal {
namespace {

std::string plugin_element_name(const MeasurePluginLatency& plugin) {
  if (!plugin.gst_element_name.empty()) {
    return plugin.gst_element_name;
  }
  return plugin.stage_name;
}

bool plugin_segment_matches_node(const MeasurePluginLatency& plugin, const GraphNodeMetrics& node) {
  return plugin.pipeline_segment_id < 0 ||
         node.pipeline_segment_id == static_cast<std::size_t>(plugin.pipeline_segment_id);
}

PluginAttributionResult unique_match(std::vector<const GraphNodeMetrics*> matches,
                                     std::string source, const std::string& missing_error,
                                     const std::string& ambiguous_error) {
  PluginAttributionResult out;
  if (matches.size() == 1U) {
    out.node = matches.front();
    out.attribution_source = std::move(source);
    return out;
  }
  out.attribution_source = "unattributed";
  out.mapping_error = matches.empty() ? missing_error : ambiguous_error;
  return out;
}

} // namespace

PluginAttributionResult attribute_plugin_latency(const MeasurePluginLatency& plugin,
                                                 const std::vector<GraphNodeMetrics>& nodes) {
  if (nodes.empty()) {
    return {nullptr, "unattributed", "unattributed: node metrics unavailable"};
  }

  if (plugin.runtime_node_id >= 0 && plugin.pipeline_segment_id >= 0) {
    std::vector<const GraphNodeMetrics*> matches;
    for (const GraphNodeMetrics& node : nodes) {
      if (node.runtime_node_id != kInvalidRuntimeNodeId &&
          node.runtime_node_id == static_cast<RuntimeNodeId>(plugin.runtime_node_id) &&
          plugin_segment_matches_node(plugin, node)) {
        matches.push_back(&node);
      }
    }
    return unique_match(std::move(matches), "lttng_v2_identity",
                        "unattributed: LTTng runtime_node_id did not match a node metric",
                        "unattributed: LTTng runtime_node_id matched multiple node metrics");
  }

  if (plugin.public_node_id >= 0) {
    const std::string public_id = "p" + std::to_string(plugin.public_node_id);
    std::vector<const GraphNodeMetrics*> matches;
    for (const GraphNodeMetrics& node : nodes) {
      if (std::find(node.public_node_ids.begin(), node.public_node_ids.end(), public_id) !=
          node.public_node_ids.end()) {
        matches.push_back(&node);
      }
    }
    return unique_match(std::move(matches), "lttng_public_node_id",
                        "unattributed: plugin public_node_id did not match a node metric",
                        "unattributed: plugin public_node_id matched multiple node metrics");
  }

  if (!plugin.public_node_ids.empty()) {
    std::vector<const GraphNodeMetrics*> matches;
    for (const GraphNodeMetrics& node : nodes) {
      for (const std::string& public_id : plugin.public_node_ids) {
        if (std::find(node.public_node_ids.begin(), node.public_node_ids.end(), public_id) !=
            node.public_node_ids.end()) {
          matches.push_back(&node);
          break;
        }
      }
    }
    return unique_match(std::move(matches), "lttng_public_node_ids",
                        "unattributed: plugin public_node_ids did not match a node metric",
                        "unattributed: plugin public_node_ids matched multiple node metrics");
  }

  const std::string element = plugin_element_name(plugin);
  if (element.empty()) {
    return {nullptr, "unattributed", "unattributed: LTTng plugin event has no element/stage name"};
  }

  std::vector<const GraphNodeMetrics*> matches;
  for (const GraphNodeMetrics& node : nodes) {
    if (!plugin_segment_matches_node(plugin, node)) {
      continue;
    }
    if (std::find(node.element_names.begin(), node.element_names.end(), element) !=
        node.element_names.end()) {
      matches.push_back(&node);
    }
  }
  return unique_match(std::move(matches), "lttng_element_name",
                      "unattributed: LTTng element did not match a graph element",
                      "unattributed: LTTng element matched multiple graph nodes");
}

void inherit_plugin_node_identity(MeasurePluginLatency* plugin, const GraphNodeMetrics& node) {
  if (!plugin) {
    return;
  }
  if (plugin->pipeline_segment_id < 0 && node.pipeline_segment_id != static_cast<std::size_t>(-1)) {
    plugin->pipeline_segment_id = static_cast<std::int32_t>(node.pipeline_segment_id);
  }
  if (plugin->runtime_node_id < 0 && node.runtime_node_id != kInvalidRuntimeNodeId) {
    plugin->runtime_node_id = static_cast<std::int32_t>(node.runtime_node_id);
  }
  for (const std::string& public_id : node.public_node_ids) {
    if (std::find(plugin->public_node_ids.begin(), plugin->public_node_ids.end(), public_id) ==
        plugin->public_node_ids.end()) {
      plugin->public_node_ids.push_back(public_id);
    }
  }
}

void attribute_plugin_latency_to_nodes(const std::vector<GraphNodeMetrics>& nodes,
                                       std::vector<MeasurePluginLatency>* attributed,
                                       std::vector<MeasurePluginLatency>* unattributed) {
  if (!attributed) {
    return;
  }
  std::vector<MeasurePluginLatency> kept;
  kept.reserve(attributed->size());
  for (MeasurePluginLatency plugin : *attributed) {
    PluginAttributionResult result = attribute_plugin_latency(plugin, nodes);
    plugin.attribution_source = result.attribution_source;
    plugin.mapping_error = result.mapping_error;
    if (result.node) {
      inherit_plugin_node_identity(&plugin, *result.node);
      kept.push_back(std::move(plugin));
    } else if (unattributed) {
      unattributed->push_back(std::move(plugin));
    }
  }
  *attributed = std::move(kept);
}

} // namespace simaai::neat::pipeline_internal
