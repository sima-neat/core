#include "CustomerGraphView.h"

#include <algorithm>
#include <string>

namespace simaai::neat::runtime {
namespace {

using json = nlohmann::ordered_json;

std::string node_ref(graph::NodeId id) {
  return id == graph::kInvalidNode ? std::string() : "n" + std::to_string(id);
}

bool has_nodes(const json& view) {
  return view.is_object() && view.contains("nodes") && view.at("nodes").is_array() &&
         !view.at("nodes").empty();
}

json lowered_execution_view_as_customer(const json& lowered_view) {
  json out = lowered_view;
  out["topology_source"] = "execution_plan_lowered_view";
  out["mapping_status"] = "mapped_lowered_view";
  out["warnings"] = json::array(
      {"No explicit public/customer graph view was exported; using the execution-plan lowered "
       "topology. For legacy linear Graph::add() runs this is the customer-authored node order."});

  json normalized_nodes = json::array();
  for (const json& node : lowered_view.value("nodes", json::array())) {
    if (!node.is_object()) {
      continue;
    }
    json n = node;
    const std::string id = n.value("id", std::string());
    if (id.empty()) {
      n["id"] =
          "customer_node_" + std::to_string(static_cast<std::size_t>(normalized_nodes.size()));
      n["lowered_node_ids"] = json::array();
    } else {
      n["lowered_node_ids"] = json::array({id});
    }
    normalized_nodes.push_back(std::move(n));
  }
  out["nodes"] = std::move(normalized_nodes);

  json normalized_edges = json::array();
  for (const json& edge : lowered_view.value("edges", json::array())) {
    if (!edge.is_object()) {
      continue;
    }
    json e = edge;
    const std::string id = e.value("id", std::string());
    if (id.empty()) {
      e["id"] =
          "customer_edge_" + std::to_string(static_cast<std::size_t>(normalized_edges.size()));
      e["lowered_edge_ids"] = json::array();
    } else {
      e["lowered_edge_ids"] = json::array({id});
    }
    e["mapping_status"] = "mapped";
    normalized_edges.push_back(std::move(e));
  }
  out["edges"] = std::move(normalized_edges);
  return out;
}

} // namespace

json customer_view_to_json(const ExecutionGraphPlan& plan, const json& public_view,
                           const json& lowered_view) {
  if (!plan.public_nodes.empty()) {
    json out;
    out["topology_source"] = "execution_plan_customer_view";
    out["mapping_status"] = "mapped";
    out["warnings"] = json::array();
    out["nodes"] = json::array();
    for (const auto& node : plan.public_nodes) {
      json n;
      n["id"] = "p" + std::to_string(node.id);
      n["kind"] = node.kind;
      n["label"] = node.label;
      n["endpoint_name"] = node.endpoint_name.empty() ? json(nullptr) : json(node.endpoint_name);
      n["input_endpoint"] = node.input_endpoint;
      n["output_endpoint"] = node.output_endpoint;
      n["runtime_node"] = node_ref(node.runtime_node);
      n["lowered_node_ids"] = node.runtime_node == graph::kInvalidNode
                                  ? json::array()
                                  : json::array({node_ref(node.runtime_node)});
      n["compiler_generated"] = false;
      out["nodes"].push_back(std::move(n));
    }

    out["edges"] = json::array();
    for (const auto& edge : plan.public_edges) {
      json e;
      e["id"] = "pe" + std::to_string(edge.id);
      e["from"] = "p" + std::to_string(edge.from);
      e["to"] = "p" + std::to_string(edge.to);
      e["kind"] = edge.kind;
      e["from_endpoint"] = edge.from_endpoint.empty() ? json(nullptr) : json(edge.from_endpoint);
      e["to_endpoint"] = edge.to_endpoint.empty() ? json(nullptr) : json(edge.to_endpoint);
      e["runtime_from"] = node_ref(edge.runtime_from);
      e["runtime_to"] = node_ref(edge.runtime_to);
      e["lowered_edge_ids"] = json::array();
      for (std::size_t lowered_index : edge.runtime_edge_indices) {
        e["lowered_edge_ids"].push_back("e" + std::to_string(lowered_index));
      }
      e["mapping_status"] = edge.runtime_edge_indices.empty() ? "partial" : "mapped";
      out["edges"].push_back(std::move(e));
    }
    return out;
  }

  if (has_nodes(public_view)) {
    json out = public_view;
    out["topology_source"] = "execution_plan_public_view";
    out["mapping_status"] = "mapped_public_fallback";
    return out;
  }

  if (has_nodes(lowered_view) &&
      lowered_view.value("topology_source", std::string()) == "execution_plan_lowered_view") {
    return lowered_execution_view_as_customer(lowered_view);
  }

  return fallback_customer_view_from_lowered(
      lowered_view, "lowered_view_fallback",
      {"Customer graph view fell back to lowered topology because no public nodes were exported."});
}

json fallback_customer_view_from_lowered(const json& lowered_view, std::string topology_source,
                                         std::vector<std::string> warnings) {
  json out;
  out["topology_source"] = std::move(topology_source);
  out["mapping_status"] = "fallback";
  out["warnings"] = warnings;
  out["nodes"] = json::array();
  out["edges"] = json::array();
  if (!lowered_view.is_object()) {
    return out;
  }

  for (const json& node : lowered_view.value("nodes", json::array())) {
    if (!node.is_object()) {
      continue;
    }
    json n = node;
    const std::string id = n.value("id", std::string());
    n["id"] = id.empty() ? std::string("customer_node_") +
                               std::to_string(static_cast<std::size_t>(out["nodes"].size()))
                         : id;
    n["lowered_node_ids"] = id.empty() ? json::array() : json::array({id});
    n["customer_fallback"] = true;
    out["nodes"].push_back(std::move(n));
  }

  for (const json& edge : lowered_view.value("edges", json::array())) {
    if (!edge.is_object()) {
      continue;
    }
    json e = edge;
    const std::string id = e.value("id", std::string());
    e["id"] = id.empty() ? std::string("customer_edge_") +
                               std::to_string(static_cast<std::size_t>(out["edges"].size()))
                         : id;
    e["lowered_edge_ids"] = id.empty() ? json::array() : json::array({id});
    e["mapping_status"] = "fallback";
    out["edges"].push_back(std::move(e));
  }
  return out;
}

} // namespace simaai::neat::runtime
