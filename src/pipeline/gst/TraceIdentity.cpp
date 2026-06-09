#define SIMA_NEAT_INTERNAL 1
#include "TraceIdentity.h"

#include "pipeline/runtime/ExecutionGraphRuntime.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/runtime/RunInternal.h"

#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <memory>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace {

bool object_has_property(GObject* object, const char* name) {
  if (!object || !name || !*name) {
    return false;
  }
  GObjectClass* klass = G_OBJECT_GET_CLASS(object);
  return klass && g_object_class_find_property(klass, name) != nullptr;
}

void set_bool_property_if_present(GObject* object, const char* name, gboolean value) {
  if (object_has_property(object, name)) {
    g_object_set(object, name, value, nullptr);
  }
}

void set_uint64_property_if_present(GObject* object, const char* name, guint64 value) {
  if (object_has_property(object, name)) {
    g_object_set(object, name, value, nullptr);
  }
}

void set_uint_property_if_present(GObject* object, const char* name, guint value) {
  if (object_has_property(object, name)) {
    g_object_set(object, name, value, nullptr);
  }
}

void set_int_property_if_present(GObject* object, const char* name, gint value) {
  if (object_has_property(object, name)) {
    g_object_set(object, name, value, nullptr);
  }
}

void set_string_property_if_present(GObject* object, const char* name, const std::string& value) {
  if (object_has_property(object, name)) {
    g_object_set(object, name, value.c_str(), nullptr);
  }
}

bool get_bool_property_if_present(GObject* object, const char* name, gboolean* value) {
  if (!value || !object_has_property(object, name)) {
    return false;
  }
  g_object_get(object, name, value, nullptr);
  return true;
}

int parse_public_node_id(const GraphNodeMetrics& node) {
  for (const std::string& public_id : node.public_node_ids) {
    if (public_id.size() < 2 || public_id[0] != 'p') {
      continue;
    }
    char* end = nullptr;
    const long parsed = std::strtol(public_id.c_str() + 1, &end, 10);
    if (end && *end == '\0' && parsed >= 0 && parsed <= INT32_MAX) {
      return static_cast<int>(parsed);
    }
  }
  return -1;
}

std::string sanitize_instance_token(std::string value) {
  for (char& c : value) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (!std::isalnum(uc) && c != '_' && c != '-' && c != '.') {
      c = '_';
    }
  }
  return value;
}

std::string make_plugin_instance_id(std::uint64_t run_id_hash, std::uint64_t graph_id_hash,
                                    std::size_t segment_id, graph::NodeId runtime_node_id,
                                    const std::string& element_name) {
  std::ostringstream os;
  os << "r" << std::hex << run_id_hash << ".g" << graph_id_hash << std::dec << ".s" << segment_id
     << ".n";
  if (runtime_node_id == graph::kInvalidNode) {
    os << "invalid";
  } else {
    os << static_cast<std::size_t>(runtime_node_id);
  }
  os << "." << sanitize_instance_token(element_name);
  return os.str();
}

void apply_to_pipeline(GstElement* pipeline, const std::vector<const GraphNodeMetrics*>& nodes,
                       std::uint64_t run_id_hash, std::uint64_t graph_id_hash, bool enable) {
  if (!pipeline || !GST_IS_BIN(pipeline)) {
    return;
  }

  for (const GraphNodeMetrics* node : nodes) {
    if (!node) {
      continue;
    }
    std::vector<std::string> elements = node->element_names;
    for (const GraphElementMetrics& element : node->elements) {
      if (!element.name.empty() &&
          std::find(elements.begin(), elements.end(), element.name) == elements.end()) {
        elements.push_back(element.name);
      }
    }
    for (const std::string& element_name : elements) {
      if (element_name.empty()) {
        continue;
      }
      GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline), element_name.c_str());
      if (!element) {
        continue;
      }
      GObject* object = G_OBJECT(element);
      set_bool_property_if_present(object, "trace-enabled", enable ? TRUE : FALSE);
      if (enable) {
        const guint segment_id = static_cast<guint>(
            std::min<std::size_t>(node->pipeline_segment_id, static_cast<std::size_t>(G_MAXUINT)));
        const gint runtime_node_id = (node->runtime_node_id != graph::kInvalidNode &&
                                      node->runtime_node_id <= static_cast<graph::NodeId>(G_MAXINT))
                                         ? static_cast<gint>(node->runtime_node_id)
                                         : -1;
        const gint public_node_id = static_cast<gint>(parse_public_node_id(*node));
        const std::string instance_id =
            make_plugin_instance_id(run_id_hash, graph_id_hash, node->pipeline_segment_id,
                                    node->runtime_node_id, element_name);
        set_uint64_property_if_present(object, "trace-run-id-hash", run_id_hash);
        set_uint64_property_if_present(object, "trace-graph-id-hash", graph_id_hash);
        set_uint_property_if_present(object, "trace-segment-id", segment_id);
        set_int_property_if_present(object, "trace-runtime-node-id", runtime_node_id);
        set_int_property_if_present(object, "trace-public-node-id", public_node_id);
        set_string_property_if_present(object, "trace-plugin-instance-id", instance_id);
      }
      gst_object_unref(element);
    }
  }
}

void apply_to_trace_capable_fallback(GstElement* pipeline, std::size_t pipeline_segment_id,
                                     std::uint64_t run_id_hash, std::uint64_t graph_id_hash,
                                     bool enable) {
  if (!pipeline || !GST_IS_BIN(pipeline)) {
    return;
  }

  GstIterator* it = gst_bin_iterate_recurse(GST_BIN(pipeline));
  if (!it) {
    return;
  }

  GValue item = G_VALUE_INIT;
  bool done = false;
  while (!done) {
    switch (gst_iterator_next(it, &item)) {
    case GST_ITERATOR_OK: {
      GstObject* object_value = GST_OBJECT(g_value_get_object(&item));
      if (object_value && GST_IS_ELEMENT(object_value)) {
        GstElement* element = GST_ELEMENT(object_value);
        GObject* object = G_OBJECT(element);
        if (object_has_property(object, "trace-enabled")) {
          gboolean already_enabled = FALSE;
          const bool had_enabled =
              get_bool_property_if_present(object, "trace-enabled", &already_enabled);
          // Node-attributed application above is more specific.  Do not overwrite its
          // runtime-node identity while enabling, but always disable trace-capable plugins.
          if (!enable || !had_enabled || !already_enabled) {
            set_bool_property_if_present(object, "trace-enabled", enable ? TRUE : FALSE);
            if (enable) {
              const std::string element_name = GST_ELEMENT_NAME(element)
                                                   ? std::string(GST_ELEMENT_NAME(element))
                                                   : std::string();
              const guint segment_id = static_cast<guint>(
                  std::min<std::size_t>(pipeline_segment_id, static_cast<std::size_t>(G_MAXUINT)));
              const std::string instance_id =
                  make_plugin_instance_id(run_id_hash, graph_id_hash, pipeline_segment_id,
                                          graph::kInvalidNode, element_name);
              set_uint64_property_if_present(object, "trace-run-id-hash", run_id_hash);
              set_uint64_property_if_present(object, "trace-graph-id-hash", graph_id_hash);
              set_uint_property_if_present(object, "trace-segment-id", segment_id);
              set_int_property_if_present(object, "trace-runtime-node-id", -1);
              set_int_property_if_present(object, "trace-public-node-id", -1);
              set_string_property_if_present(object, "trace-plugin-instance-id", instance_id);
            }
          }
        }
      }
      g_value_unset(&item);
      break;
    }
    case GST_ITERATOR_RESYNC:
      gst_iterator_resync(it);
      break;
    case GST_ITERATOR_ERROR:
    case GST_ITERATOR_DONE:
      done = true;
      break;
    }
  }
  gst_iterator_free(it);
}

} // namespace

void apply_lttng_trace_identity(const Run& run, const std::vector<GraphNodeMetrics>& nodes,
                                std::uint64_t run_id_hash, std::uint64_t graph_id_hash, bool enable,
                                bool enable_message_events) {
  const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
  if (!core) {
    return;
  }

  if (!core->graph_execution_) {
    std::vector<const GraphNodeMetrics*> linear_nodes;
    linear_nodes.reserve(nodes.size());
    for (const GraphNodeMetrics& node : nodes) {
      linear_nodes.push_back(&node);
    }
    GstElement* pipeline = core->pipeline.stream.pipeline_handle();
    apply_to_pipeline(pipeline, linear_nodes, run_id_hash, graph_id_hash, enable);
    apply_to_trace_capable_fallback(pipeline, 0U, run_id_hash, graph_id_hash, enable);
    return;
  }

  auto& execution = const_cast<runtime::ExecutionGraphRuntime&>(core->graph_execution());
  execution.message_trace_enabled.store(enable && enable_message_events, std::memory_order_release);
  execution.trace_run_id_hash.store(enable ? run_id_hash : 0, std::memory_order_relaxed);
  execution.trace_graph_id_hash.store(enable ? graph_id_hash : 0, std::memory_order_relaxed);
  std::unordered_map<std::size_t, std::vector<const GraphNodeMetrics*>> nodes_by_segment;
  for (const GraphNodeMetrics& node : nodes) {
    nodes_by_segment[node.pipeline_segment_id].push_back(&node);
  }
  for (const auto& pipe : execution.pipelines) {
    if (!pipe || !pipe->run_core) {
      continue;
    }
    const auto it = nodes_by_segment.find(pipe->seg.id);
    if (it == nodes_by_segment.end()) {
      apply_to_trace_capable_fallback(pipe->run_core->pipeline.stream.pipeline_handle(),
                                      pipe->seg.id, run_id_hash, graph_id_hash, enable);
      continue;
    }
    apply_to_pipeline(pipe->run_core->pipeline.stream.pipeline_handle(), it->second, run_id_hash,
                      graph_id_hash, enable);
    apply_to_trace_capable_fallback(pipe->run_core->pipeline.stream.pipeline_handle(), pipe->seg.id,
                                    run_id_hash, graph_id_hash, enable);
  }
}

} // namespace simaai::neat::pipeline_internal
