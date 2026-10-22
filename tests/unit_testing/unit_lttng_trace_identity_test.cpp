#include "pipeline/gst/TraceIdentity.h"
#include "test_main.h"

#include <gst/gst.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {
struct Snapshot {
  std::uint64_t run = 0;
  std::uint64_t graph = 0;
  unsigned segment = 0;
  int runtime_node = -1;
  int public_node = -1;
  std::string instance;
};
struct ProbeState {
  bool enabled = false;
  Snapshot current;
  std::vector<Snapshot> committed;
  std::vector<std::string> writes;
};
struct TraceProbe {
  GstElement parent;
  ProbeState* state;
};
struct TraceProbeClass {
  GstElementClass parent;
};
G_DEFINE_TYPE(TraceProbe, trace_probe, GST_TYPE_ELEMENT)

enum Property { None, Enabled, Run, Graph, Segment, RuntimeNode, PublicNode, Instance };

void set_property(GObject* object, guint id, const GValue* value, GParamSpec* spec) {
  auto& s = *reinterpret_cast<TraceProbe*>(object)->state;
  s.writes.emplace_back(spec->name);
  switch (id) {
  case Enabled:
    s.enabled = g_value_get_boolean(value);
    if (s.enabled)
      s.committed.push_back(s.current);
    break;
  case Run:
    s.current.run = g_value_get_uint64(value);
    break;
  case Graph:
    s.current.graph = g_value_get_uint64(value);
    break;
  case Segment:
    s.current.segment = g_value_get_uint(value);
    break;
  case RuntimeNode:
    s.current.runtime_node = g_value_get_int(value);
    break;
  case PublicNode:
    s.current.public_node = g_value_get_int(value);
    break;
  case Instance:
    s.current.instance = g_value_get_string(value);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
  }
}
void get_property(GObject* object, guint id, GValue* value, GParamSpec* spec) {
  const auto& s = *reinterpret_cast<TraceProbe*>(object)->state;
  if (id == Enabled)
    g_value_set_boolean(value, s.enabled);
  else if (id == Instance)
    g_value_set_string(value, s.current.instance.c_str());
  else
    G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
}
void finalize(GObject* object) {
  delete reinterpret_cast<TraceProbe*>(object)->state;
  G_OBJECT_CLASS(trace_probe_parent_class)->finalize(object);
}
void trace_probe_class_init(TraceProbeClass* klass) {
  auto* object = G_OBJECT_CLASS(klass);
  object->set_property = set_property;
  object->get_property = get_property;
  object->finalize = finalize;
  constexpr auto flags = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
  g_object_class_install_property(
      object, Enabled, g_param_spec_boolean("trace-enabled", "enabled", "enabled", FALSE, flags));
  g_object_class_install_property(
      object, Run,
      g_param_spec_uint64("trace-run-id-hash", "run", "run", 0, G_MAXUINT64, 0, flags));
  g_object_class_install_property(
      object, Graph,
      g_param_spec_uint64("trace-graph-id-hash", "graph", "graph", 0, G_MAXUINT64, 0, flags));
  g_object_class_install_property(
      object, Segment,
      g_param_spec_uint("trace-segment-id", "segment", "segment", 0, G_MAXUINT, 0, flags));
  g_object_class_install_property(
      object, RuntimeNode,
      g_param_spec_int("trace-runtime-node-id", "runtime", "runtime", -1, G_MAXINT, -1, flags));
  g_object_class_install_property(
      object, PublicNode,
      g_param_spec_int("trace-public-node-id", "public", "public", -1, G_MAXINT, -1, flags));
  g_object_class_install_property(
      object, Instance,
      g_param_spec_string("trace-plugin-instance-id", "instance", "instance", "", flags));
}
void trace_probe_init(TraceProbe* probe) {
  probe->state = new ProbeState;
}
void require_transaction(const ProbeState& state) {
  require(
      state.writes.size() == 8 && state.writes.front() == "trace-enabled" &&
          state.writes.back() == "trace-enabled",
      "each cold session must disable, write all identity fields, and enable last exactly once");
}
} // namespace

RUN_TEST("unit_lttng_trace_identity_test", ([] {
           using namespace simaai::neat;
           using pipeline_internal::configure_lttng_trace_pipeline;
           gst_init(nullptr, nullptr);
           auto* pipeline = gst_pipeline_new("trace-test");
           auto* owned = reinterpret_cast<TraceProbe*>(
               g_object_new(trace_probe_get_type(), "name", "owned", nullptr));
           auto* fallback = reinterpret_cast<TraceProbe*>(
               g_object_new(trace_probe_get_type(), "name", "fallback", nullptr));
           gst_bin_add_many(GST_BIN(pipeline), GST_ELEMENT(owned), GST_ELEMENT(fallback), nullptr);
           GraphNodeMetrics node;
           node.element_names = {"owned"};
           node.pipeline_segment_id = 2;
           node.runtime_node_id = 5;
           node.public_node_ids = {"p3"};
           const std::vector<const GraphNodeMetrics*> nodes{&node};
           std::vector<std::string> configured;
           configure_lttng_trace_pipeline(pipeline, nodes, 2, 17, 23, true, 1, &configured);
           require(configured.size() == 2, "collect exact attributed and fallback identities");
           require_transaction(*owned->state);
           require_transaction(*fallback->state);
           const auto first = owned->state->committed.back();
           require(first.run == 17 && first.graph == 23 && first.segment == 2 &&
                       first.runtime_node == 5 && first.public_node == 3 &&
                       first.instance.ends_with(".m1"),
                   "enable must commit the complete attributed identity, not zero/stale fields");
           require(fallback->state->committed.back().runtime_node == -1 &&
                       fallback->state->committed.back().instance.ends_with("fallback.m1"),
                   "fallback attribution remains explicitly unknown");
           require(std::find(configured.begin(), configured.end(), first.instance) !=
                       configured.end(),
                   "returned allowlist must match the actual enabled instance");

           // Deliberately leave old enabled state installed: fallback must be refreshed, not
           // skipped.
           owned->state->writes.clear();
           fallback->state->writes.clear();
           configured.clear();
           configure_lttng_trace_pipeline(pipeline, nodes, 2, 17, 23, true, 2, &configured);
           require_transaction(*owned->state);
           require_transaction(*fallback->state);
           require(owned->state->committed.size() == 2 && fallback->state->committed.size() == 2 &&
                       owned->state->committed.back().instance.ends_with(".m2") &&
                       fallback->state->committed.back().instance.ends_with(".m2"),
                   "a repeated measurement must replace both attributed and fallback epochs");
           require(first.instance != owned->state->committed.back().instance &&
                       first.run == owned->state->committed.back().run &&
                       first.graph == owned->state->committed.back().graph,
                   "measurement epoch must not change logical graph/run identity");
           configured.clear();
           configure_lttng_trace_pipeline(pipeline, nodes, 2, 17, 23, false, 2, &configured);
           require(!owned->state->enabled && !fallback->state->enabled && configured.empty(),
                   "stop disarms every trace-capable element without inventing enabled identities");
           require(owned->state->committed.size() == 2 && fallback->state->committed.size() == 2,
                   "disable must not commit another session or mutate an earlier snapshot");
           gst_object_unref(pipeline);
         }));
