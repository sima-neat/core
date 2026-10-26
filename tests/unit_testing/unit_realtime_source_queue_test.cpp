#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/RunCore.h"
#include "pipeline/runtime/RunInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>
#include <memory>
#include <string>

namespace {

class LiveCpuSource final : public simaai::neat::Node {
public:
  std::string kind() const override {
    return "LiveCpuSource";
  }
  std::string backend_fragment(int index) const override {
    return "fakesrc name=n" + std::to_string(index) +
           "_source is-live=true sizetype=fixed sizemax=1 datarate=1 sync=true "
           "format=time ! application/octet-stream";
  }
  std::vector<std::string> element_names(int index) const override {
    return {"n" + std::to_string(index) + "_source"};
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }
  simaai::neat::InputRole input_role() const override {
    return simaai::neat::InputRole::Source;
  }
};

void check_queue(GstElement* pipeline, const char* name, guint buffers, guint bytes, guint64 time,
                 gint leaky) {
  auto queue =
      std::shared_ptr<GstElement>(gst_bin_get_by_name(GST_BIN(pipeline), name), gst_object_unref);
  require(bool(queue), std::string("built source queue missing: ") + name);
  guint actual_buffers = 0, actual_bytes = 0;
  guint64 actual_time = 0;
  gint actual_leaky = -1;
  g_object_get(queue.get(), "max-size-buffers", &actual_buffers, "max-size-bytes", &actual_bytes,
               "max-size-time", &actual_time, "leaky", &actual_leaky, nullptr);
  require(actual_buffers == buffers && actual_bytes == bytes && actual_time == time &&
              actual_leaky == leaky,
          std::string("source graph changed queue limits: ") + name);
}

} // namespace

RUN_TEST("unit_realtime_source_queue_test", ([] {
           using namespace simaai::neat;
           for (const auto preset : {RunPreset::Realtime, RunPreset::Reliable}) {
             GraphOptions graph_options;
             graph_options.async_queue_depth = 3;
             Graph graph("source_queue_policy", graph_options);
             graph.add(std::make_shared<LiveCpuSource>());
             QueueOptions explicit_queue;
             explicit_queue.max_buffers = 7;
             explicit_queue.overflow_policy = OverflowPolicy::KeepLatest;
             graph.add(nodes::Queue(explicit_queue));
             graph.add(nodes::Queue());
             graph.add(nodes::Output(OutputOptions::EveryFrame(4)));
             RunOptions options;
             options.preset = preset;
             auto run = graph.build(options);
             const auto core = run_internal::core(run);
             auto* pipeline = core->pipeline.stream.pipeline_handle();
             require(pipeline, "source graph has no materialized pipeline");
             check_queue(pipeline, "queue_neat_async_0", 3, 0, 0, 0);
             check_queue(pipeline, "n1_queue", 7, 0, 0, 2);
             check_queue(pipeline, "n2_queue", 200, 10U * 1024U * 1024U, GST_SECOND, 0);
             run.stop();
           }
         }));
