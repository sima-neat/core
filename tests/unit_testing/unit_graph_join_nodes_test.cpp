#include "graph/nodes/JoinBundle.h"
#include "graph/nodes/JoinEncodedWithMeta.h"
#include "graph/nodes/StampFrameId.h"
#include "pipeline/EncodedSampleUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace {

using simaai::neat::graph::PortId;
using simaai::neat::graph::StageMsg;
using simaai::neat::graph::StageOutMsg;
using simaai::neat::graph::StagePorts;

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

simaai::neat::Sample make_tensor_sample(int frame_id, const std::string& stream_id,
                                        int64_t pts_ns = -1) {
  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(8, 6, simaai::neat::ImageSpec::PixelFormat::RGB);
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  sample.pts_ns = pts_ns;
  return sample;
}

bool has_field_name(const simaai::neat::Sample& bundle, const std::string& name) {
  for (const auto& field : bundle.fields) {
    if (field.port_name == name)
      return true;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_graph_join_nodes_test", ([] {
      using simaai::neat::graph::nodes::JoinBundle;
      using simaai::neat::graph::nodes::JoinBundleOptions;
      using simaai::neat::graph::nodes::JoinEncodedWithMeta;
      using simaai::neat::graph::nodes::JoinEncodedWithMetaOptions;
      using simaai::neat::graph::nodes::JoinKeyPolicy;
      using simaai::neat::graph::nodes::StampFrameId;

      // StampFrameId: deterministic per-stream id assignment.
      {
        StampFrameId stamp;
        StagePorts ports;
        ports.out["out"] = static_cast<PortId>(9);
        stamp.set_ports(ports);

        std::vector<StageOutMsg> out;

        StageMsg m0;
        m0.in_port = 1;
        m0.sample = make_tensor_sample(-1, "");
        stamp.on_input(std::move(m0), out);
        require(out.size() == 1, "StampFrameId should emit one output per input");
        require(out.back().out_port == static_cast<PortId>(9), "StampFrameId output port mismatch");
        require(out.back().sample.stream_id == "stream0", "StampFrameId should default stream_id");
        require(out.back().sample.frame_id == 0, "StampFrameId should start frame ids at zero");

        StageMsg m1;
        m1.sample = make_tensor_sample(-1, "stream0");
        stamp.on_input(std::move(m1), out);
        require(out.back().sample.frame_id == 1,
                "StampFrameId should increment frame ids per stream");

        StageMsg m2;
        m2.sample = make_tensor_sample(-1, "camera-b");
        stamp.on_input(std::move(m2), out);
        require(out.back().sample.frame_id == 0,
                "StampFrameId should keep independent counters per stream");

        StageMsg m3;
        m3.sample = make_tensor_sample(42, "camera-b");
        stamp.on_input(std::move(m3), out);
        require(out.back().sample.frame_id == 42,
                "StampFrameId should preserve existing frame ids");
      }

      // JoinBundle: deterministic bundle routing/cardinality and ordered fields.
      {
        JoinBundleOptions opt;
        opt.inputs = {"encoded", "meta"};
        opt.emit_partial = false;

        JoinBundle join(opt);
        StagePorts ports;
        const PortId encoded_port = 11;
        const PortId meta_port = 12;
        const PortId bundle_port = 19;
        ports.in["encoded"] = encoded_port;
        ports.in["meta"] = meta_port;
        ports.out["bundle"] = bundle_port;
        join.set_ports(ports);

        std::vector<StageOutMsg> out;

        StageMsg encoded_msg;
        encoded_msg.in_port = encoded_port;
        encoded_msg.sample = make_tensor_sample(7, "cam0");
        join.on_input(std::move(encoded_msg), out);
        require(out.empty(), "JoinBundle should wait for required ports");

        StageMsg meta_msg;
        meta_msg.in_port = meta_port;
        meta_msg.sample = make_tensor_sample(7, "cam0");
        join.on_input(std::move(meta_msg), out);

        require(out.size() == 1, "JoinBundle should emit one bundle when all ports arrive");
        require(out[0].out_port == bundle_port, "JoinBundle output port mismatch");
        require(out[0].sample.kind == simaai::neat::SampleKind::Bundle,
                "JoinBundle output kind should be Bundle");
        require(out[0].sample.fields.size() == 2,
                "JoinBundle should emit both encoded/meta fields");
        require(out[0].sample.fields[0].port_name == "encoded",
                "JoinBundle should preserve input order in bundle fields");
        require(out[0].sample.fields[1].port_name == "meta",
                "JoinBundle should preserve input order in bundle fields");
      }

      // JoinBundle: key policy validation and eviction behavior.
      {
        JoinBundleOptions opt;
        opt.inputs = {"encoded", "meta"};
        opt.key_policy = JoinKeyPolicy::StreamPts;
        opt.max_pending_keys = 1;

        JoinBundle join(opt);
        StagePorts ports;
        const PortId encoded_port = 21;
        const PortId meta_port = 22;
        ports.in["encoded"] = encoded_port;
        ports.in["meta"] = meta_port;
        ports.out["bundle"] = 29;
        join.set_ports(ports);

        std::vector<StageOutMsg> out;

        StageMsg p0;
        p0.in_port = encoded_port;
        p0.sample = make_tensor_sample(10, "cam", 1000);
        join.on_input(std::move(p0), out);

        StageMsg p1;
        p1.in_port = encoded_port;
        p1.sample = make_tensor_sample(11, "cam", 2000);
        join.on_input(std::move(p1), out);

        StageMsg good_meta;
        good_meta.in_port = meta_port;
        good_meta.sample = make_tensor_sample(11, "cam", 2000);
        join.on_input(std::move(good_meta), out);
        require(out.size() == 1, "JoinBundle should emit only for non-evicted key");
        require(out[0].sample.pts_ns == 2000, "JoinBundle should route by pts in StreamPts mode");

        const size_t emitted_before_stale = out.size();
        StageMsg stale_meta;
        stale_meta.in_port = meta_port;
        stale_meta.sample = make_tensor_sample(10, "cam", 1000);
        join.on_input(std::move(stale_meta), out);
        require(out.size() == emitted_before_stale,
                "JoinBundle should not emit for an evicted stale key");

        StageMsg invalid_key;
        invalid_key.in_port = encoded_port;
        invalid_key.sample = make_tensor_sample(-1, "cam", -1);
        require(throws_with(
                    [&]() {
                      std::vector<StageOutMsg> local;
                      join.on_input(std::move(invalid_key), local);
                    },
                    "missing pts/frame_id"),
                "JoinBundle should reject samples without key fields in StreamPts mode");
      }

      // JoinBundle: required port validation and timeout eviction.
      {
        JoinBundleOptions bad_required;
        bad_required.inputs = {"encoded", "meta"};
        bad_required.required = {"encoded", "not_a_port"};
        JoinBundle bad_join(bad_required);

        StagePorts ports;
        ports.in["encoded"] = 31;
        ports.in["meta"] = 32;
        ports.out["bundle"] = 39;

        require(throws_with([&]() { bad_join.set_ports(ports); }, "unknown required port"),
                "JoinBundle should reject unknown required ports");

        JoinBundleOptions timeout_opt;
        timeout_opt.inputs = {"encoded", "meta"};
        timeout_opt.timeout_ms = 1;
        JoinBundle timeout_join(timeout_opt);
        timeout_join.set_ports(ports);

        std::vector<StageOutMsg> out;
        StageMsg encoded_msg;
        encoded_msg.in_port = 31;
        encoded_msg.sample = make_tensor_sample(4, "cam");
        timeout_join.on_input(std::move(encoded_msg), out);

        timeout_join.on_tick(std::numeric_limits<int64_t>::max() / 2, out);

        StageMsg meta_msg;
        meta_msg.in_port = 32;
        meta_msg.sample = make_tensor_sample(4, "cam");
        timeout_join.on_input(std::move(meta_msg), out);
        require(out.empty(), "JoinBundle should drop expired pending entries on tick");
      }

      // JoinEncodedWithMeta: deterministic join behavior and naming.
      {
        JoinEncodedWithMetaOptions opt;
        opt.encoded_port = 41;
        opt.port_names.emplace(42, "meta");
        opt.max_pending = 8;

        JoinEncodedWithMeta join(opt);
        StagePorts ports;
        ports.out["bundle"] = 49;
        join.set_ports(ports);

        std::vector<StageOutMsg> out;

        StageMsg meta_first;
        meta_first.in_port = 42;
        meta_first.sample = make_tensor_sample(9, "cam-x", 9000);
        join.on_input(std::move(meta_first), out);
        require(out.empty(), "JoinEncodedWithMeta should not emit before encoded payload arrives");

        StageMsg enc;
        enc.in_port = 41;
        enc.sample = simaai::neat::make_encoded_sample(std::vector<uint8_t>{0x00, 0x00, 0x01, 0x67},
                                                       "video/x-h264", 9000, 9000, 33333);
        enc.sample.stream_id = "cam-x";
        enc.sample.frame_id = 9;

        join.on_input(std::move(enc), out);
        require(out.size() == 1, "JoinEncodedWithMeta should emit one bundle once encoded arrives");
        require(out[0].out_port == 49, "JoinEncodedWithMeta output port mismatch");
        require(out[0].sample.kind == simaai::neat::SampleKind::Bundle,
                "JoinEncodedWithMeta output should be a bundle");
        require(out[0].sample.fields.size() == 2,
                "JoinEncodedWithMeta should include encoded and meta fields");
        require(has_field_name(out[0].sample, "encoded"),
                "JoinEncodedWithMeta bundle should include encoded field name");
        require(has_field_name(out[0].sample, "meta"),
                "JoinEncodedWithMeta bundle should include mapped meta field name");
      }

      // JoinEncodedWithMeta: key validation and max_pending eviction.
      {
        JoinEncodedWithMetaOptions opt;
        opt.encoded_port = simaai::neat::graph::kInvalidPort;
        opt.emit_partial = false;
        opt.max_pending = 1;

        JoinEncodedWithMeta join(opt);
        StagePorts ports;
        ports.out["bundle"] = 59;
        join.set_ports(ports);

        std::vector<StageOutMsg> out;

        StageMsg meta0;
        meta0.in_port = 52;
        meta0.sample = make_tensor_sample(1, "s0");
        join.on_input(std::move(meta0), out);
        require(out.empty(),
                "JoinEncodedWithMeta should not emit partial bundles when emit_partial=false");

        StageMsg meta1;
        meta1.in_port = 52;
        meta1.sample = make_tensor_sample(2, "s0");
        join.on_input(std::move(meta1), out);
        require(out.empty(),
                "JoinEncodedWithMeta should defer emission until encoded sample arrives");

        StageMsg enc0;
        enc0.in_port = 51;
        enc0.sample = simaai::neat::make_encoded_sample(
            std::vector<uint8_t>{0x00, 0x00, 0x01, 0x67}, "video/x-h264");
        enc0.sample.stream_id = "s0";
        enc0.sample.frame_id = 1;
        join.on_input(std::move(enc0), out);
        require(out.size() == 1, "JoinEncodedWithMeta should emit once encoded sample arrives");
        require(out[0].sample.fields.size() == 1,
                "JoinEncodedWithMeta max_pending eviction should drop stale meta-only key");
        require(has_field_name(out[0].sample, "encoded"),
                "JoinEncodedWithMeta emitted bundle should retain encoded field");

        StageMsg stale_meta;
        stale_meta.in_port = 52;
        stale_meta.sample = make_tensor_sample(1, "s0");
        join.on_input(std::move(stale_meta), out);
        require(
            out.size() == 1,
            "JoinEncodedWithMeta stale non-encoded input should not emit without encoded partner");

        StageMsg good_meta;
        good_meta.in_port = 52;
        good_meta.sample = make_tensor_sample(2, "s0");
        join.on_input(std::move(good_meta), out);
        require(out.size() == 1, "JoinEncodedWithMeta stale pending-only updates should not emit");

        StageMsg bad_key;
        bad_key.in_port = 51;
        bad_key.sample = make_tensor_sample(-1, "", -1);
        require(throws_with(
                    [&]() {
                      std::vector<StageOutMsg> local;
                      join.on_input(std::move(bad_key), local);
                    },
                    "missing pts/frame_id"),
                "JoinEncodedWithMeta should reject samples without key fields");
      }
    }));
