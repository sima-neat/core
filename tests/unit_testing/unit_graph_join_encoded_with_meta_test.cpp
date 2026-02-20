#include "graph/nodes/JoinEncodedWithMeta.h"
#include "graph_test_utils.h"
#include "pipeline/EncodedSampleUtil.h"
#include "test_main.h"

#include <string>
#include <vector>

RUN_TEST(
    "unit_graph_join_encoded_with_meta_test", ([] {
      using simaai::neat::graph::PortId;
      using simaai::neat::graph::StageMsg;
      using simaai::neat::graph::StageOutMsg;
      using simaai::neat::graph::StagePorts;
      using simaai::neat::graph::nodes::JoinEncodedWithMeta;
      using simaai::neat::graph::nodes::JoinEncodedWithMetaOptions;

      const PortId encoded_port = 31;
      const PortId meta_a_port = 32;
      const PortId meta_b_port = 33;
      const PortId meta_c_port = 34;
      const PortId out_port = 39;

      // Explicit encoded_port behavior + field naming precedence.
      {
        JoinEncodedWithMetaOptions opt;
        opt.encoded_port = encoded_port;
        opt.emit_partial = false;
        opt.encoded_name = "enc";
        opt.port_names.emplace(meta_a_port, "mapped_meta");

        JoinEncodedWithMeta join(opt);
        StagePorts ports;
        ports.out["bundle"] = out_port;
        join.set_ports(ports);

        std::vector<StageOutMsg> out;

        StageMsg meta_a;
        meta_a.in_port = meta_a_port;
        meta_a.sample = sima_test::make_tensor_sample(7, "cam-meta", 7000);
        join.on_input(std::move(meta_a), out);

        StageMsg meta_b;
        meta_b.in_port = meta_b_port;
        meta_b.sample = sima_test::make_tensor_sample(7, "cam-meta", 7000);
        meta_b.sample.port_name = "sample_meta";
        join.on_input(std::move(meta_b), out);

        StageMsg meta_c;
        meta_c.in_port = meta_c_port;
        meta_c.sample = sima_test::make_tensor_sample(7, "cam-meta", 7000);
        join.on_input(std::move(meta_c), out);

        require(out.empty(),
                "JoinEncodedWithMeta should wait for encoded sample when emit_partial=false");

        StageMsg encoded;
        encoded.in_port = encoded_port;
        encoded.sample = simaai::neat::make_encoded_sample(
            std::vector<uint8_t>{0x00, 0x00, 0x01, 0x67}, "video/x-h264", 7000, 7000, 33333);
        encoded.sample.stream_id = "cam-meta";
        encoded.sample.frame_id = 7;
        join.on_input(std::move(encoded), out);

        require(out.size() == 1,
                "JoinEncodedWithMeta should emit one bundle after encoded arrives");
        require(out[0].out_port == out_port, "JoinEncodedWithMeta out_port mismatch");
        require(out[0].sample.fields.size() == 4,
                "JoinEncodedWithMeta should include encoded + all meta fields");

        require(sima_test::has_field_name(out[0].sample, "enc"),
                "JoinEncodedWithMeta encoded_name precedence mismatch");
        require(sima_test::has_field_name(out[0].sample, "mapped_meta"),
                "JoinEncodedWithMeta port_names precedence mismatch");
        require(sima_test::has_field_name(out[0].sample, "sample_meta"),
                "JoinEncodedWithMeta sample.port_name fallback mismatch");
        require(sima_test::has_field_name(out[0].sample, "port34"),
                "JoinEncodedWithMeta default port-name fallback mismatch");
      }

      // Implicit encoded detection via tensor semantic.
      {
        JoinEncodedWithMetaOptions opt;
        opt.encoded_port = simaai::neat::graph::kInvalidPort;
        opt.emit_partial = false;

        JoinEncodedWithMeta join(opt);
        StagePorts ports;
        ports.out["bundle"] = out_port;
        join.set_ports(ports);

        std::vector<StageOutMsg> out;

        StageMsg meta;
        meta.in_port = meta_a_port;
        meta.sample = sima_test::make_tensor_sample(5, "implicit", 5000);
        join.on_input(std::move(meta), out);

        StageMsg encoded;
        encoded.in_port = meta_b_port;
        encoded.sample = simaai::neat::make_encoded_sample(std::vector<uint8_t>{0x11, 0x22, 0x33},
                                                           "video/x-h264", 5000, 5000, 33333);
        encoded.sample.stream_id = "implicit";
        encoded.sample.frame_id = 5;
        join.on_input(std::move(encoded), out);

        require(out.size() == 1,
                "JoinEncodedWithMeta should detect encoded sample implicitly from tensor semantic");
        require(sima_test::has_field_name(out[0].sample, "encoded"),
                "JoinEncodedWithMeta implicit encoded field should use default encoded name");
      }

      // max_pending eviction behavior.
      {
        JoinEncodedWithMetaOptions opt;
        opt.encoded_port = encoded_port;
        opt.emit_partial = false;
        opt.max_pending = 1;

        JoinEncodedWithMeta join(opt);
        StagePorts ports;
        ports.out["bundle"] = out_port;
        join.set_ports(ports);

        std::vector<StageOutMsg> out;

        StageMsg meta0;
        meta0.in_port = meta_a_port;
        meta0.sample = sima_test::make_tensor_sample(1, "evict", 1000);
        join.on_input(std::move(meta0), out);

        StageMsg meta1;
        meta1.in_port = meta_a_port;
        meta1.sample = sima_test::make_tensor_sample(2, "evict", 2000);
        join.on_input(std::move(meta1), out);

        StageMsg stale_encoded;
        stale_encoded.in_port = encoded_port;
        stale_encoded.sample = simaai::neat::make_encoded_sample(std::vector<uint8_t>{0x01},
                                                                 "video/x-h264", 1000, 1000, 1);
        stale_encoded.sample.stream_id = "evict";
        stale_encoded.sample.frame_id = 1;
        join.on_input(std::move(stale_encoded), out);

        require(out.size() == 1,
                "JoinEncodedWithMeta should emit encoded-only bundle after oldest-key eviction");
        require(out[0].sample.fields.size() == 1,
                "JoinEncodedWithMeta eviction path should not retain evicted meta fields");
        require(sima_test::has_field_name(out[0].sample, "encoded"),
                "JoinEncodedWithMeta eviction path encoded field missing");

        out.clear();

        StageMsg fresh_encoded;
        fresh_encoded.in_port = encoded_port;
        fresh_encoded.sample = simaai::neat::make_encoded_sample(std::vector<uint8_t>{0x02},
                                                                 "video/x-h264", 2000, 2000, 1);
        fresh_encoded.sample.stream_id = "evict";
        fresh_encoded.sample.frame_id = 2;
        join.on_input(std::move(fresh_encoded), out);

        require(out.size() == 1,
                "JoinEncodedWithMeta should emit deterministically after pending-key churn");
        require(out[0].sample.fields.size() == 1,
                "JoinEncodedWithMeta churn path should not emit stale multi-field bundle");
      }

      // Missing key fields should fail deterministically.
      {
        JoinEncodedWithMeta join(JoinEncodedWithMetaOptions{});
        std::vector<StageOutMsg> out;

        StageMsg bad;
        bad.in_port = 3;
        bad.sample = sima_test::make_tensor_sample(-1, "", -1);
        require(sima_test::throws_with([&]() { join.on_input(std::move(bad), out); },
                                       "missing pts/frame_id"),
                "JoinEncodedWithMeta should reject samples without key fields");
      }
    }));
