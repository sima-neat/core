#include "graph/Compiler.h"
#include "graph/nodes/PipelineNode.h"
#include "graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#include <memory>
#include <vector>

RUN_TEST(
    "graph_migration_legacy_graph_compiler_topology_test", ([] {
      using simaai::neat::OutputSpec;
      using simaai::neat::graph::Compiler;
      using simaai::neat::graph::Graph;
      using simaai::neat::graph::PortDesc;
      using simaai::neat::graph::nodes::PipelineNode;

      Compiler compiler;

      // Cycle rejection.
      {
        Graph g;
        const auto a = g.add(sima_test::make_stage_passthrough("a"));
        const auto b = g.add(sima_test::make_stage_passthrough("b"));
        g.connect(a, b);
        g.connect(b, a);

        require(sima_test::throws_with([&]() { (void)compiler.compile(g); }, "graph must be a DAG"),
                "Compiler should reject cyclic graphs");
      }

      // Unknown input port wiring rejection.
      {
        Graph g;
        const auto src = g.add(sima_test::make_stage_source("src", OutputSpec{}));
        const auto sink = g.add(sima_test::make_stage_passthrough("sink"));
        g.connect(src, sink, "out", "missing_port");

        require(sima_test::throws_with([&]() { (void)compiler.compile(g); }, "unknown input port"),
                "Compiler should reject edges wired to undeclared input ports");
      }

      // Pipeline invalid fan-in rejection.
      {
        Graph g;
        const auto src0 =
            g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Input(), "src0"));
        const auto src1 =
            g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Input(), "src1"));
        const auto conv =
            g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::VideoConvert(), "conv"));
        g.connect(src0, conv);
        g.connect(src1, conv);

        require(sima_test::throws_with([&]() { (void)compiler.compile(g); }, "multiple inputs"),
                "Compiler should reject pipeline-node fan-in without explicit stage join");
      }

      // Pipeline invalid fan-out rejection.
      {
        Graph g;
        const auto src = g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Input(), "src"));
        const auto conv =
            g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::VideoConvert(), "conv"));
        const auto sink0 =
            g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Output(), "sink0"));
        const auto sink1 =
            g.add(std::make_shared<PipelineNode>(simaai::neat::nodes::Output(), "sink1"));
        g.connect(src, conv);
        g.connect(conv, sink0);
        g.connect(conv, sink1);

        require(sima_test::throws_with([&]() { (void)compiler.compile(g); }, "multiple outputs"),
                "Compiler should reject pipeline-node fan-out without explicit stage fan-out");
      }

      // Duplicate input port-name rejection.
      {
        Graph g;
        std::vector<PortDesc> in_ports = {PortDesc{.name = "in", .spec = OutputSpec{}},
                                          PortDesc{.name = "in", .spec = OutputSpec{}}};
        std::vector<PortDesc> out_ports = {PortDesc{.name = "out", .spec = OutputSpec{}}};
        g.add(sima_test::make_stage_node("dup_ports", std::move(in_ports), std::move(out_ports)));

        require(sima_test::throws_with([&]() { (void)compiler.compile(g); }, "duplicate port name"),
                "Compiler should reject duplicate stage input port names");
      }

      // Empty input port-name rejection.
      {
        Graph g;
        std::vector<PortDesc> in_ports = {PortDesc{.name = "", .spec = OutputSpec{}}};
        std::vector<PortDesc> out_ports = {PortDesc{.name = "out", .spec = OutputSpec{}}};
        g.add(sima_test::make_stage_node("empty_port", std::move(in_ports), std::move(out_ports)));

        require(sima_test::throws_with([&]() { (void)compiler.compile(g); }, "empty port name"),
                "Compiler should reject empty stage input port names");
      }

      // Max input-edge contract enforcement.
      {
        Graph g;
        const OutputSpec spec = OutputSpec{
            .media_type = "video/x-raw", .format = "RGB", .width = 32, .height = 24, .depth = 3};
        const auto src_a = g.add(sima_test::make_stage_source("src_a", spec));
        const auto src_b = g.add(sima_test::make_stage_source("src_b", spec));
        const auto sink = g.add(sima_test::make_stage_passthrough("sink", 1));
        g.connect(src_a, sink, "out", "in");
        g.connect(src_b, sink, "out", "in");

        require(
            sima_test::throws_with([&]() { (void)compiler.compile(g); }, "exceeds max_in_edges"),
            "Compiler should enforce max_in_edges contracts");
      }

      // Merged input spec consistency enforcement.
      {
        Graph g;
        const auto rgb_src = g.add(sima_test::make_stage_source(
            "rgb_src",
            OutputSpec{.media_type = "video/x-raw", .format = "RGB", .width = 64, .height = 48}));
        const auto nv12_src = g.add(sima_test::make_stage_source(
            "nv12_src",
            OutputSpec{.media_type = "video/x-raw", .format = "NV12", .width = 64, .height = 48}));
        const auto merge = g.add(sima_test::make_stage_passthrough("merge", 0));
        g.connect(rgb_src, merge, "out", "in");
        g.connect(nv12_src, merge, "out", "in");

        require(sima_test::throws_with([&]() { (void)compiler.compile(g); }, "input spec mismatch"),
                "Compiler should reject conflicting merged input specs");
      }
    }));
