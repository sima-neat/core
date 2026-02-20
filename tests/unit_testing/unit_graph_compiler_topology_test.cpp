#include "graph/Compiler.h"
#include "graph_test_utils.h"
#include "graph/nodes/PipelineNode.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/Input.h"
#include "test_main.h"

#include <memory>
#include <vector>

RUN_TEST(
    "unit_graph_compiler_topology_test", ([] {
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
    }));
