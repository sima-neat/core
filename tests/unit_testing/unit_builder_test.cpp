#include "builder/Builder.h"
#include "builder/Graph.h"
#include "builder/GraphPrinter.h"

#include "test_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FakeNode final : public simaai::neat::Node {
public:
  explicit FakeNode(std::string label) : label_(std::move(label)) {}

  std::string kind() const override {
    return "FakeNode";
  }
  std::string user_label() const override {
    return label_;
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }

  std::string backend_fragment(int node_index) const override {
    return "identity name=n" + std::to_string(node_index) + "_fake";
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {"n" + std::to_string(node_index) + "_fake"};
  }

private:
  std::string label_;
};

} // namespace

int main() {
  try {
    using simaai::neat::Builder;
    using simaai::neat::Graph;
    using simaai::neat::NodeGroup;

    auto n0 = std::make_shared<FakeNode>("a");
    auto n1 = std::make_shared<FakeNode>("b");

    NodeGroup group({n0, n1});
    require(group.size() == 2, "NodeGroup size mismatch");
    require(!group.empty(), "NodeGroup should not be empty");

    Builder b;
    b.add(n0).then(n1);
    require(b.size() == 2, "Builder size mismatch");
    NodeGroup built = b.build();
    require(built.size() == 2, "Builder build mismatch");

    Graph g;
    Graph::NodeId a = g.add(n0);
    Graph::NodeId b_id = g.add(n1);
    g.add_edge(a, b_id);
    auto order = g.topo_order();
    require(order.size() == 2, "Graph topo order size mismatch");
    require(order[0] == a && order[1] == b_id, "Graph topo order wrong");

    g.add_edge(b_id, a);
    bool threw = false;
    try {
      (void)g.topo_order();
    } catch (...) {
      threw = true;
    }
    require(threw, "Graph should detect cycle");

    simaai::neat::GraphPrinter::Options opt;
    std::string text = simaai::neat::GraphPrinter::to_text(group, opt);
    require_contains(text, "FakeNode", "GraphPrinter text missing kind");

    std::string dot = simaai::neat::GraphPrinter::to_dot(g, opt);
    require_contains(dot, "digraph", "GraphPrinter dot missing header");

    std::string mermaid = simaai::neat::GraphPrinter::to_mermaid(g, opt);
    require_contains(mermaid, "flowchart", "GraphPrinter mermaid missing header");

    std::cout << "[OK] unit_builder_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
