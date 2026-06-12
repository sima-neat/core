#include "builder/Builder.h"
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
    auto n0 = std::make_shared<FakeNode>("a");
    auto n1 = std::make_shared<FakeNode>("b");

    std::vector<std::shared_ptr<simaai::neat::Node>> group{n0, n1};
    require(group.size() == 2, "node list size mismatch");
    require(!group.empty(), "node list should not be empty");

    Builder b;
    b.add(n0).then(n1);
    require(b.size() == 2, "Builder size mismatch");
    auto built = b.build();
    require(built.size() == 2, "Builder build mismatch");

    simaai::neat::GraphPrinter::Options opt;
    std::string text = simaai::neat::GraphPrinter::to_text(group, opt);
    require_contains(text, "FakeNode", "GraphPrinter text missing kind");
    require_contains(text, "[a]", "GraphPrinter text missing first label");
    require_contains(text, "[b]", "GraphPrinter text missing second label");

    std::cout << "[OK] unit_builder_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
