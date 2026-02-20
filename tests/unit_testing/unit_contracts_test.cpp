#include "contracts/ContractRegistry.h"
#include "contracts/Validators.h"

#include "test_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FakeNode final : public simaai::neat::Node {
public:
  FakeNode(std::string kind, std::string label)
      : kind_(std::move(kind)), label_(std::move(label)) {}

  std::string kind() const override {
    return kind_;
  }
  std::string user_label() const override {
    return label_;
  }
  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Static;
  }
  std::string backend_fragment(int) const override {
    return "identity";
  }
  std::vector<std::string> element_names(int) const override {
    return {};
  }

private:
  std::string kind_;
  std::string label_;
};

} // namespace

int main() {
  try {
    using simaai::neat::ContractRegistry;
    using simaai::neat::NodeGroup;
    using simaai::neat::ValidationContext;

    ContractRegistry reg;
    reg.add(simaai::neat::validators::NonEmptyPipeline());
    reg.add(simaai::neat::validators::NoNullNodes());
    reg.add(simaai::neat::validators::SinkLastForRun());

    ValidationContext ctx;
    ctx.mode = ValidationContext::Mode::Validate;
    ctx.strict = true;

    NodeGroup empty;
    auto rep = reg.validate(empty, ctx);
    require(rep.has_errors(), "Empty pipeline should fail");

    auto sink = std::make_shared<FakeNode>("Output", "");
    auto other = std::make_shared<FakeNode>("Other", "");

    NodeGroup run_ok({other, sink});
    ctx.mode = ValidationContext::Mode::Run;
    rep = reg.validate(run_ok, ctx);
    require(!rep.has_errors(), "Run mode should accept sink last");

    std::cout << "[OK] unit_contracts_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
