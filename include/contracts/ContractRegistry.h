// include/contracts/ContractRegistry.h
/**
 * @file
 * @ingroup contracts
 * @brief Contract registry for builder-level validation.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "contracts/Contract.h"
#include "contracts/ValidationReport.h"

namespace simaai::neat {

/**
 * @brief Holds a set of Contracts and runs them to produce a ValidationReport.
 *
 * This is intentionally small and STL-only.
 */
class ContractRegistry final {
public:
  using ContractPtr = std::shared_ptr<Contract>;

  ContractRegistry() = default;

  /// Add/replace a contract by id().
  ContractRegistry& add(ContractPtr c) {
    if (!c)
      return *this;
    const std::string cid = c->id();
    if (cid.empty())
      return *this;

    auto it = by_id_.find(cid);
    if (it == by_id_.end()) {
      order_.push_back(cid);
    }
    by_id_[cid] = std::move(c);
    return *this;
  }

  /// Convenience: construct + add.
  template <class T, class... Args> ContractRegistry& emplace(Args&&... args) {
    return add(std::make_shared<T>(std::forward<Args>(args)...));
  }

  /// Remove a contract by id. Returns true if removed.
  bool remove(const std::string& id) {
    auto it = by_id_.find(id);
    if (it == by_id_.end())
      return false;
    by_id_.erase(it);

    // Keep deterministic order_: erase id if present.
    for (auto oit = order_.begin(); oit != order_.end(); ++oit) {
      if (*oit == id) {
        order_.erase(oit);
        break;
      }
    }
    return true;
  }

  void clear() {
    by_id_.clear();
    order_.clear();
  }

  std::size_t size() const noexcept {
    return by_id_.size();
  }
  bool empty() const noexcept {
    return by_id_.empty();
  }

  /// Get a contract by id (nullptr if missing).
  ContractPtr get(const std::string& id) const {
    auto it = by_id_.find(id);
    return (it == by_id_.end()) ? nullptr : it->second;
  }

  /// Deterministic list of ids in insertion order.
  std::vector<std::string> ids() const {
    return order_;
  }

  /**
   * @brief Run all contracts and return a ValidationReport.
   *
   * Defensive behavior:
   * - contract violations should be reported (not thrown)
   * - if a Contract throws, registry converts that into an internal ERROR issue
   */
  ValidationReport validate(const NodeGroup& nodes, const ValidationContext& ctx) const {
    ValidationReport report;
    report.set_mode(static_cast<int>(ctx.mode));

    for (const auto& id : order_) {
      auto it = by_id_.find(id);
      if (it == by_id_.end() || !it->second)
        continue;

      report.note_contract_run(id);

      try {
        it->second->validate(nodes, ctx, report);
      } catch (const std::exception& e) {
        report.add_issue({
            .severity = ValidationSeverity::Error,
            .contract_id = id,
            .code = "CONTRACT_THREW",
            .message = std::string("Contract threw exception: ") + e.what(),
            .node_index = -1,
        });
      } catch (...) {
        report.add_issue({
            .severity = ValidationSeverity::Error,
            .contract_id = id,
            .code = "CONTRACT_THREW",
            .message = "Contract threw unknown exception",
            .node_index = -1,
        });
      }
    }

    return report;
  }

private:
  std::unordered_map<std::string, ContractPtr> by_id_;
  std::vector<std::string> order_;
};

} // namespace simaai::neat
