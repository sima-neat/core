/**
 * @file
 * @ingroup contracts
 * @brief Built-in contract implementations and default registry.
 *
 * Provides the small set of header-only `Contract` factories that ship with
 * the framework — non-empty pipeline, no-null nodes, sink-last-for-run,
 * RTSP source presence — plus `DefaultRegistry()`, the recommended starting
 * point for most callers. Library code can compose its own registry by
 * cloning `DefaultRegistry()` and adding/removing contracts.
 *
 * @see Contract
 * @see ContractRegistry
 */
// include/contracts/Validators.h
#pragma once

#include <memory>
#include <string>
#include <utility>

#include "builder/Node.h"
#include "builder/NodeGroup.h"
#include "contracts/Contract.h"
#include "contracts/ContractRegistry.h"
#include "contracts/ValidationReport.h"

namespace simaai::neat {
namespace validators {

// -----------------------------
// Built-in Contract factories
// -----------------------------

/**
 * @brief Ensures NodeGroup is not empty.
 *
 * Issues `EMPTY_PIPELINE` error when validated against an empty NodeGroup.
 *
 * @ingroup contracts
 * @return Shared pointer to a fresh `Contract` instance.
 */
inline std::shared_ptr<Contract> NonEmptyPipeline() {
  class C final : public Contract {
  public:
    std::string id() const override {
      return "NonEmptyPipeline";
    }
    std::string description() const override {
      return "Pipeline must contain at least one node.";
    }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx,
                  ValidationReport& r) const override {
      (void)ctx;
      if (nodes.empty()) {
        r.add_error(id(), "EMPTY_PIPELINE", "No nodes were added to the pipeline.");
      }
    }
  };
  return std::make_shared<C>();
}

/**
 * @brief Ensures there are no null node pointers in the NodeGroup.
 *
 * Issues a `NULL_NODE` error per offending index.
 *
 * @ingroup contracts
 * @return Shared pointer to a fresh `Contract` instance.
 */
inline std::shared_ptr<Contract> NoNullNodes() {
  class C final : public Contract {
  public:
    std::string id() const override {
      return "NoNullNodes";
    }
    std::string description() const override {
      return "All nodes must be non-null shared_ptr.";
    }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx,
                  ValidationReport& r) const override {
      (void)ctx;
      const auto& v = nodes.nodes();
      for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        if (!v[static_cast<std::size_t>(i)]) {
          r.add_error(id(), "NULL_NODE", "Null node pointer in NodeGroup.", i);
        }
      }
    }
  };
  return std::make_shared<C>();
}

/**
 * @brief Ensures the configured sink kind exists and is last when `ctx.mode == Run`.
 *
 * This is the builder-level version of the "sink last" contract described in
 * the architecture. Issues `SINK_NOT_LAST` if the last Node isn't of the
 * expected kind, and `MULTIPLE_SINKS` if a sink-kind Node is found earlier
 * in the chain.
 *
 * @ingroup contracts
 * @param sink_kind The Node kind expected as the terminal (default `"Output"`).
 * @return Shared pointer to a fresh `Contract` instance.
 */
inline std::shared_ptr<Contract> SinkLastForRun(std::string sink_kind = "Output") {
  class C final : public Contract {
  public:
    explicit C(std::string kind) : sink_kind_(std::move(kind)) {}
    std::string id() const override {
      return "SinkLastForRun";
    }
    std::string description() const override {
      return "When running, the pipeline must end with the terminal appsink node.";
    }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx,
                  ValidationReport& r) const override {
      if (ctx.mode != ValidationContext::Mode::Run)
        return;
      const auto& v = nodes.nodes();
      if (v.empty())
        return;

      int last_idx = static_cast<int>(v.size()) - 1;
      const auto& last = v.back();
      const std::string last_kind = last ? last->kind() : "";

      // Require sink kind last.
      if (!last || last_kind != sink_kind_) {
        r.add_error(id(), "SINK_NOT_LAST", "Last node must be " + sink_kind_ + " for run().",
                    last_idx, last_kind, last ? last->user_label() : "");
      }

      // Disallow additional sinks earlier (best-effort sanity).
      for (int i = 0; i < last_idx; ++i) {
        const auto& n = v[static_cast<std::size_t>(i)];
        if (n && n->kind() == sink_kind_) {
          r.add_error(id(), "MULTIPLE_SINKS",
                      "Found " + sink_kind_ + " before the end of the pipeline.", i, n->kind(),
                      n->user_label());
        }
      }
    }

  private:
    std::string sink_kind_;
  };
  return std::make_shared<C>(std::move(sink_kind));
}

/**
 * @brief Ensures an RTSP source node exists when `ctx.mode == Rtsp`.
 *
 * Builder-level: we only check presence of `StillImageInput` (or another
 * configured kind). Issues `RTSP_SOURCE_MISSING` if no Node of the expected
 * kind is found in the NodeGroup.
 *
 * @ingroup contracts
 * @param source_kind The Node kind expected to act as the RTSP source.
 * @return Shared pointer to a fresh `Contract` instance.
 */
inline std::shared_ptr<Contract> RtspRequiresSource(std::string source_kind = "StillImageInput") {
  class C final : public Contract {
  public:
    explicit C(std::string k) : src_kind_(std::move(k)) {}
    std::string id() const override {
      return "RtspRequiresSource";
    }
    std::string description() const override {
      return "RTSP mode requires a server-side source node (e.g., StillImageInput).";
    }

    void validate(const NodeGroup& nodes, const ValidationContext& ctx,
                  ValidationReport& r) const override {
      if (ctx.mode != ValidationContext::Mode::Rtsp)
        return;

      const auto& v = nodes.nodes();
      bool found = false;
      for (int i = 0; i < static_cast<int>(v.size()); ++i) {
        const auto& n = v[static_cast<std::size_t>(i)];
        if (n && n->kind() == src_kind_) {
          found = true;
          break;
        }
      }

      if (!found) {
        r.add_error(id(), "RTSP_SOURCE_MISSING",
                    "RTSP mode requires a node of kind \"" + src_kind_ + "\".", -1, src_kind_, "");
      }
    }

  private:
    std::string src_kind_;
  };
  return std::make_shared<C>(std::move(source_kind));
}

// -----------------------------
// Default registry
// -----------------------------

/**
 * @brief Reasonable default set of builder-level contracts.
 *
 * Bundles `NonEmptyPipeline`, `NoNullNodes`, `SinkLastForRun`, and
 * `RtspRequiresSource` into a fresh registry. Keep this purely structural
 * (no GStreamer); domain-specific contracts should be added on top.
 *
 * @ingroup contracts
 * @return New `ContractRegistry` populated with the default contracts.
 */
inline ContractRegistry DefaultRegistry() {
  ContractRegistry reg;
  reg.add(NonEmptyPipeline());
  reg.add(NoNullNodes());
  reg.add(SinkLastForRun());
  reg.add(RtspRequiresSource());
  return reg;
}

} // namespace validators
} // namespace simaai::neat
