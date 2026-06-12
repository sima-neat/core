// include/contracts/ValidationReport.h
/**
 * @file
 * @ingroup contracts
 * @brief Structured validation results for builder-level checks.
 *
 * Defines `ValidationSeverity`, `ValidationIssue`, and `ValidationReport` —
 * the data types that `ContractRegistry::validate()` produces. The report is
 * STL-only and JSON-serializable so CI tools and dashboards can consume it
 * without pulling in any framework dependencies.
 *
 * @see Contract
 * @see ContractRegistry
 */
#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

/**
 * @brief Severity level for validation issues.
 * @ingroup contracts
 */
enum class ValidationSeverity {
  Info = 0, ///< Informational only; never blocks a run.
  Warning,  ///< Soft contract violation; pipeline may still run.
  Error,    ///< Hard contract violation; pipeline must not run.
};

/// @brief Stable string label for a `ValidationSeverity` (for reports/logs).
inline const char* to_string(ValidationSeverity s) {
  switch (s) {
  case ValidationSeverity::Info:
    return "INFO";
  case ValidationSeverity::Warning:
    return "WARNING";
  case ValidationSeverity::Error:
    return "ERROR";
  }
  return "UNKNOWN";
}

/**
 * @brief A single reported issue from a Contract.
 *
 * Carries severity, the contract that emitted it, a machine-stable `code`
 * (suitable for filtering/asserting in tests), a human-readable `message`,
 * and best-effort Node location info.
 *
 * @ingroup contracts
 */
struct ValidationIssue {
  ValidationSeverity severity = ValidationSeverity::Error; ///< Severity of this issue.

  // Contract metadata
  std::string contract_id; ///< `Contract::id()` of the emitting contract.
  std::string code;        ///< Stable machine code (e.g., `"SINK_NOT_LAST"`).
  std::string message;     ///< Human-readable diagnostic.

  // Best-effort node location (builder-level)
  int node_index = -1;    ///< Index in the node list, or -1 if not Node-specific.
  std::string node_kind;  ///< `Node::kind()` of the offending Node, if any.
  std::string node_label; ///< User label of the offending Node, if any.
};

/**
 * @brief Report produced by running a ContractRegistry.
 *
 * Aggregates all issues emitted by every contract that was run, plus the
 * list of contract ids that participated. STL-only, CI-friendly, and safe
 * to serialize via `to_string()` (text) or `to_json()` (JSON).
 *
 * @ingroup contracts
 * @see ValidationIssue
 * @see ContractRegistry
 */
class ValidationReport final {
public:
  /// @brief Construct an empty report.
  ValidationReport() = default;

  // -------- Recording --------

  /// @brief Append an issue to the report.
  void add_issue(ValidationIssue issue) {
    issues_.push_back(std::move(issue));
  }

  /// @brief Append an `Info` issue.
  void add_info(std::string contract_id, std::string code, std::string msg, int node_index = -1,
                std::string node_kind = {}, std::string node_label = {}) {
    add_issue({ValidationSeverity::Info, std::move(contract_id), std::move(code), std::move(msg),
               node_index, std::move(node_kind), std::move(node_label)});
  }

  /// @brief Append a `Warning` issue.
  void add_warning(std::string contract_id, std::string code, std::string msg, int node_index = -1,
                   std::string node_kind = {}, std::string node_label = {}) {
    add_issue({ValidationSeverity::Warning, std::move(contract_id), std::move(code), std::move(msg),
               node_index, std::move(node_kind), std::move(node_label)});
  }

  /// @brief Append an `Error` issue.
  void add_error(std::string contract_id, std::string code, std::string msg, int node_index = -1,
                 std::string node_kind = {}, std::string node_label = {}) {
    add_issue({ValidationSeverity::Error, std::move(contract_id), std::move(code), std::move(msg),
               node_index, std::move(node_kind), std::move(node_label)});
  }

  /// @brief Record that a contract id participated in this validation pass.
  void note_contract_run(std::string id) {
    contracts_run_.push_back(std::move(id));
  }

  /// @brief Record which Graph mode produced this report (integer to avoid Contract.h include).
  void set_mode(int mode) {
    mode_ = mode;
  }
  /// @brief Mode integer set via `set_mode()`.
  int mode() const {
    return mode_;
  }

  // -------- Query --------

  /// @brief All issues collected, in insertion order.
  const std::vector<ValidationIssue>& issues() const noexcept {
    return issues_;
  }
  /// @brief Ids of all contracts that ran (in evaluation order).
  const std::vector<std::string>& contracts_run() const noexcept {
    return contracts_run_;
  }

  /// @brief True if the report contains no errors (warnings/info are still allowed).
  bool ok() const noexcept {
    return !has_errors();
  }

  /// @brief True if at least one issue with `Error` severity is present.
  bool has_errors() const noexcept {
    for (const auto& i : issues_) {
      if (i.severity == ValidationSeverity::Error)
        return true;
    }
    return false;
  }

  /// @brief Count of `Error` issues in the report.
  std::size_t error_count() const noexcept {
    std::size_t n = 0;
    for (const auto& i : issues_)
      if (i.severity == ValidationSeverity::Error)
        ++n;
    return n;
  }

  /// @brief Count of `Warning` issues in the report.
  std::size_t warning_count() const noexcept {
    std::size_t n = 0;
    for (const auto& i : issues_)
      if (i.severity == ValidationSeverity::Warning)
        ++n;
    return n;
  }

  /// @brief Count of `Info` issues in the report.
  std::size_t info_count() const noexcept {
    std::size_t n = 0;
    for (const auto& i : issues_)
      if (i.severity == ValidationSeverity::Info)
        ++n;
    return n;
  }

  // -------- Formatting --------

  /// @brief Render the report as a deterministic, human-readable text block.
  std::string to_string() const {
    std::ostringstream oss;
    oss << (ok() ? "OK" : "FAILED") << " (errors=" << error_count()
        << ", warnings=" << warning_count() << ", info=" << info_count() << ")\n";

    for (const auto& i : issues_) {
      oss << "- [" << ::simaai::neat::to_string(i.severity) << "] "
          << (i.contract_id.empty() ? "<contract?>" : i.contract_id);

      if (!i.code.empty())
        oss << " {" << i.code << "}";

      if (i.node_index >= 0) {
        oss << " @node[" << i.node_index << "]";
        if (!i.node_kind.empty())
          oss << ":" << i.node_kind;
        if (!i.node_label.empty())
          oss << " [" << i.node_label << "]";
      }

      oss << ": " << i.message << "\n";
    }
    return oss.str();
  }

  /// @brief Serialize the report as a JSON string (CI/dashboard-friendly).
  std::string to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"ok\":" << (ok() ? "true" : "false") << ",";
    oss << "\"mode\":" << mode_ << ",";
    oss << "\"errors\":" << error_count() << ",";
    oss << "\"warnings\":" << warning_count() << ",";
    oss << "\"info\":" << info_count() << ",";

    // contracts_run
    oss << "\"contracts_run\":[";
    for (std::size_t i = 0; i < contracts_run_.size(); ++i) {
      if (i)
        oss << ",";
      oss << "\"" << json_escape_(contracts_run_[i]) << "\"";
    }
    oss << "],";

    // issues
    oss << "\"issues\":[";
    for (std::size_t i = 0; i < issues_.size(); ++i) {
      if (i)
        oss << ",";
      const auto& it = issues_[i];
      oss << "{";
      oss << "\"severity\":\"" << json_escape_(::simaai::neat::to_string(it.severity)) << "\",";
      oss << "\"contract_id\":\"" << json_escape_(it.contract_id) << "\",";
      oss << "\"code\":\"" << json_escape_(it.code) << "\",";
      oss << "\"message\":\"" << json_escape_(it.message) << "\",";
      oss << "\"node_index\":" << it.node_index << ",";
      oss << "\"node_kind\":\"" << json_escape_(it.node_kind) << "\",";
      oss << "\"node_label\":\"" << json_escape_(it.node_label) << "\"";
      oss << "}";
    }
    oss << "]";

    oss << "}";
    return oss.str();
  }

private:
  static std::string json_escape_(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
      switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        // Control chars -> escape as \u00XX
        if (static_cast<unsigned char>(c) < 0x20) {
          const char* hex = "0123456789abcdef";
          out += "\\u00";
          out += hex[(c >> 4) & 0xF];
          out += hex[c & 0xF];
        } else {
          out += c;
        }
        break;
      }
    }
    return out;
  }

  int mode_ = 0;
  std::vector<std::string> contracts_run_;
  std::vector<ValidationIssue> issues_;
};

} // namespace simaai::neat
