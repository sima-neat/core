#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simaai::neat::pipeline_internal::gst_launch {

struct ByteSpan {
  std::size_t begin = 0;
  std::size_t end = 0;

  bool empty() const noexcept {
    return begin >= end;
  }
};

struct Diagnostic {
  ByteSpan span;
  std::string message;
};

struct Assignment {
  std::string key;
  std::string canonical_value;
  ByteSpan token_span;
  ByteSpan value_span;
  char quote = '\0';
};

struct Reference {
  std::string canonical_element_name;
  ByteSpan token_span;
  ByteSpan element_span;
};

struct Analysis {
  std::vector<Assignment> assignments;
  std::vector<Reference> references;
  std::vector<Diagnostic> diagnostics;
  bool has_topology_syntax = false;
  bool complete = true;
};

struct RewriteResult {
  std::string text;
  std::vector<Diagnostic> diagnostics;
  bool complete = true;
  bool changed = false;
};

struct AssignmentValueReplacement {
  ByteSpan value_span;
  std::string canonical_value;
  char quote = '\0';
};

using NameMapping = std::unordered_map<std::string, std::string>;

// Lossless, purpose-limited analysis of gst-launch assignments and named element/pad references.
// This is deliberately not a grammar or topology parser.
Analysis analyze(std::string_view launch);

std::vector<const Assignment*> explicit_name_bindings(const Analysis& analysis);

RewriteResult rewrite(std::string_view launch, const Analysis& analysis, const NameMapping& mapping,
                      std::span<const std::string_view> alias_properties = {});

// Apply occurrence-specific property-value edits using spans returned by analyze().
RewriteResult rewrite_assignment_values(std::string_view launch, const Analysis& analysis,
                                        std::span<const AssignmentValueReplacement> replacements);

} // namespace simaai::neat::pipeline_internal::gst_launch
