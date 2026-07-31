#include "pipeline/internal/GstLaunchBindings.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <unordered_set>
#include <utility>

namespace simaai::neat::pipeline_internal::gst_launch {
namespace {

bool ascii_space(char c) {
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

bool identifier_start(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool identifier_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '-' || c == '%' ||
         c == ':';
}

bool protocol_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '+' || c == '-' || c == '.';
}

std::string unescape_value(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '\\' && i + 1U < value.size()) {
      out.push_back(value[++i]);
    } else {
      out.push_back(value[i]);
    }
  }
  return out;
}

std::string canonical_assignment_value(std::string_view raw, char* quote) {
  *quote = '\0';
  if (raw.size() >= 2U && raw.front() == '"' && raw.back() == '"') {
    *quote = '"';
    raw.remove_prefix(1U);
    raw.remove_suffix(1U);
  } else if (raw.size() >= 2U && raw.front() == '\'' && raw.back() == '\'') {
    // gst_parse_split_assignment() does not strip single quotes. Preserve them as part of the
    // actual string-property value while still recording the lexical form for lossless rewriting.
    *quote = '\'';
  }
  return unescape_value(raw);
}

std::size_t scan_quoted(std::string_view text, std::size_t begin, char quote, bool* complete) {
  std::size_t i = begin + 1U;
  while (i < text.size()) {
    if (text[i] == '\\') {
      if (i + 1U >= text.size()) {
        *complete = false;
        return text.size();
      }
      i += 2U;
      continue;
    }
    if (text[i] == quote) {
      return i + 1U;
    }
    ++i;
  }
  *complete = false;
  return text.size();
}

std::size_t scan_value(std::string_view text, std::size_t begin, bool* complete) {
  if (begin >= text.size()) {
    *complete = false;
    return begin;
  }
  if (text[begin] == '"' || text[begin] == '\'') {
    return scan_quoted(text, begin, text[begin], complete);
  }
  std::size_t i = begin;
  while (i < text.size() && !ascii_space(text[i])) {
    if (text[i] == '\\') {
      if (i + 1U >= text.size()) {
        *complete = false;
        return text.size();
      }
      i += 2U;
      continue;
    }
    ++i;
  }
  return i;
}

bool begins_protocol_url(std::string_view text, std::size_t begin, std::size_t* scheme_end) {
  if (begin >= text.size() || !std::isalpha(static_cast<unsigned char>(text[begin]))) {
    return false;
  }
  std::size_t i = begin + 1U;
  while (i < text.size() && protocol_char(text[i])) {
    ++i;
  }
  if (i + 2U >= text.size() || text.substr(i, 3U) != "://") {
    return false;
  }
  *scheme_end = i + 3U;
  return true;
}

std::size_t skip_url(std::string_view text, std::size_t begin, bool* complete) {
  std::size_t scheme_end = begin;
  if (!begins_protocol_url(text, begin, &scheme_end)) {
    return begin;
  }
  if (scheme_end < text.size() && (text[scheme_end] == '"' || text[scheme_end] == '\'')) {
    return scan_quoted(text, scheme_end, text[scheme_end], complete);
  }
  std::size_t i = scheme_end;
  while (i < text.size() && !ascii_space(text[i])) {
    if (text[i] == '\\') {
      if (i + 1U >= text.size()) {
        *complete = false;
        return text.size();
      }
      i += 2U;
      continue;
    }
    ++i;
  }
  return i;
}

bool looks_like_caps_start(std::string_view text, std::size_t begin) {
  while (begin < text.size() && ascii_space(text[begin])) {
    ++begin;
  }
  std::size_t i = begin;
  bool saw_slash = false;
  while (i < text.size() && !ascii_space(text[i]) && text[i] != '!' && text[i] != ':') {
    if (text[i] == '/') {
      saw_slash = i > begin && i + 1U < text.size();
      break;
    }
    if (!(std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '-')) {
      break;
    }
    ++i;
  }
  return saw_slash;
}

// A caps link is lexically one link token (`! caps !` or `: caps :`). Skip its body so caps fields
// such as `name=` are never confused with element properties.
std::optional<std::size_t> caps_link_end(std::string_view text, std::size_t operator_pos,
                                         bool* complete) {
  if (operator_pos >= text.size() || (text[operator_pos] != '!' && text[operator_pos] != ':')) {
    return std::nullopt;
  }
  std::size_t body = operator_pos + 1U;
  if (!looks_like_caps_start(text, body)) {
    return std::nullopt;
  }
  bool in_double = false;
  bool in_single = false;
  int paren_depth = 0;
  int bracket_depth = 0;
  int brace_depth = 0;
  const char closing_operator = text[operator_pos];
  for (std::size_t i = body; i < text.size(); ++i) {
    if (text[i] == '\\') {
      if (i + 1U >= text.size()) {
        *complete = false;
        return text.size();
      }
      ++i;
      continue;
    }
    if (text[i] == '"' && !in_single) {
      in_double = !in_double;
      continue;
    }
    if (text[i] == '\'' && !in_double) {
      in_single = !in_single;
      continue;
    }
    if (in_double || in_single) {
      continue;
    }
    switch (text[i]) {
    case '(':
      ++paren_depth;
      continue;
    case ')':
      if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0) {
        // The caps expression is the last item in an enclosing Gst bin, for example
        // `( appsrc name=src ! video/x-raw,format=NV12 )`. The closing parenthesis belongs to
        // the surrounding bin rather than the caps value.
        return i;
      }
      paren_depth = std::max(0, paren_depth - 1);
      continue;
    case '[':
      ++bracket_depth;
      continue;
    case ']':
      bracket_depth = std::max(0, bracket_depth - 1);
      continue;
    case '{':
      ++brace_depth;
      continue;
    case '}':
      brace_depth = std::max(0, brace_depth - 1);
      continue;
    default:
      break;
    }
    if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 && text[i] == ';') {
      // A semicolon starts a separate top-level chain. It is also a valid terminator for caps
      // when there is no trailing link operator.
      return i;
    }
    if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 && text[i] == closing_operator) {
      return i + 1U;
    }
  }
  if (in_double || in_single || paren_depth != 0 || bracket_depth != 0 || brace_depth != 0) {
    *complete = false;
  }
  return text.size();
}

struct Replacement {
  ByteSpan span;
  std::string text;
};

std::string encode_replacement(std::string_view replacement, char quote) {
  const bool needs_double_quotes = quote == '"' || quote == '\'' ||
                                   std::any_of(replacement.begin(), replacement.end(), [](char c) {
                                     return ascii_space(c) || c == '!' || c == ';';
                                   });
  if (needs_double_quotes) {
    // Single quotes are literal characters in a Gst launch assignment, not string delimiters.
    // Emit mapped canonical values with Gst's real (double-quote) quoting so a rewritten value
    // re-analyzes to exactly `replacement` rather than gaining literal quote characters.
    std::string out;
    out.reserve(replacement.size() + 2U);
    out.push_back('"');
    for (char c : replacement) {
      if (c == '"' || c == '\\') {
        out.push_back('\\');
      }
      out.push_back(c);
    }
    out.push_back('"');
    return out;
  }
  return std::string(replacement);
}

RewriteResult apply_replacements(std::string_view launch, const Analysis& analysis,
                                 std::vector<Replacement> replacements) {
  RewriteResult result;
  result.complete = analysis.complete;
  result.diagnostics = analysis.diagnostics;
  if (!analysis.complete) {
    result.text = std::string(launch);
    return result;
  }

  std::sort(replacements.begin(), replacements.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.span.begin < rhs.span.begin; });
  std::size_t cursor = 0;
  result.text.reserve(launch.size());
  for (const auto& replacement : replacements) {
    if (replacement.span.begin < cursor || replacement.span.end > launch.size() ||
        replacement.span.begin > replacement.span.end) {
      result.complete = false;
      result.diagnostics.push_back(
          Diagnostic{.span = replacement.span, .message = "overlapping or invalid rewrite span"});
      result.text = std::string(launch);
      result.changed = false;
      return result;
    }
    result.text.append(launch.substr(cursor, replacement.span.begin - cursor));
    result.text.append(replacement.text);
    cursor = replacement.span.end;
    result.changed = true;
  }
  result.text.append(launch.substr(cursor));
  return result;
}

} // namespace

Analysis analyze(std::string_view launch) {
  Analysis result;
  std::size_t i = 0;
  while (i < launch.size()) {
    if (ascii_space(launch[i])) {
      ++i;
      continue;
    }

    if (launch[i] == '!' || launch[i] == ':') {
      result.has_topology_syntax = true;
      if (const auto end = caps_link_end(launch, i, &result.complete)) {
        i = *end;
      } else {
        ++i;
      }
      continue;
    }

    std::size_t scheme_end = i;
    if (begins_protocol_url(launch, i, &scheme_end)) {
      i = skip_url(launch, i, &result.complete);
      continue;
    }

    if (launch[i] == '"' || launch[i] == '\'') {
      const std::size_t begin = i;
      i = scan_quoted(launch, i, launch[i], &result.complete);
      if (!result.complete) {
        result.diagnostics.push_back(
            Diagnostic{.span = {begin, launch.size()}, .message = "unterminated quoted token"});
      }
      continue;
    }

    if (launch[i] == '(' || launch[i] == ')' || launch[i] == ';') {
      result.has_topology_syntax = true;
      ++i;
      continue;
    }

    if (!identifier_start(launch[i])) {
      if (launch[i] == '\\' && i + 1U < launch.size()) {
        i += 2U;
      } else if (launch[i] == '\\') {
        result.complete = false;
        result.diagnostics.push_back(Diagnostic{.span = {i, i + 1U}, .message = "dangling escape"});
        ++i;
      } else {
        ++i;
      }
      continue;
    }

    const std::size_t token_begin = i;
    std::size_t identifier_end = i + 1U;
    while (identifier_end < launch.size() && identifier_char(launch[identifier_end])) {
      ++identifier_end;
    }

    std::size_t after_key = identifier_end;
    while (after_key < launch.size() && ascii_space(launch[after_key])) {
      ++after_key;
    }
    if (after_key < launch.size() && launch[after_key] == '=') {
      std::size_t value_begin = after_key + 1U;
      while (value_begin < launch.size() && ascii_space(launch[value_begin])) {
        ++value_begin;
      }
      bool value_complete = true;
      const std::size_t value_end = scan_value(launch, value_begin, &value_complete);
      if (!value_complete) {
        result.complete = false;
        result.diagnostics.push_back(Diagnostic{.span = {token_begin, launch.size()},
                                                .message = "unterminated assignment value"});
      }
      char quote = '\0';
      const std::string canonical =
          canonical_assignment_value(launch.substr(value_begin, value_end - value_begin), &quote);
      result.assignments.push_back(Assignment{
          .key = std::string(launch.substr(token_begin, identifier_end - token_begin)),
          .canonical_value = canonical,
          .token_span = {token_begin, value_end},
          .value_span = {value_begin, value_end},
          .quote = quote,
      });
      if (value_begin >= launch.size()) {
        result.diagnostics.push_back(
            Diagnostic{.span = {token_begin, launch.size()}, .message = "assignment has no value"});
      }
      i = std::max(value_end, value_begin);
      continue;
    }

    std::size_t dot = identifier_end;
    while (dot < launch.size() && ascii_space(launch[dot])) {
      ++dot;
    }
    if (dot < launch.size() && launch[dot] == '.') {
      std::size_t suffix = dot + 1U;
      while (suffix < launch.size() && ascii_space(launch[suffix])) {
        ++suffix;
      }
      if (suffix < launch.size() && launch[suffix] != '(' && identifier_start(launch[suffix])) {
        std::size_t suffix_end = suffix + 1U;
        while (suffix_end < launch.size() && identifier_char(launch[suffix_end])) {
          ++suffix_end;
        }
        result.references.push_back(Reference{
            .canonical_element_name =
                std::string(launch.substr(token_begin, identifier_end - token_begin)),
            .token_span = {token_begin, suffix_end},
            .element_span = {token_begin, identifier_end},
        });
        i = suffix_end;
        continue;
      }
      // A bare `element.` is also a real Gst launch reference (all pads / a pad supplied by the
      // adjacent link clause). Exclude `bin.(`, which opens a named bin rather than referring to
      // a previously declared element.
      if (suffix >= launch.size() || launch[suffix] != '(') {
        result.references.push_back(Reference{
            .canonical_element_name =
                std::string(launch.substr(token_begin, identifier_end - token_begin)),
            .token_span = {token_begin, dot + 1U},
            .element_span = {token_begin, identifier_end},
        });
        i = dot + 1U;
        continue;
      }
    }

    i = identifier_end;
  }
  return result;
}

std::vector<const Assignment*> explicit_name_bindings(const Analysis& analysis) {
  std::vector<const Assignment*> out;
  for (const auto& assignment : analysis.assignments) {
    if (assignment.key == "name") {
      out.push_back(&assignment);
    }
  }
  return out;
}

RewriteResult rewrite(std::string_view launch, const Analysis& analysis, const NameMapping& mapping,
                      std::span<const std::string_view> alias_properties) {
  if (!analysis.complete || mapping.empty()) {
    RewriteResult result;
    result.complete = analysis.complete;
    result.diagnostics = analysis.diagnostics;
    result.text = std::string(launch);
    return result;
  }

  std::unordered_set<std::string_view> aliases(alias_properties.begin(), alias_properties.end());
  std::vector<Replacement> replacements;
  replacements.reserve(analysis.assignments.size() + analysis.references.size());

  for (const auto& assignment : analysis.assignments) {
    if (assignment.key != "name" && aliases.find(assignment.key) == aliases.end()) {
      continue;
    }
    const auto it = mapping.find(assignment.canonical_value);
    if (it == mapping.end() || it->second == assignment.canonical_value) {
      continue;
    }
    replacements.push_back(Replacement{.span = assignment.value_span,
                                       .text = encode_replacement(it->second, assignment.quote)});
  }
  for (const auto& reference : analysis.references) {
    const auto it = mapping.find(reference.canonical_element_name);
    if (it == mapping.end() || it->second == reference.canonical_element_name) {
      continue;
    }
    replacements.push_back(Replacement{.span = reference.element_span, .text = it->second});
  }

  return apply_replacements(launch, analysis, std::move(replacements));
}

RewriteResult rewrite_assignment_values(std::string_view launch, const Analysis& analysis,
                                        std::span<const AssignmentValueReplacement> replacements) {
  std::vector<Replacement> encoded;
  encoded.reserve(replacements.size());
  for (const auto& replacement : replacements) {
    encoded.push_back(Replacement{
        .span = replacement.value_span,
        .text = encode_replacement(replacement.canonical_value, replacement.quote),
    });
  }
  return apply_replacements(launch, analysis, std::move(encoded));
}

} // namespace simaai::neat::pipeline_internal::gst_launch
