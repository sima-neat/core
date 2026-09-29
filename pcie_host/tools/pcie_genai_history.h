/**
 * @file
 * @brief Pure helpers for the pcie-genai REPL command history.
 *
 * The interactive REPL edits and recalls past prompts with GNU readline (see
 * pcie_genai_readline.h), matching the LLiMa devkit console. The two decisions
 * that do not need readline -- where the history file lives, and whether a line
 * is worth keeping -- live here so they can be unit-tested with no tty and no
 * readline dependency.
 */
#pragma once

#include <filesystem>
#include <string_view>

namespace simaai::neat::pcie::genai::tools {

// Where the persistent history is stored: "$HOME/.pcie_genai_history", or
// "./.pcie_genai_history" when HOME is unset or empty (so a stripped env keeps
// the file somewhere readable instead of trying to write under "/").
std::filesystem::path history_file_path();

// True if `line` should be added to history. Skips an empty line and a line
// identical to the immediately-previous one (`previous`), matching the LLiMa
// console's consecutive-duplicate suppression. Pass an empty `previous` for the
// first line of a session.
bool should_record_history(std::string_view line, std::string_view previous);

} // namespace simaai::neat::pcie::genai::tools
