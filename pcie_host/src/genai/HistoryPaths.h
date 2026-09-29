/**
 * @file
 * @brief Show `print history` with the host image paths, like the devkit does.
 *
 * The card's Chat stores the path of each pulled image on the card (for example
 * /data/recv/pcie-genai/h1-3-0.jpg). The host knows which host file it sent
 * under each name, so it can put the user's path back before printing.
 */
#pragma once

#include <map>
#include <string>

namespace simaai::neat::pcie::genai::internal {

/// Replace card image paths in a history JSON array with host paths. @p sent
/// maps a sent image name ("pcie-genai/h1-3-0.jpg") to the host path. An image
/// whose path equals a name, ends with "/<name>", or has the same file name
/// (the card stores it as <recv root>/h1-3-0.jpg) is replaced; anything else
/// is left as it is. Never throws: a body that is not a JSON array is returned
/// unchanged.
std::string map_history_image_paths(const std::string& history_json,
                                    const std::map<std::string, std::string>& sent);

} // namespace simaai::neat::pcie::genai::internal
