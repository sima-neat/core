// include/gst/GstHelpers.h
/**
 * @file
 * @ingroup gst
 * @brief Miscellaneous GStreamer helper utilities.
 */
#pragma once

#include <string>

namespace simaai::neat {

bool element_exists(const char* factory);
std::string factory_plugin_path(const char* factory);

void require_element(const char* factory, const char* context);
void require_tensordecoder(const char* context);

} // namespace simaai::neat
