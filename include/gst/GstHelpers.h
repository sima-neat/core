// include/gst/GstHelpers.h
/**
 * @file
 * @ingroup gst
 * @brief Miscellaneous GStreamer factory and element helpers.
 *
 * Small free-function utilities the framework uses to interrogate the local GStreamer
 * registry: check whether a factory is installed, whether it exposes a particular
 * property, and where its plugin shared object lives. The `require_*` variants raise
 * a SessionError when the dependency is missing, with `context` baked into the message
 * so callers can pinpoint which Node triggered the failure.
 */
#pragma once

#include <string>

namespace simaai::neat {

/// Returns true if a GStreamer element factory with `factory` name is registered.
bool element_exists(const char* factory);

/// Returns true if `factory` exists and exposes a property named `property_name`.
bool element_property_exists(const char* factory, const char* property_name);

/// Returns the filesystem path of the plugin `.so` providing `factory`, or empty if unknown.
std::string factory_plugin_path(const char* factory);

/**
 * @brief Throw a SessionError if `factory` is not registered.
 * @param factory GStreamer element factory name (e.g., `"h264parse"`).
 * @param context Caller context interpolated into the error message (typically the Node kind).
 */
void require_element(const char* factory, const char* context);

/**
 * @brief Throw a SessionError if SiMa's `tensordecoder` element is not available.
 * @param context Caller context for the error message.
 */
void require_tensordecoder(const char* context);

} // namespace simaai::neat
