/**
 * @file
 * @ingroup gst
 * @brief Bus logging and message helpers.
 */
#pragma once

#include <string>

struct _GstElement;
struct _GstMessage;
typedef struct _GstElement GstElement;
typedef struct _GstMessage GstMessage;

namespace simaai::neat {

using BusMessageFn = void (*)(const char* type, const char* src, const std::string& line,
                              void* user_data);
using BusErrorFn = void (*)(const std::string& line, void* user_data);

std::string gst_message_to_string(GstMessage* msg);

void drain_bus(GstElement* pipeline, BusMessageFn on_message, void* user_data);

void throw_if_bus_error(GstElement* pipeline, BusMessageFn on_message, void* user_data,
                        BusErrorFn on_error, void* error_user_data);

} // namespace simaai::neat
