#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include <gst/gst.h>
#include <memory>
namespace simaai::neat {
namespace camera_copy {
class Backend;
}
bool register_neat_v4l2_copy_source();
GstElement* make_v4l2_copy_source_for_test(std::shared_ptr<camera_copy::Backend> backend);
} // namespace simaai::neat
