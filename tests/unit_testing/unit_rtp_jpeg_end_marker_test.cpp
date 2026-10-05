#include "gst/GstInit.h"
#include "pipeline/internal/RtpJpegEndMarker.h"
#include "test_main.h"
#include "test_utils.h"

#include <array>
#include <vector>

RUN_TEST("unit_rtp_jpeg_end_marker_test", ([] {
           simaai::neat::gst_init_once();
           for (const auto tail :
                {std::array<guint8, 2>{0xff, 0xd9}, {0xb0, 0xd9}, {0xff, 0x00}, {0x00, 0x00}}) {
             std::vector<guint8> bytes = {0xff, 0xd8, tail[0], tail[1]};
             GstBuffer* buffer = gst_buffer_new_allocate(nullptr, bytes.size(), nullptr);
             gst_buffer_fill(buffer, 0, bytes.data(), bytes.size());
             GST_BUFFER_PTS(buffer) = 123;
             GST_BUFFER_DTS(buffer) = 100;
             GST_BUFFER_DURATION(buffer) = 33;
             GstBuffer* original = gst_buffer_ref(buffer);
             GstPadProbeInfo info{};
             info.type = GST_PAD_PROBE_TYPE_BUFFER;
             info.data = buffer;
             require(simaai::neat::pipeline_internal::repair_rtp_jpeg_end_marker(
                         nullptr, &info, nullptr) == GST_PAD_PROBE_OK,
                     "JPEG repair did not forward the buffer");
             GstBuffer* result = GST_PAD_PROBE_INFO_BUFFER(&info);
             if (tail == std::array<guint8, 2>{0xff, 0xd9}) {
               require(result == original, "valid JPEG must remain unchanged");
             } else {
               bytes.insert(bytes.end(), {0xff, 0xd9});
             }
             std::vector<guint8> actual(gst_buffer_get_size(result));
             gst_buffer_extract(result, 0, actual.data(), actual.size());
             require(actual == bytes, "JPEG payload or end marker changed incorrectly");
             require(gst_buffer_peek_memory(result, 0) == gst_buffer_peek_memory(original, 0),
                     "JPEG repair copied the original payload");
             require(GST_BUFFER_PTS(result) == 123 && GST_BUFFER_DTS(result) == 100 &&
                         GST_BUFFER_DURATION(result) == 33,
                     "JPEG repair lost timing metadata");
             require(gst_buffer_get_size(original) == 4, "JPEG repair mutated a shared input");
             gst_buffer_unref(original);
             gst_buffer_unref(result);
           }
         }));
