// Exercise public BoxDecode stages with synthetic dense heads and no inference.
#include "model/Model.h"
#include "pipeline/StageRun.h"
#include "gst/GstInit.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "model_archive_fixture_utils.h"
#include "test_utils.h"
#include "test_main.h"
#include <nlohmann/json.hpp>
#include <gst/gst.h>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {
sima_test::ModelArchiveFixture
make_rfdetr_feature_geometry_fixture(simaai::neat::BoxDecodeType type) {
  // RF transformer ingress is a feature map; image geometry comes from upstream metadata.

  auto mpk = nlohmann::json::parse(R"json({
    "name": "rfdetr_feature_geometry",
    "model_sdk_version": "2.1.0",
    "input_nodes": [{"name":"features","type":"buffer","size":1327104,
                     "dtype":"float32","shape":[1,36,36,256]}],
    "plugins": [{
      "name": "MLA_0", "sequence": 1, "processor": "MLA",
      "config_params": {
        "desired_batch_size": 1, "actual_batch_size": 1,
        "number_of_quads_to_user": 1,
        "input_types": [{"scalar":"float32","shape":[1,36,36,256]}],
        "output_types": [
          {"scalar":"float32","shape":[1,1,7,4]},
          {"scalar":"float32","shape":[1,1,7,5]},
          {"scalar":"float32","shape":[1,3,6,7]}
        ]
      },
      "input_nodes": [{"name":"features","type":"buffer","size":1327104,
                       "dtype":"float32","shape":[1,36,36,256]}],
      "output_nodes": [
        {"name":"output_0","type":"buffer","size":112,
         "dtype":"float32","shape":[1,1,7,4],"layout":"normal"},
        {"name":"output_1","type":"buffer","size":140,
         "dtype":"float32","shape":[1,1,7,5],"layout":"normal"},
        {"name":"output_2","type":"buffer","size":504,
         "dtype":"float32","shape":[1,3,6,7],"layout":"normal"}
      ],
      "type": "sgpProcess", "resources": {"executable":"placeholder.elf"}
    }, {
      "name": "boxdecode_rf", "sequence": 2, "processor": "A65",
      "config_params": {"kernel":"boxdecode","params":{"decode_type":"rfdetr_seg"}},
      "input_nodes": [],
      "output_nodes": [{"name":"detections","type":"buffer","size":4096}]
    }]
  })json");
  auto& mla = mpk["plugins"][0];
  auto& terminal = mpk["plugins"][1];
  if (type == simaai::neat::BoxDecodeType::RfDetr) {
    mla["output_nodes"].erase(2);
    mla["config_params"]["output_types"].erase(2);
    terminal["config_params"]["params"]["decode_type"] = "rfdetr";
  }
  terminal["input_nodes"] = mla["output_nodes"];
  return sima_test::make_model_archive_fixture(
      "rfdetr_feature_geometry", {{"etc/rfdetr_feature_geometry_mpk.json", mpk.dump()}});
}

using namespace simaai::neat;

Tensor head(std::vector<int64_t> shape, std::vector<float> values, int index,
            bool attach_meta = true) {
  Tensor tensor;
  tensor.shape = std::move(shape);
  tensor.dtype = TensorDType::Float32;
  tensor.strides_bytes.resize(tensor.shape.size());
  int64_t stride = sizeof(float);
  for (std::size_t axis = tensor.shape.size(); axis-- > 0;) {
    tensor.strides_bytes[axis] = stride;
    stride *= tensor.shape[axis];
  }
  tensor.device = {DeviceType::CPU, 0};
  tensor.route.logical_index = index;
  tensor.route.backend_output_index = index;
  tensor.route.physical_index = index;
  tensor.route.memory_index = index;
  tensor.route.name = "output_" + std::to_string(index);
  tensor.route.backend_name = tensor.route.name;
  tensor.route.segment_name = tensor.route.name;
  PreprocessRuntimeMeta meta;
  meta.original_width = 64;
  meta.original_height = 48;
  meta.resized_width = meta.scaled_width = 64;
  meta.resized_height = meta.scaled_height = 48;
  // Identity image preprocessing covers the whole source frame.
  meta.resize_mode = "stretch";
  meta.color_in = meta.color_out = "RGB";
  if (attach_meta) {
    tensor.semantic.preprocess = meta;
  }

  // Public stages validate the GstBuffer metadata on the retained sample holder,
  // not just the Tensor's semantic copy. Use the same storage adapter as real outputs.
  gst_init_once();
  const std::size_t bytes = values.size() * sizeof(float);
  const std::unique_ptr<GstBuffer, decltype(&gst_buffer_unref)> buffer(
      gst_buffer_new_allocate(nullptr, bytes, nullptr), &gst_buffer_unref);
  require(buffer != nullptr, "allocate synthetic RF head GstBuffer");
  require(gst_buffer_fill(buffer.get(), 0, values.data(), bytes) == bytes,
          "fill synthetic RF head GstBuffer");
  if (attach_meta) {
    require(write_simaai_preprocess_meta(buffer.get(), meta),
            "attach synthetic RF head preprocess metadata");
  }
  const std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample(
      gst_sample_new(buffer.get(), nullptr, nullptr, nullptr), &gst_sample_unref);
  require(sample != nullptr, "wrap synthetic RF head GstSample");
  tensor.storage = pipeline_internal::make_gst_sample_storage(sample.get());
  require(tensor.storage != nullptr, "retain synthetic RF head sample storage");
  // Match an owned Model output while retaining canonical preprocessing metadata.
  auto owned = pipeline_internal::copy_tensor_from_sample_memory(tensor, 0, false);
  owned.route = tensor.route;
  return owned;
}

Sample inputs(bool attach_meta = true) {
  std::vector<float> boxes(7 * 4);
  for (int q = 0; q < 7; ++q) {
    boxes[q * 4] = boxes[q * 4 + 1] = 0.5F;
    boxes[q * 4 + 2] = boxes[q * 4 + 3] = 0.25F;
  }
  // Every query/class score and every mask pixel has sigmoid(0) = .5.
  return Sample{sample_from_tensors(
      TensorList{head({1, 1, 7, 4}, boxes, 0, attach_meta),
                 head({1, 1, 7, 5}, std::vector<float>(35, 0.0F), 1, attach_meta),
                 head({1, 3, 6, 7}, std::vector<float>(126, 0.0F), 2, attach_meta)})};
}

void check_masks(const SegmentationDecodeTensors& decoded, int count) {
  require(decoded.boxes.shape == std::vector<int64_t>({count, 6}), "stage box count");
  require(decoded.masks.shape == std::vector<int64_t>({count, 3, 6}),
          "stage native mask dimensions");
  require(decoded.masks.dtype == TensorDType::Float32, "stage native FP32 masks");
  const auto bytes = decoded.masks.copy_payload_bytes();
  for (std::size_t i = 0; i < bytes.size(); i += sizeof(float)) {
    float value = 0;
    std::memcpy(&value, bytes.data() + i, sizeof(value));
    require(value == 0.5F, "stage mask probability must retain sigmoid(0)");
  }
}
} // namespace

RUN_TEST("rfdetr_stage_options_test", ([] {
           const auto fixture = make_rfdetr_feature_geometry_fixture(BoxDecodeType::RfDetrSeg);
           Model::Options base;
           base.decode_type = BoxDecodeType::RfDetrSeg;
           base.preprocess.kind = InputKind::Tensor;
           base.preprocess.enable = AutoFlag::Off;
           base.boxdecode_resize_mode = ResizeMode::Stretch;
           base.score_threshold = 0.75F;
           base.top_k = 1;
           const Model model(fixture.tar_path, base);
           // Repeated calls exercise both public result paths and runner reuse.
           for (const double threshold : {0.25, 0.4, 0.5}) {
             stages::BoxDecodeOptions opt(BoxDecodeType::RfDetrSeg);
             opt.detection_threshold = threshold;
             opt.top_k = 2;
             const auto outputs = stages::BoxDecode(inputs(), model, opt);
             require(outputs.size() == 1, "one public stage output");
             const auto tensors = stages::Tensors(outputs.front());
             require(tensors.size() == 1, "one RF wire tensor");
             check_masks(decode_segmentation(tensors).front(), 2);
             const auto results = stages::BoxDecodeResults(inputs(), model, opt);
             require(results.size() == 1 && results.front().boxes.size() == 2,
                     "structured stage count");
             // BoxDecodeResults preserves the typed raw wire buffer, including its mask section.
             Tensor wire;
             wire.shape = {static_cast<int64_t>(results.front().raw.size())};
             wire.dtype = TensorDType::UInt8;
             wire.storage = make_cpu_owned_storage(results.front().raw.size());
             {
               auto mapping = wire.storage->map(MapMode::Write);
               std::memcpy(mapping.data, results.front().raw.data(), results.front().raw.size());
             }
             tag_detection_format(wire, "RFDETR_SEG_V1");
             check_masks(decode_segmentation({wire}).front(), 2);
           }
           stages::BoxDecodeOptions zero(BoxDecodeType::RfDetrSeg);
           // Explicit zeros must override the original Model's .75 threshold and top_k=1.
           const auto zero_out = stages::BoxDecode(inputs(), model, zero);
           check_masks(decode_segmentation(stages::Tensors(zero_out.front())).front(), 35);
           const auto zero_results = stages::BoxDecodeResults(inputs(), model, zero);
           require(zero_results.front().boxes.size() == 35,
                   "RF zero score/top_k controls must survive cloning");
           // Metadata-free Stretch uses explicit source geometry. The same pack
           // must not reuse a runner prepared for another source size.
           base.score_threshold = 0.25F;
           base.top_k = 3;
           for (const int scale : {1, 2}) {
             base.boxdecode_original_width = 64 * scale;
             base.boxdecode_original_height = 48 * scale;
             const Model sized_model(fixture.tar_path, base);
             const auto output = stages::Postprocess(inputs(false), sized_model);
             const auto decoded = decode_segmentation(stages::Tensors(output)).front();
             check_masks(decoded, 3); // Source geometry must not resize native masks.
             const auto bytes = decoded.boxes.copy_payload_bytes();
             float box[4];
             std::memcpy(box, bytes.data(), sizeof(box));
             require(box[0] == 24 * scale && box[1] == 18 * scale && box[2] == 40 * scale &&
                         box[3] == 30 * scale,
                     "postprocess boxes must use the requested source geometry");
           }
         }));
