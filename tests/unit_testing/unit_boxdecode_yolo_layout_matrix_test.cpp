#include "pipeline/internal/sima/BoxDecodeStaticContractExtractor.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/stagesemantics/BoxDecodeStageSemantics.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

RUN_TEST(
    "unit_boxdecode_yolo_layout_matrix_test", ([] {
      using namespace simaai::neat;
      using namespace simaai::neat::pipeline_internal::sima;
      using namespace simaai::neat::pipeline_internal::sima::stagesemantics;

      auto make_flags = [](bool quant_needed, bool tess_needed) {
        return ModelManagedRouteFlags{
            .quant_needed = quant_needed,
            .tess_needed = tess_needed,
            .pre_cast_needed = false,
            .quant_contract_required = quant_needed,
            .include_pre_stage = false,
            .boxdecode_selected = true,
        };
      };

      auto tensor = [](int index, std::string name, std::string dtype,
                       std::vector<std::int64_t> shape, std::size_t size_bytes,
                       MpkShapeSemantics semantics = MpkShapeSemantics::Geometry,
                       std::vector<std::int64_t> logical_shape = {}, std::string segment_name = {},
                       int physical_index = 0) {
        MpkTensorContract out;
        out.tensor_index = index;
        out.physical_index = physical_index;
        out.source_physical_index = physical_index;
        out.name = std::move(name);
        out.segment_name = std::move(segment_name);
        out.dtype = std::move(dtype);
        out.mpk_shape = std::move(shape);
        out.shape_semantics = semantics;
        out.size_bytes = size_bytes;
        out.logical_shape = std::move(logical_shape);
        return out;
      };

      auto add_edge = [](MpkContract& mpk, std::size_t src, int src_out, std::size_t dst,
                         int dst_in, const std::string& tensor_name) {
        mpk.edges.push_back(MpkContractEdge{
            .src_plugin_index = src,
            .src_output_index = src_out,
            .dst_plugin_index = dst,
            .dst_input_index = dst_in,
            .src_plugin = mpk.plugins[src].name,
            .dst_plugin = mpk.plugins[dst].name,
            .tensor_name = tensor_name,
        });
      };

      auto add_mla = [&](MpkContract& mpk, const std::string& parent_name, const std::string& dtype,
                         std::size_t size_bytes,
                         const std::vector<std::int64_t>& logical_shape = {}) {
        MpkPluginIoContract mla;
        mla.name = "MLA_0";
        mla.sequence = 1;
        mla.processor = "MLA";
        mla.kernel = "mla";
        mla.canonical_output_dtype = dtype;
        mla.output_tensors.push_back(tensor(0, parent_name, dtype,
                                            {1, static_cast<std::int64_t>(size_bytes)}, size_bytes,
                                            MpkShapeSemantics::PackedExtent, logical_shape, {}, 0));
        mpk.plugins.push_back(std::move(mla));
      };

      auto add_unpack = [&](MpkContract& mpk, const std::string& input_name,
                            const std::string& output_name, const std::string& parent_name,
                            const std::string& dtype,
                            const std::vector<std::int64_t>& physical_shape,
                            const std::vector<std::int64_t>& logical_nhwc,
                            std::size_t physical_size_bytes,
                            MpkShapeSemantics output_semantics = MpkShapeSemantics::Geometry) {
        MpkPluginIoContract unpack;
        unpack.name = "MLA_0_ofm_unpack_transform";
        unpack.sequence = static_cast<int>(mpk.plugins.size()) + 1;
        unpack.kernel = "ofm_unpack";
        unpack.input_tensors.push_back(tensor(
            0, input_name, dtype, {1, static_cast<std::int64_t>(physical_size_bytes)},
            physical_size_bytes, MpkShapeSemantics::PackedExtent, logical_nhwc, parent_name, 0));
        unpack.output_tensors.push_back(tensor(0, output_name, dtype, physical_shape,
                                               physical_size_bytes, output_semantics, logical_nhwc,
                                               parent_name, 0));
        mpk.plugins.push_back(std::move(unpack));
        add_edge(mpk, 0, 0, mpk.plugins.size() - 1U, 0, input_name);
      };

      auto add_slice = [&](MpkContract& mpk, const std::string& input_name,
                           const std::string& output_name, const std::string& dtype,
                           const std::vector<std::int64_t>& physical_nhwc,
                           const std::vector<std::int64_t>& logical_nhwc,
                           std::size_t physical_size_bytes, std::size_t logical_size_bytes) {
        const std::size_t producer = mpk.plugins.size() - 1U;
        MpkPluginIoContract slice;
        slice.name = "slice_transform";
        slice.sequence = static_cast<int>(mpk.plugins.size()) + 1;
        slice.kernel = "slice_transform";
        slice.slice_begin = {0, 0, 0, 0};
        slice.input_tensors.push_back(tensor(0, input_name, dtype, physical_nhwc,
                                             physical_size_bytes, MpkShapeSemantics::Geometry,
                                             logical_nhwc, {}, 0));
        slice.output_tensors.push_back(tensor(0, output_name, dtype, logical_nhwc,
                                              logical_size_bytes, MpkShapeSemantics::Geometry,
                                              logical_nhwc, {}, 0));
        mpk.plugins.push_back(std::move(slice));
        add_edge(mpk, producer, 0, mpk.plugins.size() - 1U, 0, input_name);
      };

      auto add_evtess = [&](MpkContract& mpk, const std::string& input_name,
                            const std::string& output_name, const std::string& dtype,
                            const std::vector<std::int64_t>& logical_nhwc,
                            std::size_t packed_size_bytes) {
        const std::size_t producer = mpk.plugins.size() - 1U;
        MpkPluginIoContract evtess;
        evtess.name = "evtess_transform";
        evtess.sequence = static_cast<int>(mpk.plugins.size()) + 1;
        evtess.processor = "EVXX";
        evtess.kernel = "tessellation_transform";
        evtess.input_tensors.push_back(tensor(0, input_name, dtype, logical_nhwc, packed_size_bytes,
                                              MpkShapeSemantics::Geometry, logical_nhwc, {}, 0));
        evtess.output_tensors.push_back(
            tensor(0, output_name, dtype, {1, static_cast<std::int64_t>(packed_size_bytes)},
                   packed_size_bytes, MpkShapeSemantics::PackedExtent, logical_nhwc, {}, 0));
        mpk.plugins.push_back(std::move(evtess));
        add_edge(mpk, producer, 0, mpk.plugins.size() - 1U, 0, input_name);
      };

      auto add_detess =
          [&](MpkContract& mpk, const std::string& input_name, const std::string& output_name,
              const std::string& dtype, const std::vector<std::int64_t>& frame_hwc,
              const std::vector<std::int64_t>& slice_hwc, std::size_t packed_size_bytes) {
            const std::size_t producer = mpk.plugins.size() - 1U;
            MpkPluginIoContract detess;
            detess.name = "detessellation_transform";
            detess.sequence = static_cast<int>(mpk.plugins.size()) + 1;
            detess.kernel = "detessellation_transform";
            detess.frame_shape = frame_hwc;
            detess.frame_type = dtype;
            detess.slice_shape = slice_hwc;
            detess.has_cblock = true;
            detess.cblock = true;
            detess.has_align_c16 = true;
            detess.align_c16 = true;
            detess.input_tensors.push_back(
                tensor(0, input_name, dtype, {1, static_cast<std::int64_t>(packed_size_bytes)},
                       packed_size_bytes, MpkShapeSemantics::PackedExtent, frame_hwc, {}, 0));
            detess.output_tensors.push_back(
                tensor(0, output_name, dtype, frame_hwc,
                       static_cast<std::size_t>(frame_hwc[0] * frame_hwc[1] * frame_hwc[2]) *
                           (dtype == "BF16" ? 2U : 1U),
                       MpkShapeSemantics::Geometry, frame_hwc, {}, 0));
            mpk.plugins.push_back(std::move(detess));
            add_edge(mpk, producer, 0, mpk.plugins.size() - 1U, 0, input_name);
          };

      struct Case {
        const char* name;
        MpkContract mpk;
        ModelManagedRouteFlags flags;
        BoxDecodeType decode_type;
        BoxDecodeSourceStorageKind storage;
        std::vector<int> input_shape;
        std::vector<int> slice_shape;
        std::string dtype;
      };

      auto mlatess_packed = [&] {
        MpkContract mpk;
        add_mla(mpk, "mla_tess_parent", "BF16", 80U * 80U * 80U * 2U, {80, 80, 80});
        add_unpack(mpk, "mla_tess_parent", "class_logit_0_packed", "mla_tess_parent", "BF16",
                   {1, static_cast<std::int64_t>(80U * 80U * 80U * 2U)}, {80, 80, 80},
                   80U * 80U * 80U * 2U, MpkShapeSemantics::PackedExtent);
        add_detess(mpk, "class_logit_0_packed", "class_logit_0", "BF16", {80, 80, 80}, {16, 4, 80},
                   80U * 80U * 80U * 2U);
        return mpk;
      }();

      auto unpack_slice_dense = [&] {
        MpkContract mpk;
        add_mla(mpk, "unpack_slice_parent", "INT8", 80U * 80U * 96U, {1, 80, 80, 80});
        add_unpack(mpk, "unpack_slice_parent", "class_logit_dense", "unpack_slice_parent", "INT8",
                   {1, 80, 80, 96}, {1, 80, 80, 80}, 80U * 80U * 96U);
        add_slice(mpk, "class_logit_dense", "class_logit_0", "INT8", {1, 80, 80, 96},
                  {1, 80, 80, 80}, 80U * 80U * 96U, 80U * 80U * 80U);
        return mpk;
      }();

      auto contiguous_dense = [&] {
        MpkContract mpk;
        add_mla(mpk, "contiguous_parent", "BF16", 80U * 80U * 80U * 2U, {1, 80, 80, 80});
        add_unpack(mpk, "contiguous_parent", "class_logit_0", "contiguous_parent", "BF16",
                   {1, 80, 80, 80}, {1, 80, 80, 80}, 80U * 80U * 80U * 2U);
        return mpk;
      }();

      auto evtess_packed = [&] {
        MpkContract mpk;
        add_mla(mpk, "evtess_dense_parent", "INT8", 80U * 80U * 80U, {1, 80, 80, 80});
        add_unpack(mpk, "evtess_dense_parent", "evtess_unpacked_parent", "evtess_dense_parent",
                   "INT8", {1, static_cast<std::int64_t>(80U * 80U * 80U)}, {1, 80, 80, 80},
                   80U * 80U * 80U, MpkShapeSemantics::PackedExtent);
        add_evtess(mpk, "evtess_unpacked_parent", "evtess_class_packed", "INT8", {1, 80, 80, 80},
                   80U * 80U * 80U);
        add_detess(mpk, "evtess_class_packed", "class_logit_0", "INT8", {80, 80, 80}, {16, 16, 80},
                   80U * 80U * 80U);
        return mpk;
      }();

      const std::vector<Case> cases = {
          {"yolov8_bf16_mlatess_packed_cblock",
           std::move(mlatess_packed),
           make_flags(false, true),
           BoxDecodeType::YoloV8,
           BoxDecodeSourceStorageKind::PackedCBlock,
           {80, 80, 80},
           {16, 4, 80},
           "BF16"},
          {"yolo26_unpack_slice_dense_hwc",
           std::move(unpack_slice_dense),
           make_flags(false, false),
           BoxDecodeType::YoloV26,
           BoxDecodeSourceStorageKind::DenseHwcPhysical,
           {80, 80, 96},
           {80, 80, 80},
           "INT8"},
          {"yolox_contiguous_dense_hwc",
           std::move(contiguous_dense),
           make_flags(false, false),
           BoxDecodeType::YoloX,
           BoxDecodeSourceStorageKind::DenseHwcPhysical,
           {80, 80, 80},
           {80, 80, 80},
           "BF16"},
          {"yolov6_int8_evtess_packed_cblock",
           std::move(evtess_packed),
           make_flags(false, true),
           BoxDecodeType::YoloV6,
           BoxDecodeSourceStorageKind::PackedCBlock,
           {80, 80, 80},
           {16, 16, 80},
           "INT8"},
      };

      for (const auto& c : cases) {
        std::string error;
        auto extracted = build_boxdecode_static_contract_from_mpk(c.mpk, c.flags, &error);
        require(extracted.has_value(), std::string(c.name) + ": MPK extraction failed: " + error);
        require(extracted->tensors.size() == 1U,
                std::string(c.name) + ": expected one synthetic class tensor");
        require(extracted->tensors[0].source_storage_kind == c.storage,
                std::string(c.name) + ": source storage kind mismatch");
        require(extracted->tensors[0].input_shape == c.input_shape,
                std::string(c.name) + ": input_shape mismatch");
        require(extracted->tensors[0].slice_shape == c.slice_shape,
                std::string(c.name) + ": slice_shape mismatch");
        require(extracted->tensors[0].data_type == c.dtype,
                std::string(c.name) + ": dtype mismatch");

        auto finalized = finalize_boxdecode_static_contract(
            *extracted, c.decode_type, std::nullopt, c.flags, BoxDecodeTypeOption::Auto, 0.25, 0.55,
            100, 80, {"orig_width", "orig_height"});
        auto compiled = build_boxdecode_compiled_contract(finalized);
        require(compiled.payload.decode_type == c.decode_type,
                std::string(c.name) + ": compiled decode type mismatch");
        require(compiled.payload.num_classes == 80,
                std::string(c.name) + ": compiled num_classes mismatch");
        require(compiled.payload.tensor_storage_kind.size() == 1U &&
                    compiled.payload.tensor_storage_kind[0] == static_cast<int>(c.storage),
                std::string(c.name) + ": compiled storage kind mismatch");
        require(!compiled.payload.slice_shapes.empty() &&
                    compiled.payload.slice_shapes[0].rank == 3 &&
                    compiled.payload.slice_shapes[0].sizes[0] == c.slice_shape[0] &&
                    compiled.payload.slice_shapes[0].sizes[1] == c.slice_shape[1] &&
                    compiled.payload.slice_shapes[0].sizes[2] == c.slice_shape[2],
                std::string(c.name) + ": compiled slice shape mismatch");
        require(!compiled.runtime_contract.logical_inputs.empty() &&
                    compiled.runtime_contract.logical_inputs[0].shape ==
                        std::vector<std::int64_t>(c.input_shape.begin(), c.input_shape.end()),
                std::string(c.name) + ": compiled logical input shape mismatch");
      }
    }));
