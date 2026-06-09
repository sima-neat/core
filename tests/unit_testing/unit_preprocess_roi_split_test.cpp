#include "pipeline/TensorCore.h"
#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL
#endif
#include "pipeline/internal/RenderedStageQueryTypes.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::stages::internal {
TensorList split_preproc_roi_output_for_stage(const simaai::neat::Tensor& batched,
                                              const PreprocessRuntimeMeta& meta, int roi_capacity,
                                              const PreprocOutputInfo& info);
} // namespace simaai::neat::stages::internal

RUN_TEST(
    "unit_preprocess_roi_split_test", ([] {
      constexpr int kCapacity = 3;
      constexpr int kHeight = 2;
      constexpr int kWidth = 3;
      constexpr int kDepth = 4;
      constexpr int kSlotBytes = kHeight * kWidth * kDepth;

      auto storage = simaai::neat::make_cpu_owned_storage(kCapacity * kSlotBytes);
      {
        auto mapping = storage->map(simaai::neat::MapMode::Write);
        require(mapping.data != nullptr, "ROI split: failed to map storage");
        auto* bytes = static_cast<std::uint8_t*>(mapping.data);
        for (int i = 0; i < kCapacity * kSlotBytes; ++i) {
          bytes[i] = static_cast<std::uint8_t>(i & 0xff);
        }
      }
      storage->sima_segments = {{"preproc_out", static_cast<std::size_t>(kCapacity * kSlotBytes)}};

      simaai::neat::Tensor batched;
      batched.storage = storage;
      batched.dtype = simaai::neat::TensorDType::UInt8;
      batched.layout = simaai::neat::TensorLayout::HWC;
      batched.shape = {kHeight, kWidth, kDepth};
      batched.strides_bytes = {kWidth * kDepth, kDepth, 1};
      batched.device = {simaai::neat::DeviceType::CPU, 0};
      batched.read_only = true;
      batched.route.segment_name = "preproc_out";

      simaai::neat::PreprocessRuntimeMeta meta;
      meta.original_width = 1280;
      meta.original_height = 720;
      meta.resized_width = 640;
      meta.resized_height = 640;
      meta.scaled_width = 640;
      meta.scaled_height = 640;
      meta.resize_mode = "stretch";
      meta.roi_list_enabled = true;
      meta.rois = {
          {0, 10, 20, 30, 40},
          {0, 50, 60, 70, 80},
          {0, 90, 100, 110, 120},
      };
      meta.roi_capacity = kCapacity;
      meta.roi_valid_count = kCapacity;
      meta.roi_input_count = kCapacity;
      meta.roi_source_width = 1280;
      meta.roi_source_height = 720;
      meta.roi_source_stride_bytes = 1280 * 3;
      meta.roi_affines = {
          {1.0, 0.0, 10.0, 0.0, 1.0, 20.0},
          {2.0, 0.0, 50.0, 0.0, 2.0, 60.0},
          {3.0, 0.0, 90.0, 0.0, 3.0, 100.0},
      };

      simaai::neat::stages::PreprocOutputInfo info;
      info.transport_kind = simaai::neat::stages::PreprocOutputTransportKind::Dense;
      info.logical_dims.width = kWidth;
      info.logical_dims.height = kHeight;
      info.logical_dims.depth = kDepth;

      simaai::neat::TensorList split =
          simaai::neat::stages::internal::split_preproc_roi_output_for_stage(batched, meta,
                                                                             kCapacity, info);
      require(split.size() == static_cast<std::size_t>(kCapacity),
              "ROI split: output count mismatch");
      for (int i = 0; i < kCapacity; ++i) {
        const auto& tensor = split[static_cast<std::size_t>(i)];
        require(tensor.storage == storage, "ROI split: storage should be shared");
        require(tensor.byte_offset == i * kSlotBytes, "ROI split: byte offset mismatch");
        require(tensor.shape == batched.shape, "ROI split: shape mismatch");
        require(tensor.semantic.preprocess.has_value(),
                "ROI split: missing per-slot preprocess metadata");
        const auto& pre = *tensor.semantic.preprocess;
        require(pre.rois.size() == 1U, "ROI split: per-slot ROI count mismatch");
        require(pre.rois.front().x == meta.rois[static_cast<std::size_t>(i)].x,
                "ROI split: per-slot ROI x mismatch");
        require(pre.affine_m02 == meta.roi_affines[static_cast<std::size_t>(i)].m02,
                "ROI split: per-slot affine mismatch");
      }

      {
        simaai::neat::PreprocessRuntimeMeta partial = meta;
        partial.rois = {meta.rois[0], meta.rois[1]};
        partial.roi_affines = {meta.roi_affines[0], meta.roi_affines[1]};
        partial.roi_capacity = kCapacity;
        partial.roi_valid_count = 2;
        partial.roi_input_count = kCapacity;
        partial.roi_dropped_overflow = 1;

        simaai::neat::TensorList partial_split =
            simaai::neat::stages::internal::split_preproc_roi_output_for_stage(batched, partial,
                                                                               kCapacity, info);
        require(partial_split.size() == 2U, "ROI split: partial valid_count should limit outputs");
        require(partial_split[0].byte_offset == 0 && partial_split[1].byte_offset == kSlotBytes,
                "ROI split: partial output byte offsets mismatch");
        require(partial_split[1].semantic.preprocess.has_value() &&
                    partial_split[1].semantic.preprocess->roi_dropped_overflow == 0,
                "ROI split: scalar per-slot metadata should clear aggregate drop counters");
      }

      {
        simaai::neat::PreprocessRuntimeMeta inconsistent = meta;
        inconsistent.rois = {meta.rois[0], meta.rois[1]};
        inconsistent.roi_affines = {meta.roi_affines[0], meta.roi_affines[1]};
        inconsistent.roi_capacity = kCapacity;
        inconsistent.roi_valid_count = kCapacity;

        bool threw = false;
        try {
          (void)simaai::neat::stages::internal::split_preproc_roi_output_for_stage(
              batched, inconsistent, kCapacity, info);
        } catch (const std::exception& e) {
          threw = true;
          require(std::string(e.what()).find("fewer ROIs") != std::string::npos,
                  std::string("ROI split: unexpected inconsistent metadata exception: ") +
                      e.what());
        }
        require(threw, "ROI split: inconsistent metadata should throw");
      }
    }));
