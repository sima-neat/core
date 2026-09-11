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
Tensor select_preproc_output_for_stage(const Sample& sample, const PreprocOutputInfo& info);
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
      info.roi_slot_bytes = kSlotBytes;
      info.roi_member_shape = batched.shape;
      info.roi_member_strides_bytes = batched.strides_bytes;

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

      {
        auto chw_info = info;
        chw_info.logical_layout = simaai::neat::TensorLayout::CHW;
        chw_info.roi_member_shape = {1, kDepth, kHeight, kWidth};
        chw_info.roi_member_strides_bytes = {kSlotBytes, kHeight * kWidth, kWidth, 1};
        const auto chw = simaai::neat::stages::internal::split_preproc_roi_output_for_stage(
            batched, meta, kCapacity, chw_info);
        require(chw[0].shape == std::vector<std::int64_t>({kDepth, kHeight, kWidth}) &&
                    chw[0].strides_bytes ==
                        std::vector<std::int64_t>({kHeight * kWidth, kWidth, 1}) &&
                    chw[0].axis_semantics.front() == simaai::neat::TensorAxisSemantic::C,
                "ROI dense image projection must preserve authored CHW member semantics");
      }

      for (const int capacity : {2, 3}) {
        // Same packed slot extents as the INT8 ROI fixture, but an aligned arena
        // with tail bytes and a nonzero selected-view offset. Arena/R is wrong.
        constexpr std::size_t slot = 9216U;
        constexpr std::size_t base = 128U;
        const std::size_t arena = (base + slot * capacity + 4095U) & ~std::size_t{4095U};
        auto owner = simaai::neat::make_cpu_owned_storage(arena);
        std::weak_ptr<void> lifetime = owner->holder;
        {
          auto map = owner->map(simaai::neat::MapMode::Write);
          auto* bytes = static_cast<std::uint8_t*>(map.data);
          for (std::size_t i = 0; i < arena; ++i) {
            bytes[i] = static_cast<std::uint8_t>(i % 251U);
          }
        }
        simaai::neat::Tensor packed;
        packed.storage = owner;
        packed.dtype = simaai::neat::TensorDType::Int8;
        packed.layout = simaai::neat::TensorLayout::HWC;
        packed.shape = {capacity, 48, 64, 3};
        packed.strides_bytes = {slot, 192, 3, 1};
        packed.byte_offset = base;
        packed.route.physical_byte_offset = base;
        packed.route.memory_index = 0;
        packed.route.logical_index = 1;
        packed.route.route_slot = 1;
        packed.route.name = "selected";
        packed.route.backend_name = "backend_selected";
        packed.route.segment_name = "actual_arena";
        simaai::neat::stages::PreprocOutputInfo packed_info;
        packed_info.transport_kind = simaai::neat::stages::PreprocOutputTransportKind::Packed;
        packed_info.roi_slot_bytes = slot;
        packed_info.roi_member_shape = {1, 48, 64, 3};
        packed_info.roi_member_strides_bytes = {slot, 192, 3, 1};
        packed_info.primary_output_name = "selected";
        packed_info.primary_route_slot = 1;

        // Two logical tensors on memory 0 must not be reselected by memory index.
        simaai::neat::Sample sample;
        auto other = packed;
        other.route.name = "other";
        other.route.backend_name = "backend_other";
        other.route.logical_index = 0;
        other.route.route_slot = 0;
        other.byte_offset = 0;
        sample.tensors = {other, packed};
        auto selected =
            simaai::neat::stages::internal::select_preproc_output_for_stage(sample, packed_info);
        require(selected.byte_offset == base && selected.route.name == "selected" &&
                    selected.route.segment_name == "actual_arena" && selected.storage == owner,
                "ROI selection must retain the logical view and owner, not memory0");
        auto packed_meta = meta;
        packed_meta.roi_capacity = capacity;
        packed_meta.roi_valid_count = capacity;
        auto views = simaai::neat::stages::internal::split_preproc_roi_output_for_stage(
            selected, packed_meta, capacity, packed_info);
        for (int i = 0; i < capacity; ++i) {
          const auto& view = views[static_cast<std::size_t>(i)];
          require(view.shape == packed_info.roi_member_shape &&
                      view.strides_bytes == packed_info.roi_member_strides_bytes &&
                      view.axis_semantics == std::vector<simaai::neat::TensorAxisSemantic>(
                                                 {simaai::neat::TensorAxisSemantic::N,
                                                  simaai::neat::TensorAxisSemantic::H,
                                                  simaai::neat::TensorAxisSemantic::W,
                                                  simaai::neat::TensorAxisSemantic::C}),
                  "ROI packed view must have coherent one-member semantics");
          require(view.byte_offset == static_cast<std::int64_t>(base + i * slot) &&
                      view.route.physical_byte_offset == view.byte_offset,
                  "ROI packed slot must exclude arena tail bytes");
          const auto payload = view.copy_payload_bytes();
          require(payload.size() == slot, "ROI packed member byte count mismatch");
          for (std::size_t b = 0; b < slot; ++b) {
            require(payload[b] == (base + i * slot + b) % 251U,
                    "ROI packed member payload mismatch");
          }
        }
        auto survivor = views.back();
        views.clear();
        sample.tensors.clear();
        selected = {};
        packed = {};
        other = {};
        owner.reset();
        require(!lifetime.expired(), "ROI surviving sibling must retain allocation");
        auto retained_map = survivor.map_read();
        survivor = {};
        require(!lifetime.expired() && retained_map.data != nullptr &&
                    static_cast<const std::uint8_t*>(retained_map.data)[0] ==
                        (base + (capacity - 1) * slot) % 251U,
                "ROI mapping must remain valid after all tensor views are released");
        retained_map = {};
        require(lifetime.expired(), "ROI final Mapping release must release allocation");
      }
    }));
