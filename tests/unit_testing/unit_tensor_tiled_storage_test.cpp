#include "pipeline/TensorCore.h"
#include "pipeline/internal/TensorTessellationMetadata.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/sima/KernelTensorLayout.h"
#include "test_main.h"

#include <array>
#include <cstring>
#include <numeric>

using namespace simaai::neat;
namespace pi = simaai::neat::pipeline_internal;

namespace {
template <class F> void rejects(F action, const char* why) {
  bool rejected = false;
  try {
    action();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, why);
}
} // namespace

RUN_TEST("unit_tensor_tiled_storage_test", ([] {
           gst_init(nullptr, nullptr);
           namespace rt = ::simaai::scheduler::client;
           rt::MlaTiled cells{{0, 1, 2, 3}, {1, 1, 1, 16}, rt::MlaTiled::Channels::Compact};
           rt::TensorSpec spec{{1, 4, 6, 16}, rt::DType::Int8, cells};
           require(!pi::sima::kernel_tensor_requires_detessellation(spec),
                   "compact full-channel INT8 cell tiles lost dense compatibility");
           spec.dtype = rt::DType::BFloat16;
           require(pi::sima::kernel_tensor_requires_detessellation(spec),
                   "split-byte BF16 was mistaken for dense storage");
           spec.dtype = rt::DType::Int8;
           cells.tile_shape = {1, 2, 2, 16};
           spec.layout = cells;
           require(pi::sima::kernel_tensor_requires_detessellation(spec),
                   "spatially blocked INT8 was mistaken for dense storage");
           cells.tile_shape = {1, 4, 6, 16};
           spec.layout = cells;
           require(!pi::sima::kernel_tensor_requires_detessellation(spec),
                   "single whole-volume INT8 tile lost dense compatibility");
           cells.tile_shape = {2, 8, 12, 16};
           spec.layout = cells;
           require(!pi::sima::kernel_tensor_requires_detessellation(spec),
                   "boundary-clipped oversized whole tile lost dense compatibility");
           spec.shape = {1, 2, 3, 3};
           cells.tile_shape = {2, 4, 6, 3};
           spec.layout = cells;
           require(!pi::sima::kernel_tensor_requires_detessellation(spec),
                   "single-tile trailing alignment was mistaken for internal padding");
           spec.shape = {1, 2, 3, 15};
           cells.tile_shape = {1, 1, 1, 15};
           spec.layout = cells;
           require(pi::sima::kernel_tensor_requires_detessellation(spec),
                   "inter-cell alignment gaps were mistaken for dense INT8 storage");

           // Nonzero carrier offset and trailing bytes catch accidental whole-carrier
           // copying; the tile extent is larger than the logical value extent.
           std::array<uint8_t, 64> carrier{};
           std::iota(carrier.begin(), carrier.end(), uint8_t{1});
           Tensor tensor;
           tensor.dtype = TensorDType::BFloat16;
           tensor.shape = {1, 1, 1, 9};
           tensor.storage = make_cpu_external_storage(carrier.data(), carrier.size());
           tensor.byte_offset = 8;
           TessSpec tess;
           tess.format = "MLA_TILED_BF16_PAD16";
           tess.slice_shape = {1, 1, 1, 16};
           tess.axes = {0, 1, 2, 3};
           tess.storage_bytes = 32;
           tensor.semantic.tess = tess;
           const std::vector<uint8_t> expected(carrier.begin() + 8, carrier.begin() + 40);
           require(tensor.has_tiled_storage() && !tensor.is_contiguous(),
                   "explicit tiled storage was lost");
           rejects([&] { (void)tensor.contiguous(); },
                   "contiguous silently reinterpreted tiled storage");
           require(tensor.copy_payload_bytes() == expected,
                   "payload copy lost padding or copied its neighbor");
           std::array<uint8_t, 40> destination{};
           destination.fill(0xFE);
           require(!tensor.copy_payload_bytes_to(destination.data(), 31),
                   "short payload destination accepted");
           require(tensor.copy_payload_bytes_to(destination.data(), destination.size()) &&
                       destination[32] == 0xFE,
                   "payload copy wrote past the checked tile extent");
           require(!tensor.copy_dense_bytes_tight_to(destination.data(), destination.size()),
                   "dense copy accepted tiled values");
           rejects([&] { (void)tensor.copy_dense_bytes_tight(); },
                   "dense copy silently reinterpreted BF16");
           rejects([&] { (void)static_cast<const Tensor&>(tensor).data_ptr<uint16_t>(); },
                   "typed pointer silently reinterpreted BF16");
           require(!pi::map_dense_tensor_read(tensor).data,
                   "generic dense reader accepted tiled storage");
           const auto cloned = tensor.clone();
           require(cloned.storage != tensor.storage && cloned.byte_offset == 0 &&
                       cloned.copy_payload_bytes() == expected && cloned.has_tiled_storage() &&
                       cloned.semantic.tess->axes == tess.axes &&
                       cloned.semantic.tess->slice_shape == tess.slice_shape,
                   "clone failed to preserve owned physical storage and metadata");
           rejects([&] { (void)cloned.copy_dense_bytes_tight(); }, "clone erased physical layout");
           Tensor contradictory = tensor;
           contradictory.planes.emplace_back();
           std::string validation;
           require(!contradictory.validate(&validation),
                   "tiled/composite storage contradiction accepted");
           Tensor short_source = tensor;
           short_source.storage = make_cpu_external_storage(carrier.data(), 39);
           require(!short_source.copy_payload_bytes_to(destination.data(), destination.size()),
                   "short source carrier was accepted");
           rejects([&] { (void)short_source.clone(); }, "clone accepted a truncated tiled payload");

           auto caps = std::shared_ptr<GstCaps>(
               gst_caps_new_empty_simple("application/vnd.simaai.tensor"), gst_caps_unref);
           pi::set_tessellation_caps(caps.get(), 7, tess);
           TessSpec second_tess = tess;
           second_tess.format = "MLA_TILED_BF16_COMPACT";
           second_tess.slice_shape = {1, 1, 1, 8};
           second_tess.storage_bytes = 16;
           pi::set_tessellation_caps(caps.get(), 3, second_tess);
           gchar* serialized = gst_caps_to_string(caps.get());
           auto restored =
               std::shared_ptr<GstCaps>(gst_caps_from_string(serialized), gst_caps_unref);
           g_free(serialized);
           Tensor decoded = tensor;
           decoded.semantic.tess.reset();
           pi::apply_tessellation_caps(decoded, restored.get(), 0, 32);
           require(!decoded.has_tiled_storage(), "layout attached to the wrong logical output");
           const std::vector<int64_t> raw_shape{32};
           require(
               pi::same_numerical_tensor_view(tensor.shape, int(TensorDType::BFloat16),
                                              tensor.shape, int(TensorDType::BFloat16)) &&
                   !pi::same_numerical_tensor_view(tensor.shape, int(TensorDType::BFloat16),
                                                   raw_shape, int(TensorDType::UInt8)) &&
                   !pi::same_numerical_tensor_view(tensor.shape, int(TensorDType::BFloat16),
                                                   raw_shape, int(TensorDType::BFloat16)) &&
                   !pi::same_numerical_tensor_view(tensor.shape, int(TensorDType::UInt8), raw_shape,
                                                   int(TensorDType::UInt8)) &&
                   !pi::same_numerical_tensor_view(tensor.shape, int(TensorDType::BFloat16),
                                                   tensor.shape, int(TensorDType::UInt8)),
               "numerical view matching confused a raw byte publication with tiled BF16 values");
           Tensor raw_alias = tensor;
           raw_alias.dtype = TensorDType::UInt8;
           raw_alias.shape = {32};
           raw_alias.semantic.tess = TessSpec{{}, "EVXX_UINT8"};
           pi::apply_tessellation_caps(raw_alias, restored.get(), 8, 32);
           require(!raw_alias.has_tiled_storage() && raw_alias.copy_dense_bytes_tight() == expected,
                   "opaque byte alias was incorrectly tagged as a numerical tiled tensor");
           pi::apply_tessellation_caps(decoded, restored.get(), 7, 32);
           require(decoded.copy_payload_bytes() == expected &&
                       decoded.semantic.tess->format == tess.format &&
                       decoded.semantic.tess->axes == tess.axes &&
                       decoded.semantic.tess->slice_shape == tess.slice_shape,
                   "caps roundtrip lost exact checked layout");
           Tensor second_view = tensor;
           second_view.shape = {1, 1, 1, 8};
           second_view.byte_offset = 40;
           second_view.semantic.tess.reset();
           pi::apply_tessellation_caps(second_view, restored.get(), 3, 16);
           require(second_view.semantic.tess->storage_bytes == 16 &&
                       second_view.semantic.tess->slice_shape == second_tess.slice_shape &&
                       second_view.semantic.tess->format == second_tess.format &&
                       second_view.copy_payload_bytes() ==
                           std::vector<uint8_t>(carrier.begin() + 40, carrier.begin() + 56) &&
                       decoded.semantic.tess->storage_bytes == 32,
                   "distinct logical views shared or overwrote physical layout metadata");
           rejects([&] { pi::apply_tessellation_caps(decoded, restored.get(), 7, 31); },
                   "metadata exceeded the selected descriptor extent");
           Tensor wrong_dtype = tensor;
           wrong_dtype.dtype = TensorDType::Int8;
           rejects([&] { pi::apply_tessellation_caps(wrong_dtype, restored.get(), 7, 32); },
                   "metadata changed the numerical dtype");
           for (int defect = 0; defect < 3; ++defect) {
             auto invalid = tess;
             if (defect == 0)
               invalid.axes[0] = 3;
             if (defect == 1)
               invalid.slice_shape[0] = 0;
             if (defect == 2)
               invalid.format = "UNKNOWN";
             pi::set_tessellation_caps(caps.get(), 7, invalid);
             rejects([&] { pi::apply_tessellation_caps(decoded, caps.get(), 7, 32); },
                     "malformed physical layout metadata accepted");
           }

           // The public copy_output overlay must preserve tile padding as well as dtype.
           GstBuffer* buffer = gst_buffer_new_allocate(nullptr, carrier.size(), nullptr);
           gst_buffer_fill(buffer, 0, carrier.data(), carrier.size());
           GstSample* sample = gst_sample_new(buffer, nullptr, nullptr, nullptr);
           gst_buffer_unref(buffer);
           Tensor gst_tensor = tensor;
           gst_tensor.storage = pi::make_gst_sample_storage(sample);
           gst_sample_unref(sample);
           OutputTensorOverrideEntry entry;
           entry.shape = tensor.shape;
           entry.dtype = tensor.dtype;
           entry.format = "BF16"; // Coarse logical format must not erase exact layout.
           const auto owned = finish_output_tensor_override_entry(gst_tensor, entry, true, false);
           require(owned.storage->kind == StorageKind::CpuOwned && owned.has_tiled_storage() &&
                       owned.semantic.tess->format == tess.format && owned.strides_bytes.empty() &&
                       owned.copy_payload_bytes() == expected,
                   "Owned override truncated or reinterpreted physical tile storage");
           const auto borrowed =
               finish_output_tensor_override_entry(gst_tensor, entry, false, false);
           require(borrowed.storage == gst_tensor.storage &&
                       borrowed.copy_payload_bytes() == expected && borrowed.strides_bytes.empty(),
                   "borrowed overlay erased physical layout");

#ifdef SIMA_WITH_OPENCV
           Tensor image = tensor;
           image.dtype = TensorDType::UInt8;
           image.shape = {2, 9};
           image.semantic.image = ImageSpec{};
           image.semantic.image->format = ImageSpec::PixelFormat::GRAY8;
           require(!image.map_cv_mat_view(ImageSpec::PixelFormat::GRAY8),
                   "OpenCV view silently reinterpreted tiled storage");
#endif

           // Existing format-only annotations are not proof of non-dense storage.
           Tensor legacy = tensor;
           legacy.semantic.tess = TessSpec{{}, "BF16"};
           require(!legacy.has_tiled_storage() && legacy.copy_dense_bytes_tight().size() == 18,
                   "legacy format-only tensor behavior changed");
           require(legacy.clone().copy_dense_bytes_tight() == legacy.copy_dense_bytes_tight(),
                   "legacy clone behavior changed");
         }));
