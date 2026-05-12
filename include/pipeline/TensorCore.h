/**
 * @file
 * @ingroup tensors
 * @brief The `Tensor` type — a labeled box of numbers, plus all its supporting types.
 *
 * This is the framework's universal data carrier. A `Tensor` knows what numbers it holds
 * (`dtype`, `shape`, `strides_bytes`), where they live (`device`, `storage` of one of
 * four `StorageKind`s), and what they *represent* (`Semantic` — image, audio, tessellated
 * tile, encoded video, quantization payload, or preprocessing residue). The Tensor's storage
 * is typically zero-copy across processors via the unified IOMMU; this file defines the
 * `TensorBuffer` abstraction that hides the per-storage-kind details.
 *
 * Headline types in this file:
 *   - `Tensor` — the user-facing struct.
 *   - `TensorBuffer` (alias `Storage`) — the storage handle with `map_fn` for read/write.
 *   - `Mapping` — a scoped read/write window into a buffer (RAII-unmap on destruction).
 *   - `Plane` — per-plane info for composite formats (NV12 = Y + UV, I420 = Y + U + V).
 *   - `Semantic` — discriminated union of "what this tensor means" (image, audio, tess, etc.).
 *   - `TensorRouteMeta` — routing metadata used by multi-output models.
 *   - `Device` / `DeviceType` — where the tensor lives (CPU/CVU/MLA/APU).
 *
 * @see "Tensors: hiding the memory mess" (§0.10 of the design deep dive)
 * @see TensorTypes.h for dtype/layout/axis-semantic enums
 * @see TensorSpec.h for `TensorConstraint` (the matching contract)
 */
#pragma once

#include "pipeline/TensorTypes.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace simaai::neat {

/**
 * @brief Where a tensor's data lives — which processor can read it directly.
 *
 * The Modalix's heterogeneous compute is reflected here. CPU is the host A65; SIMA_CVU is the
 * EV74 vector processor's accessible memory; SIMA_MLA is the ML accelerator's scratchpad;
 * SIMA_APU is the audio processing unit. Tensors with the same buffer can sometimes be
 * accessed from multiple devices (zero-copy via unified IOMMU); other times an explicit
 * transfer is required.
 * @ingroup tensors
 */
enum class DeviceType {
  CPU = 0,  ///< Host A65 (general-purpose CPU).
  SIMA_APU, ///< Audio Processing Unit.
  SIMA_CVU, ///< Compute/Vector Unit (EV74).
  SIMA_MLA, ///< Machine Learning Accelerator.
  UNKNOWN,  ///< Placement not specified.
};

/// Device descriptor: type + numeric ID (for multi-device boards).
struct Device {
  DeviceType type = DeviceType::CPU; ///< Which processor this device refers to.
  int id = 0; ///< Numeric ID disambiguating multiple devices of the same type.
};

/**
 * @brief How a `TensorBuffer`'s memory was acquired and how to access it.
 *
 * Determines the framework's strategy for `map()` and lifetime management. `CpuOwned` is the
 * default — framework allocated, framework frees. `CpuExternal` wraps a foreign pointer
 * (cv::Mat data, NumPy array, mmap'd file). `GstSample` carries a reference to a GStreamer
 * buffer pool sample. `DeviceHandle` lives in accelerator memory (e.g., MLA scratchpad) and
 * requires an explicit transfer for CPU access.
 * @ingroup tensors
 */
enum class StorageKind {
  CpuOwned = 0, ///< Framework allocated; freed when the storage refcount hits zero.
  CpuExternal,  ///< Wraps a foreign pointer; lifetime tracked via `holder`.
  GstSample,    ///< References a GStreamer pool sample; refcounted via GStreamer.
  DeviceHandle, ///< Lives in accelerator memory; CPU access requires an explicit transfer.
  Unknown,
};

/**
 * @brief Placement for Tensor creators that materialize owned storage.
 *
 * The default for new owning creators is EV74: DDR allocated from the SiMa
 * allocator in the EV74-addressable window (not EV74 VCCM).  Use CPU for
 * ordinary malloc-backed host tensors, or MLA for DMS0-backed tensors when the
 * runtime allocator is available.
 */
enum class TensorMemory {
  EV74 = 0,
  CPU,
  MLA,
  A65,
  Auto,
};

/**
 * @brief Role of a plane within a composite (multi-plane) tensor.
 *
 * Composite tensors carry their planes as separate `Plane` records. NV12 has Y + UV; I420
 * has Y + U + V; single-plane formats use the default `Unknown`.
 * @ingroup tensors
 */
enum class PlaneRole {
  Unknown = 0,
  Y,  ///< Luminance plane.
  U,  ///< Chrominance-U plane (I420).
  V,  ///< Chrominance-V plane (I420).
  UV, ///< Interleaved chrominance plane (NV12).
};

/// Access mode for `TensorBuffer::map()`. Affects whether the framework treats the mapping as read
/// or write.
enum class MapMode {
  Read = 0,  ///< Read-only map; framework may skip cache invalidation if known clean.
  Write,     ///< Write-only map; framework may skip read-back; flushes cache on unmap.
  ReadWrite, ///< Both directions; framework treats as full ownership transfer for the mapping's
             ///< lifetime.
};

/**
 * @brief Image-tensor metadata: pixel format and (optional) color space.
 *
 * Set on the `Tensor::semantic.image` optional when the tensor carries pixel data. The
 * pixel format is the structural shape (channels, plane count); color space is informational.
 * @ingroup tensors
 */
struct ImageSpec {
  /// Pixel format enumeration covering the framework's supported image layouts.
  enum class PixelFormat {
    RGB = 0, ///< Packed RGB, 3 channels, 8 bits per channel.
    BGR,     ///< Packed BGR (OpenCV's native default), 3 channels, 8 bpp.
    GRAY8,   ///< Single-channel 8-bit grayscale.
    NV12,    ///< 4:2:0 chroma with interleaved UV plane (camera-native on most chips).
    I420,    ///< 4:2:0 chroma with separate U and V planes.
    UNKNOWN,
  };

  PixelFormat format = PixelFormat::UNKNOWN; ///< Pixel format of the image data.
  std::string color_space; ///< Color space hint (e.g., `"bt709"`, `"srgb"`); empty = unspecified.
};

/// Audio-tensor metadata: sample rate, channel count, interleaving.
struct AudioSpec {
  int sample_rate = 0; ///< Hz.
  int channels = 0;    ///< Channel count (1 = mono, 2 = stereo).
  bool interleaved =
      true; ///< If `true`, channels are interleaved (LRLRLR…); if `false`, planar (LL…RR…).
};

/// Token-tensor metadata for NLP-style tensors.
struct TokensSpec {
  int vocab_size = 0; ///< Vocabulary size (used for one-hot widths and bounds checks).
};

/**
 * @brief Encoded-stream tensor metadata: which codec the bytes represent.
 *
 * Set on `Tensor::semantic.encoded` when the tensor carries an encoded media bitstream
 * (e.g., output of an H.264 encoder, input to a decoder).
 * @ingroup tensors
 */
struct EncodedSpec {
  /// Codec identifier for an encoded media bitstream.
  enum class Codec {
    H264 = 0, ///< Raw H.264 Annex-B bitstream.
    H265,     ///< Raw H.265 / HEVC bitstream.
    RTP_H264, ///< RTP-packetized H.264.
    RTP_H265, ///< RTP-packetized H.265.
    JPEG,     ///< JPEG-encoded image.
    UNKNOWN,
  };

  Codec codec = Codec::UNKNOWN; ///< Codec identifier for the encoded bitstream.
};

/**
 * @brief Quantization metadata for INT8/INT16 tensors.
 *
 * For per-tensor quantization, set `scale` and `zero_point` directly. For per-channel
 * (typical for quantized weights), populate `scales` and `zero_points` and set `axis` to
 * the channel dimension index.
 * @ingroup tensors
 */
struct QuantSpec {
  float scale = 1.0f;        ///< Per-tensor scale (`x_real = (x_int - zero_point) * scale`).
  int32_t zero_point = 0;    ///< Per-tensor zero point.
  int axis = -1;             ///< Channel axis for per-channel quantization (-1 = per-tensor).
  std::vector<float> scales; ///< Per-channel scales (used when `axis >= 0`).
  std::vector<int32_t> zero_points; ///< Per-channel zero points (used when `axis >= 0`).
};

/**
 * @brief Tessellation metadata — tile geometry for the MLA's tile-block layout.
 *
 * Tensors flowing into the MLA are tile-blocked into VCCM-sized chunks. `slice_shape` is the
 * tile geometry (e.g., `[8, 8, 32]` for an 8×8×32 tile). The `format` token identifies the
 * specific tessellation variant.
 * @see "Tessellation/quant/cast" (§17 of the design deep dive)
 * @ingroup tensors
 */
struct TessSpec {
  std::vector<int64_t> slice_shape; ///< Per-axis tile dimensions.
  std::string format; ///< Tessellation format token (specific to the MLA hardware revision).

  /// Replace the slice shape (move-friendly).
  void set_slice_shape(std::vector<int64_t> shape) {
    slice_shape = std::move(shape);
  }
};

/**
 * @brief Per-buffer preprocessing context — the inverse-transform breadcrumb trail.
 *
 * When the framework's preprocess stage resizes/letterboxes/normalizes an input image, it
 * records what it did into this struct and attaches it as `Tensor::semantic.preprocess`. The
 * downstream BoxDecode (and any user code) uses these fields to map model-space coordinates
 * (e.g., detected boxes in 640×640) back to original image coordinates.
 *
 * The `affine_*` fields encode the full 2×3 affine transform from original image coordinates
 * to model input coordinates; invert this matrix to map detections back to the source frame.
 * @ingroup tensors
 */
struct PreprocessRuntimeMeta {
  int original_width = 0;  ///< Width of the source frame in pixels before preprocess.
  int original_height = 0; ///< Height of the source frame in pixels before preprocess.
  int resized_width = 0;   ///< Width after the resize step (before any padding).
  int resized_height = 0;  ///< Height after the resize step (before any padding).
  int scaled_width = 0;    ///< Width after scaling (resize × additional scale factor).
  int scaled_height = 0;   ///< Height after scaling (resize × additional scale factor).
  int pad_left = 0;        ///< Letterbox padding added on the left edge, in pixels.
  int pad_right = 0;       ///< Letterbox padding added on the right edge, in pixels.
  int pad_top = 0;         ///< Letterbox padding added on the top edge, in pixels.
  int pad_bottom = 0;      ///< Letterbox padding added on the bottom edge, in pixels.

  std::string resize_mode; ///< Resize policy token (e.g., `"letterbox"`, `"stretch"`).
  std::string color_in;    ///< Input color format token (e.g., `"bgr"`, `"nv12"`).
  std::string color_out;   ///< Output color format token after color-convert.
  /// Axis permutation applied by preprocess layout_convert, if any.
  std::vector<int> axis_perm;

  bool normalize = false;  ///< True if a normalize step (mean/scale) was applied.
  bool quantize = false;   ///< True if an INT8/INT16 quantize step was applied.
  bool tessellate = false; ///< True if a tessellate step (tile-block layout) was applied.

  double affine_m00 = 1.0;      ///< 2×3 affine matrix element (row 0, col 0): x scale.
  double affine_m01 = 0.0;      ///< 2×3 affine matrix element (row 0, col 1): x shear.
  double affine_m02 = 0.0;      ///< 2×3 affine matrix element (row 0, col 2): x translation.
  double affine_m10 = 0.0;      ///< 2×3 affine matrix element (row 1, col 0): y shear.
  double affine_m11 = 1.0;      ///< 2×3 affine matrix element (row 1, col 1): y scale.
  double affine_m12 = 0.0;      ///< 2×3 affine matrix element (row 1, col 2): y translation.
  double affine_scale_x = 1.0;  ///< Scalar X scale factor from original to model coordinates.
  double affine_scale_y = 1.0;  ///< Scalar Y scale factor from original to model coordinates.
  double affine_offset_x = 0.0; ///< Scalar X offset (typically pad_left) in model coordinates.
  double affine_offset_y = 0.0; ///< Scalar Y offset (typically pad_top) in model coordinates.

  /// True iff `axis_perm` is non-empty (a layout permutation was recorded).
  bool has_axis_perm() const noexcept {
    return !axis_perm.empty();
  }
};

/**
 * @brief Discriminated union of "what this tensor represents".
 *
 * A single tensor can be image data, audio, tokens, a tessellated tile, an encoded video
 * frame, a quantization payload, or a preprocessing residue — and sometimes more than one
 * at once (e.g., quantized image data). Each spec is an `std::optional`; only the relevant
 * ones are set. Consumers query the optional that matters to them.
 *
 * @code
 *   if (tensor.semantic.image.has_value()) {
 *     auto fmt = tensor.semantic.image->format;
 *     // ...
 *   }
 * @endcode
 *
 * @ingroup tensors
 */
struct Semantic {
  std::optional<ImageSpec> image;     ///< Set for image tensors.
  std::optional<AudioSpec> audio;     ///< Set for audio tensors.
  std::optional<TokensSpec> tokens;   ///< Set for token-stream tensors (NLP).
  std::optional<TessSpec> tess;       ///< Set for tessellated tile-layout tensors.
  std::optional<EncodedSpec> encoded; ///< Set for encoded-stream tensors (H.264, etc.).
  std::optional<QuantSpec> quant;     ///< Set for quantized integer tensors.
  std::optional<PreprocessRuntimeMeta>
      preprocess; ///< Set when the tensor was produced by a preprocess stage.
};

/**
 * @brief Scoped read/write window into a `TensorBuffer`.
 *
 * RAII: the destructor calls `unmap` (handling cache flush/sync). Move-only — a Mapping
 * uniquely owns its mapping. Created by `TensorBuffer::map(MapMode)`. Hold the Mapping while
 * you read/write `data`; let it go out of scope to release.
 *
 * @code
 *   auto m = tensor.storage->map(MapMode::Read);
 *   const uint8_t* bytes = static_cast<const uint8_t*>(m.data);
 *   // ... read m.size_bytes from bytes ...
 * @endcode
 *
 * @ingroup tensors
 */
struct Mapping {
  void* data =
      nullptr; ///< Pointer to mapped memory (CPU-readable for the duration of the Mapping).
  std::size_t size_bytes = 0; ///< Size of the mapped region in bytes.
  std::function<void()>
      unmap; ///< Cleanup callback invoked on destructor (cache flush, refcount decrement).
  std::shared_ptr<void>
      keepalive; ///< Optional lifetime guard; ensures the underlying storage outlives the Mapping.

  /// Construct an empty (unmapped) Mapping; assignable from a real mapping later.
  Mapping() = default;
  /// Mappings are non-copyable: each Mapping uniquely owns its scope.
  Mapping(const Mapping&) = delete;
  /// Mappings are non-copy-assignable: each Mapping uniquely owns its scope.
  Mapping& operator=(const Mapping&) = delete;
  /// Move-construct from another Mapping; the source becomes empty.
  Mapping(Mapping&& other) noexcept {
    *this = std::move(other);
  }
  /// Move-assign from another Mapping; runs `unmap` on the prior contents first.
  Mapping& operator=(Mapping&& other) noexcept {
    if (this != &other) {
      if (unmap)
        unmap();
      data = other.data;
      size_bytes = other.size_bytes;
      unmap = std::move(other.unmap);
      keepalive = std::move(other.keepalive);
      other.data = nullptr;
      other.size_bytes = 0;
      other.unmap = nullptr;
      other.keepalive.reset();
    }
    return *this;
  }
  /// Destructor; runs `unmap` if set, releasing the mapping.
  ~Mapping() {
    if (unmap)
      unmap();
  }
};

#if defined(SIMA_WITH_OPENCV)
/// Bundles a `Mapping` with a non-owning `cv::Mat` view over the same memory.
/// Lifetime of the cv::Mat is tied to the Mapping's lifetime.
struct CvMatView {
  Mapping mapping; ///< The underlying buffer mapping.
  cv::Mat mat;     ///< OpenCV view of `mapping.data`. Don't outlive the Mapping.
};
#endif

/// One named memory segment within a multi-segment tensor buffer (e.g., separate Y / UV planes).
struct Segment {
  std::string name;           ///< Segment label (e.g., `"Y"`, `"UV"`, `"weights"`).
  std::size_t size_bytes = 0; ///< Size of this segment in bytes.
};

/**
 * @brief Storage handle for a tensor — opaque container for one of four backing memory kinds.
 *
 * `TensorBuffer` is the abstraction over the four `StorageKind`s. The `map_fn` callback
 * unifies the access path: regardless of where the bytes live (CPU heap, GStreamer pool,
 * accelerator scratch), `map(mode)` returns a `Mapping` you can read or write. The `holder`
 * shared_ptr keeps the underlying memory alive for the lifetime of the buffer (and any
 * `Mapping` derived from it).
 *
 * Multi-segment buffers (`sima_segments`) carry several named regions in one allocation —
 * used for composite formats like NV12 (Y + UV) and packed multi-tensor MLA outputs.
 *
 * @ingroup tensors
 */
struct TensorBuffer {
  StorageKind kind = StorageKind::Unknown; ///< How the storage was acquired (CPU-owned, external,
                                           ///< GstSample, device).
  Device device{};                         ///< Where the buffer is accessible.
  std::size_t size_bytes = 0;              ///< Total backing-memory size in bytes.
  std::shared_ptr<void>
      holder; ///< Lifetime guard (shared_ptr that owns/refcounts the underlying memory).
  void* data =
      nullptr; ///< Optional direct pointer (some storage kinds set this; others rely on `map_fn`).
  std::function<Mapping(MapMode)>
      map_fn; ///< Custom map function (set for non-trivial storage kinds).
  std::uint64_t sima_mem_target_flags =
      0; ///< SIMA memory target flags (advanced; for accelerator-aware allocators).
  std::uint64_t sima_mem_flags = 0;   ///< SIMA memory flags (cache, alignment, etc.).
  std::vector<Segment> sima_segments; ///< Named segments for multi-region buffers (composite
                                      ///< formats, packed outputs).

  /**
   * @brief Map the buffer for read/write access; returns a scoped `Mapping`.
   *
   * If the storage has a custom `map_fn`, calls it. Otherwise, returns a Mapping wrapping
   * the bare `data` pointer. Always sets `keepalive` to the buffer's `holder` so the
   * Mapping safely outlives buffer destruction.
   */
  Mapping map(MapMode mode) const {
    Mapping mapping;
    if (map_fn) {
      mapping = map_fn(mode);
    } else {
      mapping.data = data;
      mapping.size_bytes = size_bytes;
    }
    if (!mapping.keepalive) {
      mapping.keepalive = holder;
    }
    return mapping;
  }
};

/// Alias for backward-compatibility. New code prefers `TensorBuffer` directly.
using Storage = TensorBuffer;

/// Allocate a CPU-owned heap buffer of `size_bytes`, initialized to zero.
std::shared_ptr<TensorBuffer> make_cpu_owned_storage(std::size_t size_bytes);
/**
 * @brief Wrap a foreign CPU pointer as a `TensorBuffer` without copying.
 *
 * Lifetime of the underlying memory is tracked via the `holder` shared_ptr — pass a custom
 * deleter or a refcounted handle that keeps the original allocation alive. Set `read_only`
 * to indicate the framework should not write through this storage.
 */
std::shared_ptr<TensorBuffer> make_cpu_external_storage(void* data, std::size_t size_bytes,
                                                        std::shared_ptr<void> holder = {},
                                                        bool read_only = true);

/**
 * @brief One plane of a composite (multi-plane) tensor.
 *
 * Composite tensors (NV12, I420) carry a `planes` vector instead of a single contiguous
 * buffer. Each Plane describes a sub-region: shape, strides, byte offset within the parent
 * storage, and the role it plays (Y, U, V, or interleaved UV).
 * @ingroup tensors
 */
struct Plane {
  PlaneRole role = PlaneRole::Unknown; ///< Role of this plane (Y / U / V / UV / Unknown).
  std::vector<int64_t> shape;          ///< Per-dimension sizes of this plane's pixel grid.
  std::vector<int64_t> strides_bytes;  ///< Per-dimension byte strides within this plane.
  int64_t byte_offset = 0; ///< Offset of this plane's first byte from the parent storage start.
};

/// Non-owning view into NV12 pixel data: Y plane + interleaved UV plane.
struct Nv12View {
  int width = 0;               ///< Image width in pixels.
  int height = 0;              ///< Image height in pixels.
  const uint8_t* y = nullptr;  ///< Pointer to the Y (luminance) plane's first byte.
  int64_t y_stride = 0;        ///< Row stride of the Y plane in bytes.
  const uint8_t* uv = nullptr; ///< Pointer to the interleaved UV plane's first byte.
  int64_t uv_stride = 0;       ///< Row stride of the UV plane in bytes.
};

/// Bundles an NV12 view with the `Mapping` that keeps its underlying buffer alive.
struct Nv12Mapped {
  Mapping mapping; ///< Underlying buffer mapping; keeps `view` pointers valid.
  Nv12View view;   ///< NV12 plane pointers and strides into `mapping.data`.
};

/// Non-owning view into I420 pixel data: separate Y, U, V planes.
struct I420View {
  int width = 0;              ///< Image width in pixels.
  int height = 0;             ///< Image height in pixels.
  const uint8_t* y = nullptr; ///< Pointer to the Y (luminance) plane's first byte.
  int64_t y_stride = 0;       ///< Row stride of the Y plane in bytes.
  const uint8_t* u = nullptr; ///< Pointer to the U (chrominance) plane's first byte.
  int64_t u_stride = 0;       ///< Row stride of the U plane in bytes.
  const uint8_t* v = nullptr; ///< Pointer to the V (chrominance) plane's first byte.
  int64_t v_stride = 0;       ///< Row stride of the V plane in bytes.
};

/// Bundles an I420 view with the `Mapping` that keeps its underlying buffer alive.
struct I420Mapped {
  Mapping mapping; ///< Underlying buffer mapping; keeps `view` pointers valid.
  I420View view;   ///< I420 plane pointers and strides into `mapping.data`.
};

/**
 * @brief Routing metadata that travels with a tensor through multi-output pipelines.
 *
 * Multi-head models produce multiple outputs that need to be matched back to logical
 * tensor names. This struct carries the routing identity: which stage produced it
 * (`stage_key`), what logical output index, where in physical memory, and the user-facing
 * name. Most application code doesn't read this — it's used by the framework's pull
 * machinery to assemble correct `Sample` payloads.
 * @ingroup tensors
 */
struct TensorRouteMeta {
  std::string stage_key;            ///< Producing stage's identifier.
  int logical_index = -1;           ///< Logical output index (user-facing).
  int backend_output_index = -1;    ///< Backend output index (per-stage; may differ from logical).
  int route_slot = -1;              ///< Internal route-graph slot ID.
  int physical_index = -1;          ///< Physical memory index for packed outputs.
  int memory_index = -1;            ///< Underlying memory pool index.
  int64_t physical_byte_offset = 0; ///< Byte offset within the physical allocation.
  std::string name;                 ///< Logical (user-facing) name of this output tensor.
  std::string backend_name;         ///< Backend (kernel-side) name of this output.
  std::string segment_name; ///< Memory segment name when packed outputs share an allocation.
};

/**
 * @brief Universal tensor type — a labeled box of numbers that flows between Nodes.
 *
 * A `Tensor` is the framework's fundamental data type. It carries:
 *   - **Storage** — a shared `TensorBuffer` describing where the bytes live and how to access them.
 *   - **Type** — `dtype` (the element type) and `layout` (legacy coarse token).
 *   - **Shape & strides** — `shape[]` (per-dimension sizes), `strides_bytes[]` (per-dimension byte
 * strides).
 *   - **Axis semantics** — `axis_semantics[]` tagging each axis (N/D/H/W/C); the long-term layout
 * vocabulary.
 *   - **Device** — where the tensor logically lives.
 *   - **Semantic** — what the numbers represent (image, audio, tessellated, encoded, quantized,
 * etc.).
 *   - **Planes** — for composite formats (NV12, I420), per-plane sub-region records.
 *   - **Route metadata** — for multi-output models, identifies which output this tensor is.
 *
 * Tensors are **shared_ptr-friendly** for storage but **value types** for the metadata. Cheap
 * to copy by value (the storage is reference-counted).
 *
 * @code
 *   sima::Tensor t = sima::Tensor::from_cv_mat(my_image, sima::ImageSpec::PixelFormat::BGR);
 *   auto m = t.storage->map(sima::MapMode::Read);
 *   // ... read m.size_bytes from m.data ...
 * @endcode
 *
 * @see TensorBuffer for the storage abstraction
 * @see Mapping for the access window
 * @see Semantic for the discriminated payload-meaning union
 * @see "Tensors: hiding the memory mess" (§0.10 of the design deep dive)
 * @ingroup tensors
 */
struct Tensor {
  std::shared_ptr<TensorBuffer>
      storage; ///< Shared storage handle (one tensor may have copies pointing at the same storage).
  simaai::neat::TensorDType dtype = simaai::neat::TensorDType::UInt8; ///< Element type.
  simaai::neat::TensorLayout layout =
      simaai::neat::TensorLayout::Unknown; ///< Coarse legacy layout token (transitional; prefer
                                           ///< axis_semantics).
  std::vector<int64_t> shape;              ///< Per-dimension sizes.
  std::vector<int64_t> strides_bytes; ///< Per-dimension strides in bytes (empty = compact row-major
                                      ///< derived from shape+dtype).
  int64_t byte_offset =
      0; ///< Offset from the start of `storage->data` to the first element of this tensor view.
  std::vector<TensorAxisSemantic> axis_semantics; ///< Per-axis role tags (N/D/H/W/C/Unknown).
  Device device{};                                ///< Where the tensor lives.
  Semantic semantic{}; ///< What the numbers represent (image, audio, tess, etc.).
  std::vector<Plane>
      planes; ///< Per-plane sub-region records for composite formats (empty for dense tensors).
  bool read_only = true;   ///< If true, framework refuses mutable `data_ptr<T>()` access.
  TensorRouteMeta route{}; ///< Routing metadata (output identity in multi-output models).

  /// True iff this tensor is single-plane (no composite plane records).
  bool is_dense() const {
    return planes.empty();
  }
  /// True iff this tensor carries composite plane records (NV12, I420, …).
  bool is_composite() const {
    return !planes.empty();
  }

  /// True iff per-axis semantic tags have been populated.
  bool has_axis_semantics() const noexcept {
    return !axis_semantics.empty();
  }

  /// True iff `axis_semantics` is empty or matches `shape.size()`.
  bool axis_semantics_match_shape() const noexcept {
    return axis_semantics.empty() || axis_semantics.size() == shape.size();
  }

  /// True iff strides describe a compact row-major layout with no internal padding.
  bool is_contiguous() const {
    if (shape.empty())
      return true;
    if (strides_bytes.empty())
      return true;
    std::size_t elem = dtype_bytes(dtype);
    if (elem == 0)
      return false;
    std::int64_t expected = static_cast<std::int64_t>(elem);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      if (strides_bytes[static_cast<size_t>(i)] != expected)
        return false;
      expected *= shape[static_cast<size_t>(i)];
    }
    return true;
  }

  /// Return a pointer to the plane with the given role, or `nullptr` if absent.
  const Plane* try_plane(PlaneRole role) const noexcept {
    for (const auto& plane : planes) {
      if (plane.role == role)
        return &plane;
    }
    return nullptr;
  }

  /// True iff a plane with the given role exists.
  bool has_plane(PlaneRole role) const noexcept {
    return try_plane(role) != nullptr;
  }

  /// Return the plane with the given role; throws if absent.
  const Plane& plane(PlaneRole role) const {
    const Plane* found = try_plane(role);
    if (!found)
      throw std::runtime_error("Tensor::plane: plane not found");
    return *found;
  }

  /// Map the underlying storage for access (delegates to `storage->map`).
  Mapping map(MapMode mode) const;

  /// Convenience: map for read-only access.
  Mapping map_read() const {
    return map(MapMode::Read);
  }
  /// Convenience: map for write-only access.
  Mapping map_write() const {
    return map(MapMode::Write);
  }
  /// Map a view of just this tensor's bytes (respecting `byte_offset`/strides).
  Mapping view(MapMode mode = MapMode::Read) const;
  /// Convenience: read-only `view`.
  Mapping view_read() const {
    return view(MapMode::Read);
  }

  /// Typed pointer into the (CPU-resident, dense, contiguous) tensor data; throws on mismatch.
  template <typename T> T* data_ptr() {
    if (read_only) {
      throw std::runtime_error("Tensor::data_ptr: tensor is read-only");
    }
    return const_cast<T*>(const_data_ptr<T>());
  }

  /// Read-only typed pointer into the (CPU-resident, dense, contiguous) tensor data.
  template <typename T> const T* data_ptr() const {
    return const_data_ptr<T>();
  }

  /// Return a contiguous-strided copy of this tensor (no-op if already contiguous).
  Tensor contiguous() const;
  /// Return a deep copy with independently-owned storage.
  Tensor clone() const;
  /// Return this tensor materialized on `target` device, copying if necessary.
  Tensor to(Device target) const;
  /// Return this tensor on the host CPU, copying if necessary.
  Tensor cpu() const;
  /// Return this tensor on the CVU (EV74), copying if necessary.
  Tensor cvu() const;
  /// Return this tensor on the MLA; pass `force=true` to always copy.
  Tensor mla(bool force = false) const;
  /// Return a CPU copy if not already on CPU; otherwise return `*this`.
  Tensor to_cpu_if_needed() const;
  /// Validate internal invariants; on failure writes a message to `*err` (if non-null) and returns
  /// false.
  bool validate(std::string* err) const;

  /// Map this tensor as an NV12 view (Y + UV); returns nullopt if not NV12.
  std::optional<Nv12Mapped> map_nv12_read() const;
  /// Bytes required to hold a tightly-packed NV12 copy of this tensor.
  std::size_t nv12_required_bytes() const;
  /// Copy NV12 pixel data into `dst` (`dst_size` ≥ `nv12_required_bytes()`); returns success.
  bool copy_nv12_contiguous_to(uint8_t* dst, std::size_t dst_size) const;
  /// Return a tightly-packed NV12 byte copy of this tensor's pixel data.
  std::vector<uint8_t> copy_nv12_contiguous() const;

  /// Map this tensor as an I420 view (Y + U + V); returns nullopt if not I420.
  std::optional<I420Mapped> map_i420_read() const;
  /// Bytes required to hold a tightly-packed I420 copy of this tensor.
  std::size_t i420_required_bytes() const;
  /// Copy I420 pixel data into `dst` (`dst_size` ≥ `i420_required_bytes()`); returns success.
  bool copy_i420_contiguous_to(uint8_t* dst, std::size_t dst_size) const;
  /// Return a tightly-packed I420 byte copy of this tensor's pixel data.
  std::vector<uint8_t> copy_i420_contiguous() const;

  /// Bytes required to hold a tightly-packed (no-stride-padding) dense copy.
  std::size_t dense_bytes_tight() const;
  /// Copy dense data into `dst` with tight strides; returns success.
  bool copy_dense_bytes_tight_to(uint8_t* dst, std::size_t dst_size) const;
  /// Return a tightly-packed dense byte copy of this tensor's data.
  std::vector<uint8_t> copy_dense_bytes_tight() const;

  /// Copy this tensor's raw payload bytes into `dst`; returns success.
  bool copy_payload_bytes_to(uint8_t* dst, std::size_t dst_size) const;
  /// Return a copy of this tensor's raw payload bytes.
  std::vector<uint8_t> copy_payload_bytes() const;

  /// Width in pixels (image tensors only).
  int width() const;
  /// Height in pixels (image tensors only).
  int height() const;
  /// Number of color channels (image tensors only).
  int channels() const;
  /// Pixel format if this tensor carries an `ImageSpec`; nullopt otherwise.
  std::optional<ImageSpec::PixelFormat> image_format() const;
  /// True iff this tensor is an NV12-format image.
  bool is_nv12() const;
  /// True iff this tensor is an I420-format image.
  bool is_i420() const;

  /// Human-readable single-line summary (for logs and error messages).
  std::string debug_string() const;

#if defined(SIMA_WITH_OPENCV)
  /// Construct a Tensor wrapping a `cv::Mat` as CPU-backed storage.
  /// @deprecated Use the TensorMemory overload. The default from_cv_mat(mat, fmt)
  ///             now creates an EV74-placed tensor; pass TensorMemory::CPU/A65
  ///             explicitly when CPU placement is intended.
  [[deprecated("Use Tensor::from_cv_mat(mat, fmt, TensorMemory::EV74/CPU/MLA); "
               "Tensor::from_cv_mat(mat, fmt) defaults to EV74 placement.")]] static Tensor
  from_cv_mat(const cv::Mat& mat, ImageSpec::PixelFormat fmt, bool read_only);
  /// Construct a Tensor from a `cv::Mat` in the requested memory placement.
  static Tensor from_cv_mat(const cv::Mat& mat,
                            ImageSpec::PixelFormat fmt = ImageSpec::PixelFormat::BGR,
                            TensorMemory memory = TensorMemory::EV74);
  /// Construct a Tensor from a `cv::Mat` in the requested memory placement.
  static Tensor from_cv_mat(const cv::Mat& mat, TensorMemory memory);
  /// Map this tensor as a `cv::Mat` view in `desired` format; nullopt if conversion needed.
  std::optional<CvMatView> map_cv_mat_view(ImageSpec::PixelFormat desired) const;
  /// Return a `cv::Mat` deep copy in `desired` format.
  cv::Mat to_cv_mat_copy(ImageSpec::PixelFormat desired) const;
#endif

private:
  template <typename T> const T* const_data_ptr() const {
    if (device.type != DeviceType::CPU) {
      throw std::runtime_error("Tensor::data_ptr: tensor is not on CPU");
    }
    if (!is_dense()) {
      throw std::runtime_error("Tensor::data_ptr: tensor is composite");
    }
    if (!is_contiguous()) {
      throw std::runtime_error("Tensor::data_ptr: call cpu().contiguous() first");
    }
    if (!storage || !storage->data) {
      throw std::runtime_error("Tensor::data_ptr: tensor storage is not mappable");
    }
    return reinterpret_cast<const T*>(static_cast<const uint8_t*>(storage->data) + byte_offset);
  }

  static std::size_t dtype_bytes(simaai::neat::TensorDType dtype) {
    switch (dtype) {
    case simaai::neat::TensorDType::UInt8:
    case simaai::neat::TensorDType::Int8:
      return 1;
    case simaai::neat::TensorDType::UInt16:
    case simaai::neat::TensorDType::Int16:
    case simaai::neat::TensorDType::BFloat16:
      return 2;
    case simaai::neat::TensorDType::Int32:
    case simaai::neat::TensorDType::Float32:
      return 4;
    case simaai::neat::TensorDType::Float64:
      return 8;
    }
    return 0;
  }

public:
  static Tensor from_vector(const std::vector<float>& data, std::vector<int64_t> shape,
                            TensorMemory memory = TensorMemory::EV74);
  static Tensor from_vector(const std::vector<uint8_t>& data, std::vector<int64_t> shape,
                            TensorMemory memory = TensorMemory::EV74);
  static Tensor from_vector(const std::vector<int8_t>& data, std::vector<int64_t> shape,
                            TensorMemory memory = TensorMemory::EV74);
  static Tensor from_vector(const std::vector<uint16_t>& data, std::vector<int64_t> shape,
                            TensorMemory memory = TensorMemory::EV74);
  static Tensor from_vector(const std::vector<int16_t>& data, std::vector<int64_t> shape,
                            TensorMemory memory = TensorMemory::EV74);
  static Tensor from_vector(const std::vector<int32_t>& data, std::vector<int64_t> shape,
                            TensorMemory memory = TensorMemory::EV74);
};

} // namespace simaai::neat
