/**
 * @file
 * @ingroup nodes_groups
 * @brief `ImageInputGroup` — still-image source NodeGroup that emits a video-shaped stream.
 *
 * Bundles a file source, an image decoder (JPEG, PNG, or auto-detected), and an
 * `imagefreeze`-style stage that promotes the single still frame into a continuous
 * video stream at a chosen frame rate. Typical placement: very first NodeGroup in a
 * Session that wants to test or demo a vision pipeline against a single image instead
 * of a live source.
 *
 * @see VideoInputGroup
 * @see RtspDecodedInput
 */
#pragma once

#include "builder/NodeGroup.h"
#include "contracts/ContractTypes.h"
#include "pipeline/FormatSpec.h"

#include <string>

namespace simaai::neat::nodes::groups {

/**
 * @brief Configuration for `ImageInputGroup`.
 *
 * Drives the image source path, the chosen decoder backend, and the optional output
 * caps that the group advertises downstream.
 *
 * @ingroup nodes_groups
 */
struct ImageInputGroupOptions {
  std::string path; ///< Filesystem path to the still image to read.
  /// Number of buffers `imagefreeze` should produce (-1 = unlimited).
  /// If `sima_decoder` is enabled and this is >0, it may be clamped to a minimum
  /// to allow decoder startup. Set `SIMA_IMAGEFREEZE_MIN_BUFFERS=0` to disable.
  int imagefreeze_num_buffers = -1;
  int fps = 30;           ///< Output frame rate produced by `imagefreeze`.
  bool sync_mode = false; ///< If true, sink elements run in sync (real-time) mode.

  bool use_videorate = false;   ///< Insert `videorate` to enforce `fps` exactly.
  bool use_videoconvert = true; ///< Insert `videoconvert` for format adaptation.
  bool use_videoscale = false;  ///< Insert `videoscale` for resolution adaptation.

  /// Optional explicit output caps applied at the group's tail.
  struct OutputCaps {
    bool enable = true;                  ///< If false, no caps filter is inserted.
    FormatSpec format = FormatTag::NV12; ///< Pixel format the group should advertise.
    int width = -1;                      ///< Output width (-1 = leave unspecified).
    int height = -1;                     ///< Output height (-1 = leave unspecified).
    int fps = -1;                        ///< Output frame rate (-1 = leave unspecified).
    simaai::neat::CapsMemory memory =
        simaai::neat::CapsMemory::SystemMemory; ///< Buffer memory domain.
  } output_caps;                                ///< Output caps applied at the group's tail.

  /// Decoder selection strategy.
  enum class Decoder {
    Auto = 0,  ///< `decodebin` — auto-detect JPEG/PNG.
    ForceJpeg, ///< Force `jpegdec`.
    ForcePng,  ///< Force `pngdec`.
    Custom,    ///< Use `custom_decoder_fragment` verbatim.
  };

  Decoder decoder = Decoder::Auto;     ///< Selected decoder backend.
  std::string custom_decoder_fragment; ///< Raw GStreamer fragment used when `decoder == Custom`.

  /// Optional SiMa hardware decoder configuration (replaces the software decoder when enabled).
  struct SimaDecoder {
    bool enable = false;                  ///< Enable the SiMa hardware decoder path.
    int sima_allocator_type = 2;          ///< SiMa allocator type for decoder output buffers.
    std::string decoder_name = "decoder"; ///< Element instance name for the decoder.
    bool raw_output = false;              ///< Request raw (non-encoded) output from the decoder.
    /// Optional: select output buffer target ("CVU" or "APU") for `neatdecoder`.
    std::string next_element;
    bool use_sw_encoder = false; ///< Insert software H264 encoder before the SiMa decoder.
    int sw_bitrate_kbps = 4000;  ///< Bitrate for the optional software H264 encoder.
  } sima_decoder; ///< SiMa hardware decoder configuration; overrides the software path when
                  ///< enabled.

  /// Optional raw GStreamer fragment inserted before `imagefreeze` (advanced use).
  std::string extra_fragment;
};

/**
 * @brief Build an image-input NodeGroup: file source, decoder, and `imagefreeze` to a video stream.
 *
 * Typical chain: `FileInput` -> image decoder (`decodebin` / `jpegdec` / `pngdec` /
 * SiMa decoder) -> optional `videoconvert`/`videorate`/`videoscale` -> `imagefreeze`
 * with the configured output caps. Use this when a Session should be driven by a
 * single still image rather than a live or file-based video source.
 *
 * @param opt Configuration controlling source path, decoder backend, and output caps.
 * @return The configured `NodeGroup` ready to be `add()`ed to a Session.
 *
 * @see VideoInputGroup
 * @see RtspDecodedInput
 * @ingroup nodes_groups
 */
simaai::neat::NodeGroup ImageInputGroup(const ImageInputGroupOptions& opt);

} // namespace simaai::neat::nodes::groups
