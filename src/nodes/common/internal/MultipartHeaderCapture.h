/**
 * @file
 * @brief Internal multipart part-header capture: allowlist normalization and a single
 *        byte-stream parser that keeps selected headers locally associated with the JPEG
 *        body they were sent with.
 *
 * Not installed. Used by the `MultipartJpegDemux` Node and by the private
 * `neatmultipartjpegdemux` element.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace simaai::neat::multipart_internal {

/// Hard safety limit for one MIME part body. This header is private to Core.
inline constexpr std::size_t kMultipartJpegMaxPartBytes = 64U * 1024U * 1024U;

/// Normalized, deduplicated, sorted capture allowlist.
///
/// @throws std::invalid_argument on an empty name, a non-token name, a name longer than
///         `kMultipartHeaderCaptureMaxNameBytes`, or more than
///         `kMultipartHeaderCaptureMaxHeaders` distinct names after normalization.
std::vector<std::string> normalize_capture_names(const std::vector<std::string>& names);

/// Join normalized names into the comma-separated form carried on the element property.
std::string join_capture_names(const std::vector<std::string>& names);

/// Split a comma-separated property value back into normalized names.
std::vector<std::string> split_capture_names(const std::string& joined);

/// True when `name` is a valid HTTP token (RFC 9110 field-name).
bool is_http_token(const std::string& name);

/// Lowercase ASCII in place, leaving non-ASCII bytes untouched.
std::string ascii_lower(std::string value);

/**
 * @brief Incremental multipart parser that emits complete JPEG parts with their selected
 *        headers attached to the same emission.
 *
 * One state machine owns both the boundary scan and the header scan, so a part's headers
 * cannot drift away from its body. The parser is chunk-boundary safe: any split point in
 * the input byte stream produces the same parts.
 */
class MultipartParser {
public:
  /// Timing carried by the input chunk where a part body begins.
  struct PartTiming {
    std::uint64_t pts = UINT64_MAX;
    std::uint64_t dts = UINT64_MAX;
    std::uint64_t duration = UINT64_MAX;
  };

  /// Receives one complete part. Return false to abort parsing.
  using PartSink =
      std::function<bool(const uint8_t* body, std::size_t size,
                         std::map<std::string, std::string>&& attributes, PartTiming timing)>;

  /// @param boundary Boundary token without the leading `--`; empty enables auto-detect.
  /// @param capture Normalized allowlist; may be empty (then no attributes are emitted).
  MultipartParser(std::string boundary, std::vector<std::string> capture,
                  std::size_t max_part_bytes = kMultipartJpegMaxPartBytes);

  /// Feed a chunk. Returns false and sets `err` on a protocol or limit violation.
  bool feed(const uint8_t* data, std::size_t size, const PartSink& sink, std::string* err,
            PartTiming timing);

  /// Flush a trailing part that ended at end-of-stream rather than at a boundary.
  bool finish(const PartSink& sink, std::string* err);

  /// Drop all buffered state, keeping the configured boundary and allowlist.
  void reset();

  /// Boundary in use once detected; empty while still auto-detecting.
  const std::string& boundary() const {
    return boundary_;
  }

private:
  enum class State {
    Preamble,    ///< Before the first boundary line.
    HeaderBlock, ///< Accumulating header lines until the blank line.
    Body,        ///< Accumulating body bytes until the next boundary.
    Epilogue,    ///< After the closing boundary; remaining bytes are ignored.
  };

  bool run(const PartSink& sink, std::string* err);
  bool parse_header_block(std::size_t block_begin, std::size_t block_end, std::string* err);
  /// Index of the next boundary delimiter at or after `from`, or npos.
  std::size_t find_boundary(std::size_t from, bool* is_closing, std::size_t* delim_len,
                            bool require_leading_crlf) const;
  void compact();

  std::string boundary_;
  std::string delimiter_; ///< "--" + boundary_, cached once known.
  std::vector<std::string> capture_;
  std::size_t max_part_bytes_;
  State state_ = State::Preamble;
  std::vector<uint8_t> buf_;
  std::size_t cursor_ = 0;                     ///< Start of unconsumed data in `buf_`.
  std::size_t body_begin_ = 0;                 ///< Body start once in State::Body.
  std::size_t scanned_ = 0;                    ///< How far the body boundary scan has advanced.
  std::map<std::string, std::string> pending_; ///< Headers for the part being assembled.
  PartTiming feed_timing_;                     ///< Timing of the chunk being consumed.
  PartTiming pending_timing_;                  ///< Timing captured when this body began.
  bool pending_timing_set_ = false;
  bool discarding_preamble_line_ = false;
  bool buffer_starts_at_line_start_ = true;
  bool saw_any_part_ = false;
};

} // namespace simaai::neat::multipart_internal
