#include "nodes/common/internal/MultipartHeaderCapture.h"

#include "nodes/common/MultipartJpegDemux.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::neat::multipart_internal {
namespace {

constexpr uint8_t kCR = 0x0D;
constexpr uint8_t kLF = 0x0A;
constexpr uint8_t kSP = 0x20;
constexpr uint8_t kHT = 0x09;
constexpr std::size_t kFeedSliceBytes = 64U * 1024U;

bool is_token_char(char c) {
  const unsigned char u = static_cast<unsigned char>(c);
  if ((u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') || (u >= '0' && u <= '9')) {
    return true;
  }
  switch (c) {
  case '!':
  case '#':
  case '$':
  case '%':
  case '&':
  case '\'':
  case '*':
  case '+':
  case '-':
  case '.':
  case '^':
  case '_':
  case '`':
  case '|':
  case '~':
    return true;
  default:
    return false;
  }
}

/// Trim surrounding SP/HTAB only. No other byte is treated as whitespace.
std::string trim_sp_ht(const std::string& value) {
  std::size_t begin = 0;
  std::size_t end = value.size();
  while (begin < end && (value[begin] == ' ' || value[begin] == '\t')) {
    ++begin;
  }
  while (end > begin && (value[end - 1U] == ' ' || value[end - 1U] == '\t')) {
    --end;
  }
  return value.substr(begin, end - begin);
}

/// Field values may not carry NUL, CR, LF, or other C0 controls; HTAB is allowed.
bool is_valid_field_value(const std::string& value) {
  for (const char c : value) {
    const unsigned char u = static_cast<unsigned char>(c);
    if (u == 0U) {
      return false;
    }
    if (u < 0x20U && u != kHT) {
      return false;
    }
    if (u == 0x7FU) {
      return false;
    }
  }
  return true;
}

bool is_jpeg_content_type(const std::string& value) {
  const std::size_t semicolon = value.find(';');
  return ascii_lower(trim_sp_ht(value.substr(0U, semicolon))) == "image/jpeg";
}

} // namespace

std::string ascii_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>((c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c);
  });
  return value;
}

bool is_http_token(const std::string& name) {
  if (name.empty()) {
    return false;
  }
  return std::all_of(name.begin(), name.end(), is_token_char);
}

std::vector<std::string> normalize_capture_names(const std::vector<std::string>& names) {
  std::set<std::string> unique;
  for (const std::string& raw : names) {
    const std::string trimmed = trim_sp_ht(raw);
    if (trimmed.empty()) {
      throw std::invalid_argument("header_capture: empty header name");
    }
    if (trimmed.size() > kMultipartHeaderCaptureMaxNameBytes) {
      throw std::invalid_argument("header_capture: header name '" + trimmed + "' exceeds " +
                                  std::to_string(kMultipartHeaderCaptureMaxNameBytes) + " bytes");
    }
    if (!is_http_token(trimmed)) {
      throw std::invalid_argument("header_capture: header name '" + trimmed +
                                  "' is not a valid HTTP token");
    }
    unique.insert(ascii_lower(trimmed));
  }
  if (unique.size() > kMultipartHeaderCaptureMaxHeaders) {
    throw std::invalid_argument("header_capture: " + std::to_string(unique.size()) +
                                " header names exceeds the limit of " +
                                std::to_string(kMultipartHeaderCaptureMaxHeaders));
  }
  return {unique.begin(), unique.end()};
}

std::string join_capture_names(const std::vector<std::string>& names) {
  std::ostringstream ss;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (i != 0U) {
      ss << ',';
    }
    ss << names[i];
  }
  return ss.str();
}

std::vector<std::string> split_capture_names(const std::string& joined) {
  std::vector<std::string> out;
  std::string current;
  std::istringstream ss(joined);
  while (std::getline(ss, current, ',')) {
    const std::string trimmed = trim_sp_ht(current);
    if (!trimmed.empty()) {
      out.push_back(ascii_lower(trimmed));
    }
  }
  return out;
}

MultipartParser::MultipartParser(std::string boundary, std::vector<std::string> capture,
                                 std::size_t max_part_bytes)
    : boundary_(std::move(boundary)), capture_(std::move(capture)),
      max_part_bytes_(max_part_bytes) {
  if (max_part_bytes_ == 0U) {
    throw std::invalid_argument("multipart maximum part size must be greater than zero");
  }
  if (!boundary_.empty()) {
    delimiter_ = "--" + boundary_;
  }
  // Lookup is a binary search, so own the ordering here rather than trusting the caller.
  // An unsorted allowlist would otherwise capture nothing at all, silently.
  for (std::string& name : capture_) {
    name = ascii_lower(trim_sp_ht(name));
  }
  std::sort(capture_.begin(), capture_.end());
  capture_.erase(std::unique(capture_.begin(), capture_.end()), capture_.end());
}

void MultipartParser::reset() {
  state_ = State::Preamble;
  buf_.clear();
  cursor_ = 0;
  body_begin_ = 0;
  scanned_ = 0;
  pending_.clear();
  saw_any_part_ = false;
}

void MultipartParser::compact() {
  if (cursor_ == 0U) {
    return;
  }
  buf_.erase(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(cursor_));
  if (body_begin_ >= cursor_) {
    body_begin_ -= cursor_;
  } else {
    body_begin_ = 0;
  }
  if (scanned_ >= cursor_) {
    scanned_ -= cursor_;
  } else {
    scanned_ = 0;
  }
  cursor_ = 0;
}

std::size_t MultipartParser::find_boundary(std::size_t from, bool* is_closing,
                                           std::size_t* delim_len,
                                           bool require_leading_crlf) const {
  if (delimiter_.empty() || buf_.size() < from) {
    return std::string::npos;
  }
  const auto* needle = reinterpret_cast<const uint8_t*>(delimiter_.data());
  const std::size_t needle_len = delimiter_.size();

  std::size_t search_from = from;
  for (;;) {
    if (buf_.size() < search_from + needle_len) {
      return std::string::npos;
    }
    const auto begin = buf_.begin() + static_cast<std::ptrdiff_t>(search_from);
    const auto found = std::search(begin, buf_.end(), needle, needle + needle_len);
    if (found == buf_.end()) {
      return std::string::npos;
    }
    const std::size_t at = static_cast<std::size_t>(found - buf_.begin());

    // RFC 2046: a delimiter line is preceded by CRLF. Requiring it keeps delimiter-like
    // byte sequences inside a compressed payload from terminating the part early.
    if (require_leading_crlf && !(at >= 2U && buf_[at - 1U] == kLF && buf_[at - 2U] == kCR)) {
      search_from = at + 1U;
      continue;
    }

    // Two bytes after the delimiter are needed to tell a separator from the closing
    // delimiter; without them, wait rather than guess.
    const std::size_t after = at + needle_len;
    if (after + 2U > buf_.size()) {
      return std::string::npos;
    }
    const bool closing = (buf_[after] == '-' && buf_[after + 1U] == '-');
    if (is_closing) {
      *is_closing = closing;
    }
    if (delim_len) {
      *delim_len = closing ? (needle_len + 2U) : needle_len;
    }
    return at;
  }
}

bool MultipartParser::parse_header_block(std::size_t block_begin, std::size_t block_end,
                                         std::string* err) {
  const std::size_t block_size = block_end - block_begin;
  if (block_size > kMultipartHeaderCaptureMaxBlockBytes) {
    if (err) {
      *err = "multipart part header block of " + std::to_string(block_size) +
             " bytes exceeds the limit of " + std::to_string(kMultipartHeaderCaptureMaxBlockBytes) +
             " bytes";
    }
    return false;
  }

  bool saw_content_type = false;
  std::size_t line_begin = block_begin;
  while (line_begin < block_end) {
    std::size_t line_end = line_begin;
    while (line_end < block_end && buf_[line_end] != kLF) {
      ++line_end;
    }
    // A bare LF (no preceding CR) is malformed framing, not a line terminator we accept.
    if (line_end == block_end || line_end == line_begin || buf_[line_end - 1U] != kCR) {
      if (err) {
        *err = "multipart part header line is not CRLF-terminated";
      }
      return false;
    }
    const std::size_t content_end = line_end - 1U; // strip CR
    const std::size_t line_len = content_end - line_begin;
    if (line_len > kMultipartHeaderCaptureMaxLineBytes) {
      if (err) {
        *err = "multipart part header line of " + std::to_string(line_len) +
               " bytes exceeds the limit of " +
               std::to_string(kMultipartHeaderCaptureMaxLineBytes) + " bytes";
      }
      return false;
    }
    if (line_len == 0U) {
      // Blank line inside the block: the caller already bounded the block, so this is
      // stray framing.
      if (err) {
        *err = "unexpected blank line inside multipart part header block";
      }
      return false;
    }

    // Obsolete line folding is rejected rather than unfolded.
    if (buf_[line_begin] == kSP || buf_[line_begin] == kHT) {
      if (err) {
        *err = "folded multipart part header line is not supported";
      }
      return false;
    }

    const std::string line(reinterpret_cast<const char*>(buf_.data() + line_begin), line_len);
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      if (err) {
        *err = "multipart part header line has no ':' separator";
      }
      return false;
    }
    const std::string name = line.substr(0, colon);
    if (!is_http_token(name)) {
      if (err) {
        *err = "multipart part header name is not a valid HTTP token";
      }
      return false;
    }
    if (name.size() > kMultipartHeaderCaptureMaxNameBytes) {
      if (err) {
        *err = "multipart part header name exceeds " +
               std::to_string(kMultipartHeaderCaptureMaxNameBytes) + " bytes";
      }
      return false;
    }
    const std::string value = trim_sp_ht(line.substr(colon + 1U));
    if (!is_valid_field_value(value)) {
      if (err) {
        *err = "multipart part header value contains control or NUL bytes";
      }
      return false;
    }

    const std::string key = ascii_lower(name);
    if (key == "content-type") {
      if (!is_jpeg_content_type(value)) {
        if (err) {
          *err = "multipart part Content-Type must be image/jpeg";
        }
        return false;
      }
      saw_content_type = true;
    }
    if (std::binary_search(capture_.begin(), capture_.end(), key)) {
      // Last value wins for a repeated selected header.
      pending_[key] = value;
    }

    line_begin = line_end + 1U;
  }
  if (!saw_content_type) {
    if (err) {
      *err = "multipart part is missing Content-Type: image/jpeg";
    }
    return false;
  }
  return true;
}

bool MultipartParser::run(const PartSink& sink, std::string* err) {
  for (;;) {
    switch (state_) {
    case State::Preamble: {
      if (delimiter_.empty()) {
        // Auto-detect: the first line starting with "--" names the boundary.
        std::size_t line_end = cursor_;
        while (line_end < buf_.size() && buf_[line_end] != kLF) {
          ++line_end;
        }
        if (line_end >= buf_.size()) {
          if (buf_.size() - cursor_ > kMultipartHeaderCaptureMaxLineBytes) {
            if (err) {
              *err = "multipart boundary line exceeds " +
                     std::to_string(kMultipartHeaderCaptureMaxLineBytes) + " bytes";
            }
            return false;
          }
          return true; // need more data
        }
        std::size_t content_end = line_end;
        if (content_end > cursor_ && buf_[content_end - 1U] == kCR) {
          --content_end;
        }
        const std::size_t len = content_end - cursor_;
        if (len >= 3U && buf_[cursor_] == '-' && buf_[cursor_ + 1U] == '-') {
          boundary_.assign(reinterpret_cast<const char*>(buf_.data() + cursor_ + 2U), len - 2U);
          delimiter_ = "--" + boundary_;
          continue; // re-enter with a known delimiter
        }
        cursor_ = line_end + 1U;
        compact();
        continue;
      }

      bool closing = false;
      std::size_t delim_len = 0;
      const std::size_t at = find_boundary(cursor_, &closing, &delim_len, false);
      if (at == std::string::npos) {
        // Keep a tail long enough to match a delimiter split across chunks.
        const std::size_t keep = delimiter_.size() + 4U;
        if (buf_.size() - cursor_ > keep) {
          cursor_ = buf_.size() - keep;
          compact();
        }
        return true;
      }
      if (closing) {
        state_ = State::Epilogue;
        cursor_ = at + delim_len;
        compact();
        continue;
      }
      cursor_ = at + delim_len;
      state_ = State::HeaderBlock;
      pending_.clear();
      compact();
      continue;
    }

    case State::HeaderBlock: {
      // The header block starts after the boundary line's CRLF and ends at CRLFCRLF.
      // Only transport padding (SP/HTAB) may follow the delimiter on its own line;
      // anything else is malformed framing rather than something to normalize away.
      std::size_t scan = cursor_;
      while (scan < buf_.size() && (buf_[scan] == kSP || buf_[scan] == kHT)) {
        ++scan;
      }
      if (scan + 1U >= buf_.size()) {
        if (buf_.size() - cursor_ > kMultipartHeaderCaptureMaxLineBytes) {
          if (err) {
            *err = "multipart boundary line exceeds " +
                   std::to_string(kMultipartHeaderCaptureMaxLineBytes) + " bytes";
          }
          return false;
        }
        return true;
      }
      if (buf_[scan] != kCR || buf_[scan + 1U] != kLF) {
        if (err) {
          *err = "multipart boundary line carries unexpected trailing bytes";
        }
        return false;
      }
      ++scan; // land on the LF
      const std::size_t block_begin = scan + 1U;

      static const uint8_t kBlockEnd[4] = {kCR, kLF, kCR, kLF};
      const auto begin = buf_.begin() + static_cast<std::ptrdiff_t>(block_begin);
      auto found = std::search(begin, buf_.end(), kBlockEnd, kBlockEnd + 4);
      std::size_t block_end = 0;
      std::size_t body_start = 0;
      if (found != buf_.end()) {
        block_end = static_cast<std::size_t>(found - buf_.begin()) + 2U; // include final CRLF
        body_start = static_cast<std::size_t>(found - buf_.begin()) + 4U;
      } else if (block_begin < buf_.size() && buf_[block_begin] == kCR &&
                 block_begin + 1U < buf_.size() && buf_[block_begin + 1U] == kLF) {
        // Empty header block.
        block_end = block_begin;
        body_start = block_begin + 2U;
      } else {
        if (buf_.size() - block_begin > kMultipartHeaderCaptureMaxBlockBytes) {
          if (err) {
            *err = "multipart part header block exceeds " +
                   std::to_string(kMultipartHeaderCaptureMaxBlockBytes) + " bytes";
          }
          return false;
        }
        return true; // need more data
      }

      if (!parse_header_block(block_begin, block_end, err)) {
        return false;
      }
      body_begin_ = body_start;
      scanned_ = body_start;
      cursor_ = body_start;
      state_ = State::Body;
      continue;
    }

    case State::Body: {
      bool closing = false;
      std::size_t delim_len = 0;
      // Resume the scan where it stopped, backing off enough to catch a split delimiter.
      const std::size_t resume = (scanned_ > body_begin_ + delimiter_.size() + 4U)
                                     ? (scanned_ - delimiter_.size() - 4U)
                                     : body_begin_;
      const std::size_t at = find_boundary(resume, &closing, &delim_len, true);
      if (at == std::string::npos) {
        if (buf_.size() - body_begin_ > max_part_bytes_) {
          if (err) {
            *err = "multipart part body exceeds " + std::to_string(max_part_bytes_) + " bytes";
          }
          return false;
        }
        scanned_ = buf_.size();
        return true;
      }
      // The delimiter is preceded by the CRLF that terminates the body.
      std::size_t body_end = at;
      if (body_end >= body_begin_ + 2U && buf_[body_end - 1U] == kLF &&
          buf_[body_end - 2U] == kCR) {
        body_end -= 2U;
      }
      if (body_end > body_begin_) {
        if (body_end - body_begin_ > max_part_bytes_) {
          if (err) {
            *err = "multipart part body exceeds " + std::to_string(max_part_bytes_) + " bytes";
          }
          return false;
        }
        std::map<std::string, std::string> attributes;
        attributes.swap(pending_);
        if (!sink(buf_.data() + body_begin_, body_end - body_begin_, std::move(attributes))) {
          if (err) {
            *err = "multipart part consumer rejected a part";
          }
          return false;
        }
        saw_any_part_ = true;
      } else {
        if (err) {
          *err = "multipart JPEG part body is empty";
        }
        return false;
      }
      cursor_ = at + delim_len;
      state_ = closing ? State::Epilogue : State::HeaderBlock;
      if (!closing) {
        pending_.clear();
      }
      compact();
      continue;
    }

    case State::Epilogue:
      cursor_ = buf_.size();
      compact();
      return true;
    }
  }
}

bool MultipartParser::feed(const uint8_t* data, std::size_t size, const PartSink& sink,
                           std::string* err) {
  if (data == nullptr || size == 0U) {
    return run(sink, err);
  }
  std::size_t offset = 0U;
  while (offset < size) {
    const std::size_t chunk = std::min(kFeedSliceBytes, size - offset);
    buf_.insert(buf_.end(), data + offset, data + offset + chunk);
    offset += chunk;
    if (!run(sink, err)) {
      return false;
    }
  }
  return true;
}

bool MultipartParser::finish(const PartSink& sink, std::string* err) {
  if (!run(sink, err)) {
    return false;
  }
  if (state_ == State::HeaderBlock) {
    if (err) {
      *err = "multipart stream ended in an unfinished part header block";
    }
    return false;
  }
  if (state_ != State::Body) {
    return true;
  }
  // A stream that ends without a closing boundary still delivers its last part.
  std::size_t body_end = buf_.size();
  if (body_end >= body_begin_ + 2U && buf_[body_end - 1U] == kLF && buf_[body_end - 2U] == kCR) {
    body_end -= 2U;
  }
  if (body_end > body_begin_) {
    if (body_end - body_begin_ > max_part_bytes_) {
      if (err) {
        *err = "multipart part body exceeds " + std::to_string(max_part_bytes_) + " bytes";
      }
      return false;
    }
    std::map<std::string, std::string> attributes;
    attributes.swap(pending_);
    if (!sink(buf_.data() + body_begin_, body_end - body_begin_, std::move(attributes))) {
      if (err) {
        *err = "multipart part consumer rejected the final part";
      }
      return false;
    }
    saw_any_part_ = true;
  } else {
    if (err) {
      *err = "multipart stream ended with an empty JPEG part body";
    }
    return false;
  }
  cursor_ = buf_.size();
  state_ = State::Epilogue;
  compact();
  return true;
}

} // namespace simaai::neat::multipart_internal
