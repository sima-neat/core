#include "nodes/common/MultipartJpegDemux.h"
#include "nodes/common/internal/MultipartHeaderCapture.h"

#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using simaai::neat::multipart_internal::MultipartParser;
using Attrs = std::map<std::string, std::string>;

struct Part {
  std::string body;
  Attrs attrs;
};

std::vector<Part> parse_all(const std::string& stream, const std::string& boundary,
                            const std::vector<std::string>& capture, std::size_t chunk, bool* ok,
                            std::string* err) {
  std::vector<Part> parts;
  MultipartParser parser(boundary, capture);
  auto sink = [&](const uint8_t* body, std::size_t size, Attrs&& attrs,
                  MultipartParser::PartTiming) {
    parts.push_back(Part{std::string(reinterpret_cast<const char*>(body), size), std::move(attrs)});
    return true;
  };
  *ok = true;
  err->clear();
  for (std::size_t off = 0; off < stream.size() && *ok; off += chunk) {
    const std::size_t n = std::min(chunk, stream.size() - off);
    if (!parser.feed(reinterpret_cast<const uint8_t*>(stream.data() + off), n, sink, err, {})) {
      *ok = false;
    }
  }
  if (*ok && !parser.finish(sink, err)) {
    *ok = false;
  }
  return parts;
}

bool parse_rejects(const std::string& stream, const std::string& boundary,
                   const std::vector<std::string>& capture) {
  bool ok = false;
  std::string err;
  (void)parse_all(stream, boundary, capture, 3U, &ok, &err);
  return !ok;
}

/// Three parts covering present, mixed-case, present-but-empty, duplicate, and missing.
std::string canonical_stream() {
  std::string s;
  s += "--myboundary\r\n";
  s += "Content-Type: image/jpeg\r\n";
  s += "Image-Index: 1\r\n";
  s += "Image-Time: 2026-08-07T12:00:00Z\r\n";
  s += "\r\n";
  s += "JPEGONE";
  s += "\r\n--myboundary\r\n";
  s += "Content-Type: image/jpeg\r\n";
  s += "image-INDEX: 2\r\n";
  s += "Image-Time:\r\n";
  s += "\r\n";
  s += "JPEGTWO!!";
  s += "\r\n--myboundary\r\n";
  s += "Content-Type: image/jpeg\r\n";
  s += "Image-Index: 3\r\n";
  s += "Image-Index: 3b\r\n";
  s += "\r\n";
  s += "THREE";
  s += "\r\n--myboundary--\r\n";
  return s;
}

void check_canonical_parts(const std::vector<Part>& parts, const std::string& ctx) {
  require(parts.size() == 3U, "expected 3 parts" + ctx);
  require(parts[0].body == "JPEGONE", "part 0 body" + ctx);
  require(parts[1].body == "JPEGTWO!!", "part 1 body" + ctx);
  require(parts[2].body == "THREE", "part 2 body" + ctx);

  require(parts[0].attrs.size() == 2U, "part 0 attribute count" + ctx);
  require(parts[0].attrs.at("image-index") == "1", "part 0 image-index" + ctx);
  require(parts[0].attrs.at("image-time") == "2026-08-07T12:00:00Z", "part 0 image-time" + ctx);

  require(parts[1].attrs.at("image-index") == "2", "part 1 case-insensitive match" + ctx);
  require(parts[1].attrs.count("image-time") == 1U, "part 1 empty header retained" + ctx);
  require(parts[1].attrs.at("image-time").empty(), "part 1 empty value preserved" + ctx);

  require(parts[2].attrs.at("image-index") == "3b", "part 2 last duplicate wins" + ctx);
  require(parts[2].attrs.count("image-time") == 0U, "part 2 missing header omitted" + ctx);
}

void test_every_chunk_split() {
  const std::vector<std::string> capture = {"image-index", "image-time"};
  const std::string stream = canonical_stream();
  for (std::size_t chunk = 1U; chunk <= stream.size(); ++chunk) {
    bool ok = false;
    std::string err;
    const std::vector<Part> parts = parse_all(stream, "myboundary", capture, chunk, &ok, &err);
    const std::string ctx = " (chunk=" + std::to_string(chunk) + " err=" + err + ")";
    require(ok, "parse must succeed at every split point" + ctx);
    check_canonical_parts(parts, ctx);
  }
}

void test_boundary_autodetect() {
  const std::vector<std::string> capture = {"image-index", "image-time"};
  const std::string stream = canonical_stream();
  for (const std::size_t chunk : {1U, 7U, 64U}) {
    bool ok = false;
    std::string err;
    const std::vector<Part> parts = parse_all(stream, "", capture, chunk, &ok, &err);
    const std::string ctx = " (autodetect chunk=" + std::to_string(chunk) + " err=" + err + ")";
    require(ok, "auto-detected boundary must parse" + ctx);
    check_canonical_parts(parts, ctx);
  }

  std::string padded_stream = stream;
  padded_stream.replace(0U, std::string("--myboundary\r\n").size(), "--myboundary \t\r\n");
  bool ok = false;
  std::string err;
  const std::vector<Part> padded_parts = parse_all(padded_stream, "", capture, 5U, &ok, &err);
  require(ok, "auto-detection must ignore transport padding: " + err);
  check_canonical_parts(padded_parts, " (padded auto-detected boundary)");
}

/// The production camera framing: no closing delimiter, and a header that is simply
/// absent on some parts.
void test_open_ended_stream_with_absent_header() {
  const std::vector<std::string> capture = {"image-index", "image-continuation"};
  std::string s;
  s += "--frame\r\nContent-Type: image/jpeg\r\nImage-Index: 10\r\n\r\nAAA\r\n";
  s += "--frame\r\nContent-Type: image/jpeg\r\nImage-Index: 11\r\nImage-Continuation: true\r\n"
       "\r\nBBBB\r\n";
  s += "--frame\r\nContent-Type: image/jpeg\r\nImage-Index: 12\r\n\r\nCC\r\n";

  for (std::size_t chunk = 1U; chunk <= s.size(); ++chunk) {
    bool ok = false;
    std::string err;
    const std::vector<Part> parts = parse_all(s, "frame", capture, chunk, &ok, &err);
    const std::string ctx = " (chunk=" + std::to_string(chunk) + " err=" + err + ")";
    require(ok, "open-ended stream must parse" + ctx);
    require(parts.size() == 3U, "open-ended stream part count" + ctx);
    require(parts[0].body == "AAA", "open-ended part 0 body" + ctx);
    require(parts[1].body == "BBBB", "open-ended part 1 body" + ctx);
    require(parts[2].body == "CC", "open-ended part 2 body" + ctx);
    require(parts[0].attrs.at("image-index") == "10", "open-ended part 0 index" + ctx);
    require(parts[0].attrs.count("image-continuation") == 0U,
            "absent header must not be synthesized" + ctx);
    require(parts[1].attrs.at("image-continuation") == "true",
            "present header must be captured" + ctx);
    require(parts[2].attrs.count("image-continuation") == 0U,
            "absent header must not persist from the previous part" + ctx);
    require(parts[2].attrs.at("image-index") == "12", "open-ended part 2 index" + ctx);
  }
}

void test_body_containing_delimiter_bytes() {
  const std::vector<std::string> capture = {"image-index"};
  const std::string body = std::string("\xFF\xD8", 2) + "payload\r\n--bJUNK\r\nstill-payload" +
                           std::string("\xFF\xD9", 2);
  std::string s =
      "--b\r\nContent-Type: image/jpeg\r\nImage-Index: 9\r\n\r\n" + body + "\r\n--b--\r\n";
  for (const std::size_t chunk : {1U, 4U, 512U}) {
    bool ok = false;
    std::string err;
    const std::vector<Part> parts = parse_all(s, "b", capture, chunk, &ok, &err);
    const std::string ctx = " (chunk=" + std::to_string(chunk) + " err=" + err + ")";
    require(ok, "binary body must parse" + ctx);
    require(parts.size() == 1U, "delimiter-like body bytes must not split the part" + ctx);
    require(parts[0].body == body, "binary body must round-trip byte for byte" + ctx);
    require(parts[0].attrs.at("image-index") == "9", "binary body attributes" + ctx);
  }
}

void test_part_keeps_timing_from_body_start_chunk() {
  MultipartParser parser("b", {});
  MultipartParser::PartTiming observed;
  std::size_t emitted = 0U;
  const auto sink = [&](const uint8_t*, std::size_t, Attrs&&, MultipartParser::PartTiming timing) {
    observed = timing;
    ++emitted;
    return true;
  };
  std::string err;
  const std::string first = "--b\r\nContent-Type: image/jpeg\r\n\r\nPART";
  const MultipartParser::PartTiming first_timing{11U, 12U, 13U};
  require(parser.feed(reinterpret_cast<const uint8_t*>(first.data()), first.size(), sink, &err,
                      first_timing),
          "the first timed chunk must parse: " + err);

  const std::string second = "END\r\n--b--\r\n";
  const MultipartParser::PartTiming second_timing{21U, 22U, 23U};
  require(parser.feed(reinterpret_cast<const uint8_t*>(second.data()), second.size(), sink, &err,
                      second_timing),
          "the completing timed chunk must parse: " + err);
  require(parser.finish(sink, &err), "the timed stream must finish: " + err);
  require(emitted == 1U, "the timed stream must emit one part");
  require(observed.pts == first_timing.pts && observed.dts == first_timing.dts &&
              observed.duration == first_timing.duration,
          "part timing must come from the chunk where its body began");
}

void test_value_trimming() {
  const std::vector<std::string> capture = {"image-index"};
  const std::string s =
      "--b\r\nContent-Type: image/jpeg\r\nImage-Index: \t 42 \t\r\n\r\nX\r\n--b--\r\n";
  bool ok = false;
  std::string err;
  const std::vector<Part> parts = parse_all(s, "b", capture, 5U, &ok, &err);
  require(ok, "trimming stream must parse: " + err);
  require(parts.size() == 1U, "trimming stream part count");
  require(parts[0].attrs.at("image-index") == "42", "surrounding SP/HTAB must be trimmed");
}

void test_malformed_input_is_rejected() {
  const std::vector<std::string> capture = {"image-index"};
  require(parse_rejects("--b\r\nImage-Index: 1\r\n\r\nBODY\r\n--b--\r\n", "b", capture),
          "a JPEG part without Content-Type must be rejected");
  require(
      parse_rejects("--b\r\nContent-Type: text/plain\r\nImage-Index: 1\r\n\r\nBODY\r\n--b--\r\n",
                    "b", capture),
      "a non-JPEG Content-Type must be rejected");
  require(parse_rejects("--b\r\nContent-Type: image/jpeg\r\nImage-Index: 1\r\n", "b", capture),
          "an unfinished header block at EOS must be rejected");
  require(parse_rejects("--b\r\nContent-Type: image/jpeg\r\n\r\n\r\n--b--\r\n", "b", capture),
          "an empty JPEG part must be rejected");
  require(
      parse_rejects("--b\r\nImage-Index: 1\r\n Folded: x\r\n\r\nBODY\r\n--b--\r\n", "b", capture),
      "folded header line must be rejected");
  require(parse_rejects("--b\r\nBad Name: 1\r\n\r\nBODY\r\n--b--\r\n", "b", capture),
          "non-token header name must be rejected");
  require(parse_rejects(std::string("--b\r\nImage-Index: a\0b\r\n\r\nBODY\r\n--b--\r\n", 39), "b",
                        capture),
          "NUL in a header value must be rejected");
  require(parse_rejects("--b\r\nImage-Index: 1\rX\r\n\r\nBODY\r\n--b--\r\n", "b", capture),
          "bare CR in a header value must be rejected");
  require(parse_rejects("--b\r\nNoColonHere\r\n\r\nBODY\r\n--b--\r\n", "b", capture),
          "header line without ':' must be rejected");
  require(parse_rejects("--b junk\r\nImage-Index: 1\r\n\r\nBODY\r\n--b--\r\n", "b", capture),
          "trailing bytes on the boundary line must be rejected");
  require(
      parse_rejects("prefix--b\r\nContent-Type: image/jpeg\r\n\r\nBODY\r\n--b--\r\n", "b", capture),
      "an opening boundary in the middle of a preamble line must be rejected");
  require(parse_rejects("--b\nImage-Index: 1\n\nBODY\n--b--\n", "b", capture),
          "bare-LF framing must be rejected");
  require(parse_rejects("", "b", capture), "EOS before the first multipart part must be rejected");
  require(parse_rejects("--", "b", capture), "EOS inside the first boundary must be rejected");
}

void test_limits_fail_rather_than_truncate() {
  const std::vector<std::string> capture = {"image-index"};

  std::string long_line = "--b\r\nImage-Index: ";
  long_line += std::string(simaai::neat::kMultipartHeaderCaptureMaxLineBytes + 1U, 'x');
  long_line += "\r\n\r\nBODY\r\n--b--\r\n";
  require(parse_rejects(long_line, "b", capture), "oversized header line must be rejected");

  std::string long_name = "--b\r\n";
  long_name += std::string(simaai::neat::kMultipartHeaderCaptureMaxNameBytes + 1U, 'a');
  long_name += ": 1\r\n\r\nBODY\r\n--b--\r\n";
  require(parse_rejects(long_name, "b", capture), "oversized header name must be rejected");

  const std::string filler_line = "X-Filler-Header-Name-Here: 0123456789\r\n";
  std::string big_block = "--b\r\n";
  std::size_t block_bytes = 0;
  while (block_bytes <= simaai::neat::kMultipartHeaderCaptureMaxBlockBytes) {
    big_block += filler_line;
    block_bytes += filler_line.size();
  }
  big_block += "\r\nBODY\r\n--b--\r\n";
  require(parse_rejects(big_block, "b", capture), "oversized header block must be rejected");

  MultipartParser parser("b", capture, 8U);
  const std::string oversized_body =
      "--b\r\nContent-Type: image/jpeg\r\n\r\n123456789\r\n--b--\r\n";
  std::string err;
  const auto sink = [](const uint8_t*, std::size_t, Attrs&&, MultipartParser::PartTiming) {
    return true;
  };
  require(!parser.feed(reinterpret_cast<const uint8_t*>(oversized_body.data()),
                       oversized_body.size(), sink, &err, {}),
          "a part body larger than the configured hard limit must be rejected");

  MultipartParser exact_parser("b", capture, 8U);
  std::size_t emitted_size = 0U;
  const auto exact_sink = [&](const uint8_t*, std::size_t size, Attrs&&,
                              MultipartParser::PartTiming) {
    emitted_size = size;
    return true;
  };
  const std::string exact_prefix = "--b\r\nContent-Type: image/jpeg\r\n\r\n12345678\r\n--b";
  require(exact_parser.feed(reinterpret_cast<const uint8_t*>(exact_prefix.data()),
                            exact_prefix.size(), exact_sink, &err, {}),
          "a size-compliant body with a split delimiter must not exceed the limit: " + err);
  const std::string exact_suffix = "--\r\n";
  require(exact_parser.feed(reinterpret_cast<const uint8_t*>(exact_suffix.data()),
                            exact_suffix.size(), exact_sink, &err, {}),
          "the completed split delimiter must parse: " + err);
  require(exact_parser.finish(exact_sink, &err),
          "the size-compliant split-delimiter stream must finish: " + err);
  require(emitted_size == 8U, "framing bytes must not count toward the body-size limit");

  MultipartParser padded_parser("b", capture, 8U);
  emitted_size = 0U;
  const std::string padded_prefix =
      "--b\r\nContent-Type: image/jpeg\r\n\r\n12345678\r\n--b--        ";
  require(padded_parser.feed(reinterpret_cast<const uint8_t*>(padded_prefix.data()),
                             padded_prefix.size(), exact_sink, &err, {}),
          "an incomplete padded delimiter must remain buffered: " + err);
  const std::string padded_suffix = "\r\n";
  require(padded_parser.feed(reinterpret_cast<const uint8_t*>(padded_suffix.data()),
                             padded_suffix.size(), exact_sink, &err, {}),
          "a completed padded delimiter must parse: " + err);
  require(padded_parser.finish(exact_sink, &err),
          "the padded-delimiter stream must finish: " + err);
  require(emitted_size == 8U, "padded delimiter bytes must not enter the part body");
}

void test_allowlist_normalization() {
  using simaai::neat::MultipartHeaderCaptureOptions;
  using simaai::neat::normalize_multipart_header_capture;

  MultipartHeaderCaptureOptions opt;
  opt.headers = {"Image-Index", "image-index", "  Image-Time  ", "IMAGE-TIME"};
  const std::vector<std::string> names = normalize_multipart_header_capture(opt);
  require(names.size() == 2U, "duplicates must collapse after normalization");
  require(names[0] == "image-index", "normalized names must be lowercase and sorted");
  require(names[1] == "image-time", "normalized names must be lowercase and sorted");

  require(normalize_multipart_header_capture(MultipartHeaderCaptureOptions{}).empty(),
          "empty allowlist must stay empty");

  bool threw = false;
  try {
    MultipartHeaderCaptureOptions bad;
    bad.headers = {"Bad Name"};
    (void)normalize_multipart_header_capture(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "non-token configured name must throw");

  threw = false;
  try {
    MultipartHeaderCaptureOptions bad;
    bad.headers = {std::string(simaai::neat::kMultipartHeaderCaptureMaxNameBytes + 1U, 'a')};
    (void)normalize_multipart_header_capture(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "over-long configured name must throw");

  threw = false;
  try {
    MultipartHeaderCaptureOptions bad;
    for (std::size_t i = 0; i <= simaai::neat::kMultipartHeaderCaptureMaxHeaders; ++i) {
      bad.headers.push_back("h" + std::to_string(i));
    }
    (void)normalize_multipart_header_capture(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "too many configured names must throw");
}

void test_reset_clears_pending_state() {
  const std::vector<std::string> capture = {"image-index"};
  MultipartParser parser("b", capture);
  std::vector<Part> parts;
  auto sink = [&](const uint8_t* body, std::size_t size, Attrs&& attrs,
                  MultipartParser::PartTiming) {
    parts.push_back(Part{std::string(reinterpret_cast<const char*>(body), size), std::move(attrs)});
    return true;
  };
  std::string err;
  const std::string head = "--b\r\nContent-Type: image/jpeg\r\nImage-Index: 1\r\n\r\nPARTIAL";
  require(parser.feed(reinterpret_cast<const uint8_t*>(head.data()), head.size(), sink, &err, {}),
          "partial feed must succeed: " + err);
  require(parts.empty(), "an unterminated part must not be emitted");

  parser.reset();
  const std::string fresh =
      "--b\r\nContent-Type: image/jpeg\r\nImage-Index: 7\r\n\r\nNEW\r\n--b--\r\n";
  require(parser.feed(reinterpret_cast<const uint8_t*>(fresh.data()), fresh.size(), sink, &err, {}),
          "post-reset feed must succeed: " + err);
  require(parts.size() == 1U, "reset must drop buffered bytes");
  require(parts[0].body == "NEW", "post-reset body");
  require(parts[0].attrs.at("image-index") == "7", "post-reset attributes must not be stale");
}

} // namespace

RUN_TEST("unit_multipart_header_capture", [] {
  test_every_chunk_split();
  test_boundary_autodetect();
  test_open_ended_stream_with_absent_header();
  test_body_containing_delimiter_bytes();
  test_part_keeps_timing_from_body_start_chunk();
  test_value_trimming();
  test_malformed_input_is_rejected();
  test_limits_fail_rather_than_truncate();
  test_allowlist_normalization();
  test_reset_clears_pending_state();
})
