#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The text GStreamer gives caps of one structure in system memory whose fields are strings and
// integers (gst_caps_to_string), written without GStreamer: the structure's name, then for each
// field in the order it was set ", <field>=(string)<value>" or ", <field>=(int)<value>". A string
// is written as gst_value_serialize writes one: as it is when it is not empty, not "NULL" and every
// character is an ASCII letter or digit or one of _-+/:. ; otherwise between double quotes, each
// such character as it is, every other printable ASCII character after a backslash, and every
// other byte as a backslash and three octal digits.

#include <cstdint>
#include <string>
#include <string_view>

namespace simaai::neat::pipeline_internal {

class CapsText {
public:
  explicit CapsText(std::string_view name) : text_(name) {}

  CapsText& string(std::string_view field, std::string_view value) {
    open(field, "string");
    append_string(value);
    return *this;
  }

  CapsText& integer(std::string_view field, std::int32_t value) {
    open(field, "int");
    text_ += std::to_string(value);
    return *this;
  }

  const std::string& text() const noexcept {
    return text_;
  }

private:
  static bool plain(unsigned char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
           c == '-' || c == '+' || c == '/' || c == ':' || c == '.';
  }

  void open(std::string_view field, const char* type) {
    text_ += ", ";
    text_ += field;
    text_ += "=(";
    text_ += type;
    text_ += ')';
  }

  void append_string(std::string_view value) {
    bool quote = value.empty() || value == "NULL";
    for (const char c : value) {
      quote = quote || !plain(static_cast<unsigned char>(c));
    }
    if (!quote) {
      text_ += value;
      return;
    }
    text_ += '"';
    for (const char ch : value) {
      const auto c = static_cast<unsigned char>(ch);
      if (plain(c)) {
        text_ += ch;
      } else if (c < 0x20U || c >= 0x7fU) {
        text_ += '\\';
        text_ += static_cast<char>('0' + (c >> 6U));
        text_ += static_cast<char>('0' + ((c >> 3U) & 7U));
        text_ += static_cast<char>('0' + (c & 7U));
      } else {
        text_ += '\\';
        text_ += ch;
      }
    }
    text_ += '"';
  }

  std::string text_;
};

} // namespace simaai::neat::pipeline_internal
