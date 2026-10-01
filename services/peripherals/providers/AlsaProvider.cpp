#include "AlsaProvider.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/GraphReport.h"
#include "pipeline/NeatError.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace simaai::neat::peripherals_internal {
namespace {

namespace fs = std::filesystem;

constexpr const char* kAlsaProvider = "daemon.microphone.alsa";

struct CardMetadata {
  std::string id;
  std::string driver;
  std::string name;
};

struct UsbMetadata {
  std::string bus_path;
  std::string interface;
  std::string vendor_id;
  std::string product_id;
  std::string manufacturer;
  std::string product;
  std::string serial;
};

struct CaptureMode {
  std::optional<unsigned> interface;
  std::optional<unsigned> altset;
  std::vector<std::string> formats;
  std::optional<unsigned> channels;
  std::optional<unsigned> sample_bits;
  std::vector<unsigned> rates_hz;
  std::optional<std::pair<unsigned, unsigned>> rate_range_hz;
  std::vector<std::string> channel_map;
};

std::string trim(std::string_view value) {
  const auto first = std::find_if_not(value.begin(), value.end(),
                                      [](const unsigned char ch) { return std::isspace(ch) != 0; });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char ch) {
                      return std::isspace(ch) != 0;
                    }).base();
  if (first >= last)
    return {};
  return {first, last};
}

std::optional<std::string> read_file(const fs::path& path) {
  std::ifstream input(path);
  if (!input)
    return std::nullopt;
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof())
    return std::nullopt;
  return trim(contents.str());
}

[[noreturn]] void throw_discovery_error(const char* code, std::string message) {
  GraphReport report;
  report.error_code = code;
  report.repro_note = std::move(message);
  const std::string description = "[" + report.error_code + "] " + report.repro_note;
  throw NeatError(description, std::move(report));
}

std::optional<unsigned> parse_unsigned(std::string_view value) {
  const std::string text = trim(value);
  if (text.empty())
    return std::nullopt;
  unsigned parsed = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (error != std::errc{} || end != text.data() + text.size())
    return std::nullopt;
  return parsed;
}

std::vector<std::string> words(std::string_view value) {
  std::istringstream input{std::string(value)};
  std::vector<std::string> result;
  std::string word;
  while (input >> word)
    result.push_back(std::move(word));
  return result;
}

std::vector<unsigned> unsigned_values(std::string_view value) {
  static const std::regex number(R"((\d+))");
  const std::string text(value);
  std::vector<unsigned> result;
  for (std::sregex_iterator it(text.begin(), text.end(), number), end; it != end; ++it) {
    if (const auto parsed = parse_unsigned((*it)[1].str()))
      result.push_back(*parsed);
  }
  return result;
}

std::vector<std::pair<unsigned, fs::path>>
numbered_entries(const fs::path& directory, const std::regex& pattern,
                 std::error_code* result_error = nullptr) {
  std::error_code error;
  fs::directory_iterator it(directory, error);
  if (error) {
    if (result_error)
      *result_error = error;
    return {};
  }
  std::vector<std::pair<unsigned, fs::path>> entries;
  const fs::directory_iterator end;
  while (it != end) {
    std::smatch match;
    const std::string name = it->path().filename().string();
    if (std::regex_match(name, match, pattern)) {
      if (const auto number = parse_unsigned(match[1].str()))
        entries.emplace_back(*number, it->path());
    }
    it.increment(error);
    if (error) {
      if (result_error)
        *result_error = error;
      return {};
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
  return entries;
}

std::map<unsigned, CardMetadata> parse_cards(std::string_view contents) {
  std::map<unsigned, CardMetadata> cards;
  std::istringstream input{std::string(contents)};
  std::string line;
  while (std::getline(input, line)) {
    const std::string stripped = trim(line);
    const auto index_end = stripped.find_first_of(" \t");
    const auto id_begin = stripped.find('[', index_end);
    const auto id_end = stripped.find(']', id_begin);
    const auto metadata_begin = stripped.find(':', id_end);
    if (index_end == std::string::npos || id_begin == std::string::npos ||
        id_end == std::string::npos || metadata_begin == std::string::npos)
      continue;
    const auto index = parse_unsigned(std::string_view(stripped).substr(0, index_end));
    if (!index)
      continue;
    const std::string metadata = trim(std::string_view(stripped).substr(metadata_begin + 1));
    const auto separator = metadata.find(" - ");
    if (separator == std::string::npos)
      continue;
    cards[*index] = {
        .id = trim(std::string_view(stripped).substr(id_begin + 1, id_end - id_begin - 1)),
        .driver = trim(std::string_view(metadata).substr(0, separator)),
        .name = trim(std::string_view(metadata).substr(separator + 3)),
    };
  }
  return cards;
}

std::map<std::string, std::string> parse_key_values(std::string_view contents) {
  std::map<std::string, std::string> values;
  std::istringstream input{std::string(contents)};
  std::string line;
  while (std::getline(input, line)) {
    const auto separator = line.find(':');
    if (separator == std::string::npos)
      continue;
    values[trim(std::string_view(line).substr(0, separator))] =
        trim(std::string_view(line).substr(separator + 1));
  }
  return values;
}

std::vector<nlohmann::json> parse_capture_modes(std::string_view contents) {
  static const std::regex interface_line(R"(^Interface\s+(\d+)$)");
  static const std::regex altset_line(R"(^Altset\s+(\d+)$)");
  static const std::regex continuous_rates(R"(^(\d+)\s*-\s*(\d+)\s*\(continuous\)$)");

  std::vector<CaptureMode> parsed;
  std::optional<unsigned> current_interface;
  std::optional<CaptureMode> current;
  bool capture = false;
  const auto finish_mode = [&] {
    if (current)
      parsed.push_back(std::move(*current));
    current.reset();
  };

  std::istringstream input{std::string(contents)};
  std::string line;
  while (std::getline(input, line)) {
    const std::string stripped = trim(line);
    if (stripped.empty())
      continue;
    if (!std::isspace(static_cast<unsigned char>(line.front()))) {
      finish_mode();
      capture = stripped == "Capture:";
      current_interface.reset();
      continue;
    }
    if (!capture)
      continue;

    std::smatch match;
    if (std::regex_match(stripped, match, interface_line)) {
      current_interface = parse_unsigned(match[1].str());
      continue;
    }
    if (std::regex_match(stripped, match, altset_line)) {
      finish_mode();
      current.emplace();
      current->interface = current_interface;
      current->altset = parse_unsigned(match[1].str());
      continue;
    }
    if (!current)
      continue;

    const auto separator = stripped.find(':');
    if (separator == std::string::npos)
      continue;
    const std::string key = trim(std::string_view(stripped).substr(0, separator));
    const std::string value = trim(std::string_view(stripped).substr(separator + 1));
    if (key == "Format") {
      current->formats = words(value);
    } else if (key == "Channels") {
      current->channels = parse_unsigned(value);
    } else if (key == "Bits") {
      current->sample_bits = parse_unsigned(value);
    } else if (key == "Rates") {
      if (std::regex_match(value, match, continuous_rates)) {
        const auto minimum = parse_unsigned(match[1].str());
        const auto maximum = parse_unsigned(match[2].str());
        if (minimum && maximum && *minimum <= *maximum)
          current->rate_range_hz = std::pair(*minimum, *maximum);
      } else {
        current->rates_hz = unsigned_values(value);
        std::sort(current->rates_hz.begin(), current->rates_hz.end());
        current->rates_hz.erase(std::unique(current->rates_hz.begin(), current->rates_hz.end()),
                                current->rates_hz.end());
      }
    } else if (key == "Channel map") {
      current->channel_map = words(value);
    }
  }
  finish_mode();

  std::vector<nlohmann::json> modes;
  for (const auto& mode : parsed) {
    for (const auto& format : mode.formats) {
      nlohmann::json serialized = {{"format", format}};
      if (mode.interface)
        serialized["interface"] = *mode.interface;
      if (mode.altset)
        serialized["altset"] = *mode.altset;
      if (mode.channels)
        serialized["channels"] = *mode.channels;
      if (mode.sample_bits)
        serialized["sample_bits"] = *mode.sample_bits;
      if (!mode.rates_hz.empty())
        serialized["rates_hz"] = mode.rates_hz;
      if (mode.rate_range_hz) {
        serialized["rate_range_hz"] = {{"min", mode.rate_range_hz->first},
                                       {"max", mode.rate_range_hz->second}};
      }
      if (!mode.channel_map.empty())
        serialized["channel_map"] = mode.channel_map;
      modes.push_back(std::move(serialized));
    }
  }
  std::sort(modes.begin(), modes.end(),
            [](const auto& left, const auto& right) { return left.dump() < right.dump(); });
  modes.erase(std::unique(modes.begin(), modes.end()), modes.end());
  return modes;
}

std::optional<fs::path> canonical_path(const fs::path& path) {
  std::error_code error;
  fs::path resolved = fs::canonical(path, error);
  if (error)
    return std::nullopt;
  return resolved;
}

bool path_within(const fs::path& path, const fs::path& root) {
  auto path_it = path.begin();
  for (auto root_it = root.begin(); root_it != root.end(); ++root_it, ++path_it) {
    if (path_it == path.end() || *path_it != *root_it)
      return false;
  }
  return true;
}

std::optional<UsbMetadata> usb_metadata(const fs::path& device_path, const fs::path& sys_root) {
  const auto canonical_sys = canonical_path(sys_root);
  if (!canonical_sys)
    return std::nullopt;
  fs::path current = device_path;
  while (path_within(current, *canonical_sys) && current != *canonical_sys) {
    const auto vendor = read_file(current / "idVendor");
    const auto product = read_file(current / "idProduct");
    if (vendor && product) {
      UsbMetadata usb;
      usb.bus_path = current.filename().string();
      usb.vendor_id = *vendor;
      usb.product_id = *product;
      usb.manufacturer = read_file(current / "manufacturer").value_or("");
      usb.product = read_file(current / "product").value_or("");
      usb.serial = read_file(current / "serial").value_or("");
      for (fs::path child = device_path; child != current && child != child.root_path();
           child = child.parent_path()) {
        const std::string name = child.filename().string();
        if (name.starts_with(usb.bus_path + ":")) {
          usb.interface = name;
          break;
        }
      }
      return usb;
    }
    current = current.parent_path();
  }
  return std::nullopt;
}

std::string relative_sysfs_path(const fs::path& path, const fs::path& sys_root) {
  const auto canonical_sys = canonical_path(sys_root);
  if (!canonical_sys || !path_within(path, *canonical_sys))
    return {};
  std::error_code error;
  const fs::path relative = fs::relative(path, *canonical_sys, error);
  return error ? std::string{} : relative.generic_string();
}

std::string stable_key(const fs::path& device_path, const fs::path& sys_root, unsigned pcm_device) {
  std::ostringstream key;
  const std::string relative = relative_sysfs_path(device_path, sys_root);
  key << "sysfs:" << (relative.empty() ? device_path.generic_string() : relative);
  key << ":pcm" << pcm_device << 'c';
  return key.str();
}

std::string stable_id(std::string_view key) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char value : key) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }
  std::ostringstream encoded;
  encoded << "microphone:alsa:" << std::hex << std::setfill('0') << std::setw(16) << hash;
  return encoded.str();
}

std::optional<std::string> device_link(const fs::path& directory, std::string_view target_name,
                                       std::string_view public_directory) {
  std::error_code error;
  fs::directory_iterator it(directory, error);
  if (error)
    return std::nullopt;
  std::vector<std::string> names;
  for (const auto& entry : it) {
    const fs::path target = fs::read_symlink(entry.path(), error);
    if (error) {
      error.clear();
      continue;
    }
    if (target.filename() == target_name)
      names.push_back(entry.path().filename().string());
  }
  if (names.empty())
    return std::nullopt;
  std::sort(names.begin(), names.end());
  return std::string(public_directory) + "/" + names.front();
}

bool valid_card_id(std::string_view value) {
  return !value.empty() && std::all_of(value.begin(), value.end(), [](const unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '_' || ch == '-';
  });
}

void add_issue(nlohmann::json& issues, std::string code, std::string reason) {
  issues.push_back({{"code", std::move(code)}, {"reason", std::move(reason)}});
}

nlohmann::json availability(const std::map<std::string, std::string>& pcm_info,
                            nlohmann::json& issues) {
  nlohmann::json result = {{"state", "unknown"}};
  const auto count_it = pcm_info.find("subdevices_count");
  const auto available_it = pcm_info.find("subdevices_avail");
  const auto count =
      count_it != pcm_info.end() ? parse_unsigned(count_it->second) : std::optional<unsigned>{};
  if (!count) {
    add_issue(issues, "peripherals.availability_unknown",
              "ALSA did not report valid capture subdevice availability. Refresh after the "
              "device finishes initializing.");
    return result;
  }
  const auto available = available_it != pcm_info.end() ? parse_unsigned(available_it->second)
                                                        : std::optional<unsigned>{};
  if (!available || *count == 0 || *available > *count) {
    add_issue(issues, "peripherals.availability_unknown",
              "ALSA did not report valid capture subdevice availability. Refresh after the "
              "device finishes initializing.");
    return result;
  }
  result["subdevices"] = *count;
  result["subdevices_available"] = *available;
  result["state"] = *available == 0 ? "in_use" : "available";
  return result;
}

nlohmann::json usb_json(const UsbMetadata& usb) {
  nlohmann::json result = {
      {"vendor_id", usb.vendor_id},
      {"product_id", usb.product_id},
      {"bus_path", usb.bus_path},
  };
  if (!usb.interface.empty())
    result["interface"] = usb.interface;
  if (!usb.manufacturer.empty())
    result["manufacturer"] = usb.manufacturer;
  if (!usb.product.empty())
    result["product"] = usb.product;
  if (!usb.serial.empty())
    result["serial"] = usb.serial;
  return result;
}

} // namespace

std::vector<PeripheralRecord> discover_alsa_capture_peripherals(const AlsaDiscoveryRoots& roots) {
  const auto cards_contents = read_file(roots.proc_asound / "cards");
  if (!cards_contents) {
    throw_discovery_error(
        error_codes::kIoOpen,
        "ALSA discovery could not read " + (roots.proc_asound / "cards").string() +
            ". Confirm that ALSA procfs is mounted and the daemon can read /proc/asound.");
  }
  if (cards_contents->find("no soundcards") != std::string::npos)
    return {};

  const auto cards = parse_cards(*cards_contents);
  if (cards.empty()) {
    throw_discovery_error(error_codes::kIoParse,
                          "ALSA discovery could not parse any cards from " +
                              (roots.proc_asound / "cards").string() +
                              ". Refresh after the sound subsystem finishes initializing.");
  }
  std::error_code directory_error;
  const auto card_directories =
      numbered_entries(roots.proc_asound, std::regex(R"(^card(\d+)$)"), &directory_error);
  if (directory_error || card_directories.empty()) {
    throw_discovery_error(
        directory_error == std::errc::permission_denied ? error_codes::kPermissionDenied
                                                        : error_codes::kIoOpen,
        "ALSA discovery could not enumerate capture cards below " + roots.proc_asound.string() +
            ". Check /proc/asound permissions and refresh the catalog.");
  }
  std::vector<PeripheralRecord> records;
  for (const auto& [card_index, card_directory] : card_directories) {
    CardMetadata card;
    if (const auto found = cards.find(card_index); found != cards.end())
      card = found->second;
    if (const auto id = read_file(card_directory / "id"); id && !id->empty())
      card.id = *id;

    const fs::path class_card =
        roots.sys / "class" / "sound" / ("card" + std::to_string(card_index));
    const auto card_device = canonical_path(class_card / "device");
    if (!card_device) {
      throw_discovery_error(error_codes::kIoOpen,
                            "ALSA discovery could not resolve the stable sysfs path for card " +
                                std::to_string(card_index) +
                                ". Refresh after the device finishes initializing.");
    }
    const auto usb = usb_metadata(*card_device, roots.sys);
    const std::string connection = usb ? "usb" : "platform";
    const std::string control = "controlC" + std::to_string(card_index);
    const auto by_path = device_link(roots.dev / "snd" / "by-path", control, "/dev/snd/by-path");
    const auto by_id = device_link(roots.dev / "snd" / "by-id", control, "/dev/snd/by-id");

    directory_error.clear();
    const auto captures =
        numbered_entries(card_directory, std::regex(R"(^pcm(\d+)c$)"), &directory_error);
    if (directory_error) {
      throw_discovery_error(directory_error == std::errc::permission_denied
                                ? error_codes::kPermissionDenied
                                : error_codes::kIoOpen,
                            "ALSA discovery could not enumerate " + card_directory.string() +
                                ". Check /proc/asound permissions and refresh the catalog.");
    }
    for (const auto& [pcm_device, pcm_directory] : captures) {
      nlohmann::json issues = nlohmann::json::array();
      const auto info_contents = read_file(pcm_directory / "info");
      const auto pcm_info =
          info_contents ? parse_key_values(*info_contents) : std::map<std::string, std::string>{};
      if (!info_contents) {
        add_issue(issues, "peripherals.pcm_info_unreadable",
                  "ALSA capture metadata could not be read. Check /proc/asound permissions and "
                  "refresh the catalog.");
      }

      const std::string key = stable_key(*card_device, roots.sys, pcm_device);
      const std::string pcm_node =
          "/dev/snd/pcmC" + std::to_string(card_index) + "D" + std::to_string(pcm_device) + "c";
      const std::string pcm_name = pcm_info.contains("name") ? pcm_info.at("name") : "";
      const std::string name = !card.name.empty()  ? card.name
                               : !pcm_name.empty() ? pcm_name
                               : !card.id.empty()
                                   ? card.id
                                   : "ALSA capture PCM " + std::to_string(pcm_device);

      nlohmann::json capture_target = {
          {"card_id", card.id},
          {"device", pcm_device},
      };
      if (valid_card_id(card.id)) {
        capture_target["selector"] =
            "plughw:CARD=" + card.id + ",DEV=" + std::to_string(pcm_device);
      } else {
        add_issue(issues, "peripherals.capture_selector_unavailable",
                  "ALSA did not report a safe stable card ID. Refresh after the device finishes "
                  "initializing.");
      }

      nlohmann::json identity = {
          {"stable_key", key},
          {"card_index", card_index},
          {"pcm_node", pcm_node},
      };
      if (!card.id.empty())
        identity["card_id"] = card.id;
      if (!card.name.empty())
        identity["card_name"] = card.name;
      if (!card.driver.empty())
        identity["card_driver"] = card.driver;
      if (!pcm_name.empty())
        identity["pcm_name"] = pcm_name;
      if (by_path)
        identity["by_path"] = *by_path;
      if (by_id)
        identity["by_id"] = *by_id;
      if (usb)
        identity["usb"] = usb_json(*usb);

      std::vector<nlohmann::json> modes;
      if (const auto stream = read_file(card_directory / ("stream" + std::to_string(pcm_device))))
        modes = parse_capture_modes(*stream);
      if (modes.empty()) {
        add_issue(issues, "peripherals.capabilities_unavailable",
                  usb ? "The USB audio stream did not publish capture formats. Reconnect the "
                        "device and refresh the catalog."
                      : "This ALSA driver does not publish read-only capture formats. The daemon "
                        "does not open the PCM during discovery.");
      }

      PeripheralRecord record;
      record.id = stable_id(key);
      record.type = "microphone";
      record.provider = kAlsaProvider;
      record.details = {
          {"name", name},
          {"backend", "alsa"},
          {"connection", connection},
          {"capture_target", std::move(capture_target)},
          {"identity", std::move(identity)},
          {"modes", std::move(modes)},
          {"availability", availability(pcm_info, issues)},
      };
      if (!issues.empty())
        record.details["issues"] = std::move(issues);
      records.push_back(std::move(record));
    }
  }

  std::sort(records.begin(), records.end(),
            [](const auto& left, const auto& right) { return left.id < right.id; });
  return records;
}

std::vector<PeripheralRecord> discover_alsa_capture_peripherals() {
  return discover_alsa_capture_peripherals(AlsaDiscoveryRoots{});
}

} // namespace simaai::neat::peripherals_internal
