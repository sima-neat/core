#include "pipeline/PowerTelemetry.h"
#include "pipeline/internal/PowerTelemetryInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdint>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct Call {
  std::string op;
  int bus = 0;
  std::uint8_t addr = 0;
  std::uint8_t value = 0;
};

class FakeI2cBackend final : public simaai::neat::pipeline_internal::PowerI2cBackend {
public:
  std::map<std::tuple<int, std::uint8_t, std::uint8_t, std::uint8_t>, std::uint8_t> regs;
  std::map<std::tuple<int, std::uint8_t, std::uint8_t>, std::string> failures;
  std::map<std::pair<int, std::uint8_t>, std::uint8_t> current_page;
  std::vector<Call> calls;

  bool set_page(int bus, std::uint8_t addr, std::uint8_t page, std::string& error) override {
    calls.push_back({"set_page", bus, addr, page});
    const auto key = std::make_tuple(bus, addr, page);
    const auto it = failures.find(key);
    if (it != failures.end()) {
      error = it->second;
      return false;
    }
    current_page[{bus, addr}] = page;
    return true;
  }

  bool read_register(int bus, std::uint8_t addr, std::uint8_t reg, std::uint8_t& value,
                     std::string& error) override {
    calls.push_back({"read", bus, addr, reg});
    const std::uint8_t page = current_page[{bus, addr}];
    const auto fail = failures.find(std::make_tuple(bus, addr, reg));
    if (fail != failures.end()) {
      error = fail->second;
      return false;
    }
    const auto key = std::make_tuple(bus, addr, page, reg);
    const auto it = regs.find(key);
    if (it == regs.end()) {
      error = "missing fake register";
      return false;
    }
    value = it->second;
    return true;
  }
};

void require_near(double actual, double expected, const std::string& label) {
  const double diff = std::abs(actual - expected);
  require(diff < 1e-9,
          label + " expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
}

} // namespace

RUN_TEST("unit_power_telemetry_test", ([] {
           using namespace simaai::neat;
           using simaai::neat::pipeline_internal::read_power_snapshot_with_backend;

           const auto rails = default_modalix_som_power_rails();
           require(rails.size() == 8, "expected default SOM rail table");
           require(rails[0].i2c_addr == 0x4f && rails[0].page == 0x00,
                   "expected 0x4f page0 rail first");
           require(rails[0].iout_exponent == -3 && rails[0].pout_exponent == -2,
                   "expected special SoC/DDR current/power exponents");
           require(rails[4].i2c_addr == 0x4d && rails[4].page == 0x00, "expected 0x4d page0 rail");
           require(rails[4].iout_exponent == -3 && rails[4].pout_exponent == -2,
                   "expected special MLA current/power exponents");
           const auto convenience = modalix_som_power_monitor_options(50);
           require(convenience.enabled && convenience.sample_interval_ms == 50 &&
                       convenience.rails.size() == rails.size(),
                   "expected convenience Modalix SOM power options");
           require(convenience.profile == PowerMonitorProfile::ModalixSom,
                   "expected SOM convenience profile");
           const auto dvt_rails = default_modalix_dvt_power_rails();
           require(dvt_rails.size() == 1, "expected single stable DVT rail table");
           require(dvt_rails[0].i2c_bus == 4 && dvt_rails[0].i2c_addr == 0x4d &&
                       dvt_rails[0].page == 0x00,
                   "expected stable DVT PMIC 0x4d page0 on bus 4");
           require(dvt_rails[0].pout_exponent == -2, "expected verified DVT page0 POUT exponent");
           const auto auto_options =
               board_power_monitor_options(75, PowerMonitorProfile::ModalixDvt);
           require(auto_options.enabled && auto_options.sample_interval_ms == 75 &&
                       auto_options.profile == PowerMonitorProfile::ModalixDvt &&
                       auto_options.rails.size() == dvt_rails.size(),
                   "expected board power options for DVT profile");
           require(power_monitor_profile_name(PowerMonitorProfile::Auto) == "auto",
                   "expected profile string helper");
           const auto detected_profile = detect_default_power_monitor_profile();
           require(detected_profile == PowerMonitorProfile::ModalixSom ||
                       detected_profile == PowerMonitorProfile::ModalixDvt,
                   "expected detectable built-in profile");

           PowerMonitorOptions options;
           options.enabled = true;
           options.profile = PowerMonitorProfile::ModalixSom;
           options.rails = {rails[0], rails[1]};

           auto fake = std::make_shared<FakeI2cBackend>();
           fake->regs[{3, 0x4f, 0x00, 0x96}] = 8; // page0: 8 * 2^-2 = 2 W

           const PowerSnapshot snap = read_power_snapshot_with_backend(options, fake);
           require(snap.rails.size() == 2, "expected two rail readings");
           require(!snap.rails[0].voltage_v.available && !snap.rails[0].current_a.available,
                   "voltage/current are intentionally not sampled by PMBus power telemetry");
           require_near(snap.rails[0].power_w.value, 2.0, "special power scaling");
           require(!snap.rails[1].power_w.error.empty(),
                   "missing fake register should produce non-fatal field error");
           require_near(snap.total_watts, 2.0, "total watts should use available rails");

           require(fake->calls.size() >= 4, "expected page and POUT register calls");
           require(fake->calls[0].op == "set_page" && fake->calls[0].addr == 0x4f &&
                       fake->calls[0].value == 0x00,
                   "expected first page selection");
           require(fake->calls[1].op == "read" && fake->calls[1].value == 0x96,
                   "expected POUT-only read after page selection");

           PowerSummary summary;
           summary.enabled = true;
           summary.samples = 1;
           summary.duration_seconds = 0.25;
           summary.total_avg_watts = 2.0;
           summary.total_min_watts = 2.0;
           summary.total_max_watts = 2.0;
           summary.energy_joules = 0.5;
           require_contains(format_power_summary(summary), "PowerStats:", "text power report");
           require_contains(power_summary_to_json(summary), "\"total_avg_watts\": 2.000000",
                            "json power report");

           PowerSummary disabled;
           require(format_power_summary(disabled).empty(), "disabled power summary is omitted");
         }));
