"""Phase 6 slice (S9): pyneat.power board power telemetry submodule."""

from __future__ import annotations

import pyneat


def test_power_surface_present():
  power = pyneat.power
  for name in (
      "PowerFieldReading",
      "PowerRailReading",
      "PowerSnapshot",
      "PowerMonitor",
      "read_power_snapshot",
      "default_modalix_som_power_rails",
      "default_modalix_dvt_power_rails",
      "detect_default_power_monitor_profile",
      "power_rails_for_profile",
  ):
    assert hasattr(power, name), name


def test_power_reuses_existing_types_via_alias():
  # S9: already-bound config/summary types are aliased into pyneat.power, NOT relocated/rebound.
  assert pyneat.power.PowerSummary is pyneat.PowerSummary
  assert pyneat.power.PowerMonitorOptions is pyneat.PowerMonitorOptions
  assert pyneat.power.PowerRailConfig is pyneat.PowerRailConfig
  assert pyneat.power.PowerMonitorProfile is pyneat.PowerMonitorProfile
  assert pyneat.power.modalix_som_power_monitor_options is pyneat.modalix_som_power_monitor_options


def test_power_rail_tables_are_pure_data():
  som = pyneat.power.default_modalix_som_power_rails()
  assert len(som) > 0
  assert all(isinstance(rail, pyneat.PowerRailConfig) for rail in som)
  assert som[0].name  # rails carry a human-readable name

  by_profile = pyneat.power.power_rails_for_profile(pyneat.power.PowerMonitorProfile.ModalixSom)
  assert len(by_profile) > 0


def test_detect_default_profile():
  profile = pyneat.power.detect_default_power_monitor_profile()
  assert isinstance(profile, pyneat.PowerMonitorProfile)


def test_power_monitor_lifecycle_disabled():
  # Default options are disabled (no I2C), so this is hardware-safe and deterministic.
  monitor = pyneat.power.PowerMonitor()
  assert monitor.running() is False
  summary = monitor.summary()
  assert isinstance(summary, pyneat.PowerSummary)
