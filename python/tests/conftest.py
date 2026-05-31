"""Shared pytest configuration for installed pyneat tests."""

import os


# The Python runtime tests exercise the actual runner outputs explicitly.  The
# hidden build-time startup preflight does an extra push/pull before each run,
# which can exhaust or time out on the DevKit's finite EV/MLA channels when the
# full installed suite is run as one pytest process.  Keep the functional checks
# deterministic by default while preserving an opt-in escape hatch for focused
# preflight debugging.
os.environ.setdefault("SIMA_INPUTSTREAM_PREFLIGHT_RUN", "0")
