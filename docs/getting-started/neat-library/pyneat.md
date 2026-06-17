---
title: Install PyNeat
description: Install the PyNeat wheel in a custom Python virtual environment
sidebar_position: 3
---

The standard Neat Library installer provisions a default PyNeat virtual
environment at `~/pyneat` on the DevKit. Use this page only when you need a
custom Python environment.

The Neat Library install artifacts include a PyNeat wheel. If you need a custom
venv, conda environment, or other Python 3.11 setup, download the wheel and
install it into that environment.

Use this flow when the matching Neat Library release artifacts are already
installed. The PyNeat-only target downloads the wheel, but it does not install
or update the Neat `.deb` packages.

## Download the Wheel

<ShellCommand prompt="sdk-or-devkit">
sima-cli neat install core -t pyneat
</ShellCommand>

To download the wheel for a specific Neat Library release, include the version.

For Neat Library 0.1.0:

<ShellCommand prompt="sdk-or-devkit">
sima-cli neat install core@v0.1.0 -t pyneat
</ShellCommand>

For Neat Library 0.2.0:

<ShellCommand prompt="sdk-or-devkit">
sima-cli neat install core@0.2.0 -t pyneat
</ShellCommand>

## Create a Python Environment

Create and activate your target Python 3.11 environment:

<ShellCommand prompt="sdk-or-devkit">
python3.11 -m venv ~/my-neat-env
source ~/my-neat-env/bin/activate
</ShellCommand>

## Install the Wheel

<ShellCommand prompt="sdk-or-devkit">
pip install ./pyneat-*.whl
</ShellCommand>

The published wheel targets **CPython 3.11** on **`linux_aarch64`**, so install
it into a matching environment. For supported Neat Library, SDK, and DevKit
software combinations, see the
[Compatibility Guide](/getting-started/compatibility/).

## Next Step

Continue to [Neat CLI](/getting-started/neat-library/neat-cli/) to inspect the
installed environment.
