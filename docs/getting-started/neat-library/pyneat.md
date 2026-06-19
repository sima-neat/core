---
title: Install PyNeat
description: Install the PyNeat wheel in a custom Python virtual environment
sidebar_position: 3
---

:::note PyNeat is already installed on the DevKit
PyNeat is installed with the Neat Library on the DevKit, in a default virtual
environment at `~/pyneat`. You do not need the steps on this page unless you want
PyNeat in a custom virtual environment (for example a separate venv or conda
environment) in the SDK container or on the DevKit.
:::

The standard Neat Library installer provisions a default PyNeat virtual
environment at `~/pyneat` on the DevKit. Use this page only when you need a
custom Python environment.

The Neat Library install artifacts include a PyNeat wheel. If you need a custom
venv, conda environment, or other Python setup, download the wheel and install it
into that environment.

Run the steps below **in the environment where you will use PyNeat** — the SDK
container or the DevKit. `sima-cli neat install core -t pyneat` downloads the
wheel that matches that environment's platform, so it always matches the Python
and architecture you install into. The PyNeat-only target downloads the wheel; it
does not install or update the runtime `.deb` packages, so run it where the
matching Neat Library runtime is already installed.

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

Create and activate a virtual environment with the environment's `python3`:

<ShellCommand prompt="sdk-or-devkit">
python3 -m venv ~/my-neat-env
source ~/my-neat-env/bin/activate
</ShellCommand>

## Install the Wheel

<ShellCommand prompt="sdk-or-devkit">
pip install ./pyneat-*.whl
</ShellCommand>

The wheel matches the environment where you downloaded it: on the DevKit it is a
`linux_aarch64` wheel, and in the SDK container it matches the container's
architecture (`x86_64` or `arm64`). Install it into a virtual environment built
with that same environment's `python3` — do not move a DevKit (`aarch64`) wheel
into an x86_64 SDK container, or vice versa. For supported Neat Library, SDK, and
DevKit software combinations, see the
[Compatibility Guide](/getting-started/compatibility/).

## Next Step

Continue to [Neat CLI](/getting-started/neat-library/neat-cli/) to inspect the
installed environment.
