---
title: Install PyNeat
description: Install the PyNeat wheel in a custom Python virtual environment
sidebar_position: 3
---

:::tip PyNeat is already installed with Neat Library
PyNeat is bundled with the Neat Library and is installed automatically when you
install the Neat Library.

By default, it is installed in a virtual environment at `~/pyneat`. You can skip
this page unless you want to install PyNeat in a custom virtual environment, such
as a separate venv or conda environment on the DevKit.
:::

Run the steps below on the DevKit. This instruction does not install or update the runtime `.deb` packages, so run it where the
matching Neat Library runtime is already installed.

## Download the Wheel

<ShellCommand prompt="devkit">
sima-cli neat install core -t pyneat
</ShellCommand>

To download the wheel for a specific Neat Library release, include the version.

For Neat Library 0.2.1:

<ShellCommand prompt="devkit">
sima-cli neat install core@0.2.1 -t pyneat
</ShellCommand>

## Create a Python Environment

Create and activate a virtual environment with the environment's `python3`:

<ShellCommand prompt="devkit">
python3 -m venv ~/my-neat-env
source ~/my-neat-env/bin/activate
</ShellCommand>

## Install the Wheel

<ShellCommand prompt="devkit">
pip install ./pyneat-*.whl
</ShellCommand>

For supported Neat Library, SDK, and DevKit software combinations, see the
[Compatibility Guide](/getting-started/compatibility/).

## Next Step

Continue to [Neat CLI](/getting-started/neat-library/neat-cli/) to inspect the
installed environment.
