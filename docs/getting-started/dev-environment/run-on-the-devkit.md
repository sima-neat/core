---
title: Run on the DevKit
description: Run SDK-built applications on a paired DevKit with dk
sidebar_position: 4
---

The SDK includes the `dk` helper, also known as `devkit-run`, to run ARM64
executables on the DevKit from inside the SDK shell.

When you invoke `dk`, the SDK runs the command on the paired DevKit and
translates paths so file arguments from the container resolve correctly on the
DevKit.

## Usage

<ShellCommand prompt="sdk-container">
dk <file> [args...]
</ShellCommand>

Run an ARM64 executable built in the SDK workspace:

<ShellCommand prompt="sdk-container">
dk build/sima_neat_hello
</ShellCommand>

Run a Python entry point on the paired DevKit:

<ShellCommand prompt="sdk-container">
dk hello_neat.py
</ShellCommand>

For Python scripts, `dk` runs the script on the paired DevKit and uses the DevKit
PyNeat runtime environment. The SDK remains useful as a unified workspace and
orchestration environment, but Python-only workflows do not require the C++
cross-compilation toolchain.

:::note Where `dk` Comes From
`dk` is a shell function defined in `~/devkit-sync.rc` inside the SDK container.
The shell loads it through `~/.bashrc`, so it is available in interactive
sessions.
:::

## Next Step

To install or update the library/runtime itself, continue to
[Neat Library](/getting-started/neat-library/).
