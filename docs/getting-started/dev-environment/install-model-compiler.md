---
title: Install Model Compiler
description: Install Model Compiler in the Neat SDK or on a supported standalone host
sidebar_position: 3
---

:::tip Start here only when installing Model Compiler separately
Model Compiler installation is opt-in during SDK installation. Use this page
only if you skipped that prompt, want to install a newer compatible Model
Compiler, or need to install Model Compiler outside of the SDK on a supported
host.
:::

The Model Compiler quantizes and compiles ONNX models so they can run on
SiMa.ai's MLA. It is **required** when you compile or quantize models yourself,
including GenAI models, and is **optional** only if you exclusively use
precompiled model packages.

During SDK install/setup, `sima-cli` prompts you to install the matching Model
Compiler as an extension inside the Neat SDK. You can also install it later,
either inside the Neat SDK container or standalone on a supported Ubuntu host.
For supported version combinations and standalone host requirements, see
[Compatibility](/getting-started/compatibility/#model-compiler).

## Install Inside the SDK

If you skip Model Compiler during SDK setup, install it later from inside the
Neat SDK. Run the command that matches your Neat SDK container architecture. To
check it, run `uname -m` inside the SDK shell: `aarch64` means use the `arm64`
command, and `x86_64` means use the `amd64` command.

For `amd64` Neat SDK containers:

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli install -v 2.1.2 tools/model-compiler/amd64
</ShellCommand>

For `arm64` Neat SDK containers:

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli install -v 2.1.2 tools/model-compiler/arm64
</ShellCommand>

After installation, activate the compiler environment from inside the Neat SDK
shell:

<ShellCommand prompt="username@neat-sdk-latest">
activate-model-compiler
</ShellCommand>

To return to the default Neat SDK shell, run:

<ShellCommand prompt="username@neat-sdk-latest">
deactivate-model-compiler
</ShellCommand>

## Install on a Standalone Host

Standalone installation is supported only on host environments listed in
[Compatibility](/getting-started/compatibility/#model-compiler). Run the
matching `sima-cli install` command from the supported host environment. To
check the host architecture, run `uname -m`: `x86_64` uses the `amd64` command,
and `aarch64` uses the `arm64` command.

For Model Compiler 2.1.2 on `amd64` hosts:

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.1.2 tools/model-compiler/amd64
</ShellCommand>

For Model Compiler 2.1.2 on `arm64` hosts:

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.1.2 tools/model-compiler/arm64
</ShellCommand>

For Model Compiler 2.0.0 on `amd64` hosts:

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.0.0 tools/model-compiler/amd64
</ShellCommand>

## Next Step

Continue to [Compile a Model](/compile-a-model/).
