# Environment Layout

Use this when location matters.

## Modalix DevKit

- Use installed paths under `/usr/include`, `/usr/lib`, and `/usr/share/sima-neat`.
- Use installed headers and `find_package(SimaNeat REQUIRED)` by default.

## eLxr SDK

- Cross builds usually use `SYSROOT=/opt/toolchain/aarch64/modalix`.
- Installed package content is typically under `$SYSROOT/usr/include`, `$SYSROOT/usr/lib`, and `$SYSROOT/usr/share/sima-neat`.
- If source trees are mounted, prefer `/neat-resources/core-src` and `/neat-resources/apps-src` for code navigation and edits.
- For runtime checks from SDK, prefer `dk /workspace/...`.
- `dk` rules:
  - target must be under `/workspace`
  - non-Python binaries must be ARM64/aarch64
  - Python is supported and `dk` tries to activate `pyneat` on the DevKit
  - stdout/stderr is streamed back to the SDK shell

## Source-of-truth order

1. If the user is editing code and `/neat-resources/core-src` or `/neat-resources/apps-src` exists, use those trees first.
2. If the task is about package usage, build integration, or installed assets, inspect the installed/sysroot paths.
3. If the task needs runtime validation from SDK, prefer `dk` to execute the built target on DevKit.
4. For generated examples, keep includes on the public installed surface even when reading source trees for reference.
