# Core SDK and DevKit dependency admission

Core uses `scheduler: snap` to select the published current-branch Scheduler,
falling back to develop only when the branch is confirmed unpublished. Scheduler's
verified publication owns the exact Runner, Kernels and Codecs cohort; Core does
not select independent latest versions of those providers. LLiMa uses its own snap
selection and must declare matching firmware, platform and SDK channel.

The selected Scheduler receipt supplies the SDK firmware and exact sysroot. An
explicit `sdk.sysroot-version` supports an official release such as B2499 / 3.0.0.
Core does not duplicate Scheduler's firmware declaration in its manifest.

For official releases, `prepare_scheduler_sdk.sh` checks the cached SDK receipt,
Palette/base-files inventory, headers, firmware identity and absence of an
overlay. A mismatch requires a refreshed SDK cache. Even `--update-sysroot` does
not send official releases through daily/pre-release overlay tooling. Exact
Runner platform archives are downloaded through the signed official release
repository into a private APT cache; their hashes remain checked by Scheduler's
coordinator. Existing pre-release/develop update paths remain available.

Core already uses the shared `.github` Modalix resolver and Vulcan build on
`main`. The resolver reads the selected Scheduler policy artifact and chooses the
compatible physical firmware separately from the SDK identity. Each DevKit
verifies `devkit_firmware` before installation, with fallback disabled. Existing
capability-aware sharding and 16 GB GenAI admission remain in force.
