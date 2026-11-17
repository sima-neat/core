# Core SDK and DevKit dependency admission

Core uses `scheduler: snap` to select the published current-branch Scheduler,
falling back to develop only when the branch is confirmed unpublished. Scheduler's
verified publication owns the exact Runner, Kernels and Codecs cohort; Core does
not select independent latest versions of those providers. LLiMa uses its own snap
selection and must declare matching firmware, platform and SDK channel.

The selected Scheduler receipt supplies the SDK firmware and exact sysroot. An
explicit `sdk.sysroot-version` supports an official release such as B2499 / 3.0.0.
Core does not duplicate Scheduler's firmware declaration in its manifest.

`prepare_scheduler_sdk.sh` validates the active target firmware and exact
Palette/base-files versions. Matching target overlays are accepted even when the
container's original SDK label is older. Official releases reuse the active
sysroot; daily/pre-release updates remain explicit through `--update-sysroot`.
Scheduler compares the observed `build_platform` with its own and Runner's build
records, without downloading platform DEBs. Core records that identity, refreshes
validation after compilation, and checks the final configured sysroot and target.
Old producers missing this record must be rebuilt in Runner → Scheduler → Core
order. NEAT dependency package integrity and target qualification remain separate.

Core already uses the shared `.github` Modalix resolver and Vulcan build on
`main`. The resolver reads the selected Scheduler policy artifact and chooses the
compatible physical firmware separately from the SDK identity. Each DevKit
verifies `devkit_firmware` before installation, with fallback disabled. Existing
capability-aware sharding and 16 GB GenAI admission remain in force.
