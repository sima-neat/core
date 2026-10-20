# Firmware selection for Core CI

Core uses `deps/manifest.json` to select Internals and LLiMa artifacts. With the
`snap` policy, branch builds first request artifacts from the same branch;
existing dependency-resolution rules determine whether a fallback is used.

The DevKit runner resolver runs after the Core build and reads
`internals-manifest.json` from the `neat-core-dist` artifact. This is the manifest
of the Internals package actually consumed by the build, rather than a second
lookup of a mutable branch. Core does not duplicate its `elxr_version` selection.

Runner discovery prefers an online runner with the exact firmware label and all
required hardware, RTSP, and Vulcan environment labels. A busy online matching
runner still qualifies. `allow-version-fallback: true` permits dropping only the
firmware label when no exact match is online. Discovery errors and the absence
of a runner satisfying the remaining labels still fail. The shared resolver's
`runner-resolution` artifact records the choice; the device job reports actual
firmware and checks exact matches before installing test packages.

Older Internals artifacts without `elxr_version` retain platform-group routing.
For master firmware manifests, Core maps `3.0.0_master_B2263` to the exact APT
sysroot receipt `3.0.0~pre2263`. Legacy sysroot receipts remain supported. Daily
firmware-only manifests currently fail sysroot setup because their timestamp and
commit cannot be inferred from the build number alone.

The runner resolver currently references `codex/elxr-runner-selection` in the
shared `.github` repository while that reusable workflow change is under review.
Keep the workflow ref and `shared_workflow_ref` aligned when switching to `main`.
These jobs select runners and install test packages; they do not flash firmware.
