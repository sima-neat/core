# Offline validation baselines

`ci.py` assembles all four PCIe host variants and standalone Core from the
selected published artifacts. Every variant validates its ARM64 payload on the
same provisioned eLxr card baseline with `Docker --network=none`.

The factory B2121 image in `baselines.json` predates the sysroot used by newer
Scheduler cohorts. For example, its `libsimaai_heap.so.2` has the expected SONAME
but lacks `simaai::cache_line_size()`, required by the pre2372 Runner. Installing
Core and its Python wheel succeeds, then `import pyneat` fails while loading
`libneatdmabufmemory`.

Before dependency collection, `ci.py` builds `CardBaseline.Dockerfile` from the
verified factory image and checksum-pinned SDK/shared provisioning tools.
`sysroot_baseline.py` authenticates the incoming Scheduler tooling and its
installed `dependency-policy.json` against the selected package receipt. It
resolves that policy's firmware through the shared resolver and requires the
result to equal the exact sysroot version recorded by Scheduler. The SDK's
unmodified `sysroot update` command then installs the complete selected sysroot.
The updater reads the same by-hash package index used for resolution.

Scheduler's authenticated `describe` command identifies the native runtime
providers required by the payload. Their resolved SDK DEBs and every already-installed counterpart in the SDK
cohort are installed into the disposable card root before offline collection
and validation. This keeps reverse dependencies such as `python3-tvm` and its
exact `tvm` pin together. Every installed version is checked against its
resolved archive, and package removal is forbidden. This provisions
all declared providers together instead of adding a special SimaHeap dependency
or allowing a newer mirror candidate to replace the selected sysroot. The
pinned updater currently supports pre-release firmware policies; other policies
fail explicitly rather than reusing an unrelated baseline.

This is CI Platform provisioning. Platform DEBs remain outside the publication,
offline repository and installer archive. Customer devices still require a
compatible provisioned Platform; the published Scheduler admission tool checks
the actual installed providers. Do not preload replacement libraries or bypass
the PyNeat import test. Baseline resolution and provider reports are retained
under `/var/lib/neat-offline-ci` in the validation image.

The tooling input contains unmodified SDK `sysroot.sh`,
`setup-sdk-sysroot.sh`, `simaai_setup_sdk.py`,
`validate-sysroot-package-versions.sh`, `platform-package-patterns.txt`, and the
shared `resolve_sysroot.py`. To refresh it, archive those files from reviewed
SDK/shared commits under a new content-addressed
`core/offline-build-input-<sha256>-sdk-tools.tar.gz` key, then update its checksum
and provenance in `baselines.json`. Validate the selected Core distribution on
the resulting clean baseline before publishing offline metadata.
