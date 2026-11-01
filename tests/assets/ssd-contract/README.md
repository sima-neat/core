# Registry-backed SSD contract fixtures

`fixtures.json` pins the six published SSD-MobileNet V1/V2/V3 packages, in INT8
and BF16, used by `unit_boxdecode_kernel_recipe_test`. The source is the standard
Model Registry snapshot built with Model Compiler 3.0.0, not locally copied JSON.

The `prepare_test_fixtures` target downloads missing packages with
`sima-cli models download`, independently checks their package run ID, profile,
SHA-256 and size, then reads exactly one authoritative `mpk.json` or `*_mpk.json`.
It never extracts model executables or sidecar JSON. Missing or ambiguous
manifests and mismatched packages are preparation failures, not skips.

Generated manifests and provenance live in
`<build>/test-assets/model-archive/registry/ssd-contract/` and are included in the
test extras at the corresponding relocatable runtime-manifest location.
The native tests need neither the CLI nor network access. Their independent
shape, stride and capacity expectations remain in C++; the downloaded manifest
does not define its own expected answer.

The CLI currently selects a branch catalog, not an immutable run selector. The
qualification branch identifies the published validation snapshot of Models
PR #495. Selection must match the pinned package; there is no fallback to
`latest`, another branch, ModelZoo or an unverified local archive. After the
models are published on `develop`, update the branch and package pins together
in a reviewed change rather than silently following new builds.

Archives are cached by SHA-256 under `<build>/model-registry-cache/`. They are
not included in the test extras. The same verified archives can be passed to
the existing `--ssd-model-carrier-contract <archive>` cold Model-route check;
manifest-only tests do not qualify execution of those model programs.

For explicit offline fixture preparation, prepopulate that cache with verified
archives named `<sha256>.tar.gz`, then run:

```sh
python3 tests/tools/prepare_registry_mpk_fixtures.py \
  --fixtures tests/assets/ssd-contract/fixtures.json \
  --root build/test-assets/model-archive/registry/ssd-contract \
  --cache build/model-registry-cache --offline
```
