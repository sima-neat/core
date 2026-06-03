# Compatibility — Investigation Findings

Investigation for the §3 Compatibility-matrix doc. Every value below is verified
against **code / packaging / build artifacts**, not docs (docs have drifted —
see `error_codes.md` / `env_vars.md`). Confidence is marked per finding.

Three axes, as scoped:
- **Axis 0 — Neat C++ core:** targets Modalix; mostly a constant.
- **Axis 1 — Container / Neat SDK:** the host-side dev env — where compatibility actually varies.
- **Axis 2 — pyneat:** the Python bindings.

---

## Axis 0 — Neat C++ core (Modalix DevKit)

| Property | Value | Source | Conf |
|---|---|---|---|
| Architecture | **arm64 / aarch64 only** | all `dist/neat-*_arm64.deb` control metadata | ✅ |
| Package version line | **2.0.0** (`sima-neat` = `2.0.0+beta02.graph.changes.<sha>`) | `dist/*.deb` `Version:` | ✅ |
| MLA runtime backend | `simaai-mlart-modalix` **OR** `simaai-mlart-davinci` | `neat-appcomplex` `Depends:` | ✅ |
| glibc floor | **libc6 ≥ 2.34** | `neat-runtime` / `neat-appcomplex` `Depends:` | ✅ |
| OpenCV | **4.6.x** (`libopencv-core406`) | `sima-neat` `Depends:` | ✅ |
| GStreamer | **1.0** (`libgstreamer1.0-0`, `libgstrtspserver-1.0-0`) | `sima-neat` `Depends:` | ✅ |
| C++ standard | **C++20** | `CMakeLists.txt`, README | ✅ |
| Board platform version | modalix, **2.0.0** | `registry.json` (`type: board, compatible_with: [modalix]`) | ✅ |

### Firmware ↔ runtime matching is by shared build hash (the current mechanism)
The whole `neat-*` set is built together from one **internals** commit and versioned with that commit's **hash**. All members of a matched build carry the same suffix:

```
neat-runtime_2.0.0-develop-ed1bec0d7782_arm64.deb
neat-ev74-firmware_2.0.0-develop-ed1bec0d7782_arm64.deb
neat-gst-plugins_2.0.0-develop-ed1bec0d7782_arm64.deb
neat-appcomplex_2.0.0-develop-ed1bec0d7782_arm64.deb
neat-internals-dev_2.0.0-develop-ed1bec0d7782_arm64.deb
```

(Other builds: `…-beta-changes-0dcb08e46966`, `…-beta-changes-a08eb013645d`.) **The shared hash is the coupling key** — installing the same-hash set gives a matched runtime + firmware. The latest internals build (the `internals` / "clean-internals" folder) is the source of the matched set.

- Within a hash-matched install, runtime and firmware match **by construction**.
- The release-packaged `dist/*.deb` carry the **plain** `2.0.0` (hash stripped from the `Version:` string), which is why the Debian `Depends:` don't show the firmware pin — the matching lives in the **install-the-matched-set** process, keyed by the hash, not in apt `Depends:`.
- **The failure we reproduced** (`No channel available` / missing upsample kernel) = the DevKit had a firmware deb from a **different hash** than the runtime — an un-matched set, i.e. a partial update (new runtime, old firmware).

> **Rule (current):** deploy the matched internals bundle (same hash) together; never update `neat-runtime` without updating `neat-ev74-firmware` to the same hash. The hash is the source of truth for "are these compatible."

### Resolved: "Modalix 3.0" is just a product name — no hardware gate
Per product: **any Modalix DevKit runs Neat.** The "3.0+" in [installation/index.mdx](docs/getting-started/installation/index.mdx) is a **marketing/product name**, not a compatibility constraint. The matrix row is simply "**any Modalix DevKit**" — there is no minimum hardware version to gate on. (`registry.json`'s `2.0.0` under `type: board` is the *software platform* version, unrelated to the hardware name.)

---

## Axis 1 — Container / Neat SDK (host dev environment)

This is where compatibility genuinely varies. The SDK runs **in a Docker container** on the host; you develop in the container and deploy to a paired DevKit.

| Property | Value | Source | Conf |
|---|---|---|---|
| Host OS — Linux | **Ubuntu 22.04 or 24.04** | `reference/elxr-sdk-host-setup/ubuntu.md` | ✅ |
| Host OS — Windows | **Windows 11** + WSL2 + Docker Engine *inside* WSL | `reference/elxr-sdk-host-setup/windows.md` | ✅ |
| Host OS — macOS | **macOS** + Colima + Docker | `reference/elxr-sdk-host-setup/macos.md` | ✅ |
| SDK type | **eLxr SDK** (Palette SDK), Docker image | `build.sh` `detect_elxr_sdk()`, `/etc/sdk-release` | ✅ |
| SDK version (this image) | `SDK Version = 2.0.0_Palette_SDK_neat_main_780365a` | `/etc/sdk-release` | ✅ |
| eLxr version (this image) | `eLXr Version = 2.0.0_release_neat_main_780365a` | `/etc/sdk-release` | ✅ |
| Cross compiler | **`aarch64-linux-gnu-g++` 12.2.0** (Debian 12.2.0-14) | `aarch64-linux-gnu-g++ --version` | ✅ |
| Sysroot | **`/opt/toolchain/aarch64/modalix`** | `build.sh`, `CMakeCache` | ✅ |
| `sima-cli` | **use the latest** (no pinned floor — current latest is the target) | per product | ✅ |

**Notes:**
- The SDK is detected at build time from `/etc/sdk-release` (fields `SDK Version`, `eLXr Version`); `build.sh` flips to cross-compile mode (`ELXR_SDK=ON`) when both are present.
- macOS/Windows run the SDK **containerized** (Colima / WSL-Docker); native macOS is only relevant for **docs builds**, not the framework.

---

## Axis 2 — pyneat (Python bindings)

| Property | Value | Source | Conf |
|---|---|---|---|
| **Prebuilt wheel** | **CPython 3.11, aarch64 only** (`pyneat-…-cp311-cp311-linux_aarch64.whl`) | `dist/*.whl`, `registry.json` resources | ✅ |
| From-source floor | `requires-python >= 3.9`; CMake `find_package(Python 3.8 REQUIRED)` | `pyproject.toml`, `python/CMakeLists.txt` | ✅ |
| Runtime dep — NumPy | **`numpy >= 1.24, < 2`** (NumPy **2.x not supported**) | `pyproject.toml` `dependencies` | ✅ |
| Runtime dep — Torch | optional extra: `torch >= 2.3.0` (DLPack interop) | `pyproject.toml` `[optional-dependencies]` | ✅ |
| Build backend | `scikit-build-core >= 0.10.0`, `nanobind == 2.5.0` | `pyproject.toml` `build-system` | ✅ |
| Platform | **`linux_aarch64` only** — no x86 pyneat wheel | `dist/`, registry (only aarch64 pyneat wheels exist) | ✅ |
| Install location | `/media/nvme/pyneat` if writable NVMe, else `$HOME/pyneat`; always symlinked to `$HOME/pyneat` | `install_neat_framework.sh` `resolve_venv_dir()` / `ensure_home_pyneat_symlink()` | ✅ |
| Activation | `source ~/pyneat/bin/activate` (on the DevKit) | install_neat_framework.sh / tutorials | ✅ |

### The cp311 gate is real (and undocumented for users)
`pyproject.toml` says `>=3.9` and CMake floor is 3.8 — but the **shipped artifact** is `cp311` only. So: **install the prebuilt wheel → you need Python 3.11 on the DevKit.** Building from source can target ≥3.9. This 3.11 gate is **not stated** in any user-facing install doc → matrix should call it out.

### Two distinct Python worlds — do not conflate
- **pyneat** (this repo, runtime): **aarch64 / cp311**, runs **on the DevKit**. The host SDK cross-builds it; it installs on the DevKit even when you run the installer from inside the SDK container.
- **Model-compiler stack** (`sima_frontend`, `sima_mlc`, `sima_tvm`, `sima_mppe`, `sima_ml_kernels`, …): **x86_64 / cp310 & cp312**, runs on the **host / Palette**. (Seen in `registry.json` resources.)

These are separate toolchains with separate Python ABIs. The compatibility matrix must keep them in separate rows or it will mislead.

---

## Versions must be DERIVED, not hardcoded

Decision: the matrix page must contain **no hand-typed version numbers**. Every
version already lives in a real source; the build should emit one manifest and
the docs render from it, so a release bumps versions in exactly one place (the
packages) and the docs follow automatically.

**Source of truth → manifest → docs:**

| Field | Authoritative source (already exists) |
|---|---|
| Neat **internals build hash** (the firmware↔runtime coupling key) | the `2.0.0-<branch>-<hash>` suffix on the `neat-*` debs / the internals build folder |
| Neat package version (`neat-runtime`, `-gst-plugins`, `-appcomplex`, `-ev74-firmware`, `sima-neat`) | `.deb` `Version:` fields / `SIMANEAT_BUILDINFO_JSON` (already wired in `build.sh`) |
| pyneat version | `pyneat.__version__` (`_core.__version__`) |
| pyneat Python/ABI + platform | the wheel tag (`cp311-cp311-linux_aarch64`) |
| SDK / eLxr version | `/etc/sdk-release` (`SDK Version`, `eLXr Version`) |
| Cross toolchain | `aarch64-linux-gnu-g++ --version` |
| Dep floors (NumPy, Torch, OpenCV, glibc) | `pyproject.toml` + `.deb` `Depends:` |

**Proposed mechanism:** a build step emits `compatibility.json` from the above;
the Docusaurus matrix page imports it at doc-build time and renders the table.
Reuse `SIMANEAT_BUILDINFO_JSON` if it already carries enough; otherwise extend it.
The page's *prose* (which OSes, which Python, the firmware caveat) stays in the
`.mdx`; only the **numbers** come from the manifest.

### Matrix shape (axes; numbers injected from the manifest, shown here as `{{…}}`)
```
Neat core (DevKit):   any Modalix DevKit · arch aarch64 · platform sw {{neat_version}}
                      glibc ≥{{glibc}} · OpenCV {{opencv}} · GStreamer 1.0 · C++20
                      runtime/plugins/appcomplex MUST be exact-equal; firmware match is manual (see caveat)
Host SDK (container): Ubuntu 22.04/24.04 · Windows 11+WSL2+Docker · macOS+Colima+Docker
                      eLxr/Palette SDK {{sdk_version}} · cross-gcc {{gcc}} · sysroot /opt/toolchain/aarch64/modalix
pyneat (DevKit):      prebuilt = CPython 3.11 / aarch64 · numpy {{numpy_range}} · torch ≥{{torch}} (opt)
                      from source: Python ≥3.9
Tools:                use the latest sima-cli
```

### Two "fail-loudly" rows
- **NumPy 2.x** on the DevKit → unsupported (pin is `<2`). Easy footgun.
- **"Where's the x86 / CUDA / GPU build?"** → N/A. pyneat is aarch64-on-DevKit; the host is build-only (cross-compile + model-compiler).

---

## Open items

All compatibility questions are resolved. The remaining items are **improvement opportunities**, not blockers for the matrix:

1. *(optional hardening)* Firmware↔runtime is matched today by **shared build hash + install-the-matched-set**, not by an apt `Depends:` pin. A partial update (new runtime, old firmware) silently breaks it (`No channel available`). If you want apt itself to refuse a mismatch, the release packaging could carry the hash into a `Depends:` pin. Current mechanism works as long as the matched set is deployed together.

*(Resolved this round: Modalix "3.0" is a product name → any Modalix runs Neat, no HW gate; sima-cli = latest; firmware↔runtime matched by internals build hash; versions derived from a build manifest, never hardcoded.)*

## Additional source-verified facts (second-pass audit)

Confirmed against packaging/source; fold into the matrix:
- **`simaai-*` are real version constraints, not "latest":** `simaai-common ≥ 2.0.0`, `simaai-memory-lib = 2.0.0` (exact), MLA runtime `simaai-mlart-modalix | simaai-mlart-davinci`. These are the SiMa runtime layer under Neat.
- **`neat-internals-dev` is the umbrella** — depends on `neat-appcomplex` + `neat-runtime` + `simaai-common` at the matched hash; installing it pulls the matched stack.
- **pyneat is aarch64 at every level** — no x86/other prebuilt wheel, and source builds are sysroot-locked to aarch64 (`CMakeLists.txt` sysroot `/opt/toolchain/aarch64/modalix`). No x86 pyneat exists.
- **Install location:** venv at `/media/nvme/pyneat` (writable NVMe) else `$HOME/pyneat`, symlinked to `~/pyneat` (`install_neat_framework.sh`).
- **x86-SDK → aarch64-wheel:** `detect_elxr_wheel_platform` defaults to `linux-aarch64`, so even an x86 host SDK emits an aarch64 wheel — note it.

## Recommended compatibility-page structure (from industry review)

Benchmarked against TensorRT/DeepStream support matrices, OpenVINO system requirements, HailoRT, Core ML/coremltools, ONNX Runtime, LiteRT. The standard template and where Neat stands:

**Already coverable from this investigation:**
1. **Two-column split** — host-side (SDK/container) vs on-device (DevKit runtime). Never conflate them.
2. **Per-release matched-tuple matrix** — Neat runtime → firmware **hash** → SDK/compiler → `.tar.gz` format → host OS → pyneat/CPython/numpy. Rendered from the build manifest (no hardcoded numbers).
3. **Known-incompatible / NOT-supported list** (every leader has one): NumPy ≥ 2; mixed runtime/firmware hash; non-3.11 CPython for the prebuilt wheel; non-aarch64; host glibc < 2.34.
4. **"Verify your install" + mismatch self-check** — commands for `/etc/sdk-release`, runtime/firmware **hash**, `pyneat.__version__`. The runtime-hash vs firmware-hash check is the highest-value diagnostic (it's the failure we hit).

**Needs product / release-eng decision (section belongs on the page; the value needs an owner):**
5. **Support lifecycle / EOL** — how long a Neat release + firmware is supported (LTS vs latest).
6. **Deprecation policy + forward-looking requirements** (OpenVINO-style "future release will require X").
7. **Model-archive portability + compiler↔runtime compatibility** — the TensorRT/Hailo lesson: is a `.tar.gz` tied to a runtime/format version or board revision, and what happens on mismatch? Needs the compiler/runtime owners.

**Recommended page order:** overview → host-side/on-device columns → matched-tuple matrix → known-incompatible → verify-install + mismatch self-check → artifact portability → lifecycle/EOL + deprecation → upgrade/downgrade ordering.

## FINAL PLAN — compatibility page + version flow

**Principle:** one source of truth, no version number ever typed into a doc. The
core build already produces `buildinfo.json`; we extend it, and both the docs and
the runtime read from it. No standalone/ad-hoc generator script.

### Single artifact: `buildinfo.json` (extend the existing generator)
- Generator: `scripts/build/generate_package_buildinfo.py` (already run by `build.sh`, already installed to `/usr/share/sima-neat/buildinfo.json` on the device).
- **Add fields** (read from in-repo sources at build time): `pyneat` `{python_abi, numpy_range, torch_min}` from `pyproject.toml`; dep floors `{opencv, glibc, gstreamer}` from the `.deb` `Depends:`; `toolchain` + `sdk`/`elxr` from the build env (`/etc/sdk-release`, compiler).
- **Pin the internals hash on release:** today `dependencies.internals.spec = "latest"`. Releases must resolve it to the actual hash — that is the firmware↔runtime coupling key, and it's already the field that belongs on the device for the self-check.

### Two consumers, zero new infrastructure
1. **Runtime self-check** — `neat` / `pyneat` reads the device's `/usr/share/sima-neat/buildinfo.json` and reports version + internals hash; flags a runtime↔firmware hash mismatch (the failure we reproduced). High-value, data already shipped.
2. **Docs** — the matrix page imports a committed per-release snapshot.

### Docs page
- **File:** `docs/getting-started/compatibility.mdx`, `sidebar_position: 1.5` (after Installation, before Build). Cross-link from `installation/index.mdx` and `troubleshooting.md`.
- **Data:** release commits the `buildinfo.json` snapshot to `website/src/data/buildinfo.json`; the page does `import info from '@site/src/data/buildinfo.json'` and renders version cells from it. No hardcoded numbers.
- **Write now (derivable):**
  - host-side / on-device two-column layout,
  - matched-tuple matrix (rendered from the snapshot),
  - **Known-incompatible list** (NumPy ≥2, mismatched runtime/firmware hash, non-3.11 ABI, non-aarch64, glibc <2.34),
  - **Verify-your-install + hash self-check** (commands for `/etc/sdk-release`, buildinfo version+hash, `pyneat.__version__`),
  - the source-verified facts (simaai-* floors, `neat-internals-dev` umbrella, aarch64-only at every level, install location).
- **Stub now (owner needed), as visible callouts:** support lifecycle / EOL; deprecation policy; `.tar.gz` artifact portability + compiler↔runtime compatibility.

### Release wiring
- `build.sh` (packaging path) already runs the buildinfo generator → extend its output → on `--all`/release, copy `build/buildinfo.json` to `website/src/data/buildinfo.json` and commit. `--doc`-only builds then render from the committed snapshot.

### File-by-file work list
1. `scripts/build/generate_package_buildinfo.py` — add pyneat floors + dep floors + toolchain/SDK fields.
2. release/`build.sh` step — resolve+pin internals hash; copy snapshot to `website/src/data/buildinfo.json`.
3. `website/src/data/buildinfo.json` — committed snapshot (initial one generated from current sources).
4. `docs/getting-started/compatibility.mdx` — the page (derivable sections + policy stubs).
5. `docs/getting-started/installation/index.mdx` + `troubleshooting.md` — cross-links.
6. (follow-up) `neat`/`pyneat` runtime self-check reading on-device buildinfo.

### Phasing
- **Phase 1 (docs, no core change):** write `compatibility.mdx` + initial committed `buildinfo.json` snapshot from current sources → page renders today. Derivable sections complete; policy stubbed.
- **Phase 2 (core):** extend the buildinfo generator + pin internals hash + commit snapshot on release → numbers self-update.
- **Phase 3 (runtime):** the on-device hash mismatch self-check.

### Decisions locked
- Single artifact = `buildinfo.json` (not a separate `compatibility.json`).
- Docs consume a **committed per-release snapshot** (robust for standalone doc builds).
- No standalone `tools/` generator — extend the existing build step.
- Versions never hardcoded in the page.

### Still needs a human/owner (not derivable)
- Lifecycle/EOL windows, deprecation policy, and `.tar.gz` artifact-portability / compiler↔runtime rules. Page ships with these as marked stubs until an owner fills them.

## Method note
All ✅ values came from packaging control fields, `pyproject.toml`, `/etc/sdk-release`, `registry.json`, `build.sh`, and the compiler itself — not from prose docs. Two independent passes (write + audit) confirmed the facts; the industry-standard *sections* above were benchmarked against the named comparable stacks.
