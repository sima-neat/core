# Neat Documentation — Revisited Plan

Follow-up to the new-user audit. Captures the concrete docs work we agreed to do, with topic lists for each section so the writing pass can start immediately.

---

## 1. Replace Hello-Neat with a real end-to-end YOLOv8 sample

**Decision.** Drop the "allocate 64 bytes and print a string" smoketest. The first thing a new user runs should *actually do inference* — and it should showcase the framework's strongest single-call story.

**Target sample.** YOLOv8 detection, single `Model` call, configured with **`PreprocOptions`** + **`BoxDecodeOptions`** so the user gets:

- one model archive loaded
- one input image pushed
- bounding boxes returned, already decoded

```cpp
#include <neat.h>

namespace neat = simaai::neat;

int main() {
  neat::ModelOptions opts;
  opts.preprocess  = neat::PreprocOptions::yolov8_default(640, 640);
  opts.postprocess = neat::BoxDecodeOptions::yolov8_default(/*score=*/0.25, /*iou=*/0.45);

  neat::Model model("yolov8s.tar.gz", opts);

  neat::Tensor frame = neat::load_image("dog.jpg");
  neat::Sample out   = model.run(frame);

  for (const auto& d : out.detections) {
    std::printf("%s  %.2f  [%d %d %d %d]\n",
                d.label.c_str(), d.score, d.x0, d.y0, d.x1, d.y1);
  }
}
```

**Why this works as the first page**

- One concept (`Model`), three knobs (`ModelOptions`, `PreprocOptions`, `BoxDecodeOptions`), one return type (`Sample::detections`).
- Demonstrates that **preproc + inference + postproc collapse into one call** — the framework's biggest delta vs hand-rolled GStreamer / TensorRT pipelines.
- Same code in Python with `pyneat.Model("yolov8s.tar.gz", opts)` — no second mental model.

**What changes**

- Rename `docs/getting-started/minimal_example.md` → `docs/getting-started/first_inference.mdx`.
- Page renames in nav: **Hello Neat!** → **Your First Inference**.
- Keep the linker smoketest as a *collapsible* "If the install didn't work, try this 5-line build check" block at the bottom of the install page — not its own top-level entry.
- Required asset: `yolov8s.tar.gz` from Model Zoo + a single test image (`dog.jpg`) shipped in `examples/media/` or auto-downloaded by `build.sh`.
- Update overview SVG to show **one Model arrow producing detections**, not a generic 3-box flow.

---

## 2. Troubleshooting / FAQ page

**Decision.** Curate from *real* edge-AI / inference-runtime pain — not just incidents we happened to hit. We should look at what users of comparable stacks routinely ask about, and answer the Neat equivalent.

**Comparison stacks to mine for common issues** (just to verify our list is representative, not to copy):

- NVIDIA DeepStream / TensorRT
- Intel OpenVINO
- Qualcomm QNN / SNPE
- Hailo HailoRT
- Apple Core ML
- ONNX Runtime
- Edge Impulse
- TensorFlow Lite / LiteRT

**Topic list — at minimum these go in `docs/getting-started/troubleshooting.md`:**

Install & environment
- `sima-cli install neat` fails behind a corporate proxy / no internet
- `pyneat` import fails — wrong Python version (cp311 only), no venv activation, or wheel installed in SDK instead of DevKit
- `find_package(SimaNeat CONFIG)` cannot find the package — missing `CMAKE_PREFIX_PATH`, missing SYSROOT in SDK builds
- `dk` / `devkit-run` returns "command not found" in a fresh shell — `~/devkit-sync.rc` not sourced
- DevKit pairing rejects connection — IP unreachable, SSH key mismatch, firmware too old

Build
- Cross-built binary segfaults or aborts on the DevKit — host SDK and DevKit firmware out of sync (ABI mismatch). How to read the version on both sides and what mismatches are tolerable.
- CMake reports the wrong GStreamer / OpenCV — multiple installs on PATH
- Linker error on `simaai::neat::*` symbols — old `.deb` left over from a previous release; how to fully uninstall

Model archive
- `io.parse` on a known-good `.tar.gz` — archive corrupted by Git LFS smudge, or repacked without `mpk.json`
- `mpk.unsupported_version` — model compiled against a newer compiler than the runtime supports; pointer to the compatibility matrix
- `plan.no_kernel` — model uses an op the current Neat build doesn't include (e.g., TVM fallback disabled)
- `plan.contract_mismatch` — user passed `PreprocOptions` that conflict with what the model was compiled with; how to read `Model::describe()` to discover the contract

Runtime
- `run.pull(timeout_ms=...)` returns nothing — producer not pushing, graph not built, or downstream stage backpressured (point at `graph.validate()` + `run.stats()`)
- Wrong predictions / accuracy regression — almost always preprocessing mismatch (normalization, layout, color order); pointer to `Model::describe()` and the preprocess intent doc
- Pipeline stalls under load — queue depth too small / wrong overflow policy; pointer to runtime tuning
- "Out of memory" or `MLASHM` allocation failure — multiple models contending, or fragmentation; recovery steps
- RTSP source disconnects every N minutes — network keepalive / firewall

Comparing to other stacks
- "Where's my `.engine` / `.blob` / `.dlc` / `.hef`?" — Neat consumes `.tar.gz` archives; that's the equivalent.
- "How do I pin to a CUDA stream / OpenCL queue?" — you don't; `RunMode::Async` + `RunOptions` is the equivalent.
- "Where's the equivalent of `trtexec`?" — `neat inspect` (see §6) for archive introspection; `run.report()` + `run.stats()` for perf.
- "Why is my throughput lower than the headline TOPS?" — host overhead, queue starvation, drop policy. Link to runtime tuning.

Format: every entry is **Symptom → Likely cause → Fix in 3 lines or fewer → Link to the deeper doc**.

---

## 3. Compatibility matrix

**Decision.** One table in `docs/getting-started/installation/index.mdx` (top-of-page, before the install-path picker) and mirrored in the release notes. A new user must be able to answer "does this release run on my machine?" in under 30 seconds.

**Columns**

| Axis | Why it matters |
|---|---|
| Neat release | The thing the user is about to install |
| Modalix DevKit firmware (min / tested) | ABI gate for the runtime |
| Host OS for the SDK (Ubuntu 22.04 / 24.04 / Debian 12 / macOS for docs only) | Picks `sima-cli` install path |
| C++ toolchain (gcc / clang version) | C++20 minimum |
| Python interpreter (currently cp311) | Wheel ABI is a hard gate |
| GStreamer (system / bundled) | Plugin compatibility |
| OpenCV major version | Affects the `cv::Mat` interop in tutorials |
| Model-compiler version range | Which `.tar.gz` archives this Neat release can load (MPK version range) |
| CMake minimum | 3.16 today |
| Node.js for docs builds | 20+ |

**Two rows the matrix must surface explicitly**

- "I have an older Modalix (pre-3.0)" → **not supported**, link to the upgrade path.
- "I want CUDA / GPU acceleration on my host" → **N/A**, Neat targets Modalix; the host is build-only.

Source of truth: the matrix is generated from a YAML in `docs/getting-started/installation/compatibility.yaml` so release engineering can update one file per release.

---

## 4. Python parallel to the programming model

**Decision.** Python users currently fall off a cliff: README has a short snippet, Hello-Neat has a one-liner, then the alphabetical `reference/pythonapi/modules/pyneat/*.md` dump. Fix it by mirroring the C++ programming-model section in Python.

**Concrete work**

a. **One canonical install/activation block** for `pyneat`. Today the venv-activate instructions appear in README, Hello-Neat, and `tutorials/index.md` with three slightly different wordings. Pick one, link to it from the other two.

b. **New page: `docs/getting-started/programming-model/python_overview.mdx`** — mirrors `overview.mdx` but Python-first. Same five concepts, same call shape, plus the Python-only bits:
   - `Tensor.from_numpy` / `to_numpy`, `from_torch` / `to_torch` (DLPack zero-copy)
   - Context-manager idiom for `Run`
   - GIL behavior in async push/pull
   - How `pyneat` exceptions map to `NeatError` codes

c. **`pyneat` package map page** at `docs/reference/pythonapi/index.md`. Group the ~60 classes by role (Inputs, Options, Outputs, Runtime, Reports, Tensor types, GenAI) with a one-line purpose per entry. The alphabetical list stays but becomes the *second* tab on the page.

d. **Every C++ snippet in `docs/concepts/` gets a Python tab.** Today the `<CodeTabs>` component is used in overview.mdx but not consistently across concepts/. Audit and fill.

e. **Add a "Python ↔ C++ cheatsheet"** under `docs/reference/`. Two columns: same operation, both languages, side by side. ~25 rows covering Model load, Graph add, Run build, push/pull/run, RunOptions tweaks, common error handling, metrics access.

f. **Python-first tutorial track.** Tutorials are dual-language today, but the example commands in `tutorials/index.md` lead with C++ binaries. Add a "Python path" intro that walks straight from `pip install` to `python3 share/sima-neat/tutorials/001_run_your_first_model.py` with no C++ detour.

---

## 5. Repo cleanup — what a new user sees on first `ls`

**Decision.** A new user opening the repo should see a clear separation between **source**, **build artifacts**, and **internal scratch**. Today the root has ~20 dirs including 9 `build-*` trees, a 1.4 GB `core` blob, and three plan `.md` files that look like in-progress drafts.

**Action items**

a. **Plan `.md` files do not belong in the repo.** Confirmed with user. Move *all* of these out of the working tree:
   - `graph_documentation_plan.md`
   - `graph_implementation_final_plan.md`
   - `graph_implementation_final_plan_phase_0_and_1.md`
   - `revisited_documentation_plan.md` (this file — also temporary)

   Add a `.gitignore` entry for top-level `*_plan.md` and any `*.md` checked into root that isn't `README`, `CHANGELOG`, `CONTRIBUTING`, `LICENSE`, `THIRD_PARTY_NOTICES`, `AGENTS`. Keep plan drafts in a private location off-tree.

b. **`build-*` directories are artifacts.** None of them should be tracked. Audit `.gitignore`:
   - `build/`, `build-*/`, `_CPack_Packages/`, `dist/`, `generated/`, `distribution/` all out.
   - The 1.4 GB `core` file at root is clearly an artifact left over from packaging; not for the repo.

c. **`docs/_tmp_test.txt`** — 4-byte stray file, delete.

d. **Tutorials directory has two parallel numbering schemes.** `tutorials/001_model_in_5_minutes/` (legacy) vs `tutorials/001_run_your_first_model/` (current). Delete the legacy `001_…` through `018_…` set; the renamed set replaces them. Confirm nothing in `CMakeLists.txt` or `build.sh` still references the old names before removal.

e. **Add `docs/contribute/repo_map.md`** — one page, one table:

   | Path | What it is | New user reads? |
   |---|---|---|
   | `src/`, `include/` | Library source and public headers | No (use docs) |
   | `python/` | `pyneat` binding source | No |
   | `tutorials/` | Tutorial source | Yes — but via the installed extras folder |
   | `examples/` | Standalone reference apps | Yes |
   | `docs/` | Source for the docs site | No (read on the site) |
   | `tests/` | gtest suite | Contributor |
   | `tools/` | Helper scripts | Contributor |
   | `scripts/` | Build/release scripts | Contributor |
   | `cmake/`, `packaging/`, `distribution/` | Build glue | Contributor |
   | `deps/` | Vendored / fetched dependencies | No |

f. **Link the repo map from CONTRIBUTING.md and from `docs/contribute/architecture.md`.**

---

## 6. Model archive anatomy — exposed through the `Model` class

**Decision.** A new user must never have to `tar -xzf` a `.tar.gz` and `cat mpk.json` to find out what's inside. Inspection is a first-class part of the `Model` API. The docs follow that API rather than describing the file format in the abstract.

**API additions** (Neat-side work, not docs-only — but the docs depend on it)

```cpp
namespace simaai::neat {

struct ModelDescription {
  std::string name;
  std::string mpk_version;
  std::string compiler_version;
  std::vector<TensorSpec> inputs;        // name, dtype, shape (incl. dynamic dims), layout
  std::vector<TensorSpec> outputs;
  std::vector<StageInfo>  pipeline;      // preprocess / inference / postprocess stages present
  std::optional<PreprocContract>  preprocess_contract;
  std::optional<PostprocContract> postprocess_contract;
  std::vector<std::string> required_kernels;
  uint64_t   archive_bytes;
  std::string archive_sha256;
};

class Model {
 public:
  ModelDescription describe() const;       // structured
  std::string      describe_text() const;  // human-readable, table-style
  nlohmann::json   describe_json() const;  // for tooling / tests
};

}  // namespace simaai::neat
```

Plus a **CLI:**

```bash
neat inspect yolov8s.tar.gz          # human-readable summary
neat inspect yolov8s.tar.gz --json   # machine-readable
neat inspect yolov8s.tar.gz --verify # check signatures, kernel allowlist, MPK schema
```

The CLI is a thin wrapper around `Model::describe()` so the same code path covers programmatic and command-line use.

**Docs work that follows the API**

a. **New concept page: `docs/concepts/model_archive.md`.** Promotes `how-to/assets_model_archives.md` from how-to to a concept — the archive is the framework's primary input, it deserves concept-level placement. Page structure:
   - "What's inside a `.tar.gz`?" — one diagram, generated from `Model::describe_text()` output of a real archive so the doc never drifts from reality.
   - "How to inspect one without writing code" — `neat inspect`.
   - "How to inspect one from code" — `Model::describe()` snippet, C++ and Python.
   - "Reading the preprocess contract" — points the user at the matching `PreprocOptions` they'd pass.
   - "When the contract conflicts with what you want" — link to `plan.contract_mismatch`.

b. **Every error code that mentions MPK or contracts links here.**

c. **Hello-Neat (now "Your First Inference") shows `model.describe_text()` first**, so the user sees what they loaded before they push a frame:

   ```cpp
   neat::Model model("yolov8s.tar.gz", opts);
   std::cout << model.describe_text() << "\n";   // verify before you run
   ```

d. **Tutorial 004 (`Configure Model Options`)** is rewritten to use `Model::describe()` as the source of truth for what options to set — instead of telling the user to read the README that ships in the model archive.

---

## 7. Branding convention — "Neat", not "NEAT"

**Decision.** First-letter cap, rest lowercase. **Neat** (the framework name) and **pyneat** (the package name). The acronym expansion ("Neural Edge Acceleration Toolkit") appears once, in the glossary, as a parenthetical — the brand itself is not styled in all-caps anywhere else.

**Audit and fix**

a. **README.md** — current line 1 says "SiMa NEAT Framework", line 7 says "SiMa NEAT". Change to "SiMa Neat".

b. **AGENTS.md, CONTRIBUTING.md** — same passes, change "SiMa NEAT" → "SiMa Neat".

c. **Code comments and Doxygen `@brief` blocks** — any "NEAT" → "Neat".

d. **CI badges, repo description on GitHub** — same.

e. **Add a `docs/contribute/style_guide.md`** (or extend `naming.md`) with one section:

   > **Product name.** Write **Neat**. Never **NEAT**, **neat**, **N.E.A.T.**, or **Sima-Neat** (use **SiMa Neat** in full, **Neat** when context is clear).
   >
   > **Package name.** Write **`pyneat`** (lowercase, monospace) for the Python package.

f. **One CI lint** that fails the docs build if `\bNEAT\b` appears in any `*.md` / `*.mdx` outside the glossary expansion line. Cheap to add, catches drift.

---

## Items deferred to a later pass (mentioned in audit, not in this plan)

- **#7 in the audit — architecture diagrams** (MLA / CVU / A65 layout, Sample-flow diagram). Take when we have an illustrator round.
- **#8 in the audit — tutorials directory cleanup beyond removing the legacy 001–018 set.** Partial cleanup is in §5; deeper restructuring waits.
- **Node catalog page (concepts/nodes.md).** Worth doing, but after §6 lands — the same "describe from code, not from a hand-maintained list" principle should apply: generate the catalog from the registered Node metadata, not a manually curated table.

---

## Order of execution suggestion

1. §7 branding sweep — purely mechanical, lowest risk, makes every other doc change start from a clean baseline.
2. §5 repo cleanup — removes the "what is all this stuff?" tax before a new user opens the tree.
3. §6 `Model::describe()` API + concept page — unblocks §1 and §2.
4. §1 first-inference rewrite — depends on §6 (`describe_text()` is part of the snippet).
5. §3 compatibility matrix — small, can land in parallel with §1.
6. §2 troubleshooting page — landed last so it can reference the new structure.
7. §4 Python parallel — biggest writing pass; do once everything else is stable so the C++/Python tabs stay in sync.
