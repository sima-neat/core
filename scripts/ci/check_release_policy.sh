#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

required_files=(
  ".github/CODEOWNERS"
  ".github/PULL_REQUEST_TEMPLATE.md"
  "CONTRIBUTING.md"
  "docs/contribute/release-checklist.md"
  "scripts/ci/check_naming_and_conflicts.sh"
  "scripts/ci/check_clean_branch.sh"
  "scripts/ci/check_artifacts.sh"
  "scripts/ci/check_release_policy.sh"
  "scripts/ci/run_crash_correctness_gate.sh"
  "scripts/ci/run_model_archive_security_gate.sh"
  "scripts/ci/run_install_smoke.sh"
  "scripts/ci/run_perf_regression_gate.sh"
  "scripts/ci/run_soak_lane.sh"
  "scripts/ci/run_fuzz_nightly.sh"
  "scripts/ci/run_stress_gate.sh"
  "scripts/ci/run_sanitizer_gate.sh"
  "scripts/dev/install_hooks.sh"
  ".githooks/pre-push"
  ".github/workflows/test-crash-correctness-nightly.yml"
  ".github/workflows/model-archive-security.yml"
  ".github/workflows/install-smoke.yml"
  ".github/workflows/perf-regression.yml"
  ".github/workflows/test-soak-weekly.yml"
  ".github/workflows/test-fuzz-nightly.yml"
  ".github/workflows/sanitizers.yml"
  ".github/workflows/test-stress-nightly.yml"
  ".github/workflows/release-gate.yml"
  "CHANGELOG.md"
)

fail=0

echo "[release-policy] checking required governance and policy files..."
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: Missing required file: ${path}" >&2
    fail=1
  fi
done

for script in scripts/ci/*.sh; do
  if [[ ! -x "${script}" ]]; then
    echo "ERROR: CI policy script is not executable: ${script}" >&2
    fail=1
  fi
done

if [[ -f .github/workflows/release-gate.yml ]] &&
   ! grep -q "docs/contribute/release-checklist.md" .github/workflows/release-gate.yml; then
  echo "ERROR: release-gate workflow must reference docs/contribute/release-checklist.md" >&2
  fail=1
fi

required_checks=(
  "repo-hygiene"
  "configure-build-sanity"
  "docs-link-check"
  "crash-correctness-gate"
  "model-archive-security-gate"
  "install-smoke"
  "perf-regression-gate"
  "soak-weekly"
  "fuzz-nightly"
  "stress-gate"
  "asan-ubsan-gate"
  "release-policy-check"
)

for check_name in "${required_checks[@]}"; do
  if ! grep -q "${check_name}" docs/contribute/release-checklist.md; then
    echo "ERROR: docs/contribute/release-checklist.md must list required check: ${check_name}" >&2
    fail=1
  fi
done

required_crash_policy_lines=(
  "Crash/segfault/use-after-free/data race = immediate fix on current branch"
  "No push if local crash/correctness gate fails"
  "No merge if CI crash/correctness gate fails"
)

for policy_line in "${required_crash_policy_lines[@]}"; do
  if ! grep -q "${policy_line}" CONTRIBUTING.md; then
    echo "ERROR: CONTRIBUTING.md missing crash policy line: ${policy_line}" >&2
    fail=1
  fi
done

project_version="$(sed -n 's/^project(SimaNeat VERSION \([^ )]*\).*/\1/p' CMakeLists.txt | head -n 1)"
if [[ -z "${project_version}" ]]; then
  echo "ERROR: Unable to determine project version from CMakeLists.txt" >&2
  fail=1
fi

ref_name="${GITHUB_REF_NAME:-}"
if [[ -z "${ref_name}" ]]; then
  ref_name="$(git rev-parse --abbrev-ref HEAD)"
fi

release_version=""
if [[ "${ref_name}" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
  release_version="${BASH_REMATCH[1]}"
elif [[ "${ref_name}" =~ ^release/([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
  release_version="${BASH_REMATCH[1]}"
fi

if [[ -n "${release_version}" ]]; then
  echo "[release-policy] release ref detected (${ref_name}), validating release metadata..."
  if ! grep -qE "^## \[${release_version}\]" CHANGELOG.md; then
    echo "ERROR: CHANGELOG.md must contain entry: ## [${release_version}]" >&2
    fail=1
  fi
  if [[ "${project_version}" != "${release_version}" ]]; then
    echo "ERROR: CMake project version (${project_version}) must match release version (${release_version})." >&2
    fail=1
  fi
else
  if ! grep -qE '^## \[Unreleased\]' CHANGELOG.md; then
    echo "ERROR: CHANGELOG.md must include an [Unreleased] section." >&2
    fail=1
  fi
fi

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "[release-policy] release policy checks passed."
