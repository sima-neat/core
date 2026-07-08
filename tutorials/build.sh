#!/usr/bin/env bash
set -euo pipefail

# Resolve TUTORIALS_DIR / REPO_ROOT based on where this script lives.
# Supported layouts (extras takes precedence when both are reachable, since a
# user running build.sh from inside the extracted extras folder expects the
# output to land there):
#   1. Extras tarball:     extras-root/build.sh     -> tutorials at ./share/sima-neat/tutorials
#   2. Source tree:        core/tutorials/build.sh  -> tutorials at ../tutorials, build at ../build/...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "${SCRIPT_DIR}/share/sima-neat/tutorials/CMakeLists.txt" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
  TUTORIALS_DIR="${SCRIPT_DIR}/share/sima-neat/tutorials"
elif [[ -f "${SCRIPT_DIR}/../tutorials/CMakeLists.txt" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
  TUTORIALS_DIR="${REPO_ROOT}/tutorials"
else
  echo "build.sh: cannot locate tutorials/CMakeLists.txt (checked ./share/sima-neat/tutorials and ../tutorials)" >&2
  exit 1
fi
BUILD_DIR="${REPO_ROOT}/build/tutorials-standalone"
BUILD_TYPE="Release"
TARGET="all"
DO_CLEAN="OFF"
LIST_TARGETS="OFF"
DOWNLOAD_MODELS_ONLY="OFF"
MODEL_TARGET_FOLDER="/tmp"
JOBS=""
SIMANEAT_DIR="${SimaNeat_DIR:-}"
SIMA_CLI="${SIMA_CLI:-}"

auto_jobs() {
  local cpu_count=1
  if command -v nproc >/dev/null 2>&1; then
    cpu_count="$(nproc)"
  elif command -v getconf >/dev/null 2>&1; then
    cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  fi

  # Cap parallel compile jobs by available memory to avoid OOM kills.
  # Rule of thumb: ~3 GiB RAM per concurrent C++ compile unit.
  if [[ -r /proc/meminfo ]]; then
    local mem_kb mem_jobs
    mem_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    if [[ -n "${mem_kb}" ]]; then
      mem_jobs=$(( mem_kb / (3 * 1024 * 1024) ))
      if (( mem_jobs < 1 )); then
        mem_jobs=1
      fi
      if (( mem_jobs < cpu_count )); then
        cpu_count="${mem_jobs}"
      fi
    fi
  fi

  if (( cpu_count < 1 )); then
    cpu_count=1
  fi
  echo "${cpu_count}"
}

absolute_path() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path%/}"
  else
    printf '%s/%s\n' "${PWD%/}" "${path%/}"
  fi
}

discover_sima_cli() {
  if [[ -n "${SIMA_CLI}" ]]; then
    if [[ -x "${SIMA_CLI}" ]]; then
      printf '%s\n' "${SIMA_CLI}"
      return 0
    fi
    if type -P "${SIMA_CLI}" >/dev/null 2>&1; then
      type -P "${SIMA_CLI}"
      return 0
    fi
  fi

  if type -P sima-cli >/dev/null 2>&1; then
    type -P sima-cli
    return 0
  fi

  local -a shell_setup_files=(
    "${HOME:-}/.bash_profile"
    "${HOME:-}/.bash_login"
    "${HOME:-}/.profile"
    "${HOME:-}/.bashrc"
  )

  local setup_file
  for setup_file in "${shell_setup_files[@]}"; do
    if [[ -r "${setup_file}" ]]; then
      set +u
      # shellcheck source=/dev/null
      source "${setup_file}" >/dev/null 2>&1 || true
      set -u
      if type -P sima-cli >/dev/null 2>&1; then
        type -P sima-cli
        return 0
      fi

      local alias_line alias_value alias_cmd
      alias_line="$(alias sima-cli 2>/dev/null || true)"
      if [[ "${alias_line}" == alias\ sima-cli=\'*\' ]]; then
        alias_value="${alias_line#alias sima-cli=\'}"
        alias_value="${alias_value%\'}"
        read -r alias_cmd _ <<< "${alias_value}"
        if [[ -x "${alias_cmd}" ]]; then
          printf '%s\n' "${alias_cmd}"
          return 0
        fi
        if type -P "${alias_cmd}" >/dev/null 2>&1; then
          type -P "${alias_cmd}"
          return 0
        fi
      fi
    fi
  done

  local -a candidates=(
    "${HOME:-}/.sima-cli/.venv/bin/sima-cli"
    "${HOME:-}/.local/bin/sima-cli"
    "/data/sima-cli/.venv/bin/sima-cli"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

header_exists() {
  local header="$1"
  local -a roots=()
  if [[ -n "${SYSROOT:-}" ]]; then
    roots+=("${SYSROOT}/usr/include" "${SYSROOT}/usr/local/include")
  fi
  roots+=(
    "/usr/include"
    "/usr/local/include"
    "/opt/toolchain/aarch64/modalix/usr/include"
    "/opt/toolchain/aarch64/modalix/usr/local/include"
  )

  local root
  for root in "${roots[@]}"; do
    if [[ -f "${root}/${header}" ]]; then
      return 0
    fi
  done
  return 1
}

preflight_cpp_headers() {
  if header_exists "nlohmann/json.hpp"; then
    return 0
  fi

  cat >&2 <<'EOF'
build.sh: missing required C++ header: nlohmann/json.hpp

The installed Neat public headers include nlohmann/json.hpp, but that header is
not present in the active DevKit/SDK include paths. Install the JSON header
package, then rerun build.sh:

  sudo apt-get update && sudo apt-get install -y nlohmann-json3-dev

EOF
  return 1
}

simaaimem_library_roots() {
  local -a roots=()
  local prefix
  local simaneat_dir="${SIMANEAT_DIR%/}"

  if [[ -n "${SYSROOT:-}" ]]; then
    roots+=(
      "${SYSROOT}/usr/lib"
      "${SYSROOT}/usr/lib/aarch64-linux-gnu"
      "${SYSROOT}/usr/local/lib"
      "${SYSROOT}/usr/local/lib/aarch64-linux-gnu"
      "${SYSROOT}/lib"
      "${SYSROOT}/lib/aarch64-linux-gnu"
    )
  fi

  case "${simaneat_dir}" in
    */usr/lib/aarch64-linux-gnu/cmake/SimaNeat)
      prefix="${simaneat_dir%/lib/aarch64-linux-gnu/cmake/SimaNeat}"
      roots+=("${prefix}/lib" "${prefix}/lib/aarch64-linux-gnu")
      ;;
    */usr/lib/cmake/SimaNeat)
      prefix="${simaneat_dir%/lib/cmake/SimaNeat}"
      roots+=("${prefix}/lib" "${prefix}/lib/aarch64-linux-gnu")
      ;;
  esac

  roots+=(
    "/usr/lib"
    "/usr/lib/aarch64-linux-gnu"
    "/usr/local/lib"
    "/usr/local/lib/aarch64-linux-gnu"
    "/lib"
    "/lib/aarch64-linux-gnu"
    "/opt/toolchain/aarch64/modalix/usr/lib"
    "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu"
    "/opt/toolchain/aarch64/modalix/usr/local/lib"
    "/opt/toolchain/aarch64/modalix/usr/local/lib/aarch64-linux-gnu"
  )

  local seen=";"
  local root
  for root in "${roots[@]}"; do
    [[ -d "${root}" ]] || continue
    case "${seen}" in
      *";${root};"*) continue ;;
    esac
    seen+="${root};"
    printf '%s\n' "${root}"
  done
}

configure_simaaimem_link_fallback() {
  if [[ "${SIMANEAT_DISABLE_SIMAAIMEM_SHIM:-0}" == "1" ]]; then
    return 0
  fi

  local -a lib_roots=()
  mapfile -t lib_roots < <(simaaimem_library_roots)
  if [[ "${#lib_roots[@]}" -eq 0 ]]; then
    return 0
  fi

  local root
  for root in "${lib_roots[@]}"; do
    if [[ -e "${root}/libsimaaimem.so" ]]; then
      return 0
    fi
  done

  local versioned_lib=""
  for root in "${lib_roots[@]}"; do
    while IFS= read -r candidate; do
      if [[ -e "${candidate}" ]]; then
        versioned_lib="${candidate}"
        break 2
      fi
    done < <(find "${root}" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | sort | grep '/libsimaaimem\.so\.' || true)
  done

  if [[ -z "${versioned_lib}" ]]; then
    return 0
  fi

  local shim_dir="${BUILD_DIR}/link-shims"
  mkdir -p "${shim_dir}"
  ln -sfn "${versioned_lib}" "${shim_dir}/libsimaaimem.so"
  local linker_flags="${CMAKE_EXE_LINKER_FLAGS:-}"
  linker_flags="${linker_flags:+${linker_flags} }-L${shim_dir}"
  cmake_args+=("-DCMAKE_EXE_LINKER_FLAGS=${linker_flags}")

  cat >&2 <<EOF
build.sh: using compatibility fallback for libsimaaimem.so.

Found runtime library:
  ${versioned_lib}

The unversioned linker library libsimaaimem.so was not found in the active
SDK/DevKit library paths, so build.sh created a build-local linker shim:
  ${shim_dir}/libsimaaimem.so

This keeps tutorials buildable in older installed environments. The preferred
fix is to install or update the package that provides the development linker
symlink:
  sudo apt-get update && sudo apt-get install -y simaai-memory-lib-dev

Set SIMANEAT_DISABLE_SIMAAIMEM_SHIM=1 to disable this fallback.

EOF
}

usage() {
  cat <<EOF
Usage: tutorials/build.sh [options]

Options:
  --target <name>      Build one target (e.g. tutorial_001_model_in_5_minutes)
  --build-dir <path>   Build directory (default: build/tutorials-standalone)
  --build-type <type>  CMake build type (default: Release)
  --clean              Remove build directory before configure
  --download-models-only
                       Download tutorial models from Model Zoo and exit
  --model-target-folder <path>
                       Directory where Model Zoo downloads are written (default: /tmp)
  --list-targets       List available tutorial targets and exit
  --simaneat-dir <dir> Path to directory containing SimaNeatConfig.cmake
  -j, --jobs <N>       Parallel build jobs (default: auto)
  -h, --help           Show this help

Examples:
  tutorials/build.sh
  tutorials/build.sh --target tutorial_001_run_your_first_model
  tutorials/build.sh --target tutorial_001_run_your_first_model --download-models-only
  tutorials/build.sh --model-target-folder /data/models
  tutorials/build.sh --simaneat-dir /usr/lib/aarch64-linux-gnu/cmake/SimaNeat
  tutorials/build.sh --list-targets
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="${2:-}"
      shift 2
      ;;
    --clean)
      DO_CLEAN="ON"
      shift
      ;;
    --download-models-only)
      DOWNLOAD_MODELS_ONLY="ON"
      shift
      ;;
    --model-target-folder)
      MODEL_TARGET_FOLDER="${2:-}"
      shift 2
      ;;
    --list-targets)
      LIST_TARGETS="ON"
      shift
      ;;
    --simaneat-dir)
      SIMANEAT_DIR="${2:-}"
      shift 2
      ;;
    -j|--jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_TARGET_FOLDER}" ]]; then
  echo "build.sh: --model-target-folder requires a non-empty path" >&2
  exit 1
fi
MODEL_TARGET_FOLDER="$(absolute_path "${MODEL_TARGET_FOLDER}")"
BUILD_DIR="$(absolute_path "${BUILD_DIR}")"

if [[ "${DO_CLEAN}" == "ON" ]]; then
  rm -rf "${BUILD_DIR}"
fi

discover_simaneat_dir() {
  if [[ -n "${SIMANEAT_DIR}" ]]; then
    return 0
  fi

  local -a candidates=()
  if [[ -n "${SYSROOT:-}" ]]; then
    candidates+=(
      "${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake/SimaNeat"
      "${SYSROOT}/usr/lib/cmake/SimaNeat"
    )
  fi
  candidates+=(
    "/usr/lib/aarch64-linux-gnu/cmake/SimaNeat"
    "/usr/lib/cmake/SimaNeat"
    "/opt/toolchain/aarch64/modalix/usr/lib/aarch64-linux-gnu/cmake/SimaNeat"
    "/opt/toolchain/aarch64/modalix/usr/lib/cmake/SimaNeat"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/SimaNeatConfig.cmake" ]]; then
      SIMANEAT_DIR="${candidate}"
      return 0
    fi
  done
}

discover_simaneat_dir

discover_cmake_prefix_paths() {
  local -a candidates=()
  local prefix
  local simaneat_dir="${SIMANEAT_DIR%/}"

  if [[ -n "${SYSROOT:-}" ]]; then
    candidates+=("${SYSROOT}/usr")
  fi

  case "${simaneat_dir}" in
    */usr/lib/aarch64-linux-gnu/cmake/SimaNeat)
      prefix="${simaneat_dir%/lib/aarch64-linux-gnu/cmake/SimaNeat}"
      candidates+=("${prefix}")
      ;;
    */usr/lib/cmake/SimaNeat)
      prefix="${simaneat_dir%/lib/cmake/SimaNeat}"
      candidates+=("${prefix}")
      ;;
  esac

  local seen=";"
  local candidate
  for candidate in "${candidates[@]}"; do
    [[ -d "${candidate}" ]] || continue
    case "${seen}" in
      *";${candidate};"*) continue ;;
    esac
    seen+="${candidate};"
    printf '%s\n' "${candidate}"
  done
}

manifest_path() {
  local -a candidates=(
    "${REPO_ROOT}/deps/manifest.json"
    "${REPO_ROOT}/share/sima-neat/deps/manifest.json"
    "${REPO_ROOT}/../deps/manifest.json"
    "${SCRIPT_DIR}/deps/manifest.json"
    "${SCRIPT_DIR}/../deps/manifest.json"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

platform_version() {
  local manifest
  if ! manifest="$(manifest_path)"; then
    echo "build.sh: cannot locate deps/manifest.json for Model Zoo platform version" >&2
    return 1
  fi

  local version
  version="$(
    sed -n 's/.*"platform-version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${manifest}" \
      | head -n1
  )"
  if [[ -z "${version}" ]]; then
    echo "build.sh: deps/manifest.json does not define platform-version: ${manifest}" >&2
    return 1
  fi
  echo "${version}"
}

model_download_path() {
  local model="$1"
  local model_base="${model##*/}"
  local model_dash="${model_base//_/-}"
  local model_underscore="${model_base//-/_}"
  local -a candidates=(
    "${MODEL_TARGET_FOLDER}/tmp/${model_base}_mpk.tar.gz"
    "${MODEL_TARGET_FOLDER}/tmp/${model_dash}_mpk.tar.gz"
    "${MODEL_TARGET_FOLDER}/tmp/${model_underscore}_mpk.tar.gz"
    "${MODEL_TARGET_FOLDER}/${model_base}_mpk.tar.gz"
    "${MODEL_TARGET_FOLDER}/${model_dash}_mpk.tar.gz"
    "${MODEL_TARGET_FOLDER}/${model_underscore}_mpk.tar.gz"
    "${MODEL_TARGET_FOLDER}/tmp/${model_base}.tar.gz"
    "${MODEL_TARGET_FOLDER}/${model_base}.tar.gz"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s/%s\n' "$(dirname "${candidate}")" "$(basename "${candidate}")"
      return 0
    fi
  done
  return 1
}

download_models() {
  local cmake_file="${TUTORIALS_DIR}/CMakeLists.txt"
  if [[ ! -f "${cmake_file}" ]]; then
    echo "build.sh: cannot find tutorials CMakeLists.txt at ${cmake_file}" >&2
    return 1
  fi

  local -a models=()
  local matched_target="OFF"
  local model
  while IFS= read -r model; do
    case "${model}" in
      __matched_target__)
        matched_target="ON"
        ;;
      ?*)
        models+=("${model}")
        ;;
    esac
  done < <(
    awk -v target="${TARGET}" '
      function emit(block, fields, count, i, target_name) {
        gsub(/[()]/, " ", block)
        gsub(/[[:space:]]+/, " ", block)
        count = split(block, fields, " ")
        target_name = fields[2]
        if (target != "all" && target_name != target) {
          return
        }

        print "__matched_target__"
        for (i = 1; i <= count; ++i) {
          if (fields[i] == "MODEL" && (i + 1) <= count) {
            print fields[i + 1]
          }
        }
      }
      /^[[:space:]]*add_tutorial\(/ {
        in_block = 1
        block = $0
        next
      }
      in_block {
        block = block " " $0
        if ($0 ~ /^[[:space:]]*\)[[:space:]]*$/) {
          emit(block)
          in_block = 0
        }
      }
    ' "${cmake_file}"
  )

  if [[ "${TARGET}" != "all" && "${matched_target}" != "ON" ]]; then
    echo "build.sh: cannot find tutorial target '${TARGET}' in ${cmake_file}" >&2
    return 1
  fi

  if [[ "${TARGET}" == "all" || "${TARGET}" == tutorial_019_* || "${TARGET}" == tutorial_020_* || "${TARGET}" == tutorial_021_* || "${TARGET}" == tutorial_022_* ]]; then
    echo "Skipping GenAI tutorial model downloads: GenAI tutorials use LLiMa model directories and are not downloaded through Model Zoo. Use llima pull explicitly before running them."
  fi

  if [[ "${#models[@]}" -eq 0 ]]; then
    echo "No Model Zoo tutorial models to download."
    return 0
  fi

  local sima_cli
  if ! sima_cli="$(discover_sima_cli)"; then
    cat >&2 <<'EOF'
build.sh: sima-cli is required to download tutorial models.

The helper could not find sima-cli on PATH, in the common DevKit install
locations, or after sourcing the user's shell startup files. If sima-cli is
installed in a custom location, rerun with:

  SIMA_CLI=/path/to/sima-cli ./build.sh

EOF
    return 1
  fi

  local version
  version="$(platform_version)"

  local -a unique_models=()
  local seen=" "
  for model in "${models[@]}"; do
    if [[ "${seen}" == *" ${model} "* ]]; then
      continue
    fi
    seen+="${model} "
    unique_models+=("${model}")
  done

  mkdir -p "${MODEL_TARGET_FOLDER}"

  echo "Downloading tutorial model(s) for platform ${version}: ${unique_models[*]}"
  echo "Model download target folder: ${MODEL_TARGET_FOLDER}"
  (
    cd "${MODEL_TARGET_FOLDER}"
    for model in "${unique_models[@]}"; do
      "${sima_cli}" modelzoo -v "${version}" get "${model}"
      local downloaded_path
      if downloaded_path="$(model_download_path "${model}")"; then
        echo "Downloaded ${model}: ${downloaded_path}"
      else
        echo "Downloaded ${model}: unable to locate tarball under ${MODEL_TARGET_FOLDER}" >&2
      fi
    done
  )
}

if [[ "${DOWNLOAD_MODELS_ONLY}" == "ON" ]]; then
  download_models
  exit 0
fi

preflight_cpp_headers

cmake_args=(
  -S "${TUTORIALS_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DSIMANEAT_TUTORIAL_MODEL_TARGET_FOLDER="${MODEL_TARGET_FOLDER}"
)
if [[ -n "${SIMANEAT_DIR}" ]]; then
  cmake_args+=("-DSimaNeat_DIR=${SIMANEAT_DIR}")
  echo "Using SimaNeat_DIR=${SIMANEAT_DIR}"
else
  cat >&2 <<'EOF'
Failed to locate SimaNeatConfig.cmake.
Install Neat core package first, or pass one of:
  tutorials/build.sh --simaneat-dir <path-to-cmake/SimaNeat>
  SimaNeat_DIR=<path-to-cmake/SimaNeat> tutorials/build.sh
EOF
  exit 1
fi

mapfile -t cmake_prefix_paths < <(discover_cmake_prefix_paths)
if [[ "${#cmake_prefix_paths[@]}" -gt 0 ]]; then
  IFS=';'
  cmake_prefix_path="${cmake_prefix_paths[*]}"
  unset IFS
  cmake_args+=("-DCMAKE_PREFIX_PATH=${cmake_prefix_path}")
  echo "Using CMAKE_PREFIX_PATH=${cmake_prefix_path}"
fi

configure_simaaimem_link_fallback

cmake "${cmake_args[@]}"

if [[ "${LIST_TARGETS}" == "ON" ]]; then
  cmake --build "${BUILD_DIR}" --target help | sed -n '/tutorial_/p'
  exit 0
fi

download_models

build_cmd=(cmake --build "${BUILD_DIR}")
if [[ "${TARGET}" != "all" ]]; then
  build_cmd+=(--target "${TARGET}")
fi
if [[ -n "${JOBS}" ]]; then
  build_cmd+=(--parallel "${JOBS}")
else
  AUTO_JOBS="$(auto_jobs)"
  echo "Auto-selected parallel jobs: ${AUTO_JOBS}"
  build_cmd+=(--parallel "${AUTO_JOBS}")
fi

"${build_cmd[@]}"

echo
if [[ "${TARGET}" == "all" ]]; then
  echo "Built tutorials under: ${BUILD_DIR}"
else
  echo "Built target '${TARGET}' under: ${BUILD_DIR}"
fi
