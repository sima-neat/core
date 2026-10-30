#!/usr/bin/env bash
set -euo pipefail

# Only native packages belong in this bootstrap. In particular, never source
# the SDK activation scripts or request an architecture-qualified target package.
unset CC CXX CPPFLAGS CFLAGS CXXFLAGS LDFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
unset LIBRARY_PATH LD_LIBRARY_PATH LD_PRELOAD
unset SYSROOT SDKTARGETSYSROOT PKG_CONFIG_SYSROOT_DIR PKG_CONFIG_PATH PKG_CONFIG_LIBDIR

host_packages=(
  build-essential gcc-12 g++-12 libstdc++-12-dev clang clang-format clang-tidy ripgrep
  cmake pkg-config python3 git libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  libgstrtspserver-1.0-dev libopencv-dev nlohmann-json3-dev libcpp-httplib-dev
  libyaml-cpp-dev zlib1g-dev
)
host_modules=(
  'gstreamer-1.0 >= 1.22' gstreamer-allocators-1.0 gstreamer-app-1.0
  gstreamer-video-1.0 gstreamer-sdp-1.0 gstreamer-rtsp-server-1.0 glib-2.0
  opencv4 cpp-httplib yaml-cpp zlib
)

clang_cpp_works() {
  command -v clang++ >/dev/null 2>&1 || return 1
  printf '%s\n' '#include <memory>' '#include <nlohmann/json.hpp>' | clang++ -std=c++20 -x c++ - -fsyntax-only >/dev/null 2>&1
}

if command -v clang-format >/dev/null 2>&1 \
  && command -v clang-tidy >/dev/null 2>&1 \
  && command -v run-clang-tidy >/dev/null 2>&1 \
  && command -v rg >/dev/null 2>&1 \
  && command -v g++ >/dev/null 2>&1 \
  && command -v cmake >/dev/null 2>&1 \
  && command -v python3 >/dev/null 2>&1 \
  && command -v git >/dev/null 2>&1 \
  && command -v pkg-config >/dev/null 2>&1 \
  && pkg-config --exists "${host_modules[@]}" \
  && clang_cpp_works; then
  echo "Code hygiene tools and native analysis dependencies are already installed."
  exit 0
fi

if [[ "${HYGIENE_REQUIRE_PREINSTALLED:-0}" == "1" ]]; then
  echo "Pinned hygiene image is missing native tools or analysis dependencies; rebuild the shared media image." >&2
  exit 1
fi

if [ "$(id -u)" -eq 0 ]; then
  apt-get update
  apt-get install -y "${host_packages[@]}"
  exit 0
fi

if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo -n apt-get update
  sudo -n apt-get install -y "${host_packages[@]}"
  exit 0
fi

echo "Missing code hygiene tools or native analysis dependencies, and cannot install without passwordless sudo on this runner." >&2
printf 'Preinstall these native packages or enable passwordless sudo: %s\n' "${host_packages[*]}" >&2
exit 1
