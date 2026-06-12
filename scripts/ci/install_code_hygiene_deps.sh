#!/usr/bin/env bash
set -euo pipefail

clang_cpp_works() {
  command -v clang++ >/dev/null 2>&1 || return 1
  printf '%s\n' '#include <memory>' | clang++ -std=c++20 -x c++ - -fsyntax-only >/dev/null 2>&1
}

if command -v clang-format >/dev/null 2>&1 \
  && command -v clang-tidy >/dev/null 2>&1 \
  && command -v run-clang-tidy >/dev/null 2>&1 \
  && command -v rg >/dev/null 2>&1 \
  && command -v g++ >/dev/null 2>&1 \
  && clang_cpp_works; then
  echo "Code hygiene tools and host C++ toolchain are already installed."
  exit 0
fi

if [ "$(id -u)" -eq 0 ]; then
  apt-get update
  apt-get install -y build-essential gcc-12 g++-12 libstdc++-12-dev clang clang-format clang-tidy ripgrep
  exit 0
fi

if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo -n apt-get update
  sudo -n apt-get install -y build-essential gcc-12 g++-12 libstdc++-12-dev clang clang-format clang-tidy ripgrep
  exit 0
fi

echo "Missing code hygiene tools or host C++ toolchain packages, and cannot install without passwordless sudo on this runner." >&2
echo "Preinstall build-essential, gcc-12, g++-12, libstdc++-12-dev, clang, clang-format, clang-tidy, and ripgrep on the x86 runner or enable passwordless sudo for the runner user." >&2
exit 1
