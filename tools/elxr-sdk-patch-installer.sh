#!/usr/bin/env bash
set -euo pipefail

# elxr-sdk-patch-installer.sh
# - Optionally removes custom SIMA apt repo line from elxr.list
# - Recursively finds and extracts all .deb files from DEB_DIR into SYSROOT
# - Normalizes BLAS/LAPACK/OpenBLAS symlinks for cross-linking
# - Exports/prints pkg-config env for cross builds
# - Intended for eLxr SDK sysroot preparation before cross-compiling NEAT
#
# Usage:
#   bash scripts/elxr-sdk-patch-installer.sh [SYSROOT] [DEB_DIR]
#
# Defaults:
#   SYSROOT=/opt/toolchain/aarch64/modalix
#   DEB_DIR=current working directory

SYSROOT="${1:-/opt/toolchain/aarch64/modalix}"
DEB_DIR="${2:-$PWD}"
ELXR_LIST="/etc/apt/sources.list.d/elxr.list"
LIBDIR="$SYSROOT/usr/lib/aarch64-linux-gnu"

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
err() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

if [[ ! -d "$DEB_DIR" ]]; then
  err "Deb dir does not exist: $DEB_DIR"
fi

# Use sudo only if needed.
SUDO=""
if [[ ! -w "$SYSROOT" ]] || [[ -e "$ELXR_LIST" && ! -w "$ELXR_LIST" ]]; then
  SUDO="sudo"
fi

log "Sysroot: $SYSROOT"
log "Deb dir: $DEB_DIR"

# 1) Optional repo cleanup
if [[ -f "$ELXR_LIST" ]]; then
  log "Removing sw-web.eng.sima.ai entry from $ELXR_LIST"
  $SUDO cp "$ELXR_LIST" "${ELXR_LIST}.bak.$(date +%Y%m%d%H%M%S)"
  tmp_file="$(mktemp)"
  awk '!/sw-web\.eng\.sima\.ai\/deb\/custom/' "$ELXR_LIST" > "$tmp_file"
  $SUDO install -m 0644 "$tmp_file" "$ELXR_LIST"
  rm -f "$tmp_file"
else
  warn "$ELXR_LIST not found, skipping repo cleanup"
fi

# 2) Find all .deb files in DEB_DIR recursively
mapfile -t DEBS < <(find "$DEB_DIR" -type f -name '*.deb' | sort)
if (( ${#DEBS[@]} == 0 )); then
  err "No .deb files found under: $DEB_DIR"
fi

log "Found ${#DEBS[@]} deb(s):"
printf '  %s\n' "${DEBS[@]}"

# 3) Extract all debs into sysroot
$SUDO mkdir -p "$SYSROOT"
for d in "${DEBS[@]}"; do
  log "Extracting $(basename "$d") -> $SYSROOT"
  $SUDO dpkg-deb -x "$d" "$SYSROOT"
done

$SUDO mkdir -p "$LIBDIR"

# 4) Normalize BLAS/LAPACK symlinks for linker compatibility
log "Normalizing BLAS/LAPACK symlinks under $LIBDIR"

if [[ -f "$LIBDIR/openblas-pthread/libblas.so.3" && -f "$LIBDIR/openblas-pthread/liblapack.so.3" ]]; then
  $SUDO ln -sfn openblas-pthread/libblas.so.3 "$LIBDIR/libblas.so.3"
  $SUDO ln -sfn openblas-pthread/liblapack.so.3 "$LIBDIR/liblapack.so.3"
elif [[ -f "$LIBDIR/blas/libblas.so.3" && -f "$LIBDIR/lapack/liblapack.so.3" ]]; then
  $SUDO ln -sfn blas/libblas.so.3 "$LIBDIR/libblas.so.3"
  $SUDO ln -sfn lapack/liblapack.so.3 "$LIBDIR/liblapack.so.3"
else
  warn "Could not find expected BLAS/LAPACK provider libs in openblas-pthread or blas/lapack dirs."
fi

# Ensure unversioned linker names exist
if [[ -e "$LIBDIR/libblas.so.3" ]]; then
  $SUDO ln -sfn libblas.so.3 "$LIBDIR/libblas.so"
fi
if [[ -e "$LIBDIR/liblapack.so.3" ]]; then
  $SUDO ln -sfn liblapack.so.3 "$LIBDIR/liblapack.so"
fi

# Ensure top-level OpenBLAS SONAME links exist.
# dpkg-deb -x does not run postinst/update-alternatives, so these may be missing.
if [[ ! -e "$LIBDIR/libopenblas.so.0" ]]; then
  cand="$(find "$LIBDIR/openblas-pthread" -maxdepth 1 -type f -name 'libopenblas*.so*' | head -n1 || true)"
  if [[ -z "${cand:-}" ]]; then
    cand="$(find "$LIBDIR" -maxdepth 2 -type f -name 'libopenblas*.so*' | head -n1 || true)"
  fi
  if [[ -n "${cand:-}" ]]; then
    rel_target="$(realpath --relative-to="$LIBDIR" "$cand")"
    $SUDO ln -sfn "$rel_target" "$LIBDIR/libopenblas.so.0"
    log "Created libopenblas.so.0 -> $rel_target"
  fi
fi

if [[ -e "$LIBDIR/libopenblas.so.0" && ! -e "$LIBDIR/libopenblas.so" ]]; then
  $SUDO ln -sfn libopenblas.so.0 "$LIBDIR/libopenblas.so"
fi

log "Final symlink targets:"
readlink -f "$LIBDIR/libblas.so.3" || true
readlink -f "$LIBDIR/liblapack.so.3" || true
readlink -f "$LIBDIR/libopenblas.so.0" || true
readlink -f "$LIBDIR/libopenblas.so" || true

# Hard fail if key OpenBLAS SONAME still missing
if [[ ! -e "$LIBDIR/libopenblas.so.0" ]]; then
  err "Missing $LIBDIR/libopenblas.so.0 after extraction/symlink normalization."
fi

# 5) Cross pkg-config env
export PKG_CONFIG_SYSROOT_DIR="$SYSROOT"
export PKG_CONFIG_LIBDIR="$SYSROOT/usr/lib/aarch64-linux-gnu/pkgconfig:$SYSROOT/usr/lib/pkgconfig:$SYSROOT/usr/share/pkgconfig"
unset PKG_CONFIG_PATH || true

log "PKG_CONFIG_SYSROOT_DIR=$PKG_CONFIG_SYSROOT_DIR"
log "PKG_CONFIG_LIBDIR=$PKG_CONFIG_LIBDIR"

log "Quick checks:"
pkg-config --libs --cflags gstreamer-rtsp-server-1.0 || true
pkg-config --libs --cflags opencv4 || true

log "readelf dependency check for liblapack.so.3:"
readelf -d "$LIBDIR/liblapack.so.3" 2>/dev/null | grep NEEDED || true

log "Suggested linker flags:"
echo "export LDFLAGS=\"--sysroot=$SYSROOT -L$LIBDIR -L$LIBDIR/blas -L$LIBDIR/lapack -L$LIBDIR/openblas-pthread -L$SYSROOT/lib/aarch64-linux-gnu -Wl,-rpath-link,$LIBDIR -Wl,-rpath-link,$LIBDIR/blas -Wl,-rpath-link,$LIBDIR/lapack -Wl,-rpath-link,$LIBDIR/openblas-pthread -Wl,-rpath-link,$SYSROOT/lib/aarch64-linux-gnu \${LDFLAGS:-}\""

log "Done: extracted debs and normalized sysroot symlinks."
