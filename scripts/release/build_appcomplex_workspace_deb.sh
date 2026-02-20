#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
APP_DIR="$ROOT_DIR/tmp/core/sima-ai-appcomplex"
PKG_ROOT="$ROOT_DIR/packaging/appcomplex-workspace"
PKG_NAME="simaai-appcomplex-workspace"

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required" >&2
  exit 1
fi

if ! command -v make >/dev/null 2>&1; then
  echo "make is required" >&2
  exit 1
fi

VERSION=${APP_COMPLEX_PKG_VERSION:-}
if [ -z "$VERSION" ]; then
  if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    VERSION="$(git -C "$ROOT_DIR" describe --tags --always --dirty | tr '/' '-' | tr -cd '[:alnum:]._+-')"
  else
    VERSION="0.0.0-dev"
  fi
fi

if [[ ! "$VERSION" =~ ^[0-9] ]]; then
  VERSION="0.0.0+$VERSION"
fi

ARCH=${APP_COMPLEX_PKG_ARCH:-$(dpkg --print-architecture)}
OUT_DIR=${APP_COMPLEX_PKG_OUT_DIR:-"$ROOT_DIR/build/packages"}
BUILD_ROOT=${APP_COMPLEX_PKG_BUILD_ROOT:-"/tmp/${PKG_NAME}_build"}
STAGE_DIR="$BUILD_ROOT/stage"
DEBIAN_DIR="$STAGE_DIR/DEBIAN"

mkdir -p "$OUT_DIR"
rm -rf "$BUILD_ROOT"
mkdir -p "$DEBIAN_DIR"

echo "[build] Compiling appcomplex binaries"
make -C "$APP_DIR" -j"$(nproc)"

install -D -m 0755 "$APP_DIR/mlashmcomplex" \
  "$STAGE_DIR/opt/simaai/appcomplex-workspace/bin/mlashmcomplex"
install -D -m 0755 "$APP_DIR/init_mla_memory.sh" \
  "$STAGE_DIR/opt/simaai/appcomplex-workspace/bin/init_mla_memory.sh"

install -D -m 0755 "$APP_DIR/libevdispatch.so" \
  "$STAGE_DIR/opt/simaai/appcomplex-workspace/lib/libevdispatch.so"
install -D -m 0755 "$APP_DIR/libsgpdispatch.so" \
  "$STAGE_DIR/opt/simaai/appcomplex-workspace/lib/libsgpdispatch.so"
install -D -m 0755 "$APP_DIR/libsgptransport.so" \
  "$STAGE_DIR/opt/simaai/appcomplex-workspace/lib/libsgptransport.so"

if [ -f "$APP_DIR/helpers/libevhelpers.so" ]; then
  install -D -m 0755 "$APP_DIR/helpers/libevhelpers.so" \
    "$STAGE_DIR/opt/simaai/appcomplex-workspace/lib/libevhelpers.so"
fi

install -D -m 0644 \
  "$PKG_ROOT/lib/systemd/system/simaai-appcomplex-workspace.service" \
  "$STAGE_DIR/lib/systemd/system/simaai-appcomplex-workspace.service"
install -D -m 0644 \
  "$PKG_ROOT/etc/default/simaai-appcomplex-workspace" \
  "$STAGE_DIR/etc/default/simaai-appcomplex-workspace"

sed \
  -e "s/@PACKAGE@/$PKG_NAME/g" \
  -e "s/@VERSION@/$VERSION/g" \
  -e "s/@ARCH@/$ARCH/g" \
  "$PKG_ROOT/DEBIAN/control.in" > "$DEBIAN_DIR/control"

install -m 0755 "$PKG_ROOT/DEBIAN/postinst" "$DEBIAN_DIR/postinst"
install -m 0755 "$PKG_ROOT/DEBIAN/prerm" "$DEBIAN_DIR/prerm"
install -m 0755 "$PKG_ROOT/DEBIAN/postrm" "$DEBIAN_DIR/postrm"
install -m 0644 "$PKG_ROOT/DEBIAN/conffiles" "$DEBIAN_DIR/conffiles"

DEB_PATH="$OUT_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"
dpkg-deb --build "$STAGE_DIR" "$DEB_PATH" >/dev/null

echo "[build] Package generated: $DEB_PATH"
echo "$DEB_PATH"
