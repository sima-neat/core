#!/usr/bin/env bash
# Import the official B2798 Palette factory root as an offline test image.
set -euo pipefail

archive=${1:?Usage: import-b2798-card.sh IMAGE.img.gz LOCAL_TAG}
tag=${2:?Usage: import-b2798-card.sh IMAGE.img.gz LOCAL_TAG}
source_sha=5bd2fff65d2868f045318dfe65ae2ed88fc924b749ae04717dbc7a852d15ceb5
printf '%s  %s\n' "$source_sha" "$archive" | sha256sum --check
recipe_sha=$(sha256sum "$0")
recipe_sha=${recipe_sha%% *}

work=$(mktemp -d)
mnt=$work/root
mkdir "$mnt"
cleanup() {
    if mountpoint -q "$mnt"; then
        sudo umount "$mnt" || return
    fi
    rm -f "$work/image"
    rmdir "$mnt" "$work"
}
trap cleanup EXIT

gzip -dc -- "$archive" > "$work/image"
# In this checksum-pinned image, GPT partition 3 starts at sector 1050624.
# LVM rootfs.A is contiguous at physical extent 0, 2048 sectors into partition 3.
sudo mount -t ext4 -o "ro,noload,loop,offset=$((1052672*512)),sizelimit=$((1408*8192*512))" "$work/image" "$mnt"
sudo grep -Exq 'DISTRO_VERSION[[:space:]]*=[[:space:]]*3\.0\.1' "$mnt/etc/buildinfo"
sudo grep -Exq 'SIMA_BUILD_VERSION[[:space:]]*=[[:space:]]*3\.0\.1_master_B2798' "$mnt/etc/buildinfo"

# Keep the factory package database and filesystem ownership. Omit device,
# boot, transient state and credentials from the container fixture.
sudo tar --numeric-owner --xattrs --xattrs-include='*' --acls --one-file-system \
    --exclude='./etc/ssh/ssh_host_*' --exclude='./root/.ssh' --exclude='./home/*' \
    --exclude='./etc/apt/auth.conf' --exclude='./etc/apt/auth.conf.d/*' \
    --exclude='./etc/machine-id' --exclude='./var/lib/dbus/machine-id' \
    --exclude='./var/lib/systemd/random-seed' --exclude='./var/log/*' \
    --exclude='./var/cache/apt/archives/*.deb' --exclude='./var/lib/apt/lists/*' \
    --exclude='./boot/*' --exclude='./dev/*' --exclude='./proc/*' --exclude='./sys/*' \
    --exclude='./run/*' --exclude='./tmp/*' --exclude='./var/tmp/*' \
    --exclude='./data/*' --exclude='./lost+found' -C "$mnt" -cf - . |
    docker import --platform linux/arm64 \
        --change "LABEL ai.sima.offline.role=card ai.sima.offline.recipe=$recipe_sha ai.sima.offline.inputs=$source_sha" \
        - "$tag"
