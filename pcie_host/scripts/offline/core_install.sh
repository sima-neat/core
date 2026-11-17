#!/bin/sh
# Preserve SDK pairing before sudo filters the invoking user's environment.
set -eu
cd -- "$(dirname -- "$0")"
case "${1-}" in
    -h|--help) exec python3 -B ./install.py "$@" ;;
esac
if [ "$(id -u)" -ne 0 ]; then
    echo 'Requesting administrator privileges for offline Core installation.'
    exec sudo --preserve-env=DEVKIT_SYNC_DEVKIT_IP,DEVKIT_DEPLOY_USER,SUDO_PASSWORD,DEVKIT_PASSWORD,NEAT_INSTALLER_SKIP_DEVKIT_SYNC,SSH_AUTH_SOCK,NO_COLOR \
        -- python3 -B "$PWD/install.py" "$@"
fi
exec python3 -B ./install.py "$@"
