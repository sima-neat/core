#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mode copy|symlink] [--source DIR] [--target DIR]

Installs the packaged SiMa NEAT Codex skill into the current user's Codex home.

Defaults:
  --mode   copy
  --source /usr/share/sima-neat/skills/sima-neat
  --target \$CODEX_HOME/skills/sima-neat (or \$HOME/.codex/skills/sima-neat)
EOF
}

mode="copy"
source_dir="/usr/share/sima-neat/skills/sima-neat"
codex_home="${CODEX_HOME:-$HOME/.codex}"
target_dir="${codex_home}/skills/sima-neat"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="${2:-}"
      shift 2
      ;;
    --source)
      source_dir="${2:-}"
      shift 2
      ;;
    --target)
      target_dir="${2:-}"
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

if [[ "$mode" != "copy" && "$mode" != "symlink" ]]; then
  echo "Invalid mode: $mode (expected copy or symlink)" >&2
  exit 1
fi

if [[ ! -d "$source_dir" ]]; then
  echo "Skill source not found: $source_dir" >&2
  exit 1
fi

mkdir -p "$(dirname "$target_dir")"

if [[ -e "$target_dir" || -L "$target_dir" ]]; then
  rm -rf "$target_dir"
fi

if [[ "$mode" == "symlink" ]]; then
  ln -s "$source_dir" "$target_dir"
else
  cp -a "$source_dir" "$target_dir"
fi

echo "Installed SiMa NEAT skill to: $target_dir"
