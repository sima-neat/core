#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: pcie-setup.sh [options]

Set up passwordless SSH from this host to SiMa PCIe cards.

Options:
  --user USER             Remote user (default: sima)
  --port PORT             Remote SSH port (default: 22)
  --key PATH              SSH private key path (default: ~/.ssh/sima_neat_pcie_ed25519)
  --password PASSWORD     Bootstrap password for sshpass/ssh-copy-id
  --hosts LIST            Space/comma-separated host list
  --range FIRST-LAST      Use static card index range 10.0.<index>.2
  --no-discover           Disable 10.0.N.1 -> 10.0.N.2 discovery; requires --hosts or --range
  --non-interactive       Do not prompt; requires working key or --password when needed
  --strict                Return non-zero if any target fails
  --dry-run               Print targets and commands without changing remote hosts
  -h, --help              Show this help

Examples:
  pcie-setup.sh
  pcie-setup.sh --hosts "10.0.0.2 10.0.1.2"
  pcie-setup.sh --range 0-3
  pcie-setup.sh --password '<bootstrap-password>'
EOF
}

user="sima"
port="22"
key_path="${HOME}/.ssh/sima_neat_pcie_ed25519"
hosts=()
range_first=0
range_last=9
range_explicit=0
hosts_explicit=0
discover=1
password=""
non_interactive=0
strict=0
dry_run=0

append_unique_host() {
  local candidate="$1"
  local existing
  [[ -n "${candidate}" ]] || return 0
  for existing in "${hosts[@]}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      return 0
    fi
  done
  hosts+=("${candidate}")
}

split_hosts() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  local parts=(${raw})
  local part
  for part in "${parts[@]}"; do
    append_unique_host "${part}"
  done
}

add_range_hosts() {
  local i
  for ((i = range_first; i <= range_last; ++i)); do
    append_unique_host "10.0.${i}.2"
  done
}

discover_pcie_hosts() {
  command -v ip >/dev/null 2>&1 || return 0

  local line ip_cidr local_ip o1 o2 o3 o4
  while IFS= read -r line; do
    # Example:
    # 4: enp3s0          inet 10.0.0.1/24 brd ...
    read -r _ _ _ ip_cidr _ <<< "${line}"

    local_ip="${ip_cidr%%/*}"
    IFS=. read -r o1 o2 o3 o4 <<< "${local_ip}"
    [[ "${o1}" == "10" && "${o2}" == "0" && -n "${o3}" && "${o4}" == "1" ]] || continue

    echo "${o1}.${o2}.${o3}.2"
  done < <(ip -o -4 addr show 2>/dev/null) | sort -u
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      user="${2:?missing value for --user}"
      shift 2
      ;;
    --port)
      port="${2:?missing value for --port}"
      shift 2
      ;;
    --key)
      key_path="${2:?missing value for --key}"
      shift 2
      ;;
    --password)
      password="${2:?missing value for --password}"
      shift 2
      ;;
    --hosts)
      split_hosts "${2:?missing value for --hosts}"
      hosts_explicit=1
      discover=0
      shift 2
      ;;
    --range)
      range_value="${2:?missing value for --range}"
      if [[ ! "${range_value}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        echo "Invalid --range '${range_value}', expected FIRST-LAST" >&2
        exit 2
      fi
      range_first="${BASH_REMATCH[1]}"
      range_last="${BASH_REMATCH[2]}"
      range_explicit=1
      discover=0
      shift 2
      ;;
    --no-discover)
      discover=0
      shift
      ;;
    --non-interactive)
      non_interactive=1
      shift
      ;;
    --strict)
      strict=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "Invalid port: ${port}" >&2
  exit 2
fi
if (( range_first > range_last )); then
  echo "Invalid range: ${range_first}-${range_last}" >&2
  exit 2
fi
if (( ${#hosts[@]} == 0 )); then
  if (( range_explicit )); then
    add_range_hosts
  elif (( discover )); then
    while IFS= read -r discovered_host; do
      append_unique_host "${discovered_host}"
    done < <(discover_pcie_hosts)
  fi
  if (( ${#hosts[@]} == 0 )); then
    echo "No PCIe card management hosts found. Connect a card or pass --hosts/--range." >&2
    exit 1
  fi
fi

pub_key="${key_path}.pub"

echo "SiMa NEAT PCIe SSH provisioning"
echo "User: ${user}"
echo "Port: ${port}"
echo "Key:  ${key_path}"
if (( discover )) && (( ! range_explicit )) && (( ! hosts_explicit )); then
  echo "Discovery: local 10.0.N.1 -> card 10.0.N.2"
else
  echo "Discovery: disabled"
fi
echo "Hosts: ${hosts[*]}"

if (( dry_run )); then
  echo "[dry-run] would create key if missing: ${key_path}"
else
  mkdir -p "$(dirname "${key_path}")"
  chmod 700 "$(dirname "${key_path}")"
  if [[ ! -f "${key_path}" ]]; then
    ssh-keygen -t ed25519 -N "" -f "${key_path}" -C "sima-neat-pcie@$(hostname)" >/dev/null
  fi
  if [[ ! -f "${pub_key}" ]]; then
    ssh-keygen -y -f "${key_path}" > "${pub_key}"
  fi
  chmod 600 "${key_path}"
  chmod 644 "${pub_key}"
fi

copy_id_base=(ssh-copy-id -i "${pub_key}" -p "${port}" -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)
if (( non_interactive )) && [[ -z "${password}" ]]; then
  copy_id_base+=(-o BatchMode=yes)
fi

success=0
failed=0
skipped=0

for host in "${hosts[@]}"; do
  target="${user}@${host}"
  echo
  echo "==> ${target}:${port}"

  if (( dry_run )); then
    echo "[dry-run] ssh-keyscan -H -p ${port} ${host} >> ~/.ssh/known_hosts"
    echo "[dry-run] ${copy_id_base[*]} ${target}"
    ((success += 1))
    continue
  fi

  ssh-keyscan -H -p "${port}" "${host}" >> "${HOME}/.ssh/known_hosts" 2>/dev/null || true
  chmod 600 "${HOME}/.ssh/known_hosts" 2>/dev/null || true

  if ssh -i "${key_path}" -p "${port}" -o BatchMode=yes -o ConnectTimeout=5 \
      -o StrictHostKeyChecking=accept-new "${target}" true >/dev/null 2>&1; then
    echo "Already provisioned."
    ((success += 1))
    continue
  fi

  if (( non_interactive )) && [[ -n "${password}" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      echo "sshpass is required for --password --non-interactive." >&2
      ((failed += 1))
      continue
    fi
    if SSHPASS="${password}" sshpass -e "${copy_id_base[@]}" "${target}"; then
      ((success += 1))
    else
      ((failed += 1))
    fi
  elif (( non_interactive )); then
    echo "Skipped: key not installed and no --password provided in non-interactive mode."
    ((skipped += 1))
  else
    if "${copy_id_base[@]}" "${target}"; then
      ((success += 1))
    else
      ((failed += 1))
    fi
  fi
done

echo
echo "Provisioning summary: success=${success} failed=${failed} skipped=${skipped}"

if (( strict )) && (( failed > 0 || skipped > 0 )); then
  exit 1
fi
if (( success == 0 )); then
  exit 1
fi
