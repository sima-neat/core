#!/usr/bin/env bash
set -euo pipefail

readonly FAULT_RE='BUG:|Oops:|Kernel panic|Call trace:|KASAN:|use-after-free|refcount_t:|IOMMU[^[:cntrl:]]*fault|mla[^[:cntrl:]]*(timeout|timed out|fault|hang)|cvu[^[:cntrl:]]*(timeout|timed out|fault|hang)|remoteproc[^[:cntrl:]]*(crash|fatal)|rcu[^[:cntrl:]]*stall|blocked for more than [0-9]+ seconds'

suite=smoke
bench=
native_model=
packed_model=
evidence_dir=
plugin_dir=
library_path=
manage_services=0
plan_only=0

usage() {
  cat <<'EOF'
Usage: run_evo_a65_dmabuf_hil.sh [options]

Required for a hardware run:
  --bench PATH          AArch64 evo_tput_bench executable
  --native-model PATH   EVO multibuffer MPK
  --packed-model PATH   EVO packed/named MPK
  --evidence DIR        Empty/new evidence directory

Options:
  --suite smoke|full    Per-candidate smoke or complete release matrix
  --plugin-dir DIR      Candidate Neat GStreamer plugin directory
  --library-path PATH   Candidate colon-separated runtime library path
  --manage-services 0|1 Stop appcomplex/RPyC during the gate and restore them
  --plan-only           Print the exact matrix without touching hardware

Optional release locks:
  SIMA_EVO_HIL_EXPECT_KERNEL
  SIMA_EVO_HIL_EXPECT_CVU_FIRMWARE_SHA256
  SIMA_EVO_HIL_EXPECT_PROCESSCVU_SHA256
  SIMA_EVO_HIL_EXPECT_NATIVE_MODEL_SHA256
  SIMA_EVO_HIL_EXPECT_PACKED_MODEL_SHA256

The smoke suite runs 32 throughput cases, 8 correctness cases, and 6
release-while-in-flight cases. The full suite runs 48, 8, and 24 respectively.
EOF
}

die() {
  printf 'EVO_A65_HIL_FATAL: %s\n' "$*" >&2
  exit 98
}

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "missing value for $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      require_value "$@"
      suite=$2
      shift 2
      ;;
    --bench)
      require_value "$@"
      bench=$2
      shift 2
      ;;
    --native-model)
      require_value "$@"
      native_model=$2
      shift 2
      ;;
    --packed-model)
      require_value "$@"
      packed_model=$2
      shift 2
      ;;
    --evidence)
      require_value "$@"
      evidence_dir=$2
      shift 2
      ;;
    --plugin-dir)
      require_value "$@"
      plugin_dir=$2
      shift 2
      ;;
    --library-path)
      require_value "$@"
      library_path=$2
      shift 2
      ;;
    --manage-services)
      require_value "$@"
      manage_services=$2
      shift 2
      ;;
    --plan-only)
      plan_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$suite" == smoke || "$suite" == full ]] ||
  die "--suite must be smoke or full"
[[ "$manage_services" == 0 || "$manage_services" == 1 ]] ||
  die "--manage-services must be 0 or 1"

readonly -a PLACEMENTS=(EV74:EV74 A65:EV74 EV74:A65 A65:A65)
readonly -a A65_PLACEMENTS=(A65:EV74 EV74:A65 A65:A65)
readonly -a ROUTES=(native packed)
readonly -a MODES=(sync async)
if [[ "$suite" == full ]]; then
  readonly -a DEPTHS=(1 2 4)
  readonly WARMUP=2
  readonly MEASURED=25
  readonly RELEASE_ITERATIONS=4
else
  readonly -a DEPTHS=(1 4)
  readonly WARMUP=1
  readonly MEASURED=3
  readonly RELEASE_ITERATIONS=1
fi

print_plan() {
  local placement pre post route mode depth iteration
  local throughput=0 correctness=0 release=0
  for placement in "${PLACEMENTS[@]}"; do
    IFS=: read -r pre post <<<"$placement"
    for route in "${ROUTES[@]}"; do
      for mode in "${MODES[@]}"; do
        for depth in "${DEPTHS[@]}"; do
          printf 'EVO_A65_HIL_PLAN kind=throughput pre=%s post=%s route=%s mode=%s depth=%s\n' \
            "$pre" "$post" "$route" "$mode" "$depth"
          ((throughput += 1))
        done
      done
      printf 'EVO_A65_HIL_PLAN kind=correctness pre=%s post=%s route=%s mode=sync depth=1\n' \
        "$pre" "$post" "$route"
      ((correctness += 1))
    done
  done
  for placement in "${A65_PLACEMENTS[@]}"; do
    IFS=: read -r pre post <<<"$placement"
    for route in "${ROUTES[@]}"; do
      for ((iteration = 1; iteration <= RELEASE_ITERATIONS; ++iteration)); do
        printf 'EVO_A65_HIL_PLAN kind=release pre=%s post=%s route=%s mode=async depth=4 iteration=%d\n' \
          "$pre" "$post" "$route" "$iteration"
        ((release += 1))
      done
    done
  done
  printf 'EVO_A65_HIL_PLAN_RESULT status=PASS suite=%s throughput=%d correctness=%d release=%d total=%d\n' \
    "$suite" "$throughput" "$correctness" "$release" \
    "$((throughput + correctness + release))"
}

if [[ "$plan_only" == 1 ]]; then
  print_plan
  exit 0
fi

[[ -n "$bench" && -n "$native_model" && -n "$packed_model" &&
   -n "$evidence_dir" ]] || {
  usage >&2
  die "hardware runs require --bench, both models, and --evidence"
}
[[ -x "$bench" ]] || die "benchmark is not executable: $bench"
[[ -f "$native_model" ]] || die "native model is missing: $native_model"
[[ -f "$packed_model" ]] || die "packed model is missing: $packed_model"
[[ -z "$plugin_dir" || -d "$plugin_dir" ]] ||
  die "plugin directory is missing: $plugin_dir"
command -v flock >/dev/null || die "flock is required"
command -v timeout >/dev/null || die "timeout is required"
command -v readelf >/dev/null || die "readelf is required"
[[ "$(uname -m)" == aarch64 ]] || die "hardware gate must run on AArch64"
readelf -h "$bench" | grep -Eq 'Machine:[[:space:]]+AArch64' ||
  die "benchmark is not AArch64"
[[ -c /dev/mla ]] || die "/dev/mla is missing"
[[ -c /dev/cvu ]] || die "/dev/cvu is missing"

mkdir -p "$evidence_dir"/{accounting,kernel,logs,registries}
exec 9>"${SIMA_EVO_HIL_LOCK_FILE:-/var/lock/sima-evo-a65-dmabuf-hil.lock}"
flock -n 9 || die "another EVO A65 HIL gate owns the board"

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo -n "$@"
  fi
}

declare -a restore_services=()
restore_board_services() {
  local service
  for service in "${restore_services[@]}"; do
    as_root systemctl start "$service" || true
  done
}
trap restore_board_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$manage_services" == 1 ]]; then
  for service in simaai-appcomplex.service simaai-rpyc-server.service; do
    if systemctl is-active --quiet "$service"; then
      restore_services+=("$service")
      as_root systemctl stop "$service"
    fi
  done
fi

assert_no_competing_processes() {
  local bad
  bad=$(ps -eo pid=,comm=,args= | awk '
    $2 == "simaai-appcomplex" ||
    $2 == "mlashmcomplex" ||
    $2 == "simamem" ||
    $2 == "mla-dispatcher" ||
    $2 == "simaai-rpyc-server" ||
    $2 == "m4" ||
    $3 ~ /\/(simaai-appcomplex|mlashmcomplex|simamem|mla-dispatcher|simaai-rpyc-server)$/ { print }
  ')
  [[ -z "$bad" ]] || {
    printf '%s\n' "$bad" >&2
    die "a competing accelerator process is running"
  }
}

check_expected_sha() {
  local path=$1 expected=$2 label=$3 actual
  [[ -z "$expected" ]] && return 0
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] ||
    die "$label SHA256 mismatch: expected $expected, got $actual"
}

if [[ -n "${SIMA_EVO_HIL_EXPECT_KERNEL:-}" ]]; then
  [[ "$(uname -r)" == "$SIMA_EVO_HIL_EXPECT_KERNEL" ]] ||
    die "kernel mismatch: expected $SIMA_EVO_HIL_EXPECT_KERNEL, got $(uname -r)"
fi
check_expected_sha /lib/firmware/modalix-cvu-fw \
  "${SIMA_EVO_HIL_EXPECT_CVU_FIRMWARE_SHA256:-}" 'CVU firmware'
check_expected_sha "$native_model" \
  "${SIMA_EVO_HIL_EXPECT_NATIVE_MODEL_SHA256:-}" 'native model'
check_expected_sha "$packed_model" \
  "${SIMA_EVO_HIL_EXPECT_PACKED_MODEL_SHA256:-}" 'packed model'
if [[ -n "${SIMA_EVO_HIL_EXPECT_PROCESSCVU_SHA256:-}" ]]; then
  [[ -n "$plugin_dir" ]] ||
    die "a processcvu hash lock requires --plugin-dir"
  check_expected_sha "$plugin_dir/libgstneatprocesscvu.so" \
    "$SIMA_EVO_HIL_EXPECT_PROCESSCVU_SHA256" 'ProcessCVU plugin'
fi

assert_no_competing_processes

readonly FRAME_TIMEOUT_MS="${SIMA_EVO_HIL_FRAME_TIMEOUT_MS:-180000}"
readonly COMMAND_TIMEOUT_SECONDS="${SIMA_EVO_HIL_COMMAND_TIMEOUT_SECONDS:-600}"
readonly ACCOUNTING_RETRIES="${SIMA_EVO_HIL_ACCOUNTING_RETRIES:-15}"
readonly ACCOUNTING_RETRY_DELAY_SECONDS="${SIMA_EVO_HIL_ACCOUNTING_RETRY_DELAY_SECONDS:-1}"
readonly CMA_TOLERANCE_KB="${SIMA_EVO_HIL_CMA_TOLERANCE_KB:-65536}"
readonly EXPECTED_OUTPUTS="${SIMA_EVO_HIL_EXPECTED_OUTPUTS:-28}"

for integer_value in "$FRAME_TIMEOUT_MS" "$COMMAND_TIMEOUT_SECONDS" \
  "$ACCOUNTING_RETRIES" "$ACCOUNTING_RETRY_DELAY_SECONDS" \
  "$CMA_TOLERANCE_KB" "$EXPECTED_OUTPUTS"; do
  [[ "$integer_value" =~ ^[0-9]+$ ]] || die "invalid numeric gate setting"
done

run_benchmark() {
  local registry=$1 depth=$2
  shift 2
  (
    unset GST_PLUGIN_PATH GST_PLUGIN_PATH_1_0
    unset GST_PLUGIN_SYSTEM_PATH GST_PLUGIN_SYSTEM_PATH_1_0
    unset SIMA_PROCESSCVU_A65_ASYNC
    export SIMA_NEAT_MEMORY_BACKEND=dmabuf-plan
    export SIMA_PROCESSMLA_SAFE_ASYNC_DEPTH="$depth"
    export SIMA_GST_RESPECT_REGISTRY=1
    export GST_REGISTRY="$registry"
    [[ -z "$plugin_dir" ]] || export SIMA_GST_PLUGIN_DIR="$plugin_dir"
    if [[ -n "$library_path" ]]; then
      export LD_LIBRARY_PATH="$library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    exec timeout --foreground "${COMMAND_TIMEOUT_SECONDS}s" "$bench" "$@"
  )
}

SNAP_CMA=0
SNAP_DMA_OBJECTS=0
SNAP_DMA_BYTES=0
SNAP_MLA_INC=0
SNAP_MLA_DEC=0
SNAP_MLA_DELTA=0
SNAP_MLA_OUTSTANDING=0
SNAP_MLA_CONTEXTS=0
BASE_CMA=0
BASE_DMA_OBJECTS=0
BASE_DMA_BYTES=0

snapshot() {
  local label=$1 attempt=$2 dir
  dir="$evidence_dir/accounting/$label/attempt-$attempt"
  mkdir -p "$dir"
  date -u +%FT%TZ >"$dir/timestamp-utc.txt"
  cat /proc/meminfo >"$dir/meminfo.txt"
  as_root cat /sys/kernel/debug/dma_buf/bufinfo >"$dir/dma-buf-bufinfo.txt"
  as_root cat /sys/kernel/debug/mla/dump >"$dir/mla-dump.txt"

  SNAP_CMA=$(awk '$1 == "CmaFree:" {print $2; exit}' "$dir/meminfo.txt")
  read -r SNAP_DMA_OBJECTS SNAP_DMA_BYTES < <(
    awk '$1 == "Total" && $3 == "objects," && $5 == "bytes" {
      gsub(/,/, "", $2); gsub(/,/, "", $4); print $2, $4; exit
    }' "$dir/dma-buf-bufinfo.txt"
  )
  SNAP_MLA_INC=$(sed -n 's/^buf_inc_ops=\([0-9][0-9]*\) buf_dec_ops=[0-9][0-9]* delta=-\{0,1\}[0-9][0-9]*$/\1/p' "$dir/mla-dump.txt" | head -n1)
  SNAP_MLA_DEC=$(sed -n 's/^buf_inc_ops=[0-9][0-9]* buf_dec_ops=\([0-9][0-9]*\) delta=-\{0,1\}[0-9][0-9]*$/\1/p' "$dir/mla-dump.txt" | head -n1)
  SNAP_MLA_DELTA=$(sed -n 's/^buf_inc_ops=[0-9][0-9]* buf_dec_ops=[0-9][0-9]* delta=\(-\{0,1\}[0-9][0-9]*\)$/\1/p' "$dir/mla-dump.txt" | head -n1)
  SNAP_MLA_OUTSTANDING=$(sed -n 's/^outstanding_alloc_bytes=\([0-9][0-9]*\)$/\1/p' "$dir/mla-dump.txt" | head -n1)
  SNAP_MLA_CONTEXTS=$(sed -n 's/^=== contexts (\([0-9][0-9]*\) open) ===$/\1/p' "$dir/mla-dump.txt" | head -n1)
  [[ "$SNAP_CMA" =~ ^[0-9]+$ && "$SNAP_DMA_OBJECTS" =~ ^[0-9]+$ &&
     "$SNAP_DMA_BYTES" =~ ^[0-9]+$ && "$SNAP_MLA_INC" =~ ^[0-9]+$ &&
     "$SNAP_MLA_DEC" =~ ^[0-9]+$ && "$SNAP_MLA_DELTA" =~ ^-?[0-9]+$ &&
     "$SNAP_MLA_OUTSTANDING" =~ ^[0-9]+$ && "$SNAP_MLA_CONTEXTS" =~ ^[0-9]+$ ]] ||
    die "unparseable accounting snapshot: $label attempt $attempt"
}

assert_idle() {
  local label=$1 attempt
  for ((attempt = 1; attempt <= ACCOUNTING_RETRIES; ++attempt)); do
    snapshot "$label" "$attempt"
    if [[ "$SNAP_DMA_OBJECTS" -le "$BASE_DMA_OBJECTS" &&
          "$SNAP_DMA_BYTES" -le "$BASE_DMA_BYTES" &&
          "$SNAP_MLA_DELTA" -eq 0 && "$SNAP_MLA_OUTSTANDING" -eq 0 &&
          "$SNAP_MLA_CONTEXTS" -eq 0 &&
          $((SNAP_CMA + CMA_TOLERANCE_KB)) -ge "$BASE_CMA" ]]; then
      printf 'ACCOUNTING_RESULT status=PASS label=%s attempt=%d cma_free_kb=%d dma_objects=%d dma_bytes=%d mla_inc=%d mla_dec=%d mla_delta=%d mla_outstanding=%d mla_contexts=%d\n' \
        "$label" "$attempt" "$SNAP_CMA" "$SNAP_DMA_OBJECTS" \
        "$SNAP_DMA_BYTES" "$SNAP_MLA_INC" "$SNAP_MLA_DEC" \
        "$SNAP_MLA_DELTA" "$SNAP_MLA_OUTSTANDING" "$SNAP_MLA_CONTEXTS"
      return 0
    fi
    sleep "$ACCOUNTING_RETRY_DELAY_SECONDS"
  done
  return 1
}

DMESG_CURSOR=0
capture_kernel_baseline() {
  as_root dmesg >"$evidence_dir/kernel/dmesg-start.txt"
  DMESG_CURSOR=$(wc -l <"$evidence_dir/kernel/dmesg-start.txt")
  if grep -Eai "$FAULT_RE" "$evidence_dir/kernel/dmesg-start.txt" \
      >"$evidence_dir/kernel/preexisting-fault-scan.txt"; then
    [[ "${SIMA_EVO_HIL_ALLOW_PREEXISTING_FAULTS:-0}" == 1 ]] ||
      die "preexisting kernel fault signature"
  else
    printf 'none\n' >"$evidence_dir/kernel/preexisting-fault-scan.txt"
  fi
}

scan_kernel() {
  local label=$1 current delta
  current="$evidence_dir/kernel/dmesg-$label.txt"
  delta="$evidence_dir/kernel/dmesg-$label-delta.txt"
  as_root dmesg >"$current"
  tail -n "+$((DMESG_CURSOR + 1))" "$current" >"$delta"
  DMESG_CURSOR=$(wc -l <"$current")
  if grep -Eai "$FAULT_RE" "$delta" \
      >"$evidence_dir/kernel/dmesg-$label-fault-scan.txt"; then
    return 1
  fi
  printf 'none\n' >"$evidence_dir/kernel/dmesg-$label-fault-scan.txt"
}

{
  printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'suite=%s\n' "$suite"
  printf 'boot_id=%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
  printf 'kernel=%s\n' "$(uname -r)"
  printf 'machine=%s\n' "$(uname -m)"
  sha256sum "$bench" "$native_model" "$packed_model" \
    /lib/firmware/modalix-cvu-fw
  if [[ -n "$plugin_dir" ]]; then
    sha256sum "$plugin_dir/libgstneatprocesscvu.so"
  fi
  readelf -h "$bench" | grep -E 'Class:|Machine:'
} >"$evidence_dir/preflight.txt"

snapshot baseline 1
[[ "$SNAP_MLA_DELTA" -eq 0 && "$SNAP_MLA_OUTSTANDING" -eq 0 &&
   "$SNAP_MLA_CONTEXTS" -eq 0 ]] || die "board is not idle at baseline"
BASE_CMA=$SNAP_CMA
BASE_DMA_OBJECTS=$SNAP_DMA_OBJECTS
BASE_DMA_BYTES=$SNAP_DMA_BYTES
capture_kernel_baseline

throughput_results="$evidence_dir/throughput-results.tsv"
correctness_results="$evidence_dir/correctness-results.tsv"
release_results="$evidence_dir/release-results.tsv"
printf 'label\tpre\tpost\troute\tmode\tdepth\trc\tmarker\taccounting\tkernel\tstatus\n' \
  >"$throughput_results"
printf 'label\tpre\tpost\troute\thash\toutputs\tbytes\trc\taccounting\tkernel\tstatus\n' \
  >"$correctness_results"
printf 'label\tpre\tpost\troute\titeration\trc\tmarker\taccounting\tkernel\tstatus\n' \
  >"$release_results"

failures=0
throughput_total=0
correctness_total=0
release_total=0

run_throughput_case() {
  local pre=$1 post=$2 route=$3 mode=$4 depth=$5 model label log rc
  local expected marker=FAIL accounting=FAIL kernel=FAIL status=FAIL
  label="pre${pre}-post${post}-${route}-${mode}-depth${depth}"
  [[ "$route" == native ]] && model=$native_model || model=$packed_model
  log="$evidence_dir/logs/throughput-$label.log"
  set +e
  run_benchmark "$evidence_dir/registries/throughput-$label.bin" "$depth" \
    --model "$model" --pre "$pre" --post "$post" --warmup "$WARMUP" \
    --measured "$MEASURED" --timeout-ms "$FRAME_TIMEOUT_MS" --cleanup 1 \
    --plugin-latency 1 --startup-preflight 1 --mode "$mode" \
    --inflight "$depth" --mla-only 0 --verbose 1 >"$log" 2>&1
  rc=$?
  set -e
  [[ "$mode" == sync ]] && expected=$((MEASURED * EXPECTED_OUTPUTS)) || expected=$MEASURED
  if [[ "$rc" -eq 0 ]] &&
     grep -q "EVO_TPUT_RESULT status=PASS .* pre=$pre post=$post mode=$mode inflight=$depth measured=$MEASURED outputs=$expected" "$log"; then
    marker=PASS
  fi
  assert_idle "$label" && accounting=PASS
  scan_kernel "$label" && kernel=PASS
  if [[ "$rc" -eq 0 && "$marker" == PASS && "$accounting" == PASS &&
        "$kernel" == PASS ]]; then
    status=PASS
  else
    ((failures += 1))
  fi
  ((throughput_total += 1))
  printf '%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\n' \
    "$label" "$pre" "$post" "$route" "$mode" "$depth" "$rc" \
    "$marker" "$accounting" "$kernel" "$status" >>"$throughput_results"
  printf 'EVO_A65_HIL_CASE kind=throughput status=%s label=%s\n' "$status" "$label"
  [[ "$accounting" == PASS && "$kernel" == PASS ]] ||
    die "unsafe terminal state after $label"
}

for placement in "${PLACEMENTS[@]}"; do
  IFS=: read -r pre post <<<"$placement"
  for route in "${ROUTES[@]}"; do
    for mode in "${MODES[@]}"; do
      for depth in "${DEPTHS[@]}"; do
        run_throughput_case "$pre" "$post" "$route" "$mode" "$depth"
      done
    done
  done
done

for route in "${ROUTES[@]}"; do
  [[ "$route" == native ]] && model=$native_model || model=$packed_model
  reference=
  for placement in "${PLACEMENTS[@]}"; do
    IFS=: read -r pre post <<<"$placement"
    label="${route}-pre${pre}-post${post}"
    log="$evidence_dir/logs/correctness-$label.log"
    set +e
    run_benchmark "$evidence_dir/registries/correctness-$label.bin" 1 \
      --model "$model" --pre "$pre" --post "$post" --warmup 0 \
      --measured 1 --timeout-ms "$FRAME_TIMEOUT_MS" --cleanup 1 \
      --startup-preflight 1 --mode sync --inflight 1 --verbose 1 \
      --correctness-hash 1 >"$log" 2>&1
    rc=$?
    set -e
    line=$(grep 'EVO_CORRECTNESS_HASH status=PASS' "$log" | tail -1 || true)
    hash=$(sed -n 's/.* combined=\([^ ]*\).*/\1/p' <<<"$line")
    outputs=$(sed -n 's/.* outputs=\([^ ]*\).*/\1/p' <<<"$line")
    bytes=$(sed -n 's/.* bytes=\([^ ]*\).*/\1/p' <<<"$line")
    [[ -n "$reference" ]] || reference=$hash
    accounting=FAIL
    kernel=FAIL
    status=FAIL
    assert_idle "correctness-$label" && accounting=PASS
    scan_kernel "correctness-$label" && kernel=PASS
    if [[ "$rc" -eq 0 && -n "$hash" && "$hash" == "$reference" &&
          "$outputs" == "$EXPECTED_OUTPUTS" && "$bytes" =~ ^[1-9][0-9]*$ &&
          "$accounting" == PASS && "$kernel" == PASS ]]; then
      status=PASS
    else
      ((failures += 1))
    fi
    ((correctness_total += 1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' \
      "$label" "$pre" "$post" "$route" "$hash" "$outputs" "$bytes" \
      "$rc" "$accounting" "$kernel" "$status" >>"$correctness_results"
    printf 'EVO_A65_HIL_CASE kind=correctness status=%s label=%s hash=%s\n' \
      "$status" "$label" "$hash"
    [[ "$accounting" == PASS && "$kernel" == PASS ]] ||
      die "unsafe terminal state after correctness $label"
  done
done

for placement in "${A65_PLACEMENTS[@]}"; do
  IFS=: read -r pre post <<<"$placement"
  for route in "${ROUTES[@]}"; do
    [[ "$route" == native ]] && model=$native_model || model=$packed_model
    for ((iteration = 1; iteration <= RELEASE_ITERATIONS; ++iteration)); do
      label="pre${pre}-post${post}-${route}-async-depth4-release-${iteration}"
      log="$evidence_dir/logs/release-$label.log"
      set +e
      run_benchmark "$evidence_dir/registries/release-$label.bin" 4 \
        --model "$model" --pre "$pre" --post "$post" --warmup 0 \
        --measured 1 --timeout-ms "$FRAME_TIMEOUT_MS" --mode async \
        --inflight 4 --early-stop-after 8 --early-stop-delay-ms 0 \
        --startup-preflight 1 --verbose 1 >"$log" 2>&1
      rc=$?
      set -e
      marker=FAIL
      accounting=FAIL
      kernel=FAIL
      status=FAIL
      if [[ "$rc" -eq 0 ]] &&
         grep -q 'EVO_EARLY_STOP_RESULT status=PASS pushed=8' "$log"; then
        marker=PASS
      fi
      assert_idle "$label" && accounting=PASS
      scan_kernel "$label" && kernel=PASS
      if [[ "$rc" -eq 0 && "$marker" == PASS && "$accounting" == PASS &&
            "$kernel" == PASS ]]; then
        status=PASS
      else
        ((failures += 1))
      fi
      ((release_total += 1))
      printf '%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\n' \
        "$label" "$pre" "$post" "$route" "$iteration" "$rc" "$marker" \
        "$accounting" "$kernel" "$status" >>"$release_results"
      printf 'EVO_A65_HIL_CASE kind=release status=%s label=%s\n' "$status" "$label"
      [[ "$accounting" == PASS && "$kernel" == PASS ]] ||
        die "unsafe terminal state after release $label"
    done
  done
done

assert_idle final || die "final accelerator accounting is not idle"
scan_kernel final || die "final kernel scan failed"
assert_no_competing_processes

status=FAIL
[[ "$failures" -eq 0 ]] && status=PASS
printf 'EVO_A65_HIL_RESULT status=%s suite=%s throughput=%d correctness=%d release=%d failures=%d total=%d\n' \
  "$status" "$suite" "$throughput_total" "$correctness_total" \
  "$release_total" "$failures" \
  "$((throughput_total + correctness_total + release_total))" \
  | tee "$evidence_dir/result.txt"
(
  cd "$evidence_dir"
  find . -type f ! -name SHA256SUMS -print0 | sort -z |
    xargs -0 sha256sum >SHA256SUMS
)
[[ "$failures" -eq 0 ]]
