#!/usr/bin/env bash
set -u
BIN=/mnt/nfs/sima-neat/core/tmp/evo_route_matrix_legacy_runner
DIR=/mnt/nfs/sima-neat/tmp/evo_models/evo_testing/evo_testing/models
export LD_LIBRARY_PATH=/mnt/nfs/sima-neat/internals/tmp/install-fix-stage/lib/aarch64-linux-gnu/neat/runtime:/mnt/nfs/sima-neat/internals/tmp/install-fix-stage/lib/aarch64-linux-gnu/neat/gst-plugins:/mnt/nfs/sima-neat/core/tmp/install-fix-aarch64:${LD_LIBRARY_PATH:-}
export GST_PLUGIN_PATH=/mnt/nfs/sima-neat/internals/tmp/install-fix-stage/lib/aarch64-linux-gnu/neat/gst-plugins
export GST_PLUGIN_PATH_1_0="$GST_PLUGIN_PATH"
export SIMA_GST_PLUGIN_DIR="$GST_PLUGIN_PATH"
export SIMA_CLI_CHECK_FOR_UPDATE=0
export GST_REGISTRY=/tmp/sima_gst_registry_evo_route_matrix_$$.bin
models=(
  evo50_ev74tess_bf16_mpk.tar.gz
  evo50_ev74tess_int8_mpk.tar.gz
  evo50_mlatess_bf16_mpk.tar.gz
  evo50_mlatess_int8_mpk.tar.gz
  evo50_mlatess_multibuff_bf16_mpk.tar.gz
  evo50_mlatess_multibuff_int8_mpk.tar.gz
  evo50_v2_ev74tess_bf16_mpk.tar.gz
  evo50_v2_ev74tess_int8_mpk.tar.gz
  evo50_v2_mlatess_bf16_mpk.tar.gz
  evo50_v2_mlatess_int8_mpk.tar.gz
  evo50_v2_mlatess_multibuff_bf16_mpk.tar.gz
  evo50_v2_mlatess_multibuff_int8_mpk.tar.gz
)
routes=(EV74-A65 A65-A65 A65-EV74 EV74-EV74)
echo "RUN_START $(date -Is)"
echo "BIN=$BIN"
file "$BIN"
echo "models=${#models[@]} routes=${#routes[@]} frames=1 timeout=180s"
for m in "${models[@]}"; do
  for route in "${routes[@]}"; do
    pre=${route%-*}
    post=${route#*-}
    echo "===== MODEL=$m ROUTE=$route pre=$pre post=$post ====="
    start=$(date +%s)
    out=$(timeout 180 "$BIN" --model "$DIR/$m" --pre "$pre" --post "$post" --frames 1 2>&1)
    rc=$?
    end=$(date +%s)
    echo "$out" | sed -n '/MODEL_INIT_OK/p;/FRAME_OK/p;/EVO_RESULT/p;/status=FAIL/p;/error=/p;/invalid/p;/Usage:/p'
    if [ $rc -eq 124 ]; then
      echo "SUMMARY model=$m route=$route status=TIMEOUT seconds=$((end-start))"
    elif [ $rc -ne 0 ]; then
      echo "SUMMARY model=$m route=$route status=RC_$rc seconds=$((end-start))"
    else
      result=$(echo "$out" | grep 'EVO_RESULT' | tail -1 || true)
      if [ -n "$result" ]; then
        echo "SUMMARY model=$m route=$route $result seconds=$((end-start))"
      else
        echo "SUMMARY model=$m route=$route status=NO_RESULT rc=0 seconds=$((end-start))"
      fi
    fi
  done
done
echo "RUN_END $(date -Is)"
