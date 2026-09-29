#!/usr/bin/env bash
# Token-drop measurement. Runs two single-user load cases against
# pcie-genai-backend on the card and reports how many token events were
# dropped in transit (parsed from the CLI's "dropped N" summary).
#
# Usage:
#   PCIE_GENAI=/path/to/pcie-genai \
#   MODEL=Llama-3.2-3B-Instruct-a16w4 CARD_ID=0 \
#   ./genai_drop_loadtest.sh
set -u

PCIE_GENAI="${PCIE_GENAI:-pcie-genai}"
MODEL="${MODEL:-Llama-3.2-3B-Instruct-a16w4}"
CARD_ID="${CARD_ID:-0}"
HEAVY_TOKENS="${HEAVY_TOKENS:-2000}"   # one long/heavy answer
SESSION_PROMPTS="${SESSION_PROMPTS:-20}" # back-to-back prompts in one session
LOG="$(mktemp)"

# Sum the "dropped N", "M tokens", "T tok/s" fields over every summary line.
summarize() {
  awk '
    /\| dropped [0-9]+\]/ {
      match($0, /([0-9]+) tokens/, t); tokens += t[1];
      match($0, /dropped ([0-9]+)\]/, d); dropped += d[1];
      match($0, /([0-9.]+) tok\/s/, s); tps_sum += s[1]; runs += 1;
    }
    END {
      if (runs == 0) { print "no summary lines parsed"; exit 1; }
      printf "runs=%d tokens=%d dropped=%d drop_rate=%.4f%% avg_tps=%.2f\n",
             runs, tokens, dropped, (tokens>0 ? 100.0*dropped/tokens : 0), tps_sum/runs;
    }' "$1"
}

echo "=== Case A: one long/heavy prompt ($HEAVY_TOKENS tokens) ==="
: > "$LOG"
"$PCIE_GENAI" --model "$MODEL" --card-id "$CARD_ID" --max-new-tokens "$HEAVY_TOKENS" \
  --prompt "Write a long, detailed essay about the history of computing." \
  >/dev/null 2>"$LOG"
cat "$LOG" | grep -E '^\[' || true
summarize "$LOG"

echo "=== Case B: $SESSION_PROMPTS prompts, one session ==="
: > "$LOG"
{
  for i in $(seq 1 "$SESSION_PROMPTS"); do
    echo "In one sentence, give me fact number $i about the ocean."
  done
} | "$PCIE_GENAI" --model "$MODEL" --card-id "$CARD_ID" >/dev/null 2>"$LOG"
grep -E '^\[' "$LOG" || true
summarize "$LOG"

rm -f "$LOG"
