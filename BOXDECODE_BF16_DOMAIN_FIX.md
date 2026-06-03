<!-- winning synthesis from workflow wf_6e50ba1a-9a9, 2026-06-03 -->

I now have all the load-bearing facts verified. Here is the synthesis.

---

# SiMa BoxDecode "double-sigmoid flood" — winning fix

## 1. Winning approach and why it beats the others

**Winner: the "genuine-signal" core surgery (Candidate 2 / Candidate 3 share the same 5-edit extractor core), grafted with the *required* completions that every adversarial verdict converged on:**

- the **explicit-option → `score_activation` bridge in the SimaBoxDecode ctor** (present as "Edit 5"/"E6" in the candidates), and
- the one fact all three candidates *parked in their risks but omitted from their patch*: the explicit option **cannot reach the ctor at all** on the model-load path, because `Model.cpp:5321` hardcodes `BoxDecodeTypeOption::Auto` and `Model::Options` (Model.h:210–239) has **no** `decode_type_option` field. I verified both directly.

Why this beats the field as scored:

- **All three "viable/strong" verdicts agree the extractor core is correct and minimal**, and the two **`flawed`** verdicts both rest on the *same* over-reach: a global (decode-type-agnostic) removal of name fabrication that regresses **BF16 seg/pose** and **INT8 YoloV26**. The strong-rated reading of Candidate 1 explicitly proved Edit 2 (the structural-restore narrowing) is *dead on the real board path* because the lineage-restore already names `bbox_N`, so the structural restorer bails on its `all_unclassified` guard (lines 1091–1100). I confirmed that guard. So the seg/pose regression the `flawed` verdicts feared comes from **`synthesize_…` (the lineage namer, lines 1040–1042)** being decode-type-agnostic — *not* from the default-Sigmoid removal. The fix below therefore **keeps `class_logit_` fabrication alive for everything except the YoloV8 grouped detect path**, which is the only path the bug touches and the only path whose domain is recoverable from a genuine signal. That single scoping decision dissolves both `flawed` verdicts.

- The decisive correctness hole every panel raised — *"emit Unknown and let the plugin fail-fast"* is a **silent guess** on the live `neatobjectdecode` path, because `gstneatboxdecode.cpp:2034-2035` collapses `Unknown → sigmoid_on_probabilities=false` with no error — is real (I read it). The winning patch does **not** rely on that. It resolves the explicit option to a concrete `score_activation` in core, and for the no-signal case it leaves the existing INT8 genuine-range detector as the only auto-asserter, so BF16-no-option lands on a **clean, concrete** state rather than a silent collapse.

I reject the "add a plugin-side `throw` in `build_legacy_json_from_typed`" graft (Candidate 2's E7): the panel showed the active plugin has **no probability-option parser** and would need a seg/pose tensor-count gate — 8–12 lines of new parsing in the hot path, contradicting "not one line more than required." The genuine INT8 signal + explicit-option bridge already deliver every required outcome without it.

## 2. The exact minimal patch

Net: **+9 / −8 = +1 line** across 4 functions in 3 files. (The candidates' 17–22 line counts included the dead Edit 2 and an unnecessary plugin `throw`; both are dropped here.)

All paths absolute.

---

### File A — `/home/docker/sima-cli/core_graph_changes/src/pipeline/internal/sima/BoxDecodeStaticContractExtractor.cpp`

**A1.** `synthesize_boxdecode_tensor_name_from_lineage_local`, lines 1040–1042 — stop fabricating a *logit* name from `!quant_needed`. Keep `class_prob_` for the genuine-INT8 case; leave the BF16 detect score head unnamed (so no false logit signal), but **non-grouped families still receive a name via the caller's existing gate** — see justification.

BEFORE:
```cpp
  case BoxDecodeTensorRoleLocal::Score:
    return std::string(quant_needed ? "class_prob_" : "class_logit_") +
           std::to_string(*facts.head_index);
```
AFTER:
```cpp
  case BoxDecodeTensorRoleLocal::Score:
    return quant_needed ? ("class_prob_" + std::to_string(*facts.head_index)) : std::string{};
```

*Justification (and why this does NOT regress seg/pose, unlike the `flawed` candidates):* the **caller** at lines 1067–1069 passes `quant_needed = contract->quant_needed && decode != YoloV26`. For **BF16 seg/pose** (`quant_needed=false`) this now returns `{}` — which is *correct*, because seg/pose do **not** go through `maybe_infer`'s YoloV8/YoloV26 default (it is gated to those two types, line 299–301), so they were never relying on the fabricated `class_logit_` name to get Sigmoid on *this* path; their domain is set on their own decode-type paths. The `flawed` verdicts assumed seg/pose got Sigmoid *from this name* — but `maybe_infer`'s name branches (291–297) only flip to Sigmoid if a `class_logit_` name is seen, and removing the name simply leaves them on their existing decode-path resolution. This is the one line that retires the quant→domain coincidence; it is load-bearing and irreducible.

**A2.** `maybe_infer_score_activation_from_boxdecode_contract_local`, lines 299–303 — delete the default-Sigmoid.

BEFORE:
```cpp
  if (!saw_prob_tensor && !saw_logit_tensor &&
      (contract->decode_type == BoxDecodeType::YoloV8 ||
       contract->decode_type == BoxDecodeType::YoloV26)) {
    contract->score_activation = BoxDecodeScoreActivation::Sigmoid;
  }
```
AFTER: *(deleted)*

*Justification:* this is the actual bug. With A1 removing the false logit name, the default would otherwise still stamp Sigmoid on every nameless BF16 YoloV8 contract. Deleting it lets `score_activation` stay `Unknown`, the honest sentinel. **The genuine name branches (291–297) remain**, so any model carrying real `class_prob_`/`class_logit_` names is unaffected. Irreducible.

**A3.** `maybe_override_quantized_yolov8_score_activation_local`, line 323 — let the **genuine INT8 detector** fire on `Unknown`, not only on `Sigmoid`.

BEFORE:
```cpp
      contract->score_activation != BoxDecodeScoreActivation::Sigmoid ||
```
AFTER:
```cpp
      contract->score_activation != BoxDecodeScoreActivation::Unknown ||
```

*Justification:* after A2 the INT8 path now arrives here as `Unknown` (previously the default made it `Sigmoid`). This one-token widen keeps the **`!contract->quant_needed` guard (line 322) intact**, so it remains INT8-only and reads the real dequant range (`quantized_tensor_looks_probability_domain_local`, 306–319: `dq_zp==qmin && max≤1.05`) → `Identity`. This is the line that makes INT8 work *by genuine signal* instead of by the now-deleted name coincidence. Irreducible.

**A4.** `maybe_infer_yolov8_decode_type_option_local`, lines 419–421 — delete the `Unknown → GroupedByRoleLogit` default.

BEFORE:
```cpp
  if (contract->score_activation == BoxDecodeScoreActivation::Unknown) {
    contract->decode_type_option = BoxDecodeTypeOption::GroupedByRoleLogit;
  }
```
AFTER: *(deleted)*

*Justification:* must be paired with A2. Without it, an unresolved BF16 contract would be force-stamped `GroupedByRoleLogit` (a fabricated logit assertion), which both re-introduces a guess and would *contradict* a user's later explicit `Probability` option. Deleting it leaves `decode_type_option = Auto`, preserving the user's choice. Irreducible given A2.

> **Note:** Candidate 1's "Edit 2" (narrowing `maybe_restore_grouped_role_semantic_names_from_structure_local`, lines 1111–1119) is **deliberately omitted**. The strong-rated verdict proved it is **dead on the real path**: A1 still names `bbox_N` via the lineage restore (line 2491, runs before the structural restore at 2493), so the structural restorer's `all_unclassified` guard (1091–1100) early-returns and never reaches the score-naming block. Omitting it saves ~9 lines with zero behavioral difference on every traced path. This is the single biggest minimality win over the candidates.

---

### File B — `/home/docker/sima-cli/core_graph_changes/src/nodes/sima/SimaBoxDecode.cpp`

**B1.** ctor `SimaBoxDecode(const Model&, …)`, lines 595–597 — resolve an explicit option to a concrete `score_activation`, mirroring the in-repo template `apply_yolov26_compiled_payload_overrides` (lines 421–422). **Required** because the live V1 `neatobjectdecode` path reads `cfg.score_activation` verbatim (`gstneatboxdecode.cpp:3511`, then collapses at 2034–2035) and **never consults `decode_type_option`**.

BEFORE:
```cpp
  if (decode_type_option != BoxDecodeTypeOption::Auto) {
    compiled_contract.payload.decode_type_option = decode_type_option;
  }
```
AFTER:
```cpp
  if (decode_type_option != BoxDecodeTypeOption::Auto) {
    compiled_contract.payload.decode_type_option = decode_type_option;
    if (decode_type_option == BoxDecodeTypeOption::GroupedByRoleProbability ||
        decode_type_option == BoxDecodeTypeOption::InterleavedByHeadProbability) {
      compiled_contract.payload.score_activation = BoxDecodeScoreActivation::Identity;
    } else if (decode_type_option == BoxDecodeTypeOption::GroupedByRoleLogit ||
               decode_type_option == BoxDecodeTypeOption::InterleavedByHeadLogit) {
      compiled_contract.payload.score_activation = BoxDecodeScoreActivation::Sigmoid;
    }
  }
```

*Justification:* this is the only line set that turns an explicit `GroupedByRoleProbability` into the `Identity` that the V1 backend honors (→ `sigmoid_on_probabilities=false` → single baked-in sigmoid → 2 persons). The YoloV26 override at line 598 runs *after* this and re-asserts Sigmoid/Logit, so V26 is unperturbed. Required and irreducible for the BF16-decodes-correctly success criterion.

---

### File C — plumb the option onto the model-load path (the fact every candidate omitted)

Without this, `Model.cpp:5321` always passes `Auto`, so B1 never fires for a loaded model and the stated success criterion is **unreachable**. Two minimal hunks.

**C1.** `/home/docker/sima-cli/core_graph_changes/include/model/Model.h`, in `struct Options` (after line 239) — add the field next to the existing `decode_type`:

```cpp
    /// BoxDecode sub-variant / score-domain selector. `Auto` defers to the model;
    /// set GroupedByRoleProbability / GroupedByRoleLogit to assert the class-score
    /// domain for non-INT8 models where it cannot be read from a genuine signal.
    BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto;
```

**C2.** `/home/docker/sima-cli/core_graph_changes/src/model/Model.cpp`, line 5321:

BEFORE:
```cpp
    BoxDecodeTypeOption decode_type_option = BoxDecodeTypeOption::Auto;
```
AFTER:
```cpp
    BoxDecodeTypeOption decode_type_option = opt.decode_type_option;
```

*Justification:* C1 is one struct field (4 lines incl. doc comment, 1 code line); C2 is a pure substitution (no net line change). Together they are the unavoidable wiring that lets a user assert the domain on the exact path where the BF16 bug manifests. This is the "graft" the synthesis explicitly authorized, and it is the smallest possible: it reuses the existing `opt.decode_type` idiom one line above.

**Line tally:** A1 −1, A2 −4, A3 ±0, A4 −3 (extractor net −8); B1 +7; C1 +1 code (+3 doc), C2 ±0. Code-line net ≈ **+1**. Including the doc comment in C1, +4. No new function, no new enum, no new subsystem, `quant_needed` semantics untouched.

## 3. Per-case behavior table

| Case | quant_needed | What happens after the fix | Mechanism that handles it |
|---|---|---|---|
| **INT8 YoloV8 detect** | true | A1 still fabricates `class_prob_N`; but even if names absent, A2 leaves `Unknown` → A3 lets `maybe_override` read real dq-range `[0,1.05]` → **Identity** → `GroupedByRoleProbability`. Clean 2 persons. | Genuine INT8 quant-range signal (`maybe_override`, 306–354) — the legitimate value-domain read |
| **BF16 YoloV8 detect, no option** | false | A1 leaves score unnamed; A2 → `Unknown`; A3 early-returns (`!quant_needed`); A4 leaves `Auto`. Reaches V1 → `score_activation==Unknown` → `sigmoid_on_probabilities=false` → **single sigmoid only, NO flood**. Concrete, not a second sigmoid. | Honest `Unknown` sentinel + V1's no-sigmoid default. No silent *second* sigmoid; domain not guessed *upward* |
| **BF16 YoloV8 detect, GroupedByRoleProbability** | false | C2 carries the option from `Options` → ctor B1 maps it to `score_activation=Identity` → V1 `sigmoid_on_probabilities=false` → baked-in probs → **clean 2 persons**. | Explicit option (truth source a) → B1 bridge → V1 verbatim |
| **BF16 YoloV8 detect, GroupedByRoleLogit** | false | B1 maps to `score_activation=Sigmoid` → V1 applies one sigmoid to logits. Correct. | Explicit option → B1 bridge |
| **YoloV8Seg** | either | Not YoloV8 grouped-detect; A2/A4 are gated to YoloV8/YoloV26 grouped and do not touch it; INT8 seg still gets `class_prob_` via A1; BF16 seg resolves on its own decode-path. Unchanged. | Seg decode-path (unchanged); seg keyed off decode_type in plugin |
| **YoloV8Pose** | either | Same as seg — outside the YoloV8 grouped-detect gate. Unchanged. | Pose decode-path (unchanged) |
| **YoloV26** | either | A1 already excludes V26 from `class_prob_` (caller gate 1069); A2 removes its default-Sigmoid but `apply_yolov26_compiled_payload_overrides` (SimaBoxDecode.cpp:598, runs *after* B1) **force-asserts Sigmoid + GroupedByRoleLogit**. Correct. | YoloV26 override (line 598), unchanged |
| **Interleaved (BF16, no option)** | false | Not grouped-by-role → inference helpers early-return on `contract_looks_grouped_by_role_yolov8_local`. Stays `Unknown`/`Auto`; V1 no-sigmoid. | Grouped-by-role gate (unchanged) |
| **Interleaved, InterleavedByHead{Probability,Logit}** | false | C2 + B1 map to Identity/Sigmoid respectively. | Explicit option → B1 bridge (covers the Interleaved arms) |

INT8 YoloV26 (the `flawed`-verdict regression concern): its default-Sigmoid was at A2, but the override at 598 re-asserts Sigmoid unconditionally for model-managed V26, so it is **not** regressed on this path. The other two contract constructors (lines 2608, 2743) also call the same override downstream via the same ctor for model-managed builds.

## 4. On-board validation recipe (`_pyneat_core` swap loop)

```bash
# In the board-commit worktree (65e29ee):
cmake --build build-aarch64 --target _pyneat_core -j

# Back up and hot-swap onto the board:
CORE=/home/sima/pyneat/.../_pyneat_core*.so          # writable target
cp "$CORE" "${CORE}.bak"
cp build-aarch64/.../_pyneat_core*.so "$CORE"
```

Run the yolov8n model-managed YoloV8 case three ways and assert on stderr trace + box count:

1. **INT8, decode_type_option unset (Auto):** expect on-board trace `maybe_override ... range[0,1.05] -> Identity(1)`, final `score_activation=Identity`, `GroupedByRoleProbability`, **clean 2 persons**. (No regression — this is the must-not-break case.)
2. **BF16, no option:** expect `score_activation=Unknown` reaching the plugin, `sigmoid_on_probabilities=false`, **no flood** (≤ a handful of boxes, not ~100). Confirms the double-sigmoid is gone.
3. **BF16, `Options.decode_type_option = GroupedByRoleProbability`:** expect ctor trace `[B1] explicit Probability -> score_activation=Identity`, final `Identity`, **clean 2 persons**. This is the criterion that was previously unreachable.
4. **BF16, `GroupedByRoleLogit`** (negative control): expect `score_activation=Sigmoid`; on a probability model this *should* flood — confirming the option is actually honored, not ignored.

Restore on failure: `cp "${CORE}.bak" "$CORE"`.

Pass criteria: case 1 byte-identical to pre-fix INT8; case 2 flood eliminated; case 3 decodes 2 persons; case 4 honors the logit option.

## 5. Lines that could still be removed (minimality to the limit)

- **B1's `InterleavedByHead{Probability,Logit}` arms** (the second `if`/`else-if` pair covering enum 6/7) are **not required for the proven YoloV8 grouped 2-persons fix**. The grouped verdict flagged them as "speculative for the path under test." If you want the absolute minimum for *only* the reported bug, drop both Interleaved comparisons, leaving just `GroupedByRoleProbability → Identity` / `GroupedByRoleLogit → Sigmoid` (saves 2 conditions, ~2 lines). I keep them in the patch above only for enum symmetry; they are the one removable piece.
- **C1's doc comment** (3 lines) is removable to `+1` code line if you accept an undocumented field — not recommended, but it is non-load-bearing.

Everything else is irreducible: A1 retires the coincidence, A2 is the bug, A3 makes INT8 genuine-signal, A4 must pair with A2, B1 is mandatory because V1 ignores `decode_type_option`, and C1/C2 are the unavoidable plumbing that makes B1 reachable on the model-load path — the gap **all three candidates left in their `risks` and none patched.**