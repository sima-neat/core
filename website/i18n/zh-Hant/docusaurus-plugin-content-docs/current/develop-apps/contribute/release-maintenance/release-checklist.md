---
title: "發布檢查清單"
description: "發布閘道策略與可重複的發布步驟"
sidebar_position: 1
slug: /develop-apps/contribute/release-checklist
---

# 發布檢查清單

本文檔是正式的發布閘道策略。

## 阻止發布的條件

除非滿足以下所有條件，否則將阻止發布：

1. 追蹤的源/文檔檔案中沒有合併衝突標記。
2. 公開文檔的命名規範檢查通過。
3. 設定/建置完整性檢查通過（`cmake -S . -B ...` 和 `cmake --build ... --target sima_neat`）。
4. 文檔連結檢查在嚴格模式下通過（`DOCS_STRICT_LINKS=1`）。
5. 在生成步驟之後，工作樹是乾淨的。
6. 在提交和發布參考版本中，沒有未解決的崩潰/正確性錯誤。
7. 崩潰/正確性/壓力/消毒器閘道在發布參考版本中顯示為綠色。
8. 模型存檔安全閘道顯示為綠色（`model-archive-security-gate`）。
9. 安裝煙霧測試閘道顯示為綠色（`install-smoke`）。
10. 性能回歸閘道顯示為綠色（`perf-regression-gate`）。
11. 穩定性測試通道對於發布標籤顯示為綠色（`soak-weekly`）。
12. 模糊測試通道對於發布候選版本顯示為綠色（`fuzz-nightly`）。
13. 零跳過閘道對於嚴格測試通道顯示為綠色（`zero-skip-gate`）。
14. 必需的治理檔案存在且有效：
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
15. 發布元資料已完成：
   - `project(SimaNeat VERSION x.y.z)` 在 `CMakeLists.txt` 中更新
   - `package-version` 和 `platform-version` 在 `deps/manifest.json` 中更新（如果需要）
   - `modelzoo-version` 顯式選擇經過驗證的 Model Zoo 發布版本，當其與 `platform-version` 不同時；如果省略，Model Zoo 解析將預設為 `platform-version`
   - `abi-version` 在 `deps/manifest.json` 中，每當公共 C++ 類型佈局或導出二進制合約以不兼容的方式更改時，都會遞增；所有 C++ 應用程式和 Python 綁定都會針對該 ABI 進行重新建置
   - `CHANGELOG.md` 具有 `## [x.y.z]` 條目
   - 在發布/標籤主體中準備了版本資訊。

在發布流程中不允許存在「已知崩潰」列表。任何崩潰回歸都會阻止發布，直到修復為止。

## 必需的狀態檢查

以下檢查對於發布 PR 和發布標籤是必需的：

- `repo-hygiene`
- `configure-build-sanity`
- `docs-link-check`
- `crash-correctness-gate`
- `model-archive-security-gate`
- `install-smoke`
- `perf-regression-gate`
- `zero-skip-gate`
- `soak-weekly`（對於發布標籤是必需的）
- `fuzz-nightly`（對於發布候選版本是必需的）
- `stress-gate`
- `asan-ubsan-gate`
- `release-policy-check`

這些檢查在以下位置實施：

- `.github/workflows/release-gate.yml`
- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/model-archive-security.yml`
- `.github/workflows/install-smoke.yml`
- `.github/workflows/perf-regression.yml`
- `.github/workflows/zero-skip.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`

觸發所有權，以避免重複執行閘道：

- 非發布版 PR 合併到 `main` 時，會從其獨立的工作流程中執行 `model-archive-security`、`install-smoke`、`perf-regression` 和 `zero-skip`。
- 發布版 PR（`release/*` 分支的頂端提交）和發布版分支（`release/**`、`v*`）會從 `.github/workflows/release-gate.yml` 執行相同的流程。

## GitHub 分支和標籤保護

設定 GitHub 儲存庫設定：

1. 保護 `main`：
   - 要求在合併之前進行拉取請求。
   - 要求至少獲得一位程式碼擁有者的批准（如果有的話，建議獲得兩位）。
   - 在有新提交時，取消過時的批准。
   - 要求所有必要的狀態檢查都通過。
   - 不允許強制推送。
   - 使用僅限合併或線性歷史記錄。
2. 保護 `v*` 標籤，以限制誰可以建立發布標籤。

## 發布流程

1. 從綠色的 `main` 分支切出 `release/x.y.z`。
2. 凍結非發布版 PR 的合併。
3. 在發布版分支上執行發布閘道工作流程。
4. 建立 `vX.Y.Z-rcN` 標籤，用於候選驗證。
5. 升級到最終的 `vX.Y.Z` 標籤。
6. 將發布版分支快速合併回 `main`。
7. 發布版本資訊並發布發布後續問題。

## 運營注意事項

- 不允許從「髒」分支發布。
- 不允許從未經審查的程式碼發布。
- 當必要的檢查失敗時，不允許發布。
- 當本地崩潰/正確性閘道失敗時，不允許推送。
- 沒有用於繞過衛生檢查失敗的手動路徑。

## 效能回歸合約

- 效能閘道的入口點是 `scripts/ci/run_perf_regression_gate.sh`。
- 基線是根據 `tests/perf/baselines/v2/modalix_default/` 進行設定的：
  - `profile.json` 定義了固定的 Modalix 環境合約。
  - 每個情境 ID 都有一個情境檔案（`<scenario_id>.json`）。
- 必需的情境：
  - `runtime_session_sync_rgb`
  - `runtime_session_async_rgb`
  - `runtime_graph_fanout`
  - `runtime_graph_join_bundle`
  - `runtime_codec_mjpeg_decode`
  - `runtime_codec_h264_decode`
  - `runtime_codec_h265_decode`
  - `runtime_model_archive_load`
- 每次效能執行都會在 `build-perf-gate/perf_results/` 中發布每個情境的結果檔案。
- 每個結果都必須包含：
  - `scenario_id`
  - `modalix_profile_id`
  - `status`
  - `failure_class`
  - `reason_code`
  - `metrics`
  - `run_meta`
  - `timestamp`
- 任何 `REGRESSION`、`HARNESS_ERROR` 或 `ENV_BROKEN` 分類都會阻止該流程。
