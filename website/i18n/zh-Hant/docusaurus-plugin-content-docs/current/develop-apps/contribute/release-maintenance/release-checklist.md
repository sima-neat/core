---
title: "發布檢查清單"
description: "發布閘道策略與可重複的發布步驟"
sidebar_position: 1
slug: /develop-apps/contribute/release-checklist
---

# 發布檢查清單

本文檔是正式的發布閘道策略。

## 最新文件版本

更新 `deps/manifest.json` 後，針對此版本更新一次 `website/release-data/current.json`。執行 `npm --prefix website run sync:release-readmes`，並提交產生的含具體版本號碼的 README 命令。執行 `npm --prefix website run check:releases` 和 `./build.sh --doc`，以驗證權杖、歷史版本例外、在地化及轉譯後的命令。相容性頁面使用具體版本號碼，並以手動方式維護。歷史版本資訊中的版本必須保持固定。僅變更版本資料時，不需要更新翻譯或來源雜湊。欄位管理責任和權杖規則記載於 `website/release-data/README.md`。

## 阻止發布的條件

除非滿足以下所有條件，否則將阻止發布：

1. 追蹤的源/文檔檔案中沒有合併衝突標記。
2. 公開文檔的命名規範檢查通過。
3. 設定/建置完整性檢查通過（`cmake -S . -B ...` 和 `cmake --build ... --target sima_neat`）。
4. 文檔連結檢查在嚴格模式下通過（`DOCS_STRICT_LINKS=1`）。
5. 在生成步驟之後，工作樹是乾淨的。
6. 在提交和發布參考版本中，沒有未解決的崩潰/正確性錯誤。
7. 崩潰/正確性/壓力/消毒器閘道在發布參考版本中顯示為綠色。
8. 穩定性測試通道對於發布標籤顯示為綠色（`soak-weekly`）。
9. 模糊測試通道對於發布候選版本顯示為綠色（`fuzz-nightly`）。
10. 零跳過閘道對於嚴格測試通道顯示為綠色（`zero-skip-gate`）。
11. 必需的治理檔案存在且有效：
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
12. 發布元資料已完成：
   - `project(SimaNeat VERSION x.y.z)` 在 `CMakeLists.txt` 中更新
   - `package-version` 和 `platform-version` 在 `deps/manifest.json` 中更新（如果需要）
   - `modelzoo-version` 顯式選擇經過驗證的 Model Zoo 發布版本，當其與 `platform-version` 不同時；如果省略，Model Zoo 解析將預設為 `platform-version`
   - 只有在公開 C++ 型別佈局或匯出的二進位契約發生不相容變更，且 `develop` 仍使用與 `main` 相同的 ABI 版本時，
     才遞增 `deps/manifest.json` 中的 `abi-version`。如果 `develop` 已有供下次發布使用的較高 ABI 版本，
     後續尚未發布的 ABI 不相容變更應沿用該版本。發生不相容變更後，請使用相符的標頭與程式庫，
     重新建置所有 C++ 應用程式與 Python 綁定。
   - 在發布/標籤主體中準備了版本資訊。

在發布流程中不允許存在「已知崩潰」列表。任何崩潰回歸都會阻止發布，直到修復為止。

## 必需的狀態檢查

以下檢查對於發布 PR 和發布標籤是必需的：

- `configure-build-sanity`
- `docs-link-check`
- `crash-correctness-gate`
- `zero-skip-gate`
- `soak-weekly`（對於發布標籤是必需的）
- `fuzz-nightly`（對於發布候選版本是必需的）
- `stress-gate`
- `asan-ubsan-gate`

這些檢查在以下位置實施：

- `.github/workflows/vulcan-ci.yml`
- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`

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

1. 開啟 **Actions → Release → Run workflow** 以建立發布候選版本。例如要發布 0.7.0，請將 `release_line` 設為 `0.7`、`tag` 設為 `0.7.0-rc1`，保留 `main` 作為 `source_branch`，並開啟 `prerelease`、關閉 `latest`。此工作流程會建立或重複使用 `release-0.7`，加上 `v0.7.0-rc1` 標籤，並建立預先發布版草稿。若 `release-0.7` 已存在（例如發布 0.7.1 時），由於 `branch_exists_behavior` 預設為 `reuse`，工作流程會將標籤加在其最新提交上。請在建立第一個候選版本之前，透過拉取請求將 `main` 合併到 `release-0.7`。
2. 凍結非發布版 PR 的合併。
3. 驗證候選版本。若需要修正，請先將修正合併到 `release-0.7`，再以下一個候選標籤重新執行工作流程，例如 `0.7.0-rc2`。
4. 候選版本通過驗證後，確認 `release-0.7` 仍指向該候選版本。工作流程會將標籤加在 `release-0.7` 的最新提交上，無法標記較舊的提交。執行 `git fetch --tags origin` 後，`git rev-parse origin/release-0.7` 必須輸出與 `git rev-parse v0.7.0-rcN^{commit}` 相同的提交。其中 `rcN` 是通過驗證的候選版本。若候選版本之後有變更進入分支，請先建立新的候選版本。接著將 `tag` 設為 `0.7.0`，關閉 `prerelease` 並開啟 `latest`，然後執行工作流程。它會將 `v0.7.0` 標籤加在該提交上，並建立正式發布版草稿。此次執行也要求所有 Target Release 為 `core-0.7.0` 的 issue 皆為 Done，之後將它們的 Release Status 設為 `released`。候選版本的執行會略過此檢查。
5. 將發布版分支快速合併回 `main`。
6. 發布版本資訊並發布發布後續問題。

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
