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
7. Vulcan CI（`.github/workflows/vulcan-ci.yml`）已在要加上標籤的那個提交上通過。
8. 零跳過閘道（`scripts/ci/run_zero_skip_gate.sh`）在 Vulcan CI 測試通道中通過。
9. 必需的治理檔案存在且有效：
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
10. 發布元資料已完成：
   - 要加上標籤的提交中，`deps/manifest.json` 的 `package-version` 等於發布版本，例如 `v0.7.0` 對應 `0.7.0`
   - `platform-version` 在 `deps/manifest.json` 中更新（如果需要）
   - `modelzoo-version` 顯式選擇經過驗證的 Model Zoo 發布版本，當其與 `platform-version` 不同時；如果省略，Model Zoo 解析將預設為 `platform-version`
   - 只有在公開 C++ 型別佈局或匯出的二進位契約發生不相容變更，且 `develop` 仍使用與 `main` 相同的 ABI 版本時，
     才遞增 `deps/manifest.json` 中的 `abi-version`。如果 `develop` 已有供下次發布使用的較高 ABI 版本，
     後續尚未發布的 ABI 不相容變更應沿用該版本。發生不相容變更後，請使用相符的標頭與程式庫，
     重新建置所有 C++ 應用程式與 Python 綁定。
   - 在發布/標籤主體中準備了版本資訊。

在發布流程中不允許存在「已知崩潰」列表。任何崩潰回歸都會阻止發布，直到修復為止。

## 必需的狀態檢查

建立發布標籤之前，Vulcan CI（`.github/workflows/vulcan-ci.yml`）必須在要加上標籤的那個提交上通過。
檢查方式請見發布流程的步驟 5。

## 排程測試通道

這些通道不會阻止發布。崩潰/正確性、壓力、消毒器和模糊測試每晚執行。
穩定性測試和長時間測試每週執行。

- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`

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

直接以正式標籤發布，例如 `v0.7.0`。不要建立發布候選版本標籤。Release 工作流程
（**Actions → Release**）會建立或重複使用 `release-X.Y`，將標籤加在其最新提交上，並建立 GitHub 發布版草稿。
以下步驟是在新版本線上發布 0.7.0。在既有版本線上的修補發布（例如 0.6.3）在步驟 3 和 6 有所不同。

1. 在 `develop` 上，將 `deps/manifest.json` 中的 `package-version` 設為 `0.7.0`。當阻止發布的條件要求時，更新 `platform-version`、`modelzoo-version` 和 `abi-version`，並依照「最新文件版本」的說明更新發布資料。
2. 透過拉取請求將 `develop` 合併到 `main`。
3. 找出要加上標籤的提交。先執行 `git fetch origin`。
   - 新版本線：工作流程會從 `main` 的最新提交建立 `release-0.7`。該提交為 `git rev-parse origin/main`。
   - 既有版本線：透過拉取請求將 `main` 合併到 `release-0.6`。由於 `branch_exists_behavior` 預設為 `reuse`，工作流程會重複使用 `release-0.6` 並將標籤加在其最新提交上。該提交為 `git rev-parse origin/release-0.6`。

   在標籤建立之前，不要將任何內容合併到該分支。
4. 檢查該提交的版本。`git show <commit>:deps/manifest.json` 必須顯示 `"package-version": "0.7.0"`。Release 工作流程也會在第一個作業中執行相同的檢查。若版本不一致，它會在建立分支或標籤之前、以及將 issue 標記為已發布之前停止。
5. 確認 Vulcan CI 已在該提交上通過。`gh run list --repo sima-neat/core --workflow vulcan-ci.yml --commit <commit>` 會列出其執行。在 `main` 或發布分支上成功的執行也算數。若沒有，請在步驟 3 的分支上以 `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref main` 啟動 Vulcan CI，修補發布則使用 `--ref release-0.6`。確認新執行的 `headSha` 就是該提交，然後以 `gh run watch <run-id> --repo sima-neat/core --exit-status` 等待結果。在該提交的執行通過之前，不要加上標籤。
6. 開啟 **Actions → Release → Run workflow**。將 `release_line` 設為 `0.7`、`tag` 設為 `0.7.0`，保留 `main` 作為 `source_branch`，`prerelease` 保持關閉、`latest` 保持開啟。發布 0.6.3 時，將 `release_line` 設為 `0.6`、`tag` 設為 `0.6.3`，若較新的版本線已發布，請關閉 `latest`。工作流程會加上 `v0.7.0` 標籤並建立發布版草稿。它要求所有 Target Release 為 `core-0.7.0` 的 issue 皆為 Done，之後將它們的 Release Status 設為 `released`。若沒有任何 issue 具有該 Target Release，此檢查也會失敗。確認執行摘要中的 **Target SHA** 是步驟 3 的提交。若不是，請不要發布草稿。
7. 以 `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref v0.7.0` 對該標籤啟動 Vulcan CI。由工作流程加上的標籤本身不會啟動任何工作流程。等待 `https://artifacts.neat.sima.ai/core/v0.7.0/latest.tag` 顯示加上標籤的提交。此執行會發布 `sima-cli neat install core@v0.7.0` 所安裝的套件以及離線套件組。
8. 在發布版草稿中填寫並發布版本資訊，並開立發布後續問題。

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
