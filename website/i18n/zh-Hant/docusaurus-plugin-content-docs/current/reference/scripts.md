---
title: "工具和腳本清單"
description: "「`core/scripts/`」和「`core/tools/`」裡分別包含哪些內容，以及何時應該使用它們？"
sidebar_position: 90
---

# 工具和腳本清單

這個框架包含兩個輔助函式目錄。本頁面是索引。

## `core/tools/` — 檔案和建置輔助工具


| 腳本 | 目的 |
|---|---|
| `generate_api_docs.sh` | 在編輯公開標頭檔後，執行 doxygen2docusaurus，處理 Doxygen XML 檔案，並產生用於 C++ API 參考網站的 Markdown 格式檔案。 |
| `generate_python_api_docs.py` | 從 `pyneat` 模組的文檔字串中，產生 Python API 參考的 Markdown 格式檔案。|
| `generate_tutorial_docs.py` |（教學檔案即將停止維護——此腳本的使用將逐漸減少。）|
| `postprocess_d2d_links.py` | 在產生檔案後，修正 Doxygen 到 Docusaurus 連結的 URL 字尾。此腳本會由 `generate_api_docs.sh` 自動呼叫。|
| `strip_empty_programlisting.py` | 這是針對空的 `<programlisting>` 元素的解決方案，因為這些空的元素會讓 doxygen2docusaurus 產生混淆。|
| `compute_version.sh` | 從 `deps/manifest.json` 中的 `package-version`，以及分支建置的 Git 中繼資料，計算框架的套件版本字串。用於 CI 和封裝。|
| `expand_code_tabs.py` | 擴展教學資源中多語言標籤。|
| `run_clean_env.sh` | 在一個乾淨的 shell 環境中執行指令（避免繼承的 `LD_*` / `PATH` 導致的異常情況）。|
| `tutorial_quality_lint.py` / `tutorial_scorecard.py` | 檢查教學檔案的 Markdown 格式，並評分。（與教學檔案相關，未來可能不再使用。）|

編輯公開標頭時的典型流程：

```bash
cd core
doxygen docs/doxygen/Doxyfile      # regenerate XML
bash tools/generate_api_docs.sh    # regenerate Markdown
cd website && yarn start           # preview the site
```

## `core/scripts/` — 儲存庫層級的檢查和開發輔助工具。


| 腳本 | 目的 |
|---|---|
| `check_format.sh` | 對 C++ 程式碼執行 clang-format，如果發現有差異則失敗。|
| `check_cmake_format.sh` / `check_cmake_style.py` | 對 `CMakeLists.txt` 檔案執行 cmake-format / 程式碼風格檢查。|
| `check_duplicate_includes.{sh,py}` | 偵測標頭檔中重複的 `#include` 行。|
| `check_internal_headers.sh` | 驗證 `core/src/pipeline/internal/sima/` 跨層級的管線是否遵守公用/內部邊界。|
| `run_cpp_tidy.sh` | 在整個程式碼庫中執行 clang-tidy。|
| `route_refactor_validation.sh` | 一個針對路線規劃器的回歸測試（由 CI 觸發）。|
| `install_neat_plugins.sh` | 將框架的 GStreamer 外掛程式安裝到系統外掛程式目錄中。|
| `install_codex_skill.sh` | 安裝 Codex CLI 的 NEAT 技能（方便開發者使用）。|
| `fix_devkit_runtime.sh` | 修補全新開發工具包的執行階段函式庫/路徑，並重新啟動協處理器。只有在 `simaai-appcomplex.service` 服務正在執行時，才會啟動 M4。|
| `sync_neatdecoder.sh` / `use_neatdecoder.sh` | 在內建和外部解碼器版本之間切換。|

### `core/scripts/ci/`, `core/scripts/dev/`, `core/scripts/release/`

這些子目錄包含屬於各自工作流程的腳本——持續整合 (CI) 執行 `ci/` 腳本集，開發人員執行 `dev/` 臨時腳本，發布工程團隊執行 `release/` 腳本。請勿在應用程式程式碼中使用這些腳本。

## 從一個乾淨的程式碼儲存庫執行檔案產生器。

```bash
sudo apt-get install -y doxygen   # if not installed
cd core
doxygen docs/doxygen/Doxyfile      # generates docs/doxygen/out/xml/
bash tools/generate_api_docs.sh    # populates docs/reference/cppapi/
python3 tools/generate_python_api_docs.py   # populates docs/reference/pythonapi/
cd website && yarn install && yarn start    # serve at http://localhost:3000/
```

## 更多資訊

- 「工具和腳本」——設計深度探討的第 55 節。
- 這個儲存庫 `core/AGENTS.md` 包含一份協作者協議，其中說明了在提交程式碼之前必須執行的工具。
