---
title: "建立"
description: "使用 build.sh 腳本從原始碼建立 SiMa.ai Neat。"
sidebar_position: 1
slug: /develop-apps/contribute/build
---

# 建立 Neat

本指南涵蓋 Neat 的原始碼建立。
如需預先建置的套件安裝，請參閱 [Neat Library](/getting-started/neat-library/)。

`build.sh` 是支援的建立進入點。它會處理相依性檢查、選擇性的相依性同步、CMake 設定/建立、選擇性的檔案產生、安裝健全性檢查和封裝。

## 建立環境

`build.sh` 會自動偵測目前使用的環境：

- Modalix DevKit 原生環境
- Palette SDK 環境（跨平台編譯）

您可以在任一環境中執行相同的 `build.sh` 指令。

### 跨平台編譯的先決條件

跨平台編譯請使用 Palette SDK。依照[安裝 SDK](/getting-started/dev-environment/install-the-environment/)在支援的 Ubuntu、Windows/WSL 或 macOS 主機上完成設定，然後開啟 SDK shell：

```bash
sima-cli sdk neat
```

SDK 包含工具鏈與 `sima-cli`。Core 的 `build.sh` 會準備相符的 Scheduler 和 LLiMa 相依套件；請參閱[本機相依套件](/develop-apps/contribute/build-test/local-dependencies/)。在已配對的 SDK 中，`./build.sh --install` 也會將相符的成品部署至 DevKit。在 DevKit 上原生建置不需要 SDK。

## 建置選項

支援的 `build.sh` 選項：

- `--dev-only`：僅建置核心函式庫和標頭檔（預設）。
- `--all`：建置函式庫 + 測試 + 教程 + Python wheel；啟用檔案和依賴項。
- `--python`：除了選定的目標之外，還建置 Python 繫結 (`pyneat`)。
- `--install-neat-internals`、`--install-deps`：在建置之前下載並安裝依賴項成品。
- `--doc`：僅建置檔案。
- `--install`：在建置/封裝後，將產生的成品安裝到目前的環境中。在配對的 Palette SDK 模式下，這也會部署並安裝配對的 DevKit 上的相應成品。
- `--no-dist`：跳過發布封裝。
- `--clean`：在設定之前移除 `build/`。
- `--no-doc`：跳過檔案建置（即使使用 `--all`）。
- `--no-node`：跳過 Node.js 安裝（如果缺少 Node，則檔案建置可能會失敗）。
- `--install-deps-only`：安裝系統依賴項和依賴項標頭檔，然後退出。

## 編譯器快取

`build.sh` 會自動啟用 `sccache`，並且其快取在 `--clean` 之後仍然可用。本地建置使用使用者本地磁碟快取。Vulcan 為 `develop` 和 `main` 提供個別的受保護快取；功能分支會從其最接近的受保護基礎分支建立隔離的可寫入快取，並在分支刪除之前保留該快取。

請參閱 [Neat sccache 快速參考資料](/develop-apps/contribute/sccache)，以了解本地控制、雲端存取規則、快取命名空間、統計資訊、驗證和疑難排解。

## 典型建置

僅核心函式庫（預設）：

```bash
./build.sh
```

完整建置（函式庫、測試、教學、檔案、wheel 檔案、封裝）：

```bash
./build.sh --all
```

核心函式庫 + Python 介面：

```bash
./build.sh --dev-only --python
```

僅適用於檔案：

這個指令在 macOS 上也能正常運作。

```bash
./build.sh --doc
```

檔案建置程序會從自動檔案工具下載的 OpenAPI 規格檔案（位於 `build/autodoc/insight/neat_insight/openapi.json`）產生 Insight API 參考檔案。為了方便本地開發，您可以透過 `INSIGHT_OPENAPI_SPEC` 來覆寫該預設值。

```bash
INSIGHT_OPENAPI_SPEC=../insight/neat_insight/openapi.json ./build.sh --doc
```

相對覆寫路徑會從核心儲存庫的根目錄解析，並在 Docusaurus 建構器執行之前轉換為絕對路徑。如果選取的檔案不存在，則會跳過 Insight API 產生步驟，並回報路徑。

執行完整建構：

```bash
./build.sh --all --clean
```

在不編譯核心程式碼的情況下安裝相依套件：

```bash
./build.sh --install-deps-only
```

## 輸出

- 建構樹：`build/`
- Docusaurus 網站輸出（當執行檔案建構時）：`website/build/`
- 安裝健全性檢查的前置目錄：一個獨特的臨時目錄（`${TMPDIR:-/tmp}/sima-neat-install-test.XXXXXX`），建構期間會顯示；成功時會移除，失敗時會保留以供檢查。
- 在 Linux 完整建構中，會產生 Neat 套件成品（`*.deb`），除非使用了 `--no-dist`。
- 在 Linux 完整建構中，會產生額外套件（`*extras.tar.gz`），除非使用了 `--no-dist`。
- 當啟用 Python 建構時，會產生 Python wheel（`dist/*.whl`）。

Python wheel 會封裝主要 CMake 建構產生的 `_pyneat_core` 擴充功能。建立 wheel 不會設定或編譯第二個 CMake 樹，因此，函式庫、DEB、額外檔案和 wheel 會共用一次編譯。

## 建構設定檔與 CMake 選項

框架的頂層 `CMakeLists.txt` 公開了一些選項，用於控制要建構的內容以及建構方式。以下選項是最重要的選項。

### 建構設定檔

框架支援三個命名設定檔：

| 設定檔 | 使用案例 | 建構內容 |
|---------|----------|-----------------|
| **Production** | 面向客戶的建構 | 所有公用節點、模型檔案載入、Modalix 後端、最佳化 |
| **Developer** | 框架工程師 | Production 集合 + 偵錯節點 + 擴展診斷 + 測試 |
| **Sandbox** | 多租戶部署 | Production 集合 + 嚴格的模型檔案安全預設值 |

在設定時，透過 `-DSIMA_NEAT_PROFILE=Production|Developer|Sandbox` 進行選擇，或者在 `CMakeLists.txt` 中接受預設值。

### 常用的 CMake 選項

| 選項 | 預設值 | 效果 |
|--------|---------|--------|
| `SIMA_NEAT_BUILD_TESTS` | `ON`（Developer） | 建構 gtest 套件。為了加快生產建構的 CI，請停用。 |
| `SIMA_NEAT_BUILD_TUTORIALS` | `OFF` | 建構教學二進位檔。 |
| `SIMA_NEAT_BUILD_PYTHON` | `ON` | 建構 `pyneat` nanobind 模組。 |
| `SIMA_NEAT_BUILD_INTERNALS` | `OFF`（公用） | 建構內部穿透層（`core/src/pipeline/internal/sima/`）。 |
| `SIMA_NEAT_ENABLE_TVM_FALLBACK` | `ON` | 編譯 TVM 後端回退核心，用於 MLA 無法處理的操作。 |
| `SIMA_NEAT_ENABLE_RTSP` | `ON` | 建構 RTSP 來源/接收節點。 |
| `SIMA_NEAT_DEBUG_PLUGINS` | `OFF` | 將 GStreamer 外掛程式偵錯轉發到 stdout。 |
| `SIMA_NEAT_USE_SYSTEM_GSTREAMER` | `ON`（主機）/ `OFF`（交叉） | 連結到系統的 GStreamer，而不是捆綁。 |
| `SIMA_NEAT_WARN_AS_ERROR` | `OFF` | 將編譯警告升級為錯誤。建議用於 CI。 |

### 工具鏈旋鈕

針對 Modalix 的交叉編譯：

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/modalix.cmake \
  -DSIMA_NEAT_PROFILE=Production
```

針對主機端開發：

```bash
cmake -B build -DSIMA_NEAT_PROFILE=Developer
```

列出您的樹狀結構所公開的內容：

```bash
cmake -L -B build       # list all cache variables
cmake -LA -B build      # include advanced
```

最上層的 `CMakeLists.txt` 檔案是選項名稱的權威來源。

Core 從所選 Scheduler 套件的收據繼承韌體及確切的 sysroot，不再另行宣告韌體。收據的平台與 SDK 通道必須符合 Core。CI 將此政策保留在建置產物中，並用它選取韌體完全相符的 DevKit，不使用備援版本。安裝 LLiMa 前，Core 會將其已發佈的韌體、平台及 SDK 通道與 Scheduler 比對，並驗證所選 DEB 屬於同一份發佈產物。LLiMa 未發佈確切的 sysroot 收據，因此此檢查確認的是宣告的韌體相容性，而非 sysroot 套件清單完全相同。使用本機 `--llima PATH` 建置時，請提供包含 core/dev/cli DEB、`llima-package-manifest.json` 及 `metadata.json` 的 LLiMa 發佈根目錄。缺少證據或證據不符時會停止建置。重新建置 Core 前，請先重新建置並發佈 Kernels → Runner → Scheduler，以及相符的 LLiMa。
