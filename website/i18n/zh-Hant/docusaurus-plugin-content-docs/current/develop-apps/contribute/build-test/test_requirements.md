---
title: "測試需求"
description: "針對 SiMa.ai 對 Neat 的貢獻，我們對測試的要求如下："
sidebar_position: 2
slug: /develop-apps/contribute/test_requirements
---

# 測試需求

每次行為變更都應包含測試，以證明其正確性並保留除錯能力。

## 基本期望

對於大多數功能開發，新增或更新測試，以涵蓋：

- 管線建置的正確性（預期的片段/字串結構）。
- 解析/驗證行為（協商和驗證失敗）。
- 執行階段行為（`run`，推送/拉取，以及適用的清理路徑）。
- 診斷品質（`PipelineReport` 在失敗路徑上的可用性）。

## 依變更類型所需測試

### 新節點或節點群組

- 對確定性片段產生進行單元級別檢查。
- 對協定或連結假設進行驗證/解析覆蓋。
- 至少有一個整合路徑，證明節點在實際鏈中可以正常工作。

### 執行階段/管線協調變更

- 對預期輸出行為進行成功路徑測試。
- 對逾時、缺少外掛程式或無效圖條件進行失敗路徑測試。
- 當狀態處理方式發生變更時，進行清理/生命週期安全斷言。

### 公開 API 簽名變更

- 新增或更新編譯時的覆蓋範圍，以測試已變更的公開標頭/用法。
- 對於非破壞性擴展，驗證現有用法是否仍然可以編譯。
- 對於已批准的破壞性變更，請包含以遷移為中心的測試/範例，以驗證替代 API 路徑。

### Python 繫結變更（`python/`，`pyneat`）

- 在 `python/tests` 中新增/更新 `pytest` 覆蓋範圍。
- 涵蓋互通協定（NumPy/PyTorch DLPack 路徑、複製與零複製行為）。
- 至少新增一個針對已安裝的 wheel 或可編輯安裝進行的導入/煙霧測試。

### 診斷或可觀察性變更

- 對新增/變更的報告欄位進行測試。
- 當資料來自串流執行緒進行更新時，進行並行安全行為測試。

## 回歸和確定性原則

- 錯誤修復必須包含回歸測試。
- 優先使用斷言確定性命名和穩定管線產生的測試。
- 如果輸出有意為非確定性，請記錄原因，並將斷言限制為穩定的不變量。

## 吞吐量測試中的 CPU 成本

`async_yolov8_tput_test` 會回報整個程序每個完成影格的使用者、系統與總 CPU 微秒數，
以及 `average_busy_cores`。後者是 CPU 時間除以實際時間，不是整塊板卡的百分比或
加速器使用率。多個 CPU 執行緒並行工作時，它可以大於一。

其 `cpu_window_fps` 與 CPU 計數器採用相同視窗：從第一個受測提交之前，到最後一個
輸出釋放為止。暖機、結果驗證、回報與結束清理均不包含在內。
既有 `avg_fps` 視窗與吞吐量門檻維持不變。`output_extraction_included` 和
`output_validation` 會識別測試執行的工作；請比較等同的設定。
未完成或數值驗證失敗的執行會回報 `cpu_window_complete=0`，不提供 CPU／影格值。
若只有吞吐量門檻失敗，仍會回報其他條件皆成功完成之影格的量測值。

新的基準測試可重用 `tests/perf/process_cpu_metrics.h`，每次執行取得兩次整個程序的
快照。不要逐影格取樣 CPU 計數器，也不要只為計算 CPU／影格而新增監視執行緒。

## 跳過原則

- 將跳過路徑視為例外情況，而不是正常的控制流程。
- 嚴格測試（預設）在無法執行時（由於缺少執行階段/工具/測試環境）應失敗。
- 只有明確標記為 `long` 的測試可以使用跳過語義（`return 77`），並且這些測試會在每週執行。
- 新增的測試不應在嚴格路徑中引入 `skip_test(...)`。

## 實用指令

使用專案建置的進入點：

```bash
./build.sh --all
```

對於會影響檔案內容的變更，請同時執行：

```bash
./build.sh --doc
```

請參閱 [建立](/develop-apps/contribute/build)，以了解所有支援的建置/測試模式。

## 貢獻者檢查清單

在開啟或合併 PR 之前：

- 新增/更新的測試與變更的行為相符。
- 既有的相關測試仍然通過。
- 已更新使用者可見行為的說明檔案。
- API/架構變更已反映在 [架構](/develop-apps/contribute/architecture) 中。
- 公開 API 變更符合 [程式碼規範](/develop-apps/contribute/coding_standard) 中 API 相容性原則。

## 必要覆蓋範圍與功能設定組合

標準 CTest 通道會在執行前記錄確切選擇，並與新產生的 JUnit 結果比較。
選擇為空、結果缺漏或重複、略過、停用測試，以及執行失敗都會使必要覆蓋驗證失敗。
測試選擇器、引數與期限仍由通道負責，而不是統計工具。

非長時間測試註冊不能藉由 `SKIP_RETURN_CODE`、略過正規表示式或僅供長時間測試的
環境設定逃過檢查。必要通道也可選擇既有長時間測試，保留其標準傳回 77 註冊，
但實際略過仍算失敗。Fast 保留其標為 long 的壓力測試；不會暗中取消選擇。
來源中的略過參照數量僅供參考，不代表實際略過測試數。

當 `SIMANEAT_GUARDED_HARDWARE_TESTS=ON` 時，`cmake/NeatGuardedTests.cmake`
透過現有硬體監督程式註冊測試與教學。這不會建立硬體准入。
執行原生通道前，請先取得設定的裝置擁有權與准入。結果驗證器檢查註冊原則；
不會取代監督程式或授權硬體執行。

Python 通道會記錄已選取及明確排除的節點 ID。必要的設定、呼叫與結束清理結果都必須
通過。在選擇前設定選用覆蓋範圍：

| 設定組合 | 設定 | 需求 |
| --- | --- | --- |
| Python 執行階段 | `NEAT_CI_PYTHON_PROFILE=runtime`（預設） | 使用已準備的 ResNet 測試固定資料啟用 Graph Phase 3；明確排除 Torch。 |
| Torch 整合 | `NEAT_CI_PYTHON_PROFILE=torch` | 設定經審查的確切 `NEAT_CI_TORCH_VERSION`；安裝／匯入必須成功。 |
| MIPI 攝影機 | `SIMA_NEAT_TEST_MIPI_CAMERA=1` | 提供攝影機硬體、`SIMA_NEAT_TUTORIAL_MODEL_MIPI` 與 `SIMA_NEAT_TUTORIAL_MIPI_DECODE`。 |
| 每週 HTTP 串流 | `NEAT_CI_STREAM_PROFILE=http`（預設） | 提供 HTTP MJPEG URL 及正值 FPS。兩個 HTTP 案例與同步煙霧檢查都仍為必要。 |
| 每週 HTTP + RTSP 串流 | `NEAT_CI_STREAM_PROFILE=all` | 另外提供 H264、H265 與 MJPEG RTSP URL，以及正值 MJPEG FPS。 |

不需硬體的攝影機 API 測試始終屬於執行階段設定組合。攝影機與 Torch 的排除仍是覆蓋
缺口，不能算通過。串流設定組合涵蓋 `codec_stream` 案例註冊，不代表其他被排除的
RTSP 通道已驗證。每個串流案例都有獨立的煙霧與重複契約測試結果，因此可用的 HTTP
串流不能用彙總 PASS 掩蓋未執行的 RTSP 案例。

準備每週模型時，請為 `SIMA_CI_MNIST_MODEL`、`SIMA_CI_SSD300_MODEL`、
`SIMA_CI_MNIST_IMAGES` 與 `SIMA_CI_MNIST_LABELS` 分別設定本機路徑或 HTTPS 位置，
以及經審查的 SHA-256。每個校驗碼使用對應的 `_SHA256` 變數。
資料集輸入為壓縮的 MNIST IDX 測試檔案。準備程序檢查確切位元組，並準備既有測試的
快取／變數；輸入缺少或不符都屬於準備失敗。它絕不修改無關的既有快取。
模型準備不依賴 RTSP 是否可用。

請將選擇、結果與設定組合排除報告一併保存於通道結果。
選定設定組合通過，不代表全部選用硬體或整合設定組合都已通過。

### 本機每週準備

本機執行與 CI 都應使用 `tools/ci/prepare_weekly_fixtures.py`。
預設準備程序同時需要 `mnist` 與 `ssd300`。若只驗證 MNIST，請明確傳入
`--fixture-group mnist`；報告會標明 SSD300 尚未驗證。此選擇不會變更 CTest 選擇，
也不會讓被略過的必要案例變成通過。完整每週驗證仍需要兩個群組。

本機啟動程式使用 `--environment-json <file>`，GitHub Actions 使用
`--github-env <file>`，或同時使用兩者。兩種輸出包含相同的環境對應資料。
本機啟動程式必須先將 JSON 對應資料合併至測試程序環境，再記錄選擇或准入硬體程序。
JSON 不是 shell 指令碼，也不是測試結果。僅建立它不會匯出環境變數。

僅在選定的模型與 MNIST gzip/IDX 檔案通過驗證後，輔助工具才會匯出
`SIMA_RUN_MNIST_PREPROC_CHANNELS1=1`。它會檢查 gzip CRC、IDX 幾何尺寸、長度、
影像／標籤數量與標籤範圍，並拒絕衝突的壓縮或原始快取項目。
每次準備嘗試請使用新的輸出目錄；失敗的暫存位元組會保留供診斷使用。

### 已安裝套件與私用前綴整合的區別

請區分下列驗證範圍：

- **私用前綴整合**檢查選定的標頭、程式庫與外掛程式，不修改全域套件資料庫。
  它不代表 Debian 安裝已驗證。
- **已安裝套件驗證**會對照選定且已驗證的封存檔，檢查實際 DevKit 根檔案系統、
  `dpkg` 記錄、負載位元組、符號連結與檔案擁有權。私用前綴無法滿足此檢查。

請繼續使用 `scripts/ci/verify_core_dist.sh` 驗證發行套件組成；本機建置時，
也應包含獨立選定的 `--expected-manifest`、`--expected-receipt` 與
`--expected-deb-dir` 輸入。使用完整的產生者發行套件組，包括五種中繼資料變體。
遠端 `sima-cli` 安裝會暫存選定資源，但不保留那些中繼資料檔案。
CI 工作流程會另行下載本次執行的完整 `neat-core-dist` 成品，再以此獨立輸入比對
CLI 實際安裝的套件。不要從已安裝機器重新產生產生者中繼資料。

請明確選擇所需套件設定組合：

| 設定組合 | 必要角色 |
| --- | --- |
| `core` | Core 執行階段／開發、Kernels 開發、Runner、Scheduler 及執行階段相容性套件 |
| `core-codecs` | `core` 加上 Codecs 執行階段／開發套件對 |
| `core-codecs-llima` | `core-codecs` 加上 LLiMa 執行階段／CLI 套件對 |

選定產生者憑證所使用的選用後端提供者仍為必要項目。
不會因封存檔缺少而推斷設定組合。LLiMa 與已淘汰的記憶體實作不是僅 Core 測試的
無條件相依項目。

在 DevKit 上，**使用現有安裝程式之前**，先透過 `tools/ci/verify_core_dist.py`
以 `--capture-platform-baseline <file>` 擷取基準。安裝後，以
`--package-profile <profile>`、`--verify-installed`、`--platform-baseline <file>`
與 `--write-package-inventory <file>` 執行發行套件組驗證器。
基準記錄受追蹤的原生套件已安裝或不存在的狀態，並保留該狀態、無關原生套件版本，
以及 OTA 命令是否可用。若 OTA 命令存在，其標準路徑、擁有權與位元組必須維持不變。
歷史套件名稱與 OTA 命令不必在每個映像上都存在；Palette 仍為必要。
明確包含於已驗證發行套件組中的提供者可升級，並改為對照所選封存檔檢查。

將清單的絕對路徑作為 `SIMA_TEST_PACKAGE_INVENTORY`，安裝前基準路徑作為
`SIMA_TEST_PLATFORM_BASELINE`，傳入現有的 `devkit-package-inventory` CTest 通道。
不要在安裝後重新擷取基準。選定清單包含預期封存資料，不是安裝 PASS。
原生測試會獨立檢查實際套件版本／架構、已載入 Core／程式庫路徑與擁有權、
未選取原生套件未變更的存在或不存在狀態、已淘汰套件不存在，以及 OTA 可用性與身分。
請繼續另行編譯已安裝標頭的範例；套件存在不能證明使用端可編譯或後端執行正確。

### 原生建置工具先決條件

在 Ubuntu 上，執行 `.github/workflows/vulcan-ci.yml` 中的建置工具 pytest 選擇前，
請安裝 `build-essential`、`file`、`nlohmann-json3-dev` 與 `python3-venv`。
平台基準探查會以 `-Werror` 編譯其真正共用的 C++ 契約；缺少編譯器／標頭是測試設定
失敗，不是略過檢查。失敗時會包含編譯器 stdout/stderr，並在原生執行檔執行前以
`file` 驗證它。
