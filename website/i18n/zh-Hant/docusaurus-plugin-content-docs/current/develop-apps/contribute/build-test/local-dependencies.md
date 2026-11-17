# 使用 Scheduler 與 LLiMa 相依項目建置 Core

Core 會針對兩個相依項目建置：Scheduler 與 LLiMa。在 Palette SDK 中，從 Core 原始碼目錄執行
`build.sh`：

```bash
./build.sh --all --no-doc
```

依預設，`build.sh` 會：

1. 從 Vulcan 下載已發布的 Scheduler 套件組與 LLiMa 套件。
2. 將 Scheduler 套件組解壓縮至原始碼目錄的 `deps/` 資料夾，並將 LLiMa 安裝至 SDK sysroot。
3. 建置 Core。

`build.sh` 不會將 Scheduler 套件組安裝至 SDK sysroot，因此多個 Core 原始碼目錄可在同一個
SDK 容器中，同時針對不同的 Scheduler 套件組建置。這些原始碼目錄共用 sysroot 中的 LLiMa：
若 sysroot 已包含相同的套件，`build.sh` 會略過 LLiMa 的安裝。

## 讓 SDK sysroot 與 Scheduler 套件組一致

每個 Scheduler 套件組都會記錄其建置時所對應的 sysroot 版本。`build.sh` 會將它與
`sysroot status` 比較，若兩者不同，會顯示警告並附上確切的 `sysroot update` 命令。
若要在建置時一併更新 sysroot，請傳入 `--update-sysroot`：

```bash
./build.sh --all --no-doc --update-sysroot
```

這會連同 Core 編譯時所依據的平台套件執行 `sudo sysroot update <version>`。
CI 一律會傳入此選項。

## 使用本機 Scheduler 或 LLiMa 套件

若要針對自行建置的 Scheduler 或 LLiMa 建置 Core，請傳入其 DEB 資料夾：

```bash
./build.sh --scheduler ../scheduler-debs --llima ../llima/dist/debs
```

每個選項都可單獨使用；`build.sh` 仍會下載另一個相依項目。指定的 Scheduler 套件組會取代此原始碼目錄 `deps/` 資料夾中
已發布的套件組，指定的 LLiMa 套件則會取代 SDK sysroot 中的 LLiMa。

Scheduler 資料夾須包含 Scheduler 建置時使用的完整套件組：

- `simaai-scheduler`
- 它所依據建置的 `simaai-runner` 與 `simaai-kernels-dev` 套件
- 啟用 EV 的建置所需、相符的 `simaai-kernels-firmware` 套件
- Scheduler 使用 Codecs 時的 `neat-codecs` 與 `neat-codecs-dev` 套件對
- 選用的 `simaai-runtime`；測試建置另需 `simaai-scheduler-client-test-dev`

Core 不會選擇 Kernels、Runner 或 Codecs。若要測試其中某一項的變更，請依 Scheduler 的
建置指南，針對該變更建置 Scheduler，將產生的套件放在同一個資料夾，並以 `--scheduler`
傳入該資料夾。Scheduler 的協調器會檢查這些套件是否屬於同一組，若不是便會停止建置。

若設定了任何 `NEAT_KERNELS_*`、`NEAT_RUNNER_*`、`NEAT_SCHEDULER_*` 或 `NEAT_CODECS_*`
來源或套件變數，`build.sh` 會報錯並停止。若要在不含 LLiMa 的情況下建置，請使用
`--no-llima`。

## 平台相依項目

請以相同的 Platform 3.0 SDK／工具鏈建置 Kernels、Runner、Scheduler 與 Core。選擇非預設工具鏈時，
請設定 `CMAKE_TOOLCHAIN_FILE`。Kernels 與 Runner 憑證必須在目標／編譯器、
核心契約及使用的建置設定上一致。標準套件流程也要求確切使用的 Kernels 負載身分。
Kernels 套件變更後，即使 Runner 來源未變更，也必須針對選定的 SDK 重新建置 Runner
套件。接著 Core 會重新連結對應的 Scheduler 客戶端實作；Scheduler 使用下層數值套件。
Core 的來源優先 DEB 會固定選定的 Scheduler 與數值 SDK 版本。
Runner 固定 Kernels；Scheduler 固定 Runner、Kernels 及已啟用的編解碼後端。
啟用測試的 Core 建置還需要同一產生來源的獨立
`simaai-scheduler-client-test-dev` 套件，並放在同一個 Scheduler 資料夾中。協調器會在提供僅供測試的 CMake 元件前，
檢查其正式套件憑證、來源、負載與封存檔。僅執行階段的建置不需要此套件。
建置會組合僅含中繼資料的 `simaai-runtime` 相容性套件，並使用確切的 Runner 與
Scheduler 相依關係。標準套件各自只擁有一次其檔案；僅 Runner 擁有現有的載入器設定。
請一律安裝完整且相符的發行套件組，而不是單獨替換某個套件。

### 遷移 GStreamer 標頭擁有權

Core `0.6.0` 將 `/usr/include/gst/SimaaiGstErrors.h` 的擁有權移交給
Scheduler。包含路徑保持不變。Core 不再攜帶自己的實作或安裝該檔案；
使用者改用已安裝的 Scheduler 標頭與 `SimaScheduler::Headers` 目標。

請使用完整且相符的發行套件組，一起升級 Core 和 Scheduler。
Scheduler 會宣告0.6.0 系列套件之標頭移轉，而不會將已修正的 Core 0.6.0 套件判定為不相容。
APT 變更檔案前，安裝程式會使用經過驗證的 Scheduler 擁有權檢查：原擁有者必須由同一套件組中不再包含該檔案的套件取代，或屬於明確宣告且核准的淘汰對象。
若新套件有重複的擁有者，或與無關套件發生擁有權衝突，安裝會遭拒。

在 Platform 3.0 上，完整的拆分套件組也會淘汰單體式的
`neat-runtime`、`neat-gst-plugins` 和 `neat-internals-dev` 0.4、0.5、0.6.0 系列套件，包括開發版本。
淘汰時，新擁有者必須提供相符且限定版本範圍的 `Breaks` 與 `Replaces` 宣告；`0.6.1~` 及之後的版本不在範圍內。
這不代表可以移除無關套件，也不宣稱與已淘汰的 API 相容。
執行階段與外掛程式的檔案移轉由 Scheduler 宣告；完整的執行階段仍須使用相符的 Runner 執行相依套件。
單獨安裝 Runner 不會淘汰單體式的 0.6.0 套件。

啟用 EV 的 Modalix 發行套件必須包含 Scheduler 所選、相符的 `simaai-kernels-firmware` 套件。
韌體缺失或不相符時，准入驗證會失敗。
若要取代舊版 `neat-ev74-firmware` 套件，必須由所選的同一韌體套件宣告符合該版本的 `Replaces`，以及 `Breaks` 或 `Conflicts`；一般交易與移除檢查仍然適用。
韌體啟用只會使用唯一提供啟用輔助程式的待安裝套件。
已安裝套件的識別資訊，以及輔助程式的擁有權、執行權限模式和位元組內容，都必須與所選套件相符。
如果所選輔助程式不相符，安裝會失敗。
對於不需要 EV74 韌體的發行套件，缺少輔助程式時會保留現有韌體。
安裝程序絕不使用先前安裝留下的輔助程式。

如果 DevKit 上的安裝中斷，請重新執行完整且已驗證的發行套件組。
只有未完成的套件屬於待安裝套件組且架構相符時，安裝程式才能修復。
修復必須安裝所選的確切版本，不得變更無關套件。
只有已完成設定且未被保留的舊套件，且待安裝的完整套件組已明確宣告淘汰並通過既有移除檢查時，才可移除。
修復會先完成已核准的轉移、檢查套件狀態，再於禁止移除套件的情況下重新安裝所有選定套件。只有完成這次最終重新安裝後，安裝才算成功。
無關的套件資料庫損壞仍需另外修復；安裝程式不會執行一般性的 APT 修復，也不保證能回復原狀。

請勿在已有 Core 開發套件的環境中單獨安裝 Scheduler DEB、使用 `--force-overwrite`，或刪除 dpkg 擁有的檔案。
獨立的 Scheduler 測試映像不需要安裝 Core；既有 Core 安裝則需要協同升級。

## 增量建置與安裝

每次建置都會在 `deps/scheduler/` 中重新取得選定的 Scheduler 套件組，並將其套件暫存於
`deps/backend/`。CMake 以增量方式建置 Core；`--clean` 也會移除 Core 的建置。

連結前，Scheduler 的相依協調器會根據憑證驗證已安裝的核心標頭，並檢查 Runner
使用的契約及 Scheduler 的確切 Runner 輸入。在準備部署套件前，它會驗證每個
Debian 套件的 `/usr` 負載都與選定的安裝樹相符。不會選擇或下載舊的 Internals 套件。

Core 使用已安裝的 `SimaScheduler` 客戶端元件，而非同層來源標頭或下層開發介面。
下層執行階段身分仍保留在已驗證的相依集合內。
產生的相依選擇會記錄於 `build/resolved_manifest.json` 和 `build/buildinfo.json`。
只有完整建置才會進入最終安裝；這不保證 SDK 或裝置安裝能以交易方式回復。

套件建置資訊在 CMake 設定前屬於暫定資料。編譯後，`build.sh` 會依據該次建置的
CMake 快取、設定的編譯器、目標 Python 標頭與建置出的擴充模組更新資訊，
再執行安裝檢查或封裝。目標 Python ABI 旗標必須與擴充模組後綴一致；
不使用主機 Python 版本。停用 Python 的建置會記錄該設定，而不記錄 Python ABI。

`compatibility.host_sdk` 描述建置主機觀察到的 SDK 版本；
`compatibility.target_sdk` 描述選定的 sysroot 與編譯器。即使主機具有 SDK 版本，
缺少目標 SDK 版本時仍會保持無法取得。清單中的平台版本是獨立的相容性需求，
不能取代未觀察到的 SDK 版本。`ELXR_SDK_RELEASE_FILE` 選擇主機版本檔案。
`NEAT_TARGET_SDK_RELEASE_FILE` 可明確提供目標版本檔案，但必須與所選 sysroot 中的
版本檔案一致。`NEAT_BUILDINFO_TOOLCHAIN` 會在設定的編譯器身分之外保留明確的
顯示標籤，而不是取代編譯器身分。

中繼資料檢查會拒絕衝突的編譯 sysroot、標頭與擴充模組 ABI 標籤。
此中繼資料路徑不支援分開的編譯／連結 sysroot，以及其他 CMake 編譯器引數／目標／
工具鏈情境；它會失敗，而不是回報未驗證的目標身分。
具有條件標記的 NumPy 需求會逐字保留自 `pyproject.toml`，而非簡化成單一無條件最低版本。

來源記錄採用實際原始碼提交與未提交變更狀態。它會分別記錄已追蹤的變更，以及未被
忽略的未追蹤檔案／符號連結身分；被忽略的建置產物不是來源快照。
中繼資料輸出本身會排除於未追蹤內容摘要之外，以避免自我參照。
具有本機未提交變更的候選版本不會被標示為乾淨的已發布成品。

僅準備相依標頭時，`--install-deps-only` 使用 `--scheduler` 資料夾，或下載已發布的套件組。它複製標頭，
不建置或執行 A65/EV 程式碼。原生 clang-tidy 分析請使用 `scripts/run_cpp_tidy.sh`，
搭配選定的 Scheduler 與相符測試支援套件的前綴。`TIDY_INSTALL_DEPS=1` 會安裝原生
主機分析先決條件。指令碼直接使用各套件的公開 CMake 匯出項目，不會啟用或修改 SDK sysroot。
使用 LLiMa 的 GenAI 實作檔案不包含在此僅主機分析設定內。指令碼不會以環境中其他
SDK LLiMa 套件取代選定建置的相依項目。僅說明內容建置不會準備原生相依項目。

EV 裝置建置仍是需要授權的獨立流程。這些命令都不會隱含呼叫 MetaWare、替換韌體，
或完成韌體變更驗證。

## 已發布的 CI 相依項目 {#published-ci-dependencies}

一般 Vulcan 建置會透過 `sima-cli` 解析 `deps/manifest.json` 中的 Scheduler
原則，然後下載所選的不可變發布版本。與 LLiMa 相同，分支 `snap` 會先嘗試目前的
Core 分支。只有在解析失敗，且 Scheduler 的已發布分支索引確認該分支不存在時，
才會明確告知退回並重試 `develop:latest`。其他查詢失敗仍會視為錯誤，包括索引無法
取得或無效的情況。標籤 snap 與明確的
分支／提交選擇器絕不退回。裝置與 PCIe 主機的取得流程共用此原則；PCIe 的
`--scheduler-spec <commit>` 也維持嚴格選擇。解析後的下載、驗證、啟動或暫存失敗
仍會視為錯誤。記錄會保留實際選定的分支與不可變提交。
`tools/ci/acquire_scheduler.py` 會在執行 Scheduler 啟動程式前
驗證發布者雜湊。已安裝的 Scheduler 協調器會分割並驗證隨附的 Runner、Kernels 和
Codecs 封存檔。Core 不會獨立解析這些儲存庫。

選定的 Scheduler 套件會安裝確切的 SDK 原則。在 SDK 中，`build.sh` 會執行
`scripts/ci/prepare_scheduler_sdk.sh`，透過 Scheduler 的 `verify-sdk` 命令驗證目前
使用的目標 sysroot，並將觀察結果存至 `deps/`。共用的 `build_platform` 記錄包含
`elxr_version`、完整的 `sysroot_version` 及編譯器 `target`，必須與 Runner 和
Scheduler 的建置記錄一致。不會下載平台 DEB。每日版或預先發行版目標只有在指定
`--update-sysroot` 時才執行 `sysroot update`；正式版本使用既有的作用中目標 sysroot。
相符的覆蓋層可能比原始容器映像更新。原生 ARM64 與交叉編譯器必須使用相同的目標
ABI 及平台，但執行檔的位元組不必完全相同。此步驟不會安裝或驗證 DevKit。

`--scheduler` 資料夾絕不會與下載的套件混用，也不會被下載的套件取代。現有的
`tools/ci/materialize_dependencies.py --manifest ... --output-root ...` 命令仍是
另行審查的不可變清單可選用的傳輸方式。它不是預設流程，也沒有隱含的 `ci/dependencies.json`。

程式碼品質檢查使用 `tools/ci/acquire_scheduler.py`，依照相同的相依性原則獨立取得
已發布的 Scheduler，無須等待 Core 建置完成。Scheduler 的啟動程序會驗證套件組。
主機分析使用其前端與相符測試支援的 CMake 匯出項目，不需啟用 SDK 或在目標上執行。

受保護效能測試通道所需的目標端監督程式，請參閱[硬體測試擁有權](./hardware-test-safety.md)。

## 驗證完成的發行套件組

`build.sh --all` 會在產生套件中繼資料後檢查完成的發行套件組。
`scripts/ci/verify_core_dist.sh` 在 CI 中使用所選 `sima-cli` Python 環境執行相同的唯讀檢查。
若直譯器不在所選 CLI 執行檔旁，請明確設定 `SIMA_CLI_PYTHON`。

此檢查會驗證相依套件位元組、相容性欄位、安裝程式允許清單、所有資源校驗碼、
公開 wheel 隔離與已安裝的 CTest 樹。它使用現有的產生者憑證；不會解析或修復相依項目。
請區分以下檔案：

- `dist/manifest.json`：平台、Model Zoo 與 ABI 相容性。
- `dist/resolved-deps-manifest.json`：所選後端憑證的原樣副本。
- `build/resolved_manifest.json`：更完整的 Core 建置解析結果。

完成此檢查後，不要替換檔案或重新計算所選校驗碼。
若成品變更，請重新建置中繼資料並再次驗證完整發行套件組。


## 明確的平台提供者套件

Core 的正式方框／遮罩發布使用 int32 座標。請選擇以標準 Kernels 繪製功能及對應的
Runner 執行 API 建置的 Scheduler。`SIMANEAT_REQUIRE_INT32_RENDERER` 預設遵循
執行階段成品需求，若已安裝的 Scheduler 缺少此功能，設定會失敗。
僅在 Core 設定不使用 Core 方框／遮罩繪製時，才將其設為 `OFF`。

繪製使用選定的 Kernels/Runner 套件，而非獨立的平台繪製器封存檔。
請為要求 API 修訂版 2 一併重新建置整個套件組。

平台 SimaHeap、MLA 和 TVM 程式庫來自選定的 SDK/sysroot。請勿提供
`NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`，也不必下載平台封存檔來證明來源。Scheduler
會將實際 SDK 識別資訊與自身及 Runner 的 `build_platform` 記錄比對。Core 會把驗證後的
識別資訊寫入安裝的 `buildinfo.json`，於編譯後重新檢查，並在發布套件中繼資料前驗證
最終 CMake sysroot 與編譯器目標。

此次中繼資料轉換須依 Runner、Scheduler、Core 順序重新建置。缺少 `build_platform`
的產生端必須重新建置，舊記錄無法略過 SDK 驗證。NEAT 相依 DEB 的確切套件、校驗和及
承載資料擁有權檢查仍會保留。平台擁有的程式庫不會包含在 Core 的可安裝套件組中。
既有的裝置安裝與硬體驗證仍須另行進行；建置中繼資料無法證明數值執行或作業結束。
