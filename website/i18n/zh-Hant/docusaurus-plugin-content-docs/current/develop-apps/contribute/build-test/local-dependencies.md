# 使用本機 Kernels、Runner 與 Scheduler 建置 Core

在 Neat 開發環境中，從 Core 原始碼目錄執行一般命令：

```bash
./build.sh --all --no-doc --install
```

建置會依序準備 **Kernels → Runner → Scheduler → Core**，接著執行現有的最終
安裝／部署。子建置絕不會安裝至 SDK、部署到 DevKit、執行目標測試、編譯 EV 韌體，
或啟用韌體。

將原始碼目錄並排放置，即可自動尋找來源：

```text
development/
├── core/
├── kernels/
├── runner/
└── scheduler/
```

Runner 包含不依賴框架的執行、準備、記憶體與原生傳輸。Scheduler 包含 GStreamer
整合／外掛程式與相容性轉接器。兩者各有獨立的來源根目錄與安裝套件。
原始碼目錄不必包含 `.git` 目錄；支援工作樹與來源快照。

## 選擇來源或現有本機套件

若原始碼位於其他位置，請設定 `NEAT_KERNELS_SOURCE`、`NEAT_RUNNER_SOURCE` 或 `NEAT_SCHEDULER_SOURCE`：

```bash
NEAT_RUNNER_SOURCE=/path/to/simaai-runtime ./build.sh --all --no-doc
```

若要重用本機建置的套件，請改設 `NEAT_KERNELS_PACKAGE_DIR`、`NEAT_RUNNER_PACKAGE_DIR`
或 `NEAT_SCHEDULER_PACKAGE_DIR`。每個套件目錄必須包含其已安裝的 `usr/` 樹、建置憑證，
以及由該元件標準套件流程產生的對應 Debian 套件。Kernels 使用標準 CMake
設定／建置／安裝及其套件產生器；Runner 與 Scheduler 則使用各自根目錄的 `build.sh` 入口：

```bash
NEAT_RUNNER_PACKAGE_DIR=/path/to/runner/dist/package \
  ./build.sh --all --no-doc
```

選擇順序為明確覆寫、有效的同層來源，再到同層的 `dist/package`。
明確指定套件會優先於同層來源。對同一元件同時設定來源與套件覆寫會產生錯誤。
本機建置失敗就會停止；絕不會退回較舊的套件。

**待辦：下載原生相依成品。** 在實作完成前，請為全部三個相依元件提供本機來源或套件。
缺少輸入會在安裝相依套件前失敗。現有 LLiMa 相依處理保持不變；不需要此選用後端時，
請使用 `--no-llima`。

## 平台相依項目

全部四個儲存庫都應使用相同的 Platform 3.0 SDK／工具鏈。選擇非預設工具鏈時，
請設定 `CMAKE_TOOLCHAIN_FILE`。Kernels 與 Runner 憑證必須在目標／編譯器、
核心契約及使用的建置設定上一致。標準套件流程也要求確切使用的 Kernels 負載身分。
Kernels 套件變更後，即使 Runner 來源未變更，也必須針對選定的 SDK 重新建置 Runner
套件。接著 Core 會重新連結對應的 Scheduler 客戶端實作；Scheduler 使用下層數值套件。
Core 的來源優先 DEB 會固定選定的 Scheduler 與數值 SDK 版本。
Runner 固定 Kernels；Scheduler 固定 Runner、Kernels 及已啟用的編解碼後端。
啟用測試的 Core 建置還需要同一產生來源的獨立
`simaai-scheduler-client-test-dev` 套件。來源建置透過 `--client-test-support` 要求它；
套件建置則選擇相鄰的 `client-test-support/` 目錄或
`NEAT_SCHEDULER_CLIENT_TEST_PACKAGE_DIR`。協調器會在提供僅供測試的 CMake 元件前，
檢查其正式套件憑證、來源、負載與封存檔。僅執行階段的建置不需要此套件。
建置會組合僅含中繼資料的 `simaai-runtime` 相容性套件，並使用確切的 Runner 與
Scheduler 相依關係。標準套件各自只擁有一次其檔案；僅 Runner 擁有現有的載入器設定。
請一律安裝完整且相符的發行套件組，而不是單獨替換某個套件。

### 遷移 GStreamer 標頭擁有權

尚未發行的 Core `0.6.0` 將 `/usr/include/gst/SimaaiGstErrors.h` 的擁有權移交給
Scheduler。包含路徑保持不變。Core 不再攜帶自己的實作或安裝該檔案；
使用者改用已安裝的 Scheduler 標頭與 `SimaScheduler::Headers` 目標。

請使用完整且相符的發行套件組，一起升級 Core 和 Scheduler。
Scheduler 會宣告尚未發行的 0.6.0 系列套件之標頭移轉，而不會將已修正的 Core 0.6.0 套件判定為不相容。
APT 變更檔案前，安裝程式會使用經過驗證的 Scheduler 擁有權檢查：原擁有者必須由同一套件組中不再包含該檔案的套件取代，或屬於明確宣告且核准的淘汰對象。
若新套件有重複的擁有者，或與無關套件發生擁有權衝突，安裝會遭拒。

在 Platform 3.0 上，完整的拆分套件組也會淘汰單體式的
`neat-runtime`、`neat-gst-plugins` 和 `neat-internals-dev` 0.4、0.5、0.6.0 系列套件，包括開發版本。
淘汰時，新擁有者必須提供相符且限定版本範圍的 `Breaks` 與 `Replaces` 宣告；`0.6.1~` 及之後的版本不在範圍內。
這不代表可以移除無關套件，也不宣稱與已淘汰的 API 相容。
執行階段與外掛程式的檔案移轉由 Scheduler 宣告；完整的執行階段仍須使用相符的 Runner 執行相依套件。
單獨安裝 Runner 不會淘汰單體式的 0.6.0 套件。

韌體啟用只會使用唯一提供啟用輔助程式的待安裝套件。
已安裝套件的識別資訊，以及輔助程式的擁有權、執行權限模式和位元組內容，都必須與所選套件相符。
如果套件組未提供輔助程式，安裝程序會保留現有韌體，不會呼叫先前安裝留下的輔助程式。
如果所選輔助程式不相符，安裝會失敗。

如果 DevKit 上的安裝中斷，請重新執行完整且已驗證的發行套件組。
只有未完成的套件屬於待安裝套件組且架構相符時，安裝程式才能修復。
修復必須安裝所選的確切版本，不得變更無關套件。
只有已完成設定且未被保留的舊套件，且待安裝的完整套件組已明確宣告淘汰並通過既有移除檢查時，才可移除。
修復會先完成已核准的轉移、檢查套件狀態，再於禁止移除套件的情況下重新安裝所有選定套件。只有完成這次最終重新安裝後，安裝才算成功。
無關的套件資料庫損壞仍需另外修復；安裝程式不會執行一般性的 APT 修復，也不保證能回復原狀。

請勿在已有 Core 開發套件的環境中單獨安裝 Scheduler DEB、使用 `--force-overwrite`，或刪除 dpkg 擁有的檔案。
獨立的 Scheduler 測試映像不需要安裝 Core；既有 Core 安裝則需要協同升級。

### 後端提供者

Scheduler 需要另外安裝的平台後端。此流程不會將它們複製進 Scheduler，也不會下載它們：

- 將 `NEAT_CODECS_PACKAGE_DIR` 設為另外建置的 Codecs 套件目錄。Scheduler 啟用
  編解碼器時，此目錄必須包含它所使用的確切已安裝 `usr/` 前綴，以及相符的
  `neat-codecs` 和 `neat-codecs-dev` Debian 套件。Scheduler 的相依協調器會驗證
  其負載、擁有權與相依控制資料；Core 保留已驗證的完整相依集合，並將兩個封存檔
  納入最終發行套件組。它不會建置 Codecs 或替換成其他版本。
- 使用 SDK 對應的 MLA Runtime、繪製與追蹤／記錄程式庫。
- 若需要明確選擇 SDK 路徑，請透過 `NEAT_SCHEDULER_CMAKE_ARGS` 傳入 CMake 定義，
  例如 `-DSIMA_RUNTIME_MLART_LIBRARY=/path/to/libmlart.so` 及其對應的標頭目錄。
  請使用 Scheduler 建置指南中記載的選項名稱。

`NEAT_RUNNER_CMAKE_ARGS` 與 `NEAT_SCHEDULER_CMAKE_ARGS` 內的引數會解析為 shell
樣式的詞彙，不會當作 shell 程式碼執行。它們無法覆寫選定的相依套件或安裝佈局。

## 增量建置與安裝

由 Core 管理的相依建置與套件位於 `deps/backend/`，與各儲存庫的獨立建置目錄分開。
CMake 採增量建置；`--clean` 會清理選定的本機相依建置與 Core 建置。

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

僅準備相依標頭時，`--install-deps-only` 需要本機套件覆寫。它複製標頭，
不建置或執行 A65/EV 程式碼。原生 clang-tidy 分析請使用 `scripts/run_cpp_tidy.sh`，
搭配選定的套件覆寫與 Scheduler 前綴。`TIDY_INSTALL_DEPS=1` 會安裝原生主機分析
先決條件，並以僅標頭模式呼叫已安裝的 Scheduler 協調器；不會啟用或修改 SDK sysroot。
使用 LLiMa 的 GenAI 實作檔案不包含在此僅主機分析設定內。指令碼不會以環境中其他
SDK LLiMa 套件取代選定建置的相依項目。僅說明內容建置不會準備原生相依項目。

EV 裝置建置仍是需要授權的獨立流程。這些命令都不會隱含呼叫 MetaWare、替換韌體，
或完成韌體變更驗證。

## 已發布的 CI 相依項目

一般 Vulcan 建置會透過 `sima-cli` 解析一次 `deps/manifest.json` 中的 Scheduler
原則，然後下載該不可變的發布版本。snap 原則選擇目前分支；尚未發布的分支會失敗，
而不會退回其他分支。`tools/ci/acquire_scheduler.py` 會在執行 Scheduler 啟動程式前
驗證發布者雜湊。已安裝的 Scheduler 協調器會分割並驗證隨附的 Runner、Kernels 和
Codecs 封存檔。Core 不會獨立解析這些儲存庫。

選定的 Scheduler 套件會安裝其確切 SDK 原則。在可拋棄式 CI SDK 容器中，
`scripts/ci/prepare_scheduler_sdk.sh` 會使用該版本的官方 sysroot 設定命令，
並保留下載的平台封存檔。協調器會檢查實際的 GCC14 版本／目標與目標 SDK 身分，
再透過 `NEAT_RUNNER_PLATFORM_RUNTIME_DEBS` 選擇 Runner 所使用的確切執行階段
提供者封存檔。原生 ARM64 與交叉編譯器的執行檔位元組不必完全相同，
但目標 ABI 與所選平台必須一致。此步驟不會安裝裝置套件或驗證韌體。

本機來源／套件覆寫仍具權威性。CI 取得相依項目是明確的操作；本機來源建置失敗
絕不會下載替代品。現有的
`tools/ci/materialize_dependencies.py --manifest ... --output-root ...` 命令仍是
另行審查的不可變清單可選用的傳輸方式。它不是預設流程，也沒有隱含的 `ci/dependencies.json`。

程式碼品質檢查會下載本次執行的 `neat-core-dist`、驗證它，並以僅標頭模式呼叫真正的
Core 委派程式與已安裝的 Scheduler 協調器。協調器會檢查建置所使用的相同下層套件
與已驗證的平台建置證據。Core 分析僅能看見 Scheduler 的前端前綴。
這是主機工具驗證，而非目標執行。

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

執行 Core 的來源優先 `build.sh` 前，請提供：

- `NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`：JSON 陣列，內含 Runner 選定的確切執行階段
  提供者 DEB 的絕對路徑。必須包含其所需的 SimaHeap 執行階段封存檔；若啟用 MLA 與
  TVM 後端，也須包含它們的執行階段封存檔。不要在此陣列中加入開發封存檔。
  Runner 會驗證實際載入器名稱與程式庫位元組。Scheduler 保留確切封存檔身分與負載
  清單，作為建置證據。由 Runner 宣告為 Platform 擁有的提供者不會包含在 Core 的
  可安裝 DEB 集合內；未宣告的提供者仍維持確切封存檔組合。

即使覆寫為預先建置的 Runner/Scheduler 套件，也必須明確指定這些選擇。
不可變的四元件具體化工具不會下載平台提供者封存檔。請在呼叫 Core 前向受支援的
產生者取得它們；缺少或未被使用的提供者會導致失敗，而不會退回 sysroot。
編譯時請使用相符的開發套件。在 DevKit 上，Scheduler 經驗證的准入工具會在套件變更前，
檢查已安裝的 Platform 提供者與即將安裝的使用元件是否相容。安裝程式拒絕替換或移除
這些提供者的 APT 計畫，並在事後驗證其身分與已安裝使用元件的完整相依集合。
`FORCE_INSTALL` 不會豁免這些檢查。不要為了將私用 DSO 標成已發行的提供者而重建 DEB。

套件組合檢查會確認負載擁有權互不重疊、相依版本確切且封存檔未變更。
這些建置時檢查不代表數值執行、動態遮罩或硬體安全釋放已通過驗證。
後者需要在選定的目標套件組上執行對應的整合測試。
