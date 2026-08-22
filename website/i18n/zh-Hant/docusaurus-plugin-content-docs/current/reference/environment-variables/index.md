---
title: "環境變數"
description: "執行階段和建置環境變數"
sidebar_position: 6
slug: /reference/environment-variables
---

# 環境變數

本頁彙總了執行階段、建構工具和 SDK 工具所使用的環境變數。其中許多變數是除錯/診斷開關；大多數使用者可以忽略它們，除非需要進行疑難排解。

如需完整的已偵測清單，請參閱 [環境變數清單](./inventory)。

> 注意：有些控制旋鈕是內部使用的或僅供測試，而且可能會變更。這些旋鈕已在此檔案中列出。
> 因為它們目前會出現在程式碼執行路徑中。

## SDK 網路存取

- `NFS_SERVER_HOST_IP=<address>` — 用於建構遠端連線的 SDK 主機位址。
  Insight 和瀏覽器中的 VS Code URL。這通常由 `sima-cli sdk setup` 提供。
- `CONTAINER_HOST_IP=<address>` — 舊版備援方案，用於取得遠端主機的位址。
  當 `NFS_SERVER_HOST_IP` 主機無法使用時。
- `OPENVSCODE_SERVER_HTTPS_PORT=<port>` — 用於瀏覽器版 VS Code 的 HTTPS 連接埠。
  如果未設定， `neat` 不會顯示 VS Code 的網址。
- `OPENVSCODE_SERVER_TOKEN=<token>` — 以瀏覽器為基礎的 VS Code 存取權杖。
  sima-cli 產生的值是 URL 安全的。請將包含此值的 URL 視為憑證，切勿分享。
- `OPENVSCODE_WORKSPACE=<path>` — 由瀏覽器開啟的 VS Code 工作區
  （預設 `/workspace`）。
- `OPENVSCODE_SERVER_WITHOUT_TOKEN=1` — 啟動並宣傳基於瀏覽器的 VS Code。
  在沒有使用驗證令牌的情況下。僅在受信任的本地環境中使用；切勿在服務可以從不受信任的網路存取時啟用此功能。

## 統一的偵錯設定檔（建議採用）

- `SIMA_DEBUG_PROFILE=<components>` — 統一的除錯啟用開關，用於常見的診斷功能。
  - 元件：`pipeline`、`graph`、`gst`、`appsink`、`inputstream` 或 `all`。
  - 多個元件可以用逗號或空格分隔（例如 `pipeline,gst,inputstream`）。
- `SIMA_DEBUG_LEVEL=<0..3>` — 統一設定所使用的除錯詳細程度（預設值為 `1`）。
  - `0`：已停用
  - `1`：核心除錯日誌
  - `2`：詳細的診斷資訊／緩衝區層級的追蹤資訊
  - `3`：最高程度的詳細程度

舊有的逐變數除錯開關仍然有效，並且在明確設定時會覆寫預設設定檔。

## 核心建置/執行

- `SIMA_PIPELINE_STRING_DEBUG=1` ——在建置時印出最終的 gst-launch 字串。
- `SIMA_PIPELINE_STATE_DEBUG=1` — 額外的狀態變更日誌。
- `SIMA_PIPELINE_TEARDOWN_DEBUG=1` — 記錄管線關閉步驟。
- `SIMA_PIPELINE_DRAIN_BEFORE_TEARDOWN_MS=<ms>` — 在關閉管線之前，允許的排空時間（預設值為 1500 毫秒）。
- `SIMA_PIPELINE_DRAIN_MIN_OUTPUTS=<n>` — 在關閉管線之前，必須先處理掉的最小輸出數量（預設值為 1）。

## GStreamer 初始化 + 抑制

- `SIMA_ALLOW_GST_INIT=1` — 允許在已初始化時手動執行 `gst_init`。
- `SIMA_GST_SUPPRESS_JSON_WARNINGS=0/1` — 抑制 JSON 警告訊息（預設為 true）。
- `SIMA_GST_SUPPRESS_GOBJECT_ASSERTS=0/1` — 抑制 GLib 斷言日誌輸出（預設為啟用）。
- `SIMA_GST_SUPPRESS_DEVICE_LOGS=0/1` — 停用裝置日誌（預設為啟用）。

## GStreamer 超時時間

- `SIMA_STATE_CHANGE_TIMEOUT_MS=<ms>` — 管線狀態變更逾時時間（預設值為 15000 毫秒）。
- `SIMA_GST_TEARDOWN_TIMEOUT_MS=<ms>` — 關閉逾時時間（預設值為 2000）。
- `SIMA_GST_TEARDOWN_REAPER_MS=<ms>` — 關閉看門狗（預設值為 250）。
- `SIMA_GST_TEARDOWN_ASYNC=1` — 非同步清理。
- `SIMA_GST_POLL_SLICE_MS=<ms>` — 用於應用程式接收器（app sink）拉取資料時的輪詢間隔（預設值為 200）。
- 偏好的 API 設定選項：
  - `ValidateOptions.preroll_timeout_ms` — validate() 預先載入逾時。
  - `RunOptions.input_timeout_ms` — build()/run() 輸入模式逾時。
- 舊版備援環境變數（僅在無法傳遞選項時使用）：
  - `SIMA_GST_VALIDATE_TIMEOUT_MS=<ms>` — validate() 超時時間（預設值為 2000/10000）。
  - `SIMA_GST_RUN_INPUT_TIMEOUT_MS=<ms>` — run() 函數的輸入逾時時間（預設為 10000）。

## 診斷工具與探測器

- `SIMA_GST_DOT_DIR=<dir>` — 輸出管線失敗或除錯時的 DOT 圖。
- `SIMA_GST_BOUNDARY_PROBES=1` — 連接邊界流量探頭。
- `SIMA_GST_STAGE_TIMINGS=1` — 階段時間探測。
- `SIMA_GST_ELEMENT_TIMINGS=1` — 元素計時探測。
- `SIMA_GST_FLOW_DEBUG=1` — 元素流程探測。
- `SIMA_GST_ENFORCE_NAMES=1` — 在建置時強制執行命名規則。
- `SIMA_GST_OPTIONS_DEBUG=1` — 在建置期間記錄 GStreamer 選項。
- `SIMA_GST_BUFFER_DEBUG_LIMIT=<n>` — 限制緩衝區除錯列印的數量。
- `SIMA_GST_DETESS_INPUT_DEBUG=1` — detess 輸入除錯。
- `SIMA_GST_DETESS_OUTPUT_DEBUG=1` — detess 輸出除錯。
- `SIMA_GST_DETESS_POOL_DEBUG=1` — detess 資源池除錯。
- `SIMA_GST_APPSINK_BUFFER_DEBUG=1` — appsink 緩衝區除錯。
- `SIMA_GST_ALL_BUFFER_DEBUG=1` — 詳細的緩衝區除錯資訊。
- `SIMA_GST_RUN_INSERT_BOUNDARIES=1` ——在 run() 函數執行期間插入邊界。
- `SIMA_GST_VALIDATE_INSERT_BOUNDARIES=1` ——在 validate() 過程中插入邊界。

## 分派器／執行階段

- `SIMA_DISPATCHER_TRACE=1` — 追蹤分派器步驟。
- `SIMA_DISPATCHER_AUTO_RECOVER=0/1` — 自動恢復分派器（預設為啟用）。
- `SIMA_ASYNC_TPUT_DIAG=1` — 非同步吞吐量診斷。
- `SIMA_ASYNC_WARMUP=<n>` — 非同步預熱幀。
- `SIMA_PERF_POWER=1` — 在效能測試情境中啟用 SOM PMIC 供電軌道的監測。
- `SIMA_PERF_POWER_INTERVAL_MS=<ms>` — 電力採樣間隔（預設值為 100）。
- `SIMA_PULL_TIMEOUT_DIAG=0/1` — 報告拉取逾時的情況（預設為啟用）。
- `SIMA_STAGE_DEBUG=1` — StageRun 除錯日誌。

## 輸入串流 / 範例除錯

- `SIMA_INPUTSTREAM_DEBUG=1` ——詳細的 InputStream 日誌。
- `SIMA_INPUTSTREAM_WARN=1` ——關於 InputStream 事件的警告。
- `SIMA_INPUTSTREAM_POLL_MS=<ms>` — 輸入串流輪詢間隔（預設值為 50 毫秒）。
- `SIMA_INPUTSTREAM_DOT_ON_TIMEOUT=1` — 在逾時時輸出 DOT 資訊。
- `SIMA_INPUTSTREAM_META_DEBUG=1` — 記錄 GstSimaMeta 的詳細資訊。
- `SIMA_INPUTSTREAM_ALLOC_DEBUG=1` — 設定除錯。
- `SIMA_INPUTSTREAM_PUSH_TIMING=1` — 推送時間記錄。
- `SIMA_INPUTSTREAM_PREFLIGHT_RUN=1` ——為 InputStream 進行預先測試。
- `SIMA_SAMPLE_DEBUG=1` — 記錄範例轉換。
- `SIMA_SAMPLE_BYTES=1` — 記錄樣本位元組大小。
- `SIMA_SAMPLE_FORCE_BUNDLE=1` — 強制產生套件輸出，用於除錯。
- `SIMA_NEAT_CAPS_TRACE=1` — 張量追蹤，用於推導張量上限。

## 預處理／偵測／接線

- `SIMA_PREPROC_DEBUG_CONFIG=1` — 輸出預處理設定資訊。
- `SIMA_KEEP_DETESS_CONFIG=1` — 保留 detess 設定的輸出結果。
- `SIMA_DETESS_ASSERT_ON_ZERO=1` — 對 detess 輸出結果進行零值斷言。
- `SIMA_CLAMP_DETESS_NUM_BUFFERS=1` — 限制 detess 的緩衝區數量。
- `SIMA_DISABLE_SYNC_NUMBUFFERS_CVU_MLA=1` — 停用同步緩衝區數量限制。

## 模型（保留舊版環境變數名稱）
- `SIMA_MLA_NEXT_CPU=<domain>` — 覆寫 MLA 的 next_cpu。
- `SIMA_MPK_EXTRACT_ROOT=<dir>` — 模型檔案的載入根目錄。會解析為絕對路徑。
  路徑在每個程序中僅使用一次，因此重寫到提取的 JSON 中的路徑永遠不會依賴於工作目錄。權威性：如果無法寫入，則載入會失敗，而不是回退。如果未設定，則基礎路徑是已掛載的 NVMe 檔案系統中第一個可寫入的候選路徑，`/data`、`TMPDIR`，然後是工作目錄。NVMe 候選路徑必須是資料掛載上可寫入的 `/dev/nvme*` 區塊裝置：root、`/boot`、`/efi` 和其他系統掛載會被排除在外，vfat/ISO 檔案系統也會被排除。這些檢查適用於 NVMe 探索；`/data`、`TMPDIR` 和工作目錄回退保留其正常的檔案系統位置。NVMe 優先用於容量、可預測的位置以及避免對 eMMC 進行寫入。這並不是解碼速度的提升：此變數選擇輸出寫入的位置，而解碼是 CPU 密集型操作。

選擇的依據是可寫入性，而不是可用空間。`.tar.gz` 不限制其解壓縮後的大小，因此在解碼之前無法得知任何容量需求；空間會在解壓縮時以及從資訊清單中提取之前，針對每個區塊進行強制執行。因此，符合條件的 NVMe 將無條件地使用，如果沒有足夠的空間，則載入會失敗，並顯示 `output_storage_unavailable`，而不是回退到 eMMC。該檔案系統上的可用空間，或者 `SIMA_MPK_EXTRACT_ROOT` 指向其他位置，是解決方案。
- `SIMA_MPK_CLEANUP_EXTRACTED=0/1` — 在正常退出時，刪除每個程序提取的模型檔案資料（預設為 `1`）。
  啟用清理功能時，每個程序都會將內容解壓縮到其自己的 `proc_<pid>` 目錄中，並在程序結束時刪除該目錄。如果停用清理功能，則會保留該程序目錄以供檢查，並且不會將其納入過時目錄的垃圾回收機制中。該目錄不會被自動發現或由其他程序重複使用；當不再需要時，請手動刪除它。

`Model` 也可以接受一個已經組織好的套件根目錄，其中包含 `etc`、`lib` 和 `share`。該目錄將直接使用，無需解壓縮或複製；調用者擁有該目錄的生命週期，並且在使用模型時必須保持其不變。由 `tar -xzf` 產生的扁平目錄不是一個組織好的套件，因此不能直接使用。
- `SIMA_MPK_EXTRACT_GC_STALE_PROC=0/1` — 移除過時的無效資料。`proc_*` 啟動時提取根目錄（預設值） `1`).
- `SIMA_MODEL_TAR=<path>` — 範例/測試所使用的基本模型套件路徑。
  針對特定模型的覆寫設定（例如 `SIMA_RESNET50_TAR`、`SIMA_YOLO_TAR` 等）仍然優先生效。
- `SIMA_MPK_EXTRACT_MIN_FREE_BYTES=<bytes>` — 在準備和
  正在解壓縮模型封存檔（預設為 16 MiB）。
- `TMPDIR=<dir>`——僅被視為上述基礎的後期候選方案，並直接用於。
  由不選擇自己基礎版本的呼叫者進行暫存。模型載入不再在此處獨立進行暫存：解壓縮的快照和解壓縮的套件會共用選取的基礎版本。每次載入（包括僅檢查中繼資料的載入）都會將 `.tar.gz` 解壓縮一次到一個私有目錄中。在解壓縮期間，該檔案系統需要有足夠的空間來容納解壓縮的快照、套件以及 `SIMA_MPK_EXTRACT_MIN_FREE_BYTES`；150 MB 的參考套件的快照大約為 354 MB。請使用本機檔案系統——如果無法讀取可用空間，載入將會失敗，而網路掛載在斷開連接時會發生這種情況。載入完成後，無論成功還是失敗，都會刪除暫存目錄；如果發生意外終止或電源中斷，可能會留下一個暫存目錄。

由 `SIMA_MPK_EXTRACT_MIN_FREE_BYTES` 設定的可用空間保留量是一種盡力而為的機制。它會在解壓縮期間針對每個區塊，與檔案系統報告的可用空間進行檢查，因此，不相關的並行寫入器仍然可能在檢查之間耗盡檔案系統；然後載入將會失敗，並顯示已寫入的位元組數以及它正在使用的路徑。

## RTSP / H264

- `SIMA_H264_SDP_DUMP=<path>` — 將 H264 SDP 轉儲到檔案中。
- `SIMA_H264_SPS_FIXUP_STREAM=<path>` — 修正資料流中的 SPS。

## 測試／內部鉤子

- `SIMA_TENSOR_MAPFAIL_DEBUG=1` — 記錄張量映射失敗的情況。
