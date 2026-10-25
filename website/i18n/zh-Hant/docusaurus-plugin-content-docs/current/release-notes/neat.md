---
title: "Neat Library 版本資訊"
sidebar_position: 3
---

# Neat Library 版本資訊

SiMa.ai Neat Library 的版本資訊。

## 尚未發布

### 破壞性變更

- 模型輸出現在呈現 MPK 中定義的最終邏輯形狀與佈局，而非中間裝置形狀。例如，ResNet-50 的批次展平輸出形狀為 `[1, 1000]`，邏輯佈局未知，而不是四維 HWC 形狀。其 Float32 張量格式可能是 `EVXX_FLOAT32` 而非 `FP32`。請依張量的 dtype 與形狀解讀數值，不要從後端格式標籤推斷邏輯軸。
- Neat Library 的 C++ ABI 現在為 6，共用函式庫 SONAME 為 `libsima_neat.so.6`。張量現在攜帶特徵擷取器語意中繼資料，公開的 GenAI 請求／結果型別攜帶 ASR 任務、語言與探測中繼資料，`GraphLinkOptions` 包含即時接納限制，而 `BenchmarkOptions` 與 `BenchmarkReport` 新增了欄位。請重新建置 C++ 應用程式與外掛程式，並安裝相符的 Core 執行階段與開發套件。 `SimaArgMaxOptions` 新增了 `axis` 與 `keepdims`，`SimaArgMax` 的類別佈局也已變更。
- 即時圖組合現在使用 `GraphLinkOptions`、`Graph::connect()` 和 `Graph::build()`。預覽版 API `RealtimeGraphLinkOptions`、`connect_realtime()`、`build_fused_realtime_sources()` / `build_fused_realtime_source()` 和 `RealtimeEveryFrameByStream` 已移除。包含 `realtime_every_frame_by_stream` 的已儲存圖必須使用受支援的原則重新建立；請參閱[連接即時片段](/develop-apps/development-workflow/graph/#connect-live-fragments)。

- 尚未發行的套件版本仍為 **0.6.0**。`Model::Options` 與 `BoxDecodeOptions`
  現在包含 `yolox_seg_pose.pose_classes`；這些公開佈局新增欄位需要 ABI 6 使用元件。
  不要使用 ABI 相容性符號連結。預覽版本也可能變更公開型別，例如目前包含
  `kernel_runner_a65_workers` 的 `GraphOptions`。更新 0.6.0 預覽版本時，
  請使用相符的執行階段與開發套件重新建置應用程式、外掛程式及 Python 擴充模組。
  不要僅因不同預覽版本都使用 ABI 6，就混用它們的二進位檔案。

### 執行階段變更

- `Model::benchmark()` 會從 [Sentinel](/tools/sentinel/) 讀取板子遙測資料，而不再自行取樣電源軌功耗。延遲階段與吞吐量階段現在都會持續執行，直到各自測量了 `num_samples` 個輸入且至少已經過 `BenchmarkOptions::min_duration_ms`（預設為 10 秒），讓功耗與溫度讀數涵蓋多個 Sentinel 樣本。若要使用先前以輸入數量為準的計時方式，請將 `min_duration_ms` 設為 0。`BenchmarkReport` 新增 `latency_p50_ms`、`latency_p95_ms`、`latency_p99_ms`、`peak_power_watts`、`throughput_samples`、`throughput_duration_ms`，以及包含電源軌功耗、晶粒與板子溫度、整體與各核心的 CPU 使用率，以及 MLA、EV74 記憶體與整塊板子的 Linux 記憶體的 `telemetry`。`avg_power_watts` 與 `energy_joules` 的意義不變；`energy_joules` 為 `num_samples` 個輸入的能耗。Sentinel 無法使用時，仍會回報延遲與 FPS，並由 `telemetry.status` 說明原因。DevKit 安裝程式會透過 `sima-cli neat install sentinel` 安裝 Sentinel；若執行中的版本不是最新的正式版建置，則會將其升級。請參閱[測量模型執行效能](/develop-apps/development-workflow/model#power-and-board-telemetry)。
- 以 `enable_board_power()` 啟用的 Run 電力遙測也改從 Sentinel 讀取。Core 不再開啟 I2C 電源監測器，因此其讀取不再干擾 Sentinel 對相同電源軌的讀取，安裝程式也不再將 `sima` 加入 `i2c` 群組。Sentinel 每 2 秒取樣一次，因此短於約 2 秒的測量視窗可能沒有電力樣本。電源軌以 Sentinel 的指標命名，例如 `mla`，且只回報功耗。`sample_interval_ms`、`PowerMonitorProfile`、內建電源軌表、`PowerRailConfig` 的 I2C 欄位、`enable_modalix_som_power()`、`enable_modalix_dvt_power()`，以及 `modalix_som_power_monitor_options()` / `modalix_dvt_power_monitor_options()` 均已棄用並會被忽略。
- `TelemetryMonitor` 會在 `start()` 與 `stop()` 之間從 Sentinel 記錄板子功耗、溫度、CPU 使用率與記憶體，C++ 與 Python 皆可使用。`read()` 會傳回目前為止的摘要，而 `Telemetry` 會以 YAML 格式列印，並可儲存為 YAML 或 JSON。`Model::benchmark()` 使用它產生 `BenchmarkReport::telemetry`。`PowerMonitor`、`read_power_snapshot()`、`RunOptions::enable_board_power()`、`RunOptions::power_monitor` 與 `MeasureOptions::include_power` 已棄用，請改用 `TelemetryMonitor`；這些 API 仍可繼續運作。請參閱[記錄板子遙測](/develop-apps/development-workflow/model#record-board-telemetry)。
- `BenchmarkOptions::inputs` 可讓 `Model::benchmark()` 使用您自己的張量執行，例如以 OpenCV 讀取的影像。每個模型輸入各一個張量，形式與傳給 `run()` 時相同。與模型不符的輸入會在基準測試開始前擲回錯誤代碼為 `misconfig.input_shape` 的 `NeatError`。`inputs` 為空時會使用合成輸入。`BenchmarkReport::input` 會記錄 `"provided"` 或 `"synthetic"`，儲存的報告也會包含此欄位。請參閱[使用自己的輸入進行測量](/develop-apps/development-workflow/model#measure-with-your-own-inputs)。
- 外掛程式延遲已棄用，並將被移除。涵蓋 `MeasureOptions::include_plugin_latency` 與 `plugin_latency_source`、`BenchmarkOptions::include_plugin_latency`、`start_measurement(bool)`，以及 `Model::benchmark(int, bool)` / `Model::benchmark(bool)`。請改用 `start_measurement()` 與 `benchmark(num_samples)`。
- 硬體解碼使用直接編解碼器工作階段與 DMA-BUF 輸出。Core 不再使用解碼器常駐程式接納租約或 `SIMA_DECODER_ADMISSION_REQUIRE`。解碼器集區與調校選項仍可使用。
- C++ 和 Python 可透過 `SimaDecode` 與 `RtspDecodedInput` 使用原生 H.265/HEVC 解碼。`RtspEncodedInput` 無需解碼即可提供已剖析的 H.265 存取單元。H.265 輸入必須使用 HEVC Main 設定檔、8 位元、4:2:0。編解碼器選擇器接受 `H265` 和 `HEVC`；H.264 選擇器也接受 `AVC`。`FormatTag` / `pyneat.Format` 在編碼資料的圖邊界接受相同別名，並仍序列化為 `H264` 和 `H265`。
- `VideoSender` 透過 `VideoSenderOptions::Passthrough(codec)` / `pyneat.VideoSenderOptions.passthrough(codec)`，將已編碼的 H.264 或 H.265 以 RTP over UDP 轉送，全程不重新編碼。H.265 預設使用 RTP 負載類型 98；H.264 維持 96。`H264RtpUdpFromEncoded()` 已棄用，請改用 `Passthrough(RtspCodec::H264)`。
- 若原始 `VideoSender` 輸入是相容的 DMA 支援 NV12，且已安裝的編碼器宣告 `input-layout-aware=true`，現在會自動省略格式轉換。既有的編碼器輸入邊界會在最終編碼器儲存空間中準備其他原始輸入。`H264RtpUdpFromRaw(...)` C++ 與 Python API 維持不變。
- `InputMemoryPolicy::Auto` 區分偏好的配置與必要的輸入儲存空間。偏好裝置的路徑會保留呼叫端擁有的 `Tensor` 與 `Sample` 儲存空間，而不經由 CPU 影像轉換。相容的 DMA-BUF 輸入維持零複製；既有 CPU 輸入仍可能需要在裝置輸入處實體化。明確的 EV74／DMS0 輸入要求仍會執行。
- RTSP 輸入使用 `RtspEncodedInputOptions` 和 `RtspDecodedInputOptions` 上單一且不依賴編解碼器的 `payload_type` 欄位來選擇 RTP 負載類型：`-1` 選擇編解碼器預設值（H.264/H.265 為 96，MJPEG 為 26）、`0` 停用負載篩選，而正值則選擇指定的負載。`RtspEncodedInputOptions::h264_payload_type` 和 `mjpeg_payload_type` 已棄用；若它們改變解析後的負載，執行階段會發出一次警告。
- 一般的 `build()` 現在會針對符合條件的即時扇入自動選擇融合式 lowering。直接編碼的 H.264 或 H.265 `VideoSender` 分支會在解碼前融合，無需將解碼後的影格複製至 CPU。來源、解碼器與傳送端的編解碼器必須一致；不一致的組合仍會留在不同的管線區段中。若要進行即時預覽，請將該邊緣設為 `RealtimeLatestByStream`，讓速度較慢的視訊接收器以新存取單元取代過時的存取單元，而不是對解碼器分支施加背壓。

- 新增 C++ 和 Python `CameraInput` 檔案與教學內容，涵蓋由 MIPI/libcamera 來源擁有的圖，以及 CVU/MLA 模型路徑前的自適應 SiMaAI 記憶體交接。
- `MetadataSender` 現在會將較大的 JSON 訊息分塊，使 UDP 負載維持在 1200 位元組以內。請在升級此 Neat Library 版本前或同時，將 Insight 更新至支援中繼資料分塊重組的版本；舊版 Insight 仍支援最高 1200 位元組、未變更的 JSON 負載。

### 圖的建構與驗證

- 預設的公開 `Input` 直接連接至單一模型影像輸入時，現在會繼承其宣告的影像格式。明確的輸入選項仍具決定權；有歧義的連接必須使用明確選項。這會改變樣本的解讀方式，而不新增負載複製。
- 圖組合現在會將一個節點物件視為一個邏輯頂點。重複插入與重疊的片段匯入會以不可分割的方式失敗，而重複呼叫 `connect()` 時會重複使用既有節點進行扇出。
- 每個具體化的管線區段現在都會在 `build()` 期間驗證最終的 GStreamer 名稱，不需另外明確呼叫 `validate()`。名稱重複或遺漏時會以 `misconfig.pipeline_shape` 失敗，而不會產生截斷的管線。
- 自訂片段現在會報告所有明確名稱，並連同宣告一起轉換具名 pad 參照。名稱衝突會直接遭到拒絕，不會自動重新命名。

| 版本 | 相容的 Neat SDK | 備註 |
|---|---|---|
| 0.4.0 | 2.1.3.0 | [Neat Library 0.4.0](https://github.com/sima-neat/core/releases/tag/v0.4.0) |
| 0.3.0 | 2.1.2.3 | [Neat Library 0.3.0](https://github.com/sima-neat/core/releases/tag/v0.3.0) |
| 0.2.2 | 2.1.2.2 | [Neat Library 0.2.2](https://github.com/sima-neat/core/releases/tag/v0.2.2) |
| 0.2.1 | 2.1.2.1 | [Neat Library 0.2.1](https://github.com/sima-neat/core/releases/tag/v0.2.1) |
| 0.2.0 | 2.1.2 | [Neat Library 0.2.0](https://github.com/sima-neat/core/releases/tag/v0.2.0) |
| 0.1.0 | 2.0.0 | [Neat Library 0.1.0](https://github.com/sima-neat/core/releases/tag/v0.1.0) |
