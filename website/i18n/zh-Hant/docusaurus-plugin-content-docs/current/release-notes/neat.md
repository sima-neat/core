---
title: "Neat Library 版本資訊"
sidebar_position: 3
---

# Neat Library 版本資訊

SiMa.ai Neat Library 的版本資訊。

## 尚未發布

### 破壞性變更

- Neat Library 的 C++ ABI 現在為 4，共享函式庫的 SONAME 為 `libsima_neat.so.4`。張量現在帶有特徵擷取器的語義中繼資料；公開的 GenAI 請求/結果類型帶有 ASR 工作、語言和探測中繼資料；`GraphLinkOptions` 則包含即時准入限制。請重新建置 C++ 應用程式和外掛程式，並安裝相符的 Core 執行階段與開發套件。
- 即時圖組合現在使用 `GraphLinkOptions`、`Graph::connect()` 和 `Graph::build()`。預覽版 API `RealtimeGraphLinkOptions`、`connect_realtime()`、`build_fused_realtime_sources()` / `build_fused_realtime_source()` 和 `RealtimeEveryFrameByStream` 已移除。包含 `realtime_every_frame_by_stream` 的已儲存圖必須使用受支援的原則重新建立；請參閱[連接即時片段](/develop-apps/development-workflow/graph/#connect-live-fragments)。

### 執行階段變更

- C++ 和 Python 可透過 `SimaDecode` 與 `RtspDecodedInput` 使用原生 H.265/HEVC 解碼。`RtspEncodedInput` 無需解碼即可提供已剖析的 H.265 存取單元。H.265 輸入必須使用 HEVC Main 設定檔、8 位元、4:2:0。編解碼器選擇器接受 `H265` 和 `HEVC`；H.264 選擇器也接受 `AVC`。`FormatTag` / `pyneat.Format` 在編碼資料的圖邊界接受相同別名，並仍序列化為 `H264` 和 `H265`。
- `VideoSender` 透過 `VideoSenderOptions::Passthrough(codec)` / `pyneat.VideoSenderOptions.passthrough(codec)`，將已編碼的 H.264 或 H.265 以 RTP over UDP 轉送，全程不重新編碼。H.265 預設使用 RTP 負載類型 98；H.264 維持 96。`H264RtpUdpFromEncoded()` 已棄用，請改用 `Passthrough(RtspCodec::H264)`。
- 若原始 `VideoSender` 輸入已確認是系統或 SiMaAI 記憶體中的 NV12，且安裝的編碼器宣告 `input-layout-aware=true`，現在會自動省略格式轉換。其他原始格式、未知的記憶體/設定，以及沒有可靠格式合約的輸入，仍會依照既有行為轉換為 NV12。`H264RtpUdpFromRaw(...)` C++ 和 Python API 維持不變。
- RTSP 輸入使用 `RtspEncodedInputOptions` 和 `RtspDecodedInputOptions` 上單一且不依賴編解碼器的 `payload_type` 欄位來選擇 RTP 負載類型：`-1` 選擇編解碼器預設值（H.264/H.265 為 96，MJPEG 為 26）、`0` 停用負載篩選，而正值則選擇指定的負載。`RtspEncodedInputOptions::h264_payload_type` 和 `mjpeg_payload_type` 已棄用；若它們改變解析後的負載，執行階段會發出一次警告。
- 一般的 `build()` 現在會針對符合條件的即時扇入自動選擇融合式 lowering。直接編碼的 H.264 或 H.265 `VideoSender` 分支會在解碼前融合，無需將解碼後的影格複製至 CPU。來源、解碼器與傳送端的編解碼器必須一致；不一致的組合仍會留在不同的管線區段中。若要進行即時預覽，請將該邊緣設為 `RealtimeLatestByStream`，讓速度較慢的視訊接收器以新存取單元取代過時的存取單元，而不是對解碼器分支施加背壓。

- 新增 C++ 和 Python `CameraInput` 檔案與教學內容，涵蓋由 MIPI/libcamera 來源擁有的圖，以及 CVU/MLA 模型路徑前的自適應 SiMaAI 記憶體交接。
- `MetadataSender` 現在會將較大的 JSON 訊息分塊，使 UDP 負載維持在 1200 位元組以內。請在升級此 Neat Library 版本前或同時，將 Insight 更新至支援中繼資料分塊重組的版本；舊版 Insight 仍支援最高 1200 位元組、未變更的 JSON 負載。

### 圖的建構與驗證

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
