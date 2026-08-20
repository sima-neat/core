---
title: "傳送影片"
description: "VideoSender 的 H.264 和 H.265 RTP/UDP 線上格式"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# 傳送視訊

當圖（Graph）需要將視訊傳送到外部接收器時，請使用 `VideoSender`。`VideoSender` 會傳回一個可重複使用的 `Graph` 片段，因此請使用 `Graph::add(...)` 將其新增。

`VideoSender` 會透過 RTP/UDP 傳送 H.264 或 H.265。原始輸入會以 H.264 格式進行編碼；編碼後的 H.264 和 H.265 輸入會直接轉送，而無需重新編碼。H.264 預設使用 RTP 負載類型 96，而 H.265 則使用 98。預設 UDP 連接埠規則為 `video_port_base + channel`，以及 `video_port_base = 9000`。如果接收器在容器連接埠重新對應後端執行，請傳遞對應的主機和與之匹配的 `video_port_base`，來自應用程式。

## 原始影格

當傳送到 `VideoSender` 的管線輸入為原始視訊影格時，請使用原始路徑。
Neat 會自動選擇安全的編碼器輸入：

```text
NV12 with a proven compatible boundary:
H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput

Other or unknown raw formats:
VideoConvert -> H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput
```

自動選擇不會新增應用程式選項或變更 `H264RtpUdpFromRaw(...)` API。已驗證的 NV12 格式，無論是在系統或 SiMaAI 記憶體中，都可以直接饋送給 H.264 編碼器，前提是已安裝的編碼器宣告 `input-layout-aware=true`。RGB、BGR、灰階、I420、未知記憶體/佈局，以及沒有可靠格式協定的輸入，都會保留一次轉換為 NV12。

### 原始影格幾何和佈局

`width` 和 `height` 是可見影像的尺寸。它們不需要是 8、16 或 32 的倍數。對於 NV12 和 I420 4:2:0 格式，這兩個尺寸都必須為正數且為偶數；作用中的編碼器、設定檔、層級和硬體會定義剩餘的最小和最大限制。例如，`680x382`、`672x384` 和 `642x480` 是有效形狀，前提是已安裝的編碼器接受這些形狀。

硬體儲存對齊與可見幾何形狀是分開的。Neat 會保留 caps 中的請求尺寸，並將其分配或分階段導入到編碼器表面，以滿足硬體所需的間距和儲存高度。具有自訂物理佈局的原始緩衝區必須攜帶 `GstVideoMeta`，其中包含權威的平面偏移量和步長。如果沒有這些中繼資料，則會使用協商後的 GStreamer 佈局；由屬性驅動的檔案輸入必須包含每個緩衝區中精確一個緊密封裝的影格。無效、截斷或不受支援的佈局會同步失敗，而不是部分複製。

```cpp
simaai::neat::Graph graph;
const int channel = 0;

auto opt = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
    width, height, fps);
opt.host = "127.0.0.1";
opt.channel = channel;
opt.video_port_base = 9000;
opt.encoder.bitrate_kbps = 2500;

graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

Python：

```python
channel = 0

opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
    width=1920,
    height=1080,
    fps=30,
)
opt.host = "127.0.0.1"
opt.channel = channel
opt.video_port_base = 9000
opt.encoder.bitrate_kbps = 2500

graph = pyneat.Graph()
graph.add(pyneat.groups.video_sender(opt))
```

## 編碼的 H.264 或 H.265

對於已編碼的輸入，將串流編解碼器傳遞給傳輸工廠。Neat 會解析、封裝，並在不重新編碼的情況下傳送串流。

| 編解碼器 | C++ 工廠 | Python 工廠 | 預設 RTP 負載類型 |
|---|---|---|---|
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |

MJPEG 傳輸會被拒絕：傳送者沒有 RTP/JPEG 封裝器。

H.265 範例：

```cpp
auto opt = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(
    simaai::neat::nodes::groups::RtspCodec::H265);
opt.host = "127.0.0.1";
opt.channel = 0;
graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

```python
opt = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.H265)
opt.host = "127.0.0.1"
opt.channel = 0
graph.add(pyneat.groups.video_sender(opt))
```

### 將編碼後的 RTSP 訊號分發至推論和預覽

當一個編碼後的 RTSP 來源同時提供解碼/推論和 `VideoSender` 時，請將該來源直接連接到傳送器。對於像 Insight 這樣的即時預覽，請將編碼後的傳送器邊緣設定為 `RealtimeLatestByStream`：

```cpp
simaai::neat::GraphLinkOptions video_link;
video_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;

graph.connect(encoded_source, decoder);
graph.connect(decoder, detector, detector_link);
graph.connect(encoded_source, video_sender, video_link);
```

```python
video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

graph.connect(encoded_source, decoder)
graph.connect(decoder, detector, detector_link)
graph.connect(encoded_source, video_sender, video_link)
```

傳送端分支會在 `SimaDecode` 之前停止，因此它不會重新編碼影片或將已解碼的影格複製到 CPU。透過 `RealtimeLatestByStream`，合併後的傳送端分支最多會保留一個待處理的已編碼存取單元，如果 UDP 輸出速度變慢，則會取代過時的資料。預設的邊緣策略仍然是無損的，並且可以對共享的已編碼來源進行反壓，包括其解碼器分支。僅在保留每個存取單元比保持即時推論的新鮮度更重要時，才使用預設設定。
