---
title: "傳送視訊"
description: "VideoSender 原始影格編碼及 H.264、H.265、MJPEG RTP/UDP 輸出"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# 傳送視訊

當 Graph 需要將視訊傳送至外部接收器時，請使用 `VideoSender`。`VideoSender` 會傳回可重複使用的 `Graph` 片段，因此請以 `Graph::add(...)` 加入。

`VideoSender` 透過 RTP/UDP 傳送 H.264、H.265 或 MJPEG。`FromRaw` 使用 `SimaEncodeOptions.type` 編碼原始影格，預設為 H.264。`Passthrough` 傳送已編碼影格，無需重新編碼。預設 UDP 連接埠為 `video_port_base + channel`，其中 `video_port_base = 9000`。

| 編解碼器 | 原始編碼器型別 | 已編碼直通型別 | 預設 RTP 負載類型 |
| --- | --- | --- | ---: |
| H.264 | `SimaEncodeType::H264` | `RtspCodec::H264` | 96 |
| H.265 | `SimaEncodeType::H265` | `RtspCodec::H265` | 98 |
| MJPEG | `SimaEncodeType::MJPEG` | `RtspCodec::MJPEG` | 26 |

Insight 的 RTP/WebRTC 檢視器支援 H.264 與 H.265。MJPEG 輸出需要相容的 RTP/JPEG 接收器；該檢視器不支援此格式。

若接收器在容器連接埠重新對應之後執行，請從應用程式傳入對應的主機與相符的 `video_port_base`。

## 原始影格

當 `VideoSender` 的管線輸入為原始視訊影格時，請使用原始路徑。
Neat 會自動選擇安全的編碼器輸入：

```text
NV12 in a compatible DMA-BUF:
SimaEncode -> codec parser -> RTP payloader -> UdpOutput

CPU input or raw frames requiring conversion:
Convert/upload into encoder DMA-BUF -> SimaEncode -> codec parser -> RTP payloader -> UdpOutput
```

編解碼器會一併選擇編碼器、解析器與 RTP 負載封裝器。已棄用的
`H264RtpUdpFromRaw(...)` 工廠仍可使用，並保留 H.264 預設值。相容的 NV12 DMA-BUF 輸入會保留其底層配置。CPU 儲存的 NV12 需要上傳；RGB、BGR、灰階與 I420 需要轉換。Neat 會在編碼器輸入邊界完成此工作，直接寫入最終 DMA 表面，而不在編碼器內另外暫存一份複本。應用程式不需要選擇記憶體後端。

### 原始影格幾何與佈局

`width` 與 `height` 是可見影像尺寸，不必是 8、16 或 32 的倍數。對 NV12 與 I420 4:2:0 格式，兩者都必須為正偶數；使用中的編解碼器、設定檔、層級與硬體會決定其餘最小與最大限制。例如，已安裝的編碼器接受時，`680x382`、`672x384` 和 `642x480` 都是有效形狀。

RTP/JPEG 的限制更嚴格：MJPEG 原始傳送的尺寸必須可被八整除，且每個軸介於 8..2040。已編碼 JPEG 直通需要使用標準 Huffman 表的基準 JPEG，且尺寸必須能以 RTP/JPEG 表示。獨立編碼與 RTP 傳輸有各自的尺寸限制。

硬體儲存對齊與可見幾何分開處理。Neat 在 caps 中保留要求的尺寸，並產生符合硬體所需 pitch 與儲存高度的編碼器表面。使用自訂實體佈局的原始緩衝區必須攜帶 `GstVideoMeta`，提供具決定權的平面偏移與步長。缺少此中繼資料時，會使用協商的 GStreamer 佈局；由屬性驅動的檔案輸入，每個緩衝區必須恰好包含一個緊密排列的影格。無效、截斷或不受支援的佈局會同步失敗，不會只複製部分資料。

```cpp
simaai::neat::Graph graph;
const int channel = 0;

simaai::neat::SimaEncodeOptions encode;
encode.type = simaai::neat::SimaEncodeType::H265;
encode.fps = 30;
encode.bitrate_kbps = 2500;
encode.gop_length = 30;
auto opt = simaai::neat::nodes::groups::VideoSenderOptions::FromRaw(encode);
opt.host = "127.0.0.1";
opt.channel = channel;
opt.video_port_base = 9000;

graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

Python：

```python
channel = 0

encode = pyneat.SimaEncodeOptions()
encode.type = pyneat.SimaEncodeType.H265
encode.fps = 30
encode.bitrate_kbps = 2500
encode.gop_length = 30
opt = pyneat.VideoSenderOptions.from_raw(encode)
opt.host = "127.0.0.1"
opt.channel = channel
opt.video_port_base = 9000

graph = pyneat.Graph()
graph.add(pyneat.groups.video_sender(opt))
```

原始傳送會從輸入影格偵測解析度，並保留該解析度而不縮放。
要以不同解析度傳送原始影格，請開始新的執行。
舊版 `H264RtpUdpFromRaw(width, height, fps)` 工廠會保留固定輸入尺寸。

使用 MJPEG 時，請在 C++ 選擇 `SimaEncodeType::MJPEG`，或在 Python 選擇
`pyneat.SimaEncodeType.MJPEG`，並將 `quality` 設為 1 至 100。位元率、速率控制、設定檔、層級、GOP 與 IDR 設定請維持未設定。`FromRaw` 會複製提供的選項，因此請先完成設定再呼叫工廠。

既有的 `opt.encoder` 位元率／設定檔／層級覆寫仍適用於原始 H.264/H.265 傳送端。`sync=true` 依時間戳記安排傳送；預設的 `false` 不等待該時脈。`async=true` 允許 UDP 輸出端在啟動時等待第一個緩衝區；預設為 `false`。Python 中此選項名為 `async_`。

## 已編碼影格

對已編碼輸入，請將串流編解碼器傳入直通工廠。Neat 會解析、封裝及傳送串流，無需重新編碼。

| 編解碼器 | C++ 工廠 | Python 工廠 | 預設 RTP 負載類型 |
| --- | --- | --- | ---: |
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |
| MJPEG | `Passthrough(RtspCodec::MJPEG)` | `passthrough(pyneat.RtspCodec.MJPEG)` | 26 |

直通不建立編碼器。呼叫端選擇的編解碼器必須符合已編碼輸入；編碼器設定不適用。

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
