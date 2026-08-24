---
title: "ビデオを送信"
description: "VideoSenderのH.264およびH.265 RTP/UDPワイヤフォーマット"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# ビデオの送信

グラフが外部の受信側にビデオを送信する必要がある場合は、`VideoSender` を使用します。`VideoSender` は再利用可能な `Graph` の一部を返します。したがって、`Graph::add(...)` を使用して追加します。

`VideoSender` は、RTP/UDP を介して H.264 または H.265 を送信します。生の入力は H.264 としてエンコードされます。エンコードされた H.264 および H.265 入力は、再エンコードせずに転送されます。H.264 はデフォルトで RTP ペイロードタイプ 96 を使用し、H.265 は 98 を使用します。デフォルトの UDP ポートルールは `video_port_base + channel` で、`video_port_base = 9000` です。受信側がコンテナポートのリマッピングの背後で実行されている場合は、マッピングされたホストと、アプリからの一致する `video_port_base` を渡します。

## 生のフレーム

`VideoSender` へのパイプライン入力が生のビデオフレームである場合は、生のパスを使用します。Neat は、安全なエンコーダーの入力ポートを自動的に選択します。

```text
NV12 with a proven compatible boundary:
H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput

Other or unknown raw formats:
VideoConvert -> H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput
```

自動選択機能は、アプリケーションのオプションを追加したり、`H264RtpUdpFromRaw(...)` API を変更したりすることはありません。システムまたは SiMaAI メモリに格納された実績のある NV12 形式のデータは、インストールされたエンコーダーが `input-layout-aware=true` をサポートしている場合、H.264 エンコーダーに直接入力できます。RGB、BGR、グレースケール、I420、不明なメモリ/レイアウト、および信頼性の高いフォーマット契約がない入力は、NV12 への変換を 1 回だけ行います。

### 生のフレームのジオメトリとレイアウト

`width` と `height` は、表示される画像の寸法です。これらは、8、16、または 32 の倍数である必要はありません。NV12 および I420 4:2:0 形式の場合、両方の寸法は正の値で偶数である必要があります。アクティブなコーデック、プロファイル、レベル、およびハードウェアによって、残りの最小値と最大値が定義されます。たとえば、`680x382`、`672x384`、および `642x480` は、インストールされたエンコーダーがこれらの寸法をサポートする場合に有効な形状です。

ハードウェアストレージのアラインメントは、表示されるジオメトリとは異なります。Neat は、要求された寸法を caps に保存し、ハードウェアに必要なピッチとストレージ高さでエンコーダーのサーフェスに割り当てまたはステージングします。カスタムの物理レイアウトを持つ生のバッファーは、信頼できるプレーンオフセットとストライドを持つ `GstVideoMeta` を含める必要があります。このメタデータがない場合、ネゴシエートされた GStreamer レイアウトが使用されます。プロパティによって制御されるファイル入力には、バッファーごとに正確に 1 つの緊密にパックされたフレームが含まれている必要があります。無効、切り捨てられた、またはサポートされていないレイアウトは、部分的にコピーされるのではなく、同期的にエラーとなります。

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

## エンコードされた H.264 または H.265

エンコードされた入力の場合、ストリームコーデックをパススルーファクトリに渡します。Neat は、ストリームを再エンコードせずに解析、パケット化、送信します。

| コーデック | C++ ファクトリ | Python ファクトリ | デフォルトの RTP ペイロードタイプ |
|---|---|---|---|
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |

MJPEG パススルーは拒否されます：送信側には RTP/JPEG パケット化機能がありません。

H.265 の例：

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

### エンコードされた RTSP を推論およびプレビューに分散する

1 つのエンコードされた RTSP ソースが、デコード/推論と `VideoSender` の両方に供給される場合、ソースを直接送信先に接続します。Insight のようなライブプレビューの場合、エンコードされた送信エッジを `RealtimeLatestByStream` に設定します。

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

送信元ブランチは、`SimaDecode` の前に留まるため、ビデオを再エンコードしたり、デコードされたフレームを CPU にコピーしたりすることはありません。`RealtimeLatestByStream` を使用すると、統合された送信元ブランチは、保留中のエンコードされたアクセスユニットを最大 1 つだけ保持し、UDP 送出が遅延した場合に古いデータを置き換えます。デフォルトのエッジポリシーは、損失のない状態を維持し、そのデコーダーブランチを含む、共有されたエンコードされたソースに対してバックプレッシャーをかけることができます。すべてのアクセスユニットを保持することが、リアルタイム推論を最新の状態に保つことよりも重要な場合にのみ、デフォルトを使用してください。
