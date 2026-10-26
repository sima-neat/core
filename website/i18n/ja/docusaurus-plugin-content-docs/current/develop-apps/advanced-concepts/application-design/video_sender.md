---
title: "ビデオを送信"
description: "VideoSender の生入力エンコードと H.264、H.265、MJPEG RTP/UDP 出力"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# ビデオの送信

グラフが外部の受信側にビデオを送信する必要がある場合は、`VideoSender` を使用します。`VideoSender` は再利用可能な `Graph` の一部を返します。したがって、`Graph::add(...)` を使用して追加します。

`VideoSender` は RTP/UDP 経由で H.264、H.265、MJPEG を送信します。`FromRaw` は `SimaEncodeOptions.type` を使って生フレームをエンコードします。デフォルトは H.264 です。`Passthrough` は、エンコード済みのフレームを再エンコードせずに送信します。デフォルトの UDP ポートは `video_port_base + channel` で、`video_port_base = 9000` です。

| コーデック | 生入力のエンコーダー型 | エンコード済み入力のパススルー型 | デフォルトの RTP ペイロードタイプ |
| --- | --- | --- | ---: |
| H.264 | `SimaEncodeType::H264` | `RtspCodec::H264` | 96 |
| H.265 | `SimaEncodeType::H265` | `RtspCodec::H265` | 98 |
| MJPEG | `SimaEncodeType::MJPEG` | `RtspCodec::MJPEG` | 26 |

Insight の RTP/WebRTC ビューアーは H.264 と H.265 をサポートします。MJPEG 出力には互換性のある RTP/JPEG 受信側が必要です。このビューアーは MJPEG をサポートしません。

受信側がコンテナのポート再割り当てを使用する場合は、アプリから割り当て後のホストと対応する `video_port_base` を渡してください。

## 生のフレーム

`VideoSender` へのパイプライン入力が生のビデオフレームである場合は、生のパスを使用します。Neat は、安全なエンコーダーの入力ポートを自動的に選択します。

```text
NV12 in a compatible DMA-BUF:
SimaEncode -> codec parser -> RTP payloader -> UdpOutput

CPU input or raw frames requiring conversion:
Convert/upload into encoder DMA-BUF -> SimaEncode -> codec parser -> RTP payloader -> UdpOutput
```

コーデックの選択により、エンコーダー、パーサー、RTP ペイローダーがまとめて選択されます。非推奨の `H264RtpUdpFromRaw(...)` ファクトリーは引き続き利用でき、H.264 のデフォルトを保持します。互換性のある NV12 DMA-BUF 入力は、元の割り当てを保持します。CPU 上の NV12 はアップロードが必要です。RGB、BGR、グレースケール、I420 は変換が必要です。Neat はエンコーダー入力境界でこの処理を行い、最終的な DMA サーフェスに書き込みます。エンコーダー内部で 2 回目のコピーを行うことはありません。アプリケーションでメモリバックエンドを選択する必要はありません。

### 生のフレームのジオメトリとレイアウト

`width` と `height` は、表示される画像の寸法です。これらは、8、16、または 32 の倍数である必要はありません。NV12 および I420 4:2:0 形式の場合、両方の寸法は正の値で偶数である必要があります。アクティブなコーデック、プロファイル、レベル、およびハードウェアによって、残りの最小値と最大値が定義されます。たとえば、`680x382`、`672x384`、および `642x480` は、インストールされたエンコーダーがこれらの寸法をサポートする場合に有効な形状です。

RTP/JPEG の制限はより厳密です。MJPEG の生入力送信には、各軸が 8 の倍数で、8..2040 の範囲内の寸法が必要です。エンコード済み JPEG のパススルーには、標準ハフマンテーブルを使用するベースライン JPEG と、RTP/JPEG で表現可能な寸法が必要です。単独のエンコードと RTP 転送ではサイズ制限が異なります。

ハードウェアストレージのアラインメントは、表示されるジオメトリとは異なります。Neat は、要求された寸法を caps に保存し、ハードウェアに必要なピッチとストレージ高さでエンコーダーのサーフェスを生成します。カスタムの物理レイアウトを持つ生のバッファーは、信頼できるプレーンオフセットとストライドを持つ `GstVideoMeta` を含める必要があります。このメタデータがない場合、ネゴシエートされた GStreamer レイアウトが使用されます。プロパティによって制御されるファイル入力には、バッファーごとに正確に 1 つの緊密にパックされたフレームが含まれている必要があります。無効、切り捨てられた、またはサポートされていないレイアウトは、部分的にコピーされるのではなく、同期的にエラーとなります。

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

生入力の送信は、入力フレームから解像度を検出し、リサイズせずに保持します。異なる解像度の生フレームを送信する場合は、新しい実行を開始してください。従来の `H264RtpUdpFromRaw(width, height, fps)` ファクトリーは、固定の入力寸法を保持します。

MJPEG では、C++ の `SimaEncodeType::MJPEG` または Python の `pyneat.SimaEncodeType.MJPEG` を選択し、`quality` を 1 から 100 に設定します。ビットレート、レート制御、プロファイル、レベル、GOP、IDR 設定は未設定にしてください。`FromRaw` は渡されたオプションをコピーするため、ファクトリーを呼ぶ前に設定します。

既存の `opt.encoder` のビットレート／プロファイル／レベルの上書きは、生入力の H.264/H.265 送信で引き続き有効です。`sync=true` はタイムスタンプに合わせて送信をスケジュールします。デフォルトの `false` はクロック待機なしで送信します。`async=true` は、UDP シンクの起動時に最初のバッファーを待てるようにします。デフォルトは `false` です。Python では `async_` として公開されています。

## エンコード済みフレーム

エンコードされた入力の場合、ストリームコーデックをパススルーファクトリに渡します。Neat は、ストリームを再エンコードせずに解析、パケット化、送信します。

| コーデック | C++ ファクトリ | Python ファクトリ | デフォルトの RTP ペイロードタイプ |
|---|---|---|---|
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |
| MJPEG | `Passthrough(RtspCodec::MJPEG)` | `passthrough(pyneat.RtspCodec.MJPEG)` | 26 |

パススルーはエンコーダーを作成しません。呼び出し側のコーデック選択は、エンコード済み入力に一致する必要があります。エンコーダー設定は適用されません。

RTSP ソースがこの送信側と `SimaDecode` の両方に供給される場合、デフォルトの `async=false` では Core が送信側をソースパイプライン内の GStreamer `tee` の後段に配置するため、ソース、送信側、デコーダーは一緒に開始、停止、失敗します。別々のパイプラインに保つには `async=true`（Python では `async_`）を設定します。

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
