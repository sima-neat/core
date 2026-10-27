---
title: "USBカメラを使う"
description: "UsbCameraInputでUSB（UVC）WebカメラのMJPEGフレームを読み取り、JPEGのまま転送するか、モデル用にデコードします。デバイス番号やパイプライン文字列は不要です。"
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# USBカメラを使う

`nodes::UsbCameraInput` は、MJPEGを送るUSB（UVC）Webカメラを読み取り、カメラのJPEGフレームを出力します。カメラの固定IDを渡します。`UsbCameraInputOptions` は（[MIPIカメラ](./mipi-camera-input.md)の）`CameraInputOptions` のフィールド名と既定値を使いますが、`camera_name` は必須で、`format` はありません。

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model
```

多くのWebカメラは4:2:2のJPEGを送り、これはNV16にデコードされます。`usb_camera_info()` はカメラのJPEGがデコードされるレイアウトを返します。これを `SimaDecode` の `out_format` に使い、`NV16` が返された場合はモデルの `preprocess.color_convert.input_format` を `PreprocessColorFormat::NV16` に設定すると、フレームは変換段なしでモデルに届きます。

## Neatがサポートする範囲

- Platform 3.0で、Modalix DevKitのUSBポートに接続したUVCカメラ。
- 4:2:0または4:2:2のクロマサンプリングによるMJPEGキャプチャー。
- カメラがMJPEG用に提示する正確な幅、高さ、フレームレート。Neatはスケーリングや近いモードの選択を行いません。

H.264専用カメラ、非圧縮のYUYV/NV12キャプチャー、深度やIRのストリーム（RealSenseの深度など）、露出などのカメラ制御は、この経路の対象外です。

## カメラIDを調べる

各USBカメラには、USBポートとベンダーID、製品IDから作られる固定IDがあり、形式はlibcameraと同じです。

```text
/base/pci@7000000000-3.1:1.0-046d:08e5
```

IDは再起動や接続順序の変化では変わりません。カメラを別のUSBポートに移すと変わるため、異なるポートに接続した同じカメラ2台は別々のIDになります。接続されているカメラを一覧表示します。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
for (const std::string& id : simaai::neat::usb_camera_ids()) {
  std::cout << id << "\n";
}
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

print(pyneat.usb_camera_ids())
```

</CodeTab>
</CodeTabs>

カメラが提供するMJPEGモードを確認するには、DevKit上でフォーマットの一覧を表示します。デバイス番号はこの確認にだけ使い、NeatはIDからデバイスを見つけます。

<ShellCommand prompt="devkit">
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video1 --list-formats-ext
</ShellCommand>

## グラフを作成する

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::UsbCameraInputOptions camera;
camera.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5";
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.buffer_name = "camera0";

neat::Graph graph("usb_camera");
graph.add(neat::nodes::UsbCameraInput(camera));
graph.add(neat::nodes::Output());

neat::Run run = graph.build();
std::optional<neat::Sample> frame = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.UsbCameraInputOptions()
camera.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5"
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.buffer_name = "camera0"

graph = pyneat.Graph("usb_camera")
graph.add(pyneat.nodes.usb_camera_input(camera))
graph.add(pyneat.nodes.output())

run = graph.build()
frame = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

ノードはカメラのJPEGフレームを出力します。モデルを実行するには、次の節のように `SimaDecode` でデコードします。

`UsbCameraInput` はノードの作成時に、モードを確認して最初のフレームのクロマサンプリングを読み取るため、カメラを一度開きます。先に、カメラを使っている他のプログラムを終了してください。`usb_camera_info(camera)` はノードを作成せずに同じ確認を行い、キャプチャーデバイスと `decoded_format`（`NV12` または `NV16`）を返します。

カメラを抜いたり別のポートに移したりした後は、`UsbCameraInput` をもう一度呼び出してください。以前に作成したグラフはビルド時に失敗するため、古いデバイス番号を引き継いだ別のカメラを開くことはありません。

## カメラのJPEGを転送する

カメラの映像をビューアーに送りつつ、同じ映像でモデルも実行するには、このノードを、JPEGを再エンコードせずにRTP/JPEGとして送る `VideoSender` パススルーと、`out_format` にカメラの `decoded_format` を指定した `SimaDecode` に接続します。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::SimaDecodeOptions decode;
decode.type = neat::SimaDecodeType::MJPEG;
decode.out_format = neat::usb_camera_info(camera).decoded_format;  // NV12 or NV16
decode.dec_width = 1920;
decode.dec_height = 1080;
decode.dec_fps = 30;

auto sender = neat::nodes::groups::VideoSenderOptions::Passthrough(
    neat::nodes::groups::RtspCodec::MJPEG);
sender.host = "127.0.0.1";

neat::GraphLinkOptions video_link;
video_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

neat::Graph source;
source.add(neat::nodes::UsbCameraInput(camera));
neat::Graph decoder;
decoder.add(neat::nodes::SimaDecode(decode));

neat::Graph graph("usb_camera_forward");
graph.connect(source, decoder);
graph.connect(decoder, detector);  // your model Graph
graph.connect(source, neat::nodes::groups::VideoSender(sender), video_link);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
decode = pyneat.SimaDecodeOptions()
decode.type = pyneat.SimaDecodeType.MJPEG
decode.out_format = pyneat.usb_camera_info(camera).decoded_format  # NV12 or NV16
decode.dec_width = 1920
decode.dec_height = 1080
decode.dec_fps = 30

sender = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
sender.host = "127.0.0.1"

video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

source = pyneat.Graph()
source.add(pyneat.nodes.usb_camera_input(camera))
decoder = pyneat.Graph()
decoder.add(pyneat.nodes.sima_decode(decode))

graph = pyneat.Graph("usb_camera_forward")
graph.connect(source, decoder)
graph.connect(decoder, detector)  # your model Graph
graph.connect(source, pyneat.groups.video_sender(sender), video_link)
```

</CodeTab>
</CodeTabs>

デコーダーは、各JPEGをデコードする前に自身の入力バッファーへコピーします。送信側には[ビデオを送信](./video_sender.md)のRTP/JPEGの制限が適用されます。

## メモリーと破棄されるフレーム

| オプション | 効果 |
| --- | --- |
| `queue_depth`, `leaky_queue` | カメラの後にある小さなleakyキューが、グラフの処理が遅れたときに古いフレームを破棄します。 |

Webカメラは、JPEGの終了マーカーの後にバイトを付け足すことがよくあり、一部のフレームは不完全な状態で届いたり、USBドライバーによって破損とマークされたりします。`UsbCameraInput` はドライバーが破損とマークしたフレームを出力前に破棄するため、`VideoSender` のパススルーにもデコーダーにも届きません。デコーダーは各フレームを終了マーカーまでデコードし、不完全なフレームや破損したフレームは停止せずに破棄します。デコーダー要素は、それらを `dropped-count` プロパティで数えます。

## トラブルシューティング

| メッセージ | 対処 |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | 一覧にあるIDのいずれかを使ってください。カメラを別のポートに移すとIDが変わります。 |
| `no USB cameras are connected` | ケーブルと `lsusb` を確認してください。 |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | 一覧にあるモードを選んでください。高解像度を低いフレームレートでしか提供しないカメラもあります。 |
| `is in use by another process` | カメラを開いている他のプログラムを終了してください。 |
| `sent no usable MJPEG frame` | NeatがデコードできないJPEG（4:4:4サンプリングなど）をカメラが送っているか、フレームがまったく届いていません。 |
| `is no longer /dev/videoN` | グラフの作成後に、カメラが抜かれたか、移動されたか、番号が変わりました。`UsbCameraInput` をもう一度呼び出してください。 |
