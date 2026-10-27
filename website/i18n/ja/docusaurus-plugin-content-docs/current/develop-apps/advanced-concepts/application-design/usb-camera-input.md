---
title: "USBカメラを使う"
description: "UsbCameraInputでUSB（UVC）WebカメラのMJPEGフレームを読み取り、ハードウェアデコーダーでその場でデコードするか、JPEGのまま転送します。デバイス番号やパイプライン文字列は不要です。"
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# USBカメラを使う

`nodes::UsbCameraInput` は、MJPEGを送るUSB（UVC）Webカメラを読み取ります。カメラの固定IDを渡すと、[MIPIカメラ](./mipi-camera-input.md)と同様に、グラフはSiMaAIメモリー上のフレームを受け取ります。`UsbCameraInputOptions` は `CameraInputOptions` のフィールド名と既定値を使いますが、`camera_name` は必須で、`format` はありません。フレームはカメラ本来のレイアウトのままで、4:2:0のJPEGはNV12、4:2:2のJPEGはNV16になります。

```text
USB camera -> UsbCameraInput -> model-managed CVU preproc -> MLA -> Output
```

内部では、カメラは圧縮された各フレームを、ハードウェアMJPEGデコーダーがその場で読み取るCMAバッファーに直接書き込むため、キャプチャーとデコードの間でCPUによるコピーは発生しません。多くのWebカメラは4:2:2のJPEGを送り、これはNV16にデコードされます。モデルのCVU前処理はNV16を直接読み取ります。別の処理のためにNV12が必要なアプリは、カメラの後に自分で `ColorConvert` を追加できます。

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

モデルはMIPIカメラと同じ方法で追加します。[MIPIカメラを使う](./mipi-camera-input.md)の、生のMLAグラフを作成する節を参照してください。

`UsbCameraInput` はノードの作成時に、モードを確認して最初のフレームのクロマサンプリングを読み取るため、カメラを一度開きます。先に、カメラを使っている他のプログラムを終了してください。`usb_camera_info(camera)` はノードを作成せずに同じ確認を行い、キャプチャーデバイスと `decoded_format`（`NV12` または `NV16`）を返します。

カメラを抜いたり別のポートに移したりした後は、`UsbCameraInput` をもう一度呼び出してください。以前に作成したグラフはビルド時に失敗するため、古いデバイス番号を引き継いだ別のカメラを開くことはありません。

## カメラのJPEGを転送する

カメラの映像をデコードせずにビューアーへ送るには、`output` を `UsbCameraOutput::Jpeg` に設定します。ノードはカメラのJPEGフレームを出力します。このノードを、JPEGを再エンコードせずにRTP/JPEGとして送る `VideoSender` パススルーに接続します。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
camera.output = neat::UsbCameraOutput::Jpeg;

auto sender = neat::nodes::groups::VideoSenderOptions::Passthrough(
    neat::nodes::groups::RtspCodec::MJPEG);
sender.host = "127.0.0.1";

neat::Graph graph("usb_camera_forward");
graph.add(neat::nodes::UsbCameraInput(camera));
graph.add(neat::nodes::groups::VideoSender(sender));
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera.output = pyneat.UsbCameraOutput.Jpeg

sender = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
sender.host = "127.0.0.1"

graph = pyneat.Graph("usb_camera_forward")
graph.add(pyneat.nodes.usb_camera_input(camera))
graph.add(pyneat.groups.video_sender(sender))
```

</CodeTab>
</CodeTabs>

このモードでは、フレームはカメラドライバーのメモリーに置かれたままです。`SimaDecode` はまだこのメモリーを読み取れず、デコーダーは最初のフレームで `Failed to map input buffer` を出して停止します。モデルを実行するには `UsbCameraOutput::Frames` を使います。送信側には[ビデオを送信](./video_sender.md)のRTP/JPEGの制限が適用されます。

## メモリーと破棄されるフレーム

| オプション | 効果 |
| --- | --- |
| `allow_cpu_fallback` | USBカメラでは効果がありません。キャプチャーは常にデコーダー自身のバッファーに書き込まれるため、コピーされるフレームはありません。オプションを `CameraInputOptions` とそろえるために残しています。 |
| `queue_depth`, `leaky_queue` | デコーダーの前にある小さなleakyキューが、グラフの処理が遅れたときに古いフレームを破棄します。 |
| `capture_buffer_count`（2番目の引数） | カメラとデコーダーが共有するCMAバッファーです。0の場合は、キューの深さにデコーダーの入力スロット2つを加えた数を使います。フレーム出力専用で、`UsbCameraOutput::Jpeg` では0にする必要があります。 |

ここでのゼロコピーとは、キャプチャーとデコードの間で圧縮フレームがコピーされないことを指します。デコード済みのフレームは別のバッファーです。

Webカメラは、JPEGの終了マーカーの後にバイトを付け足すことがよくあり、一部のフレームは不完全な状態で届いたり、USBドライバーによって破損とマークされたりします。デコーダーは各フレームを終了マーカーまでデコードし、不完全なフレームや破損したフレームは停止せずに破棄します。デコーダー要素は、それらを `dropped-count` プロパティで数えます。

## トラブルシューティング

| メッセージ | 対処 |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | 一覧にあるIDのいずれかを使ってください。カメラを別のポートに移すとIDが変わります。 |
| `no USB cameras are connected` | ケーブルと `lsusb` を確認してください。 |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | 一覧にあるモードを選んでください。高解像度を低いフレームレートでしか提供しないカメラもあります。 |
| `is in use by another process` | カメラを開いている他のプログラムを終了してください。 |
| `sent no usable MJPEG frame` | NeatがデコードできないJPEG（4:4:4サンプリングなど）をカメラが送っているか、フレームがまったく届いていません。 |
| `is no longer /dev/videoN` | グラフの作成後に、カメラが抜かれたか、移動されたか、番号が変わりました。`UsbCameraInput` をもう一度呼び出してください。 |
| `requires a neatdecoder with the input-pool-buffers property` | インストールされているデコーダーが古すぎます。`simaai-scheduler` パッケージを更新してください。 |
