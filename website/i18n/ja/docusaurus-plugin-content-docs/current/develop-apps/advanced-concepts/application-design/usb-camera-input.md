---
title: "USBカメラを使う"
description: "UsbCameraInputでUSB（UVC）WebカメラのMJPEGフレームを読み取り、ハードウェアデコーダーでその場でNV12にデコードします。デバイス番号やパイプライン文字列は不要です。"
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# USBカメラを使う

`groups::UsbCameraInput` は、MJPEGを送るUSB（UVC）Webカメラを読み取ります。カメラの固定IDを渡すと、[MIPIカメラ](./mipi-camera-input.md)と同様に、グラフはSiMaAIメモリー上のNV12フレームを受け取ります。`UsbCameraInputOptions` は `CameraInputOptions` のフィールド名と既定値を使いますが、`camera_name` は必須で、`format` はありません。出力は常にNV12です。

```text
USB camera -> UsbCameraInput -> model-managed CVU preproc -> MLA -> Output
```

内部では、カメラは圧縮された各フレームを、ハードウェアMJPEGデコーダーがその場で読み取るCMAバッファーに直接書き込むため、キャプチャーとデコードの間でCPUによるコピーは発生しません。多くのWebカメラのように4:2:2のJPEGを送るカメラでは、デコーダーの後にNV16からNV12へのEV74変換が1つ追加されます。

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
graph.add(neat::nodes::groups::UsbCameraInput(camera));
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
graph.add(pyneat.groups.usb_camera_input(camera))
graph.add(pyneat.nodes.output())

run = graph.build()
frame = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

モデルはMIPIカメラと同じ方法で追加します。[MIPIカメラを使う](./mipi-camera-input.md)の、生のMLAグラフを作成する節を参照してください。

`UsbCameraInput` はグラフの作成時に、モードを確認して最初のフレームのクロマサンプリングを読み取るため、カメラを一度開きます。先に、カメラを使っている他のプログラムを終了してください。

カメラを抜いたり別のポートに移したりした後は、`UsbCameraInput` をもう一度呼び出してください。以前に作成したグラフはビルド時に失敗するため、古いデバイス番号を引き継いだ別のカメラを開くことはありません。

## メモリーと破棄されるフレーム

| オプション | 効果 |
| --- | --- |
| `allow_cpu_fallback` | USBカメラでは効果がありません。キャプチャーは常にデコーダー自身のバッファーに書き込まれるため、コピーされるフレームはありません。オプションを `CameraInputOptions` とそろえるために残しています。 |
| `queue_depth`, `leaky_queue` | デコーダーの前にある小さなleakyキューが、グラフの処理が遅れたときに古いフレームを破棄します。 |
| `capture_buffer_count`（2番目の引数） | カメラとデコーダーが共有するCMAバッファーです。0の場合は、キューの深さにデコーダーの入力スロット2つを加えた数を使います。 |

ここでのゼロコピーとは、キャプチャーとデコードの間で圧縮フレームがコピーされないことを指します。デコード済みのNV12フレームは別のバッファーで、4:2:2向けのEV74変換はさらに別のバッファーに書き込みます。

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
