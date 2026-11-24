---
title: "USBカメラを使う"
description: "UsbCameraInputでUSB（UVC）WebカメラのMJPEGまたはYUYVフレームを読み取り、JPEGを転送またはデコードするか、YUYVフレームをモデルとH.264ビューアーに渡します。デバイス番号やパイプライン文字列は不要です。"
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# USBカメラを使う

`nodes::UsbCameraInput` は、USB（UVC）Webカメラを読み取ります。既定ではMJPEGをキャプチャーしてカメラのJPEGフレームを出力します。`format` を `UsbCameraFormat::YUYV` に設定すると、[YUYVをキャプチャーする](#capture-yuyv)で説明するように、代わりに非圧縮のYUYVフレームをキャプチャーします。カメラの固定IDを渡します。`UsbCameraInputOptions` は（[MIPIカメラ](./mipi-camera-input.md)の）`CameraInputOptions` のフィールド名と既定値を使いますが、`camera_name` は必須で、`format` は `UsbCameraFormat`（既定値は `MJPEG`）です。

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model

USB camera -> UsbCameraInput (YUYV) -+-> Preproc -> model
                                     `-> ColorConvert (NV12) -> VideoSender (H.264)
```

多くのWebカメラは4:2:2のJPEGを送り、これはNV16にデコードされます。`usb_camera_info()` はカメラのJPEGがデコードされるレイアウトを返します。これを `SimaDecode` の `out_format` に使い、`NV16` が返された場合はモデルの `preprocess.color_convert.input_format` を `PreprocessColorFormat::NV16` に設定すると、フレームは変換段なしでモデルに届きます。

## Neatがサポートする範囲

- Platform 3.0で、Modalix DevKitのUSBポートに接続したUVCカメラ。
- 4:2:0または4:2:2のクロマサンプリングによるMJPEGキャプチャー。
- Platform 3.0.1で、偶数の幅、BT.601またはBT.709の色、フルレンジまたはリミテッドレンジによるYUYVキャプチャー。
- カメラが選択したフォーマット用に提示する正確な幅、高さ、フレームレート。Neatはスケーリングや近いモードの選択を行いません。

H.264専用カメラ、NV12やUYVYなどのその他の非圧縮フォーマット、深度やIRのストリーム、露出などのカメラ制御は、この経路の対象外です。

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

カメラが提供するMJPEGモードとYUYVモードを確認するには、DevKit上でフォーマットの一覧を表示します。デバイス番号はこの確認にだけ使い、NeatはIDからデバイスを見つけます。

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

`decoded_format` も必要な場合は、次の節のように `usb_camera_info` を一度だけ呼び出し、その結果を `UsbCameraInput(camera, info)` に渡してください。ノードはカメラをもう一度開かずにその確認結果を再利用し、デバイスがまだそのカメラのものであることをsysfsで確認するだけです。両方の呼び出しには同じオプションを渡してください。

カメラを抜いたり別のポートに移したりした後は、`UsbCameraInput` をもう一度呼び出してください。以前に作成したグラフはビルド時に失敗するため、古いデバイス番号を引き継いだ別のカメラを開くことはありません。

## カメラのJPEGを転送する

カメラの映像をビューアーに送りつつ、同じ映像でモデルも実行するには、このノードを、JPEGを再エンコードせずにRTP/JPEGとして送る `VideoSender` パススルーと、`out_format` にカメラの `decoded_format` を指定した `SimaDecode` に接続します。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
const neat::UsbCameraInfo info = neat::usb_camera_info(camera);  // opens the camera once

neat::SimaDecodeOptions decode;
decode.type = neat::SimaDecodeType::MJPEG;
decode.out_format = info.decoded_format;  // NV12 or NV16
decode.dec_width = 1920;
decode.dec_height = 1080;
decode.dec_fps = 30;

auto sender = neat::nodes::groups::VideoSenderOptions::Passthrough(
    neat::nodes::groups::RtspCodec::MJPEG);
sender.host = "127.0.0.1";

neat::GraphLinkOptions video_link;
video_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

neat::Graph source;
source.add(neat::nodes::UsbCameraInput(camera, info));  // reuses info, no second open
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
info = pyneat.usb_camera_info(camera)  # opens the camera once

decode = pyneat.SimaDecodeOptions()
decode.type = pyneat.SimaDecodeType.MJPEG
decode.out_format = info.decoded_format  # NV12 or NV16
decode.dec_width = 1920
decode.dec_height = 1080
decode.dec_fps = 30

sender = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
sender.host = "127.0.0.1"

video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

source = pyneat.Graph()
source.add(pyneat.nodes.usb_camera_input(camera, info))  # reuses info, no second open
decoder = pyneat.Graph()
decoder.add(pyneat.nodes.sima_decode(decode))

graph = pyneat.Graph("usb_camera_forward")
graph.connect(source, decoder)
graph.connect(decoder, detector)  # your model Graph
graph.connect(source, pyneat.groups.video_sender(sender), video_link)
```

</CodeTab>
</CodeTabs>

カメラはハードウェアデコーダーがその場で読み取るCMAバッファーにキャプチャーするため、キャプチャーからデコードまでの間にフレームはコピーされません。カーネルのUVCドライバーは、引き続きUSBパケットを各フレームバッファーへコピーします。送信側には[ビデオを送信](./video_sender.md)のRTP/JPEGの制限が適用されます。

## メモリーと破棄されるフレーム

| オプション | 効果 |
| --- | --- |
| `queue_depth`, `leaky_queue` | カメラの後にある小さなleakyキューが、グラフの処理が遅れたときに古いフレームを破棄します。カメラのキャプチャープールは `queue_depth` + 6 個のバッファーを持ち、上限は64個のため、`queue_depth` は最大58です。 |

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
| `is not at '/dev/videoN' as its UsbCameraInfo says` | `usb_camera_info` の呼び出し後に、カメラが抜かれたか、移動されたか、番号が変わったか、またはその情報が別のカメラのものです。`usb_camera_info` をもう一度呼び出し、新しい情報を渡してください。 |
| `has no YUYV mode 640x480 at 30/1 fps; YUYV modes: ...` | 一覧にあるYUYVモードを選んでください。 |
| `has no YUYV capture node` | カメラはYUYVを送りません。MJPEGをキャプチャーしてください。 |
| `YUYV requires an even width` | 幅が偶数のYUYVモードを選んでください。 |
| `adjusted the requested progressive YUYV mode` または `did not accept 30/1 fps` | カメラはこのモードを一覧に示しましたが、受け付けませんでした。一覧にある別のモードを選んでください。 |
| `has unsupported YUYV matrix/range/transfer metadata` または `has unsupported YUYV color primaries` | カメラがBT.601またはBT.709以外の色空間を報告しています。MJPEGをキャプチャーしてください。 |
| `capture info format does not match the selected mode` | `usb_camera_info` が返した `UsbCameraInfo` をそのまま渡してください。 |
| `capture info format does not match opt.format` | 情報は別の `format` でチェックされています。`UsbCameraInput` に渡すオプションで `usb_camera_info` を呼び出してください。 |
| `capture info describes another mode` | オプションが `usb_camera_info` に渡したものと異なります。両方の呼び出しに同じオプションを渡してください。 |
| カメラがストリーミングを開始するとYUYVのRunが失敗する | DevKit上で `dmesg` を実行してください。`Failed to submit URB 0 (-28)`（`ENOSPC`）は、非圧縮モードに必要な帯域幅をUSBコントローラーが確保できないことを示します。より低い解像度やフレームレートを選ぶか、同時にストリーミングしている他のUSBカメラやオーディオデバイスを停止するか、MJPEGをキャプチャーしてください。 |
| `YUYV requires explicit input color` | `PreprocOptions::input_color` に `info.color` を設定してください。 |
| `YUYV input does not support antialiased sampling`、`FP32 transforms require supported RGB/Gray tensor output`、`crop currently requires RGB, BGR or Gray input` または `ROI lists require RGB/BGR/Gray inputs/outputs` | モデルの前処理が、YUYV入力でサポートされないオプションを必要としています。[YUYVフレームでモデルを実行する](#run-a-model-on-yuyv-frames)を参照してください。 |
| `Strict encoder ingress requires EncoderNv12 output from ColorConvert` | 厳格な `VideoSender` の前に、`ImageSurfaceLayout::EncoderNv12` とNV12出力を指定した `ColorConvert` を置いてください。 |
| `Strict encoder ingress requires an exact integer frame rate` | 整数のフレームレートを持つモードをキャプチャーし、エンコーダーの `fps` をその値に設定してください。 |
| `Graph JSON does not store PreprocOptions::input_color or ColorConvertOptions::input_colorimetry` | YUYVをキャプチャーするグラフはコードで構築してください。`Graph::save` はカメラの色を保存できません。 |

## YUYVをキャプチャーする {#capture-yuyv}

多くのUVCカメラは非圧縮のYUYVも送ります。これはパックされた4:2:2フレームで、GStreamerでは `YUY2` と呼ばれます。YUYVフレームにはデコーダーが不要で、モデルの前処理と `ColorConvert` がキャプチャーバッファーからその場で読み取ります。非圧縮フレームはMJPEGよりはるかに多くのUSB帯域幅を必要とするため、カメラは通常、YUYVをより低い解像度やフレームレートで提供します。

`format` を `UsbCameraFormat::YUYV` に設定して `usb_camera_info` を呼び出し、その結果を `UsbCameraInput` に渡します。カメラが一覧に示すYUYVモードのうち、幅が偶数のものを選びます。返される `UsbCameraInfo` はフレームも記述します。`row_stride` はバイト単位の行ピッチ、`storage_bytes` は各キャプチャーバッファーのサイズ、`color` はカメラの行列（BT.601またはBT.709）とレンジ、`colorimetry` は同じ解釈をGStreamerのcolorimetry文字列で表したものです。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
camera.format = neat::UsbCameraFormat::YUYV;
camera.width = 640;
camera.height = 480;
const neat::UsbCameraInfo info = neat::usb_camera_info(camera);  // opens the camera once
auto source = neat::nodes::UsbCameraInput(camera, info);         // reuses info
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera.format = pyneat.UsbCameraFormat.YUYV
camera.width = 640
camera.height = 480
info = pyneat.usb_camera_info(camera)  # opens the camera once
source = pyneat.nodes.usb_camera_input(camera, info)  # reuses info
```

</CodeTab>
</CodeTabs>

ノードは、1画素2バイトのHWCレイアウトの `UInt8` フレームを出力します。フレームは30000/1001のような小数のレートを含めてカメラのフレームレートを保ち、奇数の高さも使えます。既定の `UsbCameraFormat::MJPEG` では、ノードはMJPEGをキャプチャーします。

### YUYVフレームでモデルを実行する {#run-a-model-on-yuyv-frames}

モデルの `preprocess.color_convert.input_format` を `PreprocessColorFormat::YUYV` に設定し、最大入力サイズをカメラのモードに合わせます。YUYVフレームは使用する行列とレンジを示さないため、`input_color` が `info.color` の `Preproc` からモデルの経路を構築し、その後にモデルの推論と後処理を続けます。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Model::Options options;
options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::YUYV;
options.preprocess.input_max_width = 640;
options.preprocess.input_max_height = 480;
neat::Model model("model.tar.gz", options);

neat::PreprocOptions preproc(model);
preproc.input_color = info.color;

neat::Graph graph("usb_camera_yuyv");
graph.add(neat::nodes::UsbCameraInput(camera, info));
graph.add(neat::nodes::Preproc(preproc));
graph.add(model.inference());
graph.add(model.postprocess());
graph.add(neat::nodes::Output());
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
options = pyneat.ModelOptions()
options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.YUYV
options.preprocess.input_max_width = 640
options.preprocess.input_max_height = 480
model = pyneat.Model("model.tar.gz", options)

preproc = pyneat.PreprocOptions(model)
preproc.input_color = info.color

graph = pyneat.Graph("usb_camera_yuyv")
graph.add(pyneat.nodes.usb_camera_input(camera, info))
graph.add(pyneat.nodes.preproc(preproc))
graph.add(model.inference())
graph.add(model.postprocess())
graph.add(pyneat.nodes.output())
```

</CodeTab>
</CodeTabs>

YUYVフレームに対するモデルの前処理は、次をサポートしません。

- アンチエイリアス付きのリサイズサンプリング（`ResizeInterpolation::BilinearAntialiased`）。
- 固定のソース演算：`rescale_factor` とそれが選択する固定の正規化、`quantization_step`、`PreprocFloatingConversion::NearestEven`、`PreprocColorConversionPlacement::BeforeResize`。
- `ResizeMode::Crop`。
- 実行時のROIリスト。

これらのいずれかを必要とする前処理を持つモデルは、グラフの構築時に失敗します。代わりにMJPEGをキャプチャーするか、先に `ColorConvert` でフレームを `ImageFormat::RGB` に変換し（`color` に `info.color`、`input_colorimetry` に `info.colorimetry` を設定します）、モデルの入力形式を `PreprocessColorFormat::RGB` に設定します。

### YUYVフレームをH.264で表示する {#view-yuyv-frames-as-h264}

カメラの映像をビューアーにも送るには、`Preproc` の前で分岐し、ビューアーの分岐を `RealtimeLatestByStream` で接続して、モデルを決して待たせないようにします。`input_colorimetry` が `info.colorimetry` の `ColorConvert` でフレームをエンコーダーのレイアウトのNV12に変換し、`SimaEncodeOptions::input_policy` が `EncoderInputPolicy::RequireDirect` の厳格なH.264の `VideoSender` でエンコードします。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::ColorConvertOptions convert;
convert.output_format = neat::ImageFormat::NV12;
convert.surface = neat::ImageSurfaceLayout::EncoderNv12;
convert.color = info.color;
convert.color.output_siting = {neat::ImageChromaSiting::Cosited,
                               neat::ImageChromaSiting::Centered};
convert.input_colorimetry = info.colorimetry;

neat::SimaEncodeOptions encode;
encode.fps = 30;
encode.input_policy = neat::EncoderInputPolicy::RequireDirect;
auto sender = neat::nodes::groups::VideoSenderOptions::FromRaw(encode);
sender.host = "127.0.0.1";

neat::Graph viewer("usb_camera_yuyv_viewer");
viewer.add(neat::nodes::ColorConvert(convert));
viewer.add(neat::nodes::groups::VideoSender(sender));
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
convert = pyneat.ColorConvertOptions()
convert.output_format = pyneat.ImageFormat.NV12
convert.surface = pyneat.ImageSurfaceLayout.EncoderNv12
convert.color = info.color
convert.color.output_siting = [pyneat.ImageChromaSiting.Cosited, pyneat.ImageChromaSiting.Centered]
convert.input_colorimetry = info.colorimetry

encode = pyneat.SimaEncodeOptions()
encode.fps = 30
encode.input_policy = pyneat.EncoderInputPolicy.RequireDirect
sender = pyneat.VideoSenderOptions.from_raw(encode)
sender.host = "127.0.0.1"

viewer = pyneat.Graph("usb_camera_yuyv_viewer")
viewer.add(pyneat.nodes.color_convert(convert))
viewer.add(pyneat.groups.video_sender(sender))
```

</CodeTab>
</CodeTabs>

厳格な送信側はCPUでフレームを変換もコピーもせず、カメラの原色、伝達特性、行列、レンジをH.264ストリームで伝えます。偶数の幅と高さ、およびエンコーダーの `fps` と等しい整数のフレームレートが必要で、それ以外の入力は丸めずに拒否します。[ビデオを送信](./video_sender.md#strict-encoder-input)を参照してください。

キャプチャーからモデルやエンコーダーまでの間にフレームはコピーされませんが、カーネルのUVCドライバーはUSBパケットを各フレームバッファーにコピーします。`UsbCameraInput` はドライバーが破損とマークしたフレームを破棄します。各フレームのドライバーのバイト数はグラフに届かないため、ドライバーがマークしない不完全なフレームは、バッファーの残りに以前のフレームのデータが残ったまま通過します。
