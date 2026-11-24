---
title: "使用 USB 攝影機"
description: "使用 UsbCameraInput 從 USB（UVC）網路攝影機讀取 MJPEG 或 YUYV 影格，轉送或解碼 JPEG，或將 YUYV 影格送入模型與 H.264 檢視器，不需要裝置編號或管線字串。"
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# 使用 USB 攝影機

`nodes::UsbCameraInput` 會讀取 USB（UVC）網路攝影機。預設會擷取 MJPEG 並輸出攝影機的 JPEG 影格；將 `format` 設為 `UsbCameraFormat::YUYV` 則改為擷取未壓縮的 YUYV 影格，詳見[擷取 YUYV](#capture-yuyv)。請傳入攝影機的穩定 ID。`UsbCameraInputOptions` 沿用（[MIPI 攝影機](./mipi-camera-input.md)的）`CameraInputOptions` 欄位名稱與預設值，但 `camera_name` 為必填，且 `format` 為 `UsbCameraFormat`，預設為 `MJPEG`。

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model

USB camera -> UsbCameraInput (YUYV) -+-> Preproc -> model
                                     `-> ColorConvert (NV12) -> VideoSender (H.264)
```

大多數網路攝影機傳送 4:2:2 JPEG，會解碼為 NV16。`usb_camera_info()` 會回報攝影機 JPEG 解碼後的排列方式：請將它作為 `SimaDecode` 的 `out_format`，並在回報 `NV16` 時將模型的 `preprocess.color_convert.input_format` 設為 `PreprocessColorFormat::NV16`，影格便會不經轉換階段直接送達模型。

## Neat 支援的範圍

- 在 Platform 3.0 上，連接到 Modalix DevKit USB 連接埠的 UVC 攝影機。
- 4:2:0 或 4:2:2 色度取樣的 MJPEG 擷取。
- 在 Platform 3.0.1 上，寬度為偶數、BT.601 或 BT.709 色彩，以及完整或有限範圍的 YUYV 擷取。
- 攝影機為所選格式列出的確切寬度、高度與影格率。Neat 不會縮放，也不會挑選相近的模式。

僅支援 H.264 的攝影機、NV12 或 UYVY 等其他原始格式、深度或紅外線串流，以及曝光等攝影機控制，都不在此路徑的範圍內。

## 找出攝影機 ID

每台 USB 攝影機都有一個由其 USB 連接埠以及廠商 ID 與產品 ID 組成的穩定 ID，格式與 libcamera 相同：

```text
/base/pci@7000000000-3.1:1.0-046d:08e5
```

ID 在重新開機或改變插入順序後都不會改變。將攝影機移到另一個 USB 連接埠時 ID 會改變，因此插在不同連接埠的兩台相同攝影機會有不同的 ID。列出已連接的攝影機：

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

若要查看攝影機提供哪些 MJPEG 與 YUYV 模式，請在 DevKit 上列出其格式。裝置編號只用於這項檢查；Neat 會依 ID 找到裝置。

<ShellCommand prompt="devkit">
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video1 --list-formats-ext
</ShellCommand>

## 建立圖

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

節點會輸出攝影機的 JPEG 影格。若要在其上執行模型，請依下一節所示以 `SimaDecode` 解碼。

`UsbCameraInput` 會在建立節點時開啟攝影機一次，以檢查模式並讀取第一個影格的色度取樣。請先關閉其他正在使用攝影機的程式。`usb_camera_info(camera)` 會執行相同的檢查但不建立節點，並傳回擷取裝置與 `decoded_format`（`NV12` 或 `NV16`）。

若也需要 `decoded_format`，請依下一節所示呼叫 `usb_camera_info` 一次，並將結果傳給 `UsbCameraInput(camera, info)`。節點會重複使用該檢查結果而不再次開啟攝影機，只透過 sysfs 確認裝置仍屬於該攝影機。請對兩次呼叫傳入相同的選項。

攝影機拔除或移到另一個連接埠後，請再次呼叫 `UsbCameraInput`。先前建立的圖在建置時會失敗，因此絕不會開啟接手舊裝置編號的另一台攝影機。

## 轉送攝影機的 JPEG

若要將攝影機的視訊傳送給檢視器，同時在其上執行模型，請將節點連接到 `VideoSender` 直通模式（不重新編碼，直接以 RTP/JPEG 傳送 JPEG），以及 `out_format` 為攝影機 `decoded_format` 的 `SimaDecode`：

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

攝影機會擷取到硬體解碼器就地讀取的 CMA 緩衝區，因此從擷取到解碼之間不會複製任何影格。核心的 UVC 驅動程式仍會把 USB 封包複製到每個影格緩衝區。傳送端適用 [傳送視訊](./video_sender.md) 中的 RTP/JPEG 限制。

## 記憶體與捨棄的影格

| 選項 | 效果 |
| --- | --- |
| `queue_depth`, `leaky_queue` | 攝影機後方的小型 leaky 佇列會在圖處理落後時捨棄過時的影格。攝影機的擷取緩衝池包含 `queue_depth` + 6 個緩衝區，最多 64 個，因此 `queue_depth` 最多為 58。 |

網路攝影機經常在 JPEG 結束標記之後補上位元組，有些影格則會不完整地抵達，或被 USB 驅動程式標示為損毀。`UsbCameraInput` 會在輸出前捨棄驅動程式標示為損毀的影格，因此 `VideoSender` 直通模式與解碼器都不會收到這些影格。解碼器會把每個影格解碼到其結束標記為止，並在不停止的情況下捨棄不完整或損毀的影格。解碼器元件會在其 `dropped-count` 屬性中計數。

## 疑難排解

| 訊息 | 處理方式 |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | 使用列出的其中一個 ID。攝影機移到另一個連接埠時 ID 會改變。 |
| `no USB cameras are connected` | 檢查纜線與 `lsusb`。 |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | 選擇列出的模式。有些攝影機只在較低的影格率下提供高解析度。 |
| `is in use by another process` | 停止目前開啟攝影機的其他程式。 |
| `sent no usable MJPEG frame` | 攝影機傳送 Neat 無法解碼的 JPEG（例如 4:4:4 取樣），或完全沒有傳送影格。 |
| `is no longer /dev/videoN` | 建立圖之後，攝影機被拔除、移動或重新編號。請再次呼叫 `UsbCameraInput`。 |
| `is not at '/dev/videoN' as its UsbCameraInfo says` | 呼叫 `usb_camera_info` 之後，攝影機被拔除、移動或重新編號，或該資訊屬於另一台攝影機。請再次呼叫 `usb_camera_info` 並傳入新的資訊。 |
| `has no YUYV mode 640x480 at 30/1 fps; YUYV modes: ...` | 選擇列出的 YUYV 模式。 |
| `has no YUYV capture node` | 攝影機不傳送 YUYV。請擷取 MJPEG。 |
| `YUYV requires an even width` | 選擇寬度為偶數的 YUYV 模式。 |
| `adjusted the requested progressive YUYV mode` 或 `did not accept 30/1 fps` | 攝影機列出了此模式，但不接受它。請選擇另一個列出的模式。 |
| `has unsupported YUYV matrix/range/transfer metadata` 或 `has unsupported YUYV color primaries` | 攝影機回報的色彩空間不是 BT.601 或 BT.709。請擷取 MJPEG。 |
| `capture info format does not match the selected mode` | 請傳入 `usb_camera_info` 傳回的完整 `UsbCameraInfo`。 |
| `capture info format does not match opt.format` | 該資訊是以另一個 `format` 檢查的。請以傳給 `UsbCameraInput` 的選項呼叫 `usb_camera_info`。 |
| `capture info describes another mode` | 選項與傳給 `usb_camera_info` 的選項不同。請在兩次呼叫中傳入相同的選項。 |
| 攝影機開始串流時 YUYV Run 失敗 | 在 DevKit 上執行 `dmesg`。`Failed to submit URB 0 (-28)`（`ENOSPC`）表示 USB 控制器無法保留未壓縮模式所需的頻寬。請選擇較低的解析度或影格率、停止同時串流的其他 USB 攝影機或音訊裝置，或改為擷取 MJPEG。 |
| `YUYV requires explicit input color` | 將 `PreprocOptions::input_color` 設為 `info.color`。 |
| `YUYV input does not support antialiased sampling`、`FP32 transforms require supported RGB/Gray tensor output`、`crop currently requires RGB, BGR or Gray input` 或 `ROI lists require RGB/BGR/Gray inputs/outputs` | 模型的前處理需要 YUYV 輸入不支援的選項。請參閱[在 YUYV 影格上執行模型](#run-a-model-on-yuyv-frames)。 |
| `Strict encoder ingress requires EncoderNv12 output from ColorConvert` | 在嚴格的 `VideoSender` 之前，放置使用 `ImageSurfaceLayout::EncoderNv12` 與 NV12 輸出的 `ColorConvert`。 |
| `Strict encoder ingress requires an exact integer frame rate` | 擷取整數影格率的模式，並將編碼器的 `fps` 設為該值。 |
| `Graph JSON does not store PreprocOptions::input_color or ColorConvertOptions::input_colorimetry` | 請以程式碼建立擷取 YUYV 的圖；`Graph::save` 無法儲存攝影機的色彩資訊。 |

## 擷取 YUYV {#capture-yuyv}

許多 UVC 攝影機也會傳送未壓縮的 YUYV：封裝的 4:2:2 影格，GStreamer 稱之為 `YUY2`。YUYV 影格不需要解碼器；模型的前處理與 `ColorConvert` 會直接從擷取緩衝區就地讀取。未壓縮影格需要的 USB 頻寬遠高於 MJPEG，因此攝影機通常只以較低的解析度或影格率列出 YUYV。

將 `format` 設為 `UsbCameraFormat::YUYV`，呼叫 `usb_camera_info`，再將其結果傳給 `UsbCameraInput`。請選擇攝影機列出且寬度為偶數的 YUYV 模式。傳回的 `UsbCameraInfo` 也會描述影格：`row_stride` 是以位元組為單位的列間距，`storage_bytes` 是每個擷取緩衝區的大小，`color` 是攝影機的矩陣（BT.601 或 BT.709）與範圍，`colorimetry` 則是以 GStreamer colorimetry 字串表示的相同解讀。

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

節點輸出每像素兩位元組、HWC 排列的 `UInt8` 影格。影格保留攝影機的影格率，包括 30000/1001 這類分數影格率，且允許奇數高度。使用預設的 `UsbCameraFormat::MJPEG` 時，節點會擷取 MJPEG。

### 在 YUYV 影格上執行模型 {#run-a-model-on-yuyv-frames}

將模型的 `preprocess.color_convert.input_format` 設為 `PreprocessColorFormat::YUYV`，並將其最大輸入尺寸設為攝影機模式的尺寸。YUYV 影格不會標示所用的矩陣與範圍，因此請從 `input_color` 為 `info.color` 的 `Preproc` 建立模型路徑，接著接上模型的推論與後處理：

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

對 YUYV 影格進行的模型前處理不支援：

- 抗鋸齒縮放取樣（`ResizeInterpolation::BilinearAntialiased`）；
- 固定的來源運算：`rescale_factor` 及其選擇的固定正規化、`quantization_step`、`PreprocFloatingConversion::NearestEven` 與 `PreprocColorConversionPlacement::BeforeResize`；
- `ResizeMode::Crop`；
- 執行階段 ROI 清單。

前處理需要上述任一項的模型會在建置圖時失敗。請改為擷取 MJPEG，或先以 `ColorConvert` 將影格轉換為 `ImageFormat::RGB`（將其 `color` 設為 `info.color`，`input_colorimetry` 設為 `info.colorimetry`），並將模型的輸入格式設為 `PreprocessColorFormat::RGB`。

### 以 H.264 檢視 YUYV 影格 {#view-yuyv-frames-as-h264}

若也要將攝影機的視訊傳送到檢視器，請在 `Preproc` 之前分流，並以 `RealtimeLatestByStream` 連接檢視分支，使其絕不拖慢模型。使用 `input_colorimetry` 為 `info.colorimetry` 的 `ColorConvert` 將影格轉換為編碼器排列方式的 NV12，再以 `SimaEncodeOptions::input_policy` 為 `EncoderInputPolicy::RequireDirect` 的嚴格 H.264 `VideoSender` 編碼：

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

嚴格傳送器絕不會在 CPU 上轉換或複製影格，並會在 H.264 串流中傳遞攝影機的原色、傳遞特性、矩陣與範圍。它需要偶數的寬度與高度，以及等於編碼器 `fps` 的整數影格率；其他輸入會遭拒絕，而非四捨五入。請參閱[傳送視訊](./video_sender.md#strict-encoder-input)。

從擷取到模型或編碼器之間不會複製任何影格；核心的 UVC 驅動程式仍會將 USB 封包複製到每個影格緩衝區。`UsbCameraInput` 會捨棄驅動程式標示為損毀的影格。驅動程式記錄的每個影格位元組數不會傳到圖中，因此驅動程式未標示的不完整影格會照常通過，其緩衝區其餘部分仍留有先前影格的資料。
