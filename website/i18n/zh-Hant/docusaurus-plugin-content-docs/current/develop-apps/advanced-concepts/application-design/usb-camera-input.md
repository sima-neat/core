---
title: "使用 USB 攝影機"
description: "使用 UsbCameraInput 從 USB（UVC）網路攝影機讀取 MJPEG 影格，直接轉送 JPEG 或解碼後交給模型，不需要裝置編號或管線字串。"
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# 使用 USB 攝影機

`nodes::UsbCameraInput` 會讀取傳送 MJPEG 的 USB（UVC）網路攝影機，並輸出攝影機的 JPEG 影格。請傳入攝影機的穩定 ID。`UsbCameraInputOptions` 沿用（[MIPI 攝影機](./mipi-camera-input.md)的）`CameraInputOptions` 欄位名稱與預設值，但 `camera_name` 為必填，且沒有 `format`。

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model
```

大多數網路攝影機傳送 4:2:2 JPEG，會解碼為 NV16。`usb_camera_info()` 會回報攝影機 JPEG 解碼後的排列方式：請將它作為 `SimaDecode` 的 `out_format`，並在回報 `NV16` 時將模型的 `preprocess.color_convert.input_format` 設為 `PreprocessColorFormat::NV16`，影格便會不經轉換階段直接送達模型。

## Neat 支援的範圍

- 在 Platform 3.0 上，連接到 Modalix DevKit USB 連接埠的 UVC 攝影機。
- 4:2:0 或 4:2:2 色度取樣的 MJPEG 擷取。
- 攝影機為 MJPEG 列出的確切寬度、高度與影格率。Neat 不會縮放，也不會挑選相近的模式。

僅支援 H.264 的攝影機、未壓縮的 YUYV/NV12 擷取、深度或紅外線串流（例如 RealSense 深度），以及曝光等攝影機控制，都不在此路徑的範圍內。

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

若要查看攝影機提供哪些 MJPEG 模式，請在 DevKit 上列出其格式。裝置編號只用於這項檢查；Neat 會依 ID 找到裝置。

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

攝影機拔除或移到另一個連接埠後，請再次呼叫 `UsbCameraInput`。先前建立的圖在建置時會失敗，因此絕不會開啟接手舊裝置編號的另一台攝影機。

## 轉送攝影機的 JPEG

若要將攝影機的視訊傳送給檢視器，同時在其上執行模型，請將節點連接到 `VideoSender` 直通模式（不重新編碼，直接以 RTP/JPEG 傳送 JPEG），以及 `out_format` 為攝影機 `decoded_format` 的 `SimaDecode`：

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

解碼器會先把每個 JPEG 複製到自己的輸入緩衝區，再進行解碼。傳送端適用 [傳送視訊](./video_sender.md) 中的 RTP/JPEG 限制。

## 記憶體與捨棄的影格

| 選項 | 效果 |
| --- | --- |
| `queue_depth`, `leaky_queue` | 攝影機後方的小型 leaky 佇列會在圖處理落後時捨棄過時的影格。 |

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
