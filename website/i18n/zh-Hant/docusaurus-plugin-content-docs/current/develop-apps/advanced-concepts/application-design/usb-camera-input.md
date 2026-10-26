---
title: "使用 USB 攝影機"
description: "使用 UsbCameraInput 從 USB（UVC）網路攝影機讀取 MJPEG 影格，由硬體解碼器就地解碼為 NV12，不需要裝置編號或管線字串。"
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# 使用 USB 攝影機

`groups::UsbCameraInput` 會讀取傳送 MJPEG 的 USB（UVC）網路攝影機。傳入攝影機的穩定 ID 後，圖會在 SiMaAI 記憶體中收到 NV12 影格，與 [MIPI 攝影機](./mipi-camera-input.md) 相同。`UsbCameraInputOptions` 沿用 `CameraInputOptions` 的欄位名稱與預設值，但 `camera_name` 為必填，且沒有 `format`：輸出一律為 NV12。

```text
USB camera -> UsbCameraInput -> model-managed CVU preproc -> MLA -> Output
```

在內部，攝影機會把每個壓縮影格直接寫入硬體 MJPEG 解碼器就地讀取的 CMA 緩衝區，因此擷取與解碼之間不會發生 CPU 複製。若攝影機傳送 4:2:2 JPEG（大多數網路攝影機都是如此），解碼器之後會加上一個將 NV16 轉為 NV12 的 EV74 轉換。

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

加入模型的方式與 MIPI 攝影機相同；請參閱 [使用 MIPI 攝影機](./mipi-camera-input.md) 中建立原始 MLA 圖的章節。

`UsbCameraInput` 會在建立圖時開啟攝影機一次，以檢查模式並讀取第一個影格的色度取樣。請先關閉其他正在使用攝影機的程式。

攝影機拔除或移到另一個連接埠後，請再次呼叫 `UsbCameraInput`。先前建立的圖在建置時會失敗，因此絕不會開啟接手舊裝置編號的另一台攝影機。

## 記憶體與捨棄的影格

| 選項 | 效果 |
| --- | --- |
| `allow_cpu_fallback` | 對 USB 攝影機沒有作用：擷取一律寫入解碼器自己的緩衝區，因此不會複製任何影格。保留此選項是為了讓選項與 `CameraInputOptions` 一致。 |
| `queue_depth`, `leaky_queue` | 解碼器前方的小型 leaky 佇列會在圖處理落後時捨棄過時的影格。 |
| `capture_buffer_count`（第二個引數） | 攝影機與解碼器共用的 CMA 緩衝區。0 表示使用佇列深度加上解碼器的兩個輸入槽。 |

這裡的零複製是指壓縮影格在擷取與解碼之間不會被複製。解碼後的 NV12 影格是另一個緩衝區，而 4:2:2 的 EV74 轉換還會寫入另一個緩衝區。

網路攝影機經常在 JPEG 結束標記之後補上位元組，有些影格則會不完整地抵達，或被 USB 驅動程式標示為損毀。解碼器會把每個影格解碼到其結束標記為止，並在不停止的情況下捨棄不完整或損毀的影格。解碼器元件會在其 `dropped-count` 屬性中計數。

## 疑難排解

| 訊息 | 處理方式 |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | 使用列出的其中一個 ID。攝影機移到另一個連接埠時 ID 會改變。 |
| `no USB cameras are connected` | 檢查纜線與 `lsusb`。 |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | 選擇列出的模式。有些攝影機只在較低的影格率下提供高解析度。 |
| `is in use by another process` | 停止目前開啟攝影機的其他程式。 |
| `sent no usable MJPEG frame` | 攝影機傳送 Neat 無法解碼的 JPEG（例如 4:4:4 取樣），或完全沒有傳送影格。 |
| `is no longer /dev/videoN` | 建立圖之後，攝影機被拔除、移動或重新編號。請再次呼叫 `UsbCameraInput`。 |
| `requires a neatdecoder with the input-pool-buffers property` | 已安裝的解碼器太舊。請更新 `simaai-scheduler` 套件。 |
