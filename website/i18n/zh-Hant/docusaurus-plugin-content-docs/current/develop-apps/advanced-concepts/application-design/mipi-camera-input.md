---
title: "使用 MIPI 相機"
description: "將支援疊加功能的 MIPI 相機新增到 Neat 圖中，並使用 CameraInput、模型預處理、MLA 推論、可選的 EV74 BoxDecode，以及輸出提取功能。"
sidebar_position: 4
slug: /develop-apps/advanced-concepts/mipi-camera-input
---

# 使用 MIPI 攝影機

當圖需要直接從 Modalix DevKit 上的 MIPI 攝影機讀取影格時，請使用 `CameraInput`。`CameraInput` 是經過驗證的 libcamera 串流與以加速器為優先的模型圖之間的 Neat 邊界：

```text
CameraInput -> model-managed CVU preproc -> MLA -> Output
CameraInput -> model-managed CVU preproc -> MLA -> EV74 BoxDecode -> Output
```

沒有 `appsrc`。使用者程式碼中沒有 `ostosima`。除非您自行新增，否則生產路徑中沒有 CPU `videoconvert` 或 `videoscale`。

## 兩個階段：先啟動，然後是 Neat

MIPI CSI-2 是相機連接。它並不能讓每個感測器都能即插即用。一個可運作的相機路徑還需要正確的板載疊加層、感測器驅動程式、libcamera 管線、ISP 行為和 caps。

將 MIPI 相機工作視為兩個階段：

1. **板載啟動：** Modalix DevKit 偵測到感測器，並且 libcamera 可以串流所需的模式。
2. **Neat 圖啟動：** `CameraInput` 將這些影格饋入模型管理的 CVU/MLA 階段。

Neat 從第二階段開始。它不會選擇 `.dtbo` 疊加層、載入感測器驅動程式或調整 ISP。對於疊加層設定、支援的疊加層名稱和 `cam` 驗證，請使用 [Modalix DevKit MIPI 相機介面指南](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces)。

## Neat 支援哪些功能

`CameraInput` 支援平台相機堆疊已啟動的相機：

- 相機連接到 Modalix DevKit MIPI 連接埠，同時板子已關閉；
- 正確的板載疊加層已啟用；
- 核心驅動程式和 libcamera 管線公開相機；
- `libcamerasrc` 可以協商所需的 `video/x-raw` caps，通常是 `NV12`；
- caps 與您在 Neat 中設定的模型預處理相符。

如果這些條件尚未滿足，請先修復相機堆疊。Neat 圖可以精確，但它無法將未綁定的感測器轉換為串流。

## 驗證相機串流

在建立圖之前，先在 libcamera/GStreamer 層驗證相機。如果此層無法運作，則 Neat 圖無法修復它。

在 DevKit 上，確認 `libcamerasrc` 是否存在：

<ShellCommand prompt="devkit">
gst-inspect-1.0 libcamerasrc
</ShellCommand>

如果可以使用 `cam`，請列出相機和檢查模式：

<ShellCommand prompt="devkit">
cam -l
cam -c 1 -I
</ShellCommand>

然後試試您打算向 Neat 要求的確切規格：

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

為了進行簡單的視覺煙霧測試，請將幾個影格編碼為 JPEG 格式：

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! videoconvert ! jpegenc ! \
  multifilesink location=/tmp/mipi-frame-%03d.jpg
</ShellCommand>

`videoconvert` 和 `jpegenc` 對於這次的單次除錯檢查來說，效果還不錯。當吞吐量很重要時，請將它們排除在模型管線之外。

## 建立原始的 MLA 煙霧圖

從一個原始的 MLA 路線開始。它證明了在您新增特定模型的後處理之前，相機畫面是否已到達 CVU 預處理和 MLA 推論階段。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::CameraInputOptions camera;
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.format = "NV12";
camera.buffer_name = "camera0";
camera.allow_cpu_fallback = true;

neat::Model::Options model_options;
model_options.preprocess.kind = neat::InputKind::Image;
model_options.preprocess.input_max_width = static_cast<int>(camera.width);
model_options.preprocess.input_max_height = static_cast<int>(camera.height);
model_options.preprocess.input_max_depth = 3;
model_options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
model_options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
model_options.preprocess.resize.enable = neat::AutoFlag::On;
model_options.preprocess.resize.width = 640;
model_options.preprocess.resize.height = 640;
model_options.preprocess.resize.mode = neat::ResizeMode::Letterbox;
model_options.preprocess.resize.pad_value = 114;
model_options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
model_options.advanced_execution.preprocess_target = "EV74";
model_options.inference_terminal.mla_only = true;

neat::Model model("/models/yolo.tar.gz", model_options);

neat::Model::RouteOptions route;
route.include_input = false;
route.include_output = true;
route.upstream_name = camera.buffer_name;
route.buffer_name = camera.buffer_name;
route.name_suffix = "_camera0";
route.advanced_execution.preprocess_target = "EV74";

neat::Graph graph("camera_mla_smoke");
graph.add(neat::nodes::CameraInput(camera));
graph.add(model.graph(route));

neat::Run run = graph.build();
std::optional<neat::Sample> output = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.CameraInputOptions()
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.format = "NV12"
camera.buffer_name = "camera0"
camera.allow_cpu_fallback = True

model_options = pyneat.ModelOptions()
model_options.preprocess.kind = pyneat.InputKind.Image
model_options.preprocess.input_max_width = int(camera.width)
model_options.preprocess.input_max_height = int(camera.height)
model_options.preprocess.input_max_depth = 3
model_options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
model_options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
model_options.preprocess.resize.enable = pyneat.AutoFlag.On
model_options.preprocess.resize.width = 640
model_options.preprocess.resize.height = 640
model_options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
model_options.preprocess.resize.pad_value = 114
model_options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
model_options.advanced_execution.preprocess_target = "EV74"
model_options.inference_terminal.mla_only = True

model = pyneat.Model("/models/yolo.tar.gz", model_options)

route = pyneat.ModelRouteOptions()
route.include_input = False
route.include_output = True
route.upstream_name = camera.buffer_name
route.buffer_name = camera.buffer_name
route.name_suffix = "_camera0"
route.advanced_execution.preprocess_target = "EV74"

graph = pyneat.Graph("camera_mla_smoke")
graph.add(pyneat.nodes.camera_input(camera))
graph.add(model.graph(route))

run = graph.build()
output = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

`inference_terminal.mla_only = true` 是刻意為之。它會將煙霧路徑維持在 `CameraInput -> CVU preproc -> MLA -> Output`，因此您可以在偵測解碼之前，先對相機和推論動作進行除錯。

## 當模型需要時，新增 EV74 BoxDecode

對於 YOLO 樣式的偵測模型，請明確啟用 BoxDecode，並將後處理保留在 EV74 上。解碼標記必須與 MPK 的輸出合約相符。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
model_options.inference_terminal.mla_only = false;
model_options.decode_type = neat::BoxDecodeType::YoloV9Seg;
model_options.advanced_execution.postprocess_target = "EV74";
model_options.score_threshold = 0.25f;
model_options.nms_iou_threshold = 0.45f;
model_options.top_k = 100;

route.advanced_execution.postprocess_target = "EV74";
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
model_options.inference_terminal.mla_only = False
model_options.decode_type = pyneat.BoxDecodeType.YoloV9Seg
model_options.advanced_execution.postprocess_target = "EV74"
model_options.score_threshold = 0.25
model_options.nms_iou_threshold = 0.45
model_options.top_k = 100

route.advanced_execution.postprocess_target = "EV74"
```

</CodeTab>
</CodeTabs>

讓 `decode_type` 保持未設定，並在您需要原始 MLA 張量時，保留 `mla_only = true`。僅在模型路徑應輸出已解碼的檢測或分割資料時，設定 `decode_type`。

## 選擇正確的記憶體模式

`CameraInput` 有兩種模式：

| 模式 | 何時使用 | 行為 |
|---|---|---|
| 嚴格的零拷貝 | 您的 `libcamerasrc` 公開 `external-buffer-mode`，且記憶體庫支援 DMA-BUF 匯出。 | Neat 需要直接將資料擷取到其下游 DMA-BUF 儲存區中，如果來源無法提供，則會失敗。 |
| 自適應回退（明確選擇） | 您需要與無法匯出裝置緩衝區的相機堆疊相容。 | Neat 接受作業系統/libcamera 緩衝區，將其複製到池化的 SiMaAI 記憶體中，以便進行 CVU/MLA 傳遞，並且在來源已經提供 SiMaAI 緩衝區時，直接傳遞這些緩衝區。 |

嚴格的零拷貝是框架的預設設定。它需要一個公開通用 `external-buffer-mode` 屬性的 `libcamerasrc`，以及一個支援 DMA-BUF 匯出的記憶體庫。僅將 `camera.allow_cpu_fallback = true` 設定為明確的相容性選項。回退複製是通往加速器管線的橋樑；它不是允許將 CPU 顏色轉換或縮放新增到熱路徑中的許可。

在兩種模式下，Neat 都在相機功能設定之後，以及任何佇列之前，立即放置其私有記憶體橋接器。該橋接器透過 `GST_QUERY_ALLOCATION` 提出一個標準緩衝區儲存區。該儲存區會從一個打包的 SiMaAI 分配中分配經過驗證的平面佈局，並為每個平面匯出一個 DMA-BUF。`libcamerasrc` 將這些 DMA-BUF 匯入到 ISP 擷取佇列中；擷取後，橋接器會解開相同的打包分配，以便進行下游處理。這是正常路徑，而不是回退複製路徑。

應用程式擁有 ISP 輸出保留策略。將 `capture_buffer_count` 參數保留給 `nodes::CameraInputWithCaptureBuffers()` 或 `pyneat.nodes.camera_input()`，設定為 `0`，以使用相機的預設值，或者在時間編碼器或非同步 ML 圖形需要更長時間保留影格時，要求更大的最小值。作用中的相機管線會驗證其自身的限制；Neat's 提供者最多支援 128 個。此計數與私有的 CSI 到 ISP RAW 傳輸環以及 `queue_depth` 分開，後者僅控制 Neat's 作用中的 GStreamer 佇列。當前影格比完整性更重要時，使用洩漏佇列；當每個影格都必須保留時，使用下游反壓。在相容性複製模式下，橋接器的私有儲存區會按需增長，因此它不會在洩漏佇列可以丟棄過時影格之前發生阻塞。

## 將預處理保留在 CVU/EV74 上

對於模型管線，請優先使用模型管理的預處理：

- 設定 `Model::Options::preprocess` 或 `pyneat.ModelOptions.preprocess`，用於調整大小、色彩轉換、正規化、量化和鑲嵌；
- 當模型路徑支援時，請在 `EV74` 上保留模型管理的 CVU 前置/後置目標；
- 除非您正在建立僅用於除錯的圖，否則請避免在模型之前插入 `VideoConvert`、`VideoScale` 或 GStreamer `videoconvert`/ `videoscale`。

相機會提供畫面。CVU 應該處理畫面計算。CPU 不應成為您意外的影像處理引擎。

## 快速疑難排解指南

| 症狀 | 首先檢查 |
|---|---|
| 沒有顯示相機 | 確認選取的 `.dtbo`、線纜方向、重新啟動電源以及核心/libcamera 日誌。 |
| 缺少 `libcamerasrc` | 安裝與 DevKit 建置相符的 Neat/runtime 相機影像或相機套件。 |
| `misconfig.media_caps` 或 `not-negotiated` | 使用 `gst-launch-1.0` 驗證確切的 `format,width,height,framerate`。嘗試使用已知的受支援模式，例如 `NV12 1920x1080@30`。 |
| 嚴格的零拷貝失敗 | 設定 `allow_cpu_fallback = true`，或使用公開 SiMaAI 零拷貝屬性的相機堆疊。 |
| 輸出色彩看起來不正確 | 確認畫面是否被解釋為 `NV12`，而不是 RGB/BGR。如果直接從 `libcamerasrc` 產生的 JPEG 也是不正確的，請在除錯 Neat 之前除錯相機 ISP/調整。 |
| 吞吐量低 | 移除 CPU 影像轉換/縮放，持續拉取，並使用優先考慮即時性的即時來源佇列原則。 |

有關更多症狀，請參閱 [疑難排解](/reference/troubleshooting)。
