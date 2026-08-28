---
title: "資料格式"
description: "格式標籤、有效載荷族群、佈局和張量語義。"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/data_formats
---

# 資料格式與張量語義

本頁說明 `InputOptions::format`、`OutputTensorOptions::format`、張量影像中繼資料和樣本酬載標籤所使用的公開格式詞彙。

對於任務層級的使用，請從 [張量與樣本](/develop-apps/development-workflow/core_types) 開始。當圖的邊界需要明確的格式合約時，請參考此頁。

## 格式標籤

`FormatTag` / `FormatSpec` 用於命名酬載格式。在 Python 中，請使用 `pyneat.Format` 或 `pyneat.FormatTag` 值來設定格式欄位。請勿將原始字串指派給 Python 格式欄位。

Python 公開了常用的使用者可見格式標籤。一些較低層級的 C++ 標籤，例如 `BBOX`、`MLA`、`ARGMAX` 和 `DETESSDEQUANT`，通常會透過張量語義中繼資料、酬載標籤或診斷資訊來呈現，而不是作為可指派的 `pyneat.Format` 值。

常見標籤：

| 標籤 | 典型酬載 | 意義 |
|---|---|---|
| `RGB` | 影像 | 封裝的 RGB，每個通道 8 位元。 |
| `BGR` | 影像 | 封裝的 BGR，每個通道 8 位元。 OpenCV 預設使用此格式。 |
| `GRAY8` | 影像 | 8 位元灰階。 |
| `NV12` | 影像/影片 | Y 平面加上交錯的 UV 平面。寬度和高度必須為偶數。 |
| `I420` | 影像/影片 | Y、U 和 V 平面。寬度和高度必須為偶數。 |
| `H264` | 編碼 | H.264 存取單元 / NAL 串流。`AVC` 是一個別名。 |
| `H265` | 編碼 | H.265 / HEVC 存取單元 / NAL 串流。`HEVC` 是一個別名。 |
| `ENCODED` | 編碼 | 通用編碼酬載。caps 字串識別不使用專用格式標籤的編解碼器。 |
| `FP32` | 張量 | Float32 張量酬載。 |
| `INT8` | 張量 | 帶符號的 INT8 張量酬載。 |
| `UINT8` | 張量 | 不帶符號的 UINT8 張量酬載。 |
| `BF16` | 張量 | BF16 張量酬載。 |
| `BBOX` | 偵測 | 封裝的邊界框酬載。 |
| `ByteStream` | 張量語義 | 由下游合約解釋的不透明位元組串流。 |

## 酬載族群

`PayloadType` 選擇跨越圖邊界的廣泛族群。

| 酬載族群 | 內部/媒體意義 | 常見中繼資料 |
|---|---|---|
| `Image` | 解碼像素 | 像素格式、寬度、高度、佈局、影像語義中繼資料 |
| `Tensor` | 模型或應用程式張量 | dtype、形狀、佈局、張量語義中繼資料 |
| `Encoded` | 編碼媒體，例如 H.264、H.265 或 JPEG | caps 字串、編解碼器格式、時間戳記 |
| `Auto` | 在可能的情況下推斷 | 僅在張量/樣本中繼資料足夠時使用 |

文字、音訊、位元組串流和不透明位元組酬載使用張量語義或專用的規格。它們不是在本次發布的公開 API 中進行審查的單獨的 `PayloadType` 列舉值。

## 原始影像對應

| 格式 | 負載類型 | 張量佈局/形狀 | 備註 |
| --- | --- | --- | --- |
| `RGB` | `Image` | `HWC`、`[H, W, 3]` | 緊密排列的像素。 |
| `BGR` | `Image` | `HWC`、`[H, W, 3]` | 用於 `cv2.imread` 或 OpenCV BGR 幀。 |
| `GRAY8` | `Image` | `HW`、`[H, W]` | 單通道灰度。 |
| `NV12` | `Image` | `HW`、`[H, W]`，以及平面元資料 | 複合的 Y + UV 平面。 |
| `I420` | `Image` | `HW`、`[H, W]`，以及平面元資料 | 複合的 Y + U + V 平面。 |

對於緊密格式，深度是通道數。對於張量負載，深度來自所選的佈局和形狀。

## 將格式、佈局和軸語義一起讀取

不要單獨讀取一個欄位：

| 欄位 | 它告訴您什麼 |
| --- | --- |
| `PixelFormat` / 圖像格式元資料 | 如何解釋像素通道，例如 RGB、BGR、GRAY8、NV12 或 I420。 |
| `TensorLayout` | 如何對張量維度進行排序，例如 HWC、CHW 或 HW。 |
| `TensorAxisSemantic` | 當張量攜帶更豐富的語義元資料時，一個軸的含義是什麼。 |
| `TensorDType` | 如何儲存每個元素，例如 UInt8、INT8、FP32 或 BF16。 |
| `ByteFormat` / 位元組流元資料 | 下一個階段應如何解釋不透明位元組。 |

位元組本身沒有意義。在重新解釋緩衝區之前，請一起使用元資料欄位。

## 輸入選項格式範例

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::InputOptions input;
input.payload_type = simaai::neat::PayloadType::Image;
input.format = simaai::neat::FormatTag::BGR;
input.width = 640;
input.height = 480;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
input_options = pyneat.InputOptions()
input_options.payload_type = pyneat.PayloadType.Image
input_options.format = pyneat.Format.BGR
input_options.width = 640
input_options.height = 480
```

</CodeTab>
</CodeTabs>

僅設定邊界所需的欄位。如果張量或樣本已經包含足夠的元資料，請避免重複的推測。

H.264 和 H.265 具有專用的 `H264` 和 `H265` 標籤。在輸入邊界上設定對應的標籤；媒體類型將從中解析。僅對沒有專用標籤的編解碼器使用 `ENCODED`，並搭配明確的 caps 字串。

## 進階影像/影片輸出配接器

對於一般的模型輸出，請使用 `nodes.output(...)` 並使用 `pull_tensors(...)` 提取張量。僅當影像或影片輸出必須轉換、調整大小或調整速率為適合 CPU 的 `UInt8` 張量時，才使用 `OutputTensorOptions`，然後再由應用程式提取。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::OutputTensorOptions output;
output.format = simaai::neat::FormatTag::BGR;
output.target_width = 640;
output.target_height = 480;

graph.add_output_tensor(output);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
output = pyneat.OutputTensorOptions()
output.format = pyneat.Format.BGR
output.target_width = 640
output.target_height = 480

graph.add_output_tensor(output)
```

</CodeTab>
</CodeTabs>

`add_output_tensor(...)` 接受 `TensorDType::UInt8`，這是預設的輸出 dtype。對於模型張量以及您希望保留完整 `Sample` 包裹的輸出，請繼續使用標準的 `nodes.output(...)` 路徑。當您需要其他 dtype 時，請新增明確的圖或應用程式端轉換。

## 樣本酬載標籤

`Sample::payload_tag` 是下游消費者的首選標籤。它取代了已棄用的 `Sample::format` 欄位。

在偵錯編碼媒體或圖邊界協商時，請一起使用 `payload_tag`、`payload_type`、`media_type` 和 `caps_string`。

## 預處理元資料和 ROI 追蹤

偵測、解碼、渲染和 ROI 工作流程需要預處理元資料，才能將模型空間座標映射回來源框架座標。

這些元資料可以包括：

- 目標寬度和高度；
- 縮放後的內容寬度和高度；
- 調整大小或寬螢幕模式；
- 填充值和幾何形狀；
- 輸入和輸出色彩格式；
- 軸置換；
- 正規化、量化和鑲嵌旗標；
- ROI 視窗、來源影像大小、ROI 批次大小以及每個 ROI 的仿射變換。

如果框或遮罩位於錯誤的位置，請檢查預處理元資料是否已到達解碼或渲染階段，然後再更改閾值。有關 ROI 列表預處理詳細資訊，請參閱 [預處理感興趣區域 (ROI) 清單](/reference/preproc_roi)。

## 參閱

- [張量與樣本](/develop-apps/development-workflow/core_types)
- [dtype 規範](/develop-apps/advanced-concepts/dtype_contract)
- [節點](/develop-apps/development-workflow/node)
