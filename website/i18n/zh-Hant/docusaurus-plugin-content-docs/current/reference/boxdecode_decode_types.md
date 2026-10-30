---
title: "BoxDecode 解碼類型"
description: "為物件偵測後處理選擇正確的 BoxDecodeType。"
sidebar_position: 6
---

# BoxDecode 解碼類型

`nodes::SimaBoxDecode` 將原始偵測頭張量轉換為偵測結果。它在模型推論後執行，套用所選模型系列的解碼運算並篩選偵測結果。NMS 與輸出編碼取決於所選系列。偵測模型可將酬載解析為方框；姿勢與分割模型也可解析方框之後的關鍵點或遮罩。

對於一般的模型套件使用，建議使用 `Model` 感知的建構函式。模型封存檔提供了解碼器所需的張量順序、佈局、量化、類別數量、調整大小的中繼資料和分數域提示。您的應用程式通常只需要選擇解碼系列和過濾閾值。

## 執行階段後端

BoxDecode 與 SuperPoint 透過 `neatkernelrunner` 使用已檢查的 Kernels 執行物件。
既有的偵測與特徵輸出格式保持不變。YOLO26 旋轉框解碼使用獨立的版本化格式。直接張量輸入請使用共用 CMA 張量
（`TensorMemory::EV74`），即使解碼在 A65 執行也一樣：此記憶體選擇提供的是
DMA-BUF 傳輸，而非執行目標。

RCNN 第 1 階段使用核心擁有的逐層 top-K 契約：先全域 NMS，再於每個模型面積層級
保留最多 `top_k` 個候選項。輸出是提案線路格式，而非一般 BBOX 格式：共四組，
每組包含一個 INT32 計數與 `top_k` 筆六 FP16 記錄（`score, 0, x1, y1, x2, y2`）。
座標轉換保留小數幾何資訊，直到最終序列化。請將其用作第二階段提案輸入，
不要搭配 `decode_bbox_tensor()` 使用。不會載入舊版外掛程式。

具有填補、交錯排列的 INT8、BF16 與 FP32 SuperPoint 輸出頭使用原生密集讀取器及
明確像素步幅。填補區不參與偵測器歸約與描述子取樣，不需要 Cast 階段或具體化。
INT8 埠保留各自設定的量化資訊。原生 MLA 平鋪輸出頭保留其宣告的實體佈局；
絕不將密集純量重新標成 MLA 位元組平面。

## 快速入門

```cpp
using namespace simaai::neat;

Model model("/path/to/yolov8_model.tar.gz");

auto boxdecode = nodes::SimaBoxDecode(
    model,
    BoxDecodeType::YoloV8,
    /* detection_threshold */ 0.25,
    /* nms_iou_threshold */ 0.45,
    /* top_k */ 100);
```

針對單獨的階段使用：

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.45;
opt.top_k = 100;
```

## 參數

| 參數 | 意義 |
| --- | --- |
| `decode_type` | 模型系列/標頭格式，例如 `BoxDecodeType::YoloV8` 或 `BoxDecodeType::YoloX`。必填。|
| `detection_threshold` | 為了保留檢測結果，所需的最低分數。請使用適合模型的數值，例如 `0.25`。|
| `nms_iou_threshold` | 非最大值抑制中使用的 IoU 閾值。設為 `0` 時會保留模型封裝的值；若未封裝任何值，YOLO26 會略過 NMS，其他解碼器會抑制所有重疊的方框。請設定適合模型的值，例如 `0.45`。|
| `top_k` | 最大輸出數量。原始邊界框解碼需要明確的正值上限。綁定模型的節點中，`0` 繼承已解析的 Model 上限。SuperPoint 在 `0` 時使用其設定組合預設值。 |
| `original_width`, `original_height` | 使用原始幾何建構函時，用於座標映射的原始影像大小。|
| `model_width`、`model_height` | 用於覆寫模型輸入大小。在使用 `Model` 建構函式時，這會更改空間解碼參數，而非已封裝的張量合約。|
| `resize_mode_override` | 僅在沒有上游 `Preproc` 階段寫入調整大小的元資料，且您需要明確指定拉伸/信箱/裁剪行為時使用。|
| `decode_type_option` | 進階子版面設定選擇器。除非您知道匯出的版面設定，否則請保留為 `Auto`，以便用於模型套件。|

## 輸入與輸出

**輸入：**來自模型的原始檢測張量。預期的張量形狀取決於模型系列。對於 MPK/模型封存檔，Neat 會從封裝的合約中讀取這些詳細資訊。

**輸出：**一個 BoxDecode 張量，其中包含已解碼的檢測結果。檢測模型使用標準 `BBOX` 負載。姿勢和分割模型保留相同的初始框，並附加其任務特定的負載：

| 模型任務 | C++ 輔助程式 | Python 輔助程式 | 解碼後的張量 |
| --- | --- | --- | --- |
| 偵測 | `decode_bbox(...)` | `pyneat.decode_bbox(...)` | `[N, 6]` float32 框：`x1, y1, x2, y2, score, class_id` |
| 旋轉物件偵測 | `decode_obb(...)` | `pyneat.decode_obb(...)` | 具有自身儲存空間的幾何資料 `[N, 5]` float32（`cx, cy, width, height, angle_radians`）、分數 `[N]` float32、類別 ID `[N]` int32 |
| 姿勢 | `decode_pose(...)` | `pyneat.decode_pose(...)` | 方框 `[N, 6]` 和關鍵點 `[N, 17, 3]` float32：`x, y, visibility` |
| 分割 | `decode_segmentation(...)` | `pyneat.decode_segmentation(...)` | 方塊 `[N, 6]` float32，以及遮罩 `[N, 160, 160]` uint8 |
| 分割 + 姿態 | `decode_segmentation_pose(...)` | `pyneat.decode_segmentation_pose(...)` | 邊界框 `[N, 6]` float32、遮罩 `[N, 160, 160]` uint8、關鍵點 `[N, 17, 3]` float32 |
| SuperPoint | `decode_superpoint(...)` | `pyneat.decode_superpoint(...)` | 關鍵點 `[N,2]`，分數 `[N]`，描述子 `[N,D]` |

軸對齊偵測結果的顯示圖可將結果傳給 `SimaRender`。`decode_bbox(...)` 僅適用於既有的方框前置格式；此函式與 `SimaRender` 都不接受旋轉框。

## YOLOv8 與 YOLO11 旋轉邊界框 {#yolov8-oriented-boxes}

選取 `BoxDecodeType::YoloV8Obb`（Python：`pyneat.BoxDecodeType.YoloV8Obb`；代碼 `yolov8-obb` 或 `yolo11-obb`）。提供九個依角色分組的 INT8、BF16 或 FP32 頭：三個 64 通道 DFL 邊界框頭、三個類別 logit 頭，以及三個角度 logit 頭，各組依步幅 8、16、32 排列。完整類別數由類別頭推導。

BoxDecode 會套用 DFL、類別 sigmoid、`(sigmoid(angle)-0.25)*pi`，以及使用 `nms_iou_threshold` 的各類別旋轉 Fast-NMS。輸入必須為原始 logit。幾何資訊與 `BBOX_OBB_V1` 輸出遵循下方的 YOLO26 契約，並使用相同的 `decode_obb` 輔助函式。

## YOLO26 旋轉框 {#yolo26-oriented-boxes}

明確選取 `BoxDecodeType::YoloV26Obb`（Python：`pyneat.BoxDecodeType.YoloV26Obb`，代碼 `yolo26-obb`）。此設定檔接收九個 FP32、BF16 或 INT8 頭，依角色分組為三個四通道距離頭、三個類別 logit 頭，以及三個單通道原始角度頭，各組步幅為 8、16、32。所有頭的完整類別數（1–256）必須一致。`Auto`、`GroupedByRole` 與 `GroupedByRoleLogit` 會選用此分組佈局；機率值類別頭與交錯佈局會遭到拒絕。

INT8 頭需要正且有限的逐張量量化中繼資料與 INT8 零點。BF16 和 FP32 頭不使用整數量化中繼資料。

使用正值 `top_k` 與 `nms_iou_threshold = 0`。此設定檔對類別 logit 套用 sigmoid，以最高啟用分數選出錨點，再選出錨點/類別配對。分數相同時保留第一階段排名與類別順序。分數門檻採嚴格不等式（`score > detection_threshold`）。原始角度已是弧度，解碼器不會再套用 sigmoid 或正規化其範圍。

前處理中繼資料必須提供明確且有限的模型座標到原圖座標轉換，X/Y 縮放係數須為正且相等，並且不得包含剪切。即使圖塊與模型尺寸相同，也必須明確提供恆等轉換。若缺少轉換或轉換為非等向性，此設定檔會拒絕，而不會近似旋轉矩形。它保留帶正負號的有限寬度/高度，不會裁切至影像邊界。進行分塊 DOTA 評估時，每個圖塊使用恆等轉換，再由應用程式還原圖塊偏移並執行跨圖塊多邊形抑制。

在來源驅動的圖中，Core 會從已驗證的上游 Preproc 契約推導初始準備用的幾何資訊。這只是準備用的初始值：每個影格仍須提供有限、正值且等比例的仿射轉換，以及與模型相符的畫布尺寸。缺少影格轉換時會回報錯誤，不會退回恆等轉換。

輸出標記為 `BBOX_OBB_V1`：32 位元組版本化標頭後接 `top_k` 筆 32 位元組記錄。未使用的記錄為零。`decode_obb_tensor(tensor)` 驗證單一載體；`decode_obb(tensors)` 為每個載體傳回一個具型別的結果。兩者都會複製到具有自身儲存空間的 CPU 張量；類別 ID 維持 INT32。例如：

```python
result = pyneat.decode_obb(outputs)[0]
geometry = result.geometry.to_numpy()  # cx, cy, width, height, angle_radians
scores = result.scores.to_numpy()
class_ids = result.class_ids.to_numpy()
```

此格式刻意不作為前置 BBOX 的擴充。請使用具型別的輔助函式，而非 `decode_bbox`、分割/姿勢解析器或軸對齊方框的繪製器。

## 超級點

SuperPoint 仍然是 BoxDecode 產品的一部分，但它會輸出特徵點，而不是假裝這些點是方框。最簡化的 A65 預設設定如下：

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.descriptor_output_dtype = TensorDType::Float32;

auto decoder = nodes::SimaBoxDecode(model, options);
```

Python 使用相同的預設值：

```python
options = pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.SuperPoint)
options.superpoint.descriptor_output_dtype = pyneat.TensorDType.Float32

decoder = pyneat.nodes.sima_box_decode(model, options=options)
```

`A65V1` 是預設設定檔。當模型需要不同的數值行為時，請明確選擇另一個設定檔；Neat 不會根據張量的形狀或值來推斷行為：

| 個人檔案 | 何時選擇 | 生產狀態 |
|---|---|---|
| `LightGlueV1` | 與 LightGlue 相容的檢測器、非最大值抑制 (NMS)、座標和描述符行為 | 已支援 |
| `MagicLeapDemoV1` | 固定位置的 Magic Leap 示範行為 | 支援 |
| `A65V1` | 與先前 A65 SuperPoint 解碼器相容 | 已支援；預設值 |
| `PaperBicubicV1` | 保留的數值 ID，用於未來完全指定的雙三次插值策略 | 暫時拒絕，直到產品定義完成 |

數值行為和輸出編碼是相互獨立的。例如，選擇 A65 數值行為，並使用預設的 V1 輸出：

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::FeaturePointsV1;
```

舊版位元組設定為選擇性功能，且具有額外的限制：

```cpp
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::LegacyA65InterleavedV0;
options.superpoint.descriptor_output_dtype = TensorDType::Int8;
```

`SuperPointProfile::Auto` 首先使用權威的 MPK `superpoint.profile` 元資料。如果 API (`Model::Options.superpoint.profile`) 或 MPK 都未提供設定檔，則會解析為 `A65V1`。
Neat 不會根據張量形狀、值、檔案名稱或下游節點來推測設定檔。

當其公開的預設值保持不變時，`detection_threshold=0.0`、`top_k=0`、`nms_radius=-1` 和 `border_margin=-1` 將從選定的設定檔中解析。`A65V1` 解析為閾值 `0.1`、Top-K `600`、NMS 半徑 `4` 和邊界間距 `0`。LightGlueV1 和 MagicLeapDemoV1 使用閾值 `0.0005` 和 `0.015`，分別；兩者都使用 Top-K `600`、NMS 半徑 `4` 和邊界間距 `4`。

`nms_iou_threshold` 不適用於 SuperPoint；請使用像素半徑 `superpoint.nms_radius`。預設輸出是版本化的 `FEATURE_POINTS_V1` 結構化陣列有效載荷。`LegacyA65InterleavedV0` 是一種明確的遷移格式，需要 256 維 INT8 描述符。請使用 `decode_superpoint`，而不是 `decode_bbox` 或 `BoxDecodeResults`。

版本化的 MPK `superpoint` 模式 v1 記錄是「失敗時關閉」的。它們必須命名設定檔、不同的檢測器和描述符張量 ID、一個帶有 64 個十六進位數字的 `sha256:` 指紋，以及受支援的輸入表示形式 `raw-logits-65` 和 `coarse-pre-l2`。模式 0 僅作為遷移/手動記錄被接受；省略的模式 0 表示形式欄位會正規化為這兩種原始輸入表示形式，並記錄在診斷資訊中作為預設值。未知的模式版本或表示形式標記會導致合約編譯失敗。
如果 API 設定檔覆蓋與為不同 MPK 設定檔蓋章的指紋衝突，則重新為選定的設定檔蓋章 MPK；Neat 不會捨棄或重新解釋該來源。

## BBOX 線路負載

偵測解碼器會針對每個輸入影格輸出一個標記為 `BBOX` 的張量。該張量是一個一維的 `UInt8` 位元組緩衝區：

| 欄位 | 值 |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | `[N_bytes]`，其中 `N_bytes` 是來自模型封存檔的已封裝緩衝區容量。 |

張量的形狀代表的是位元組數量，而不是偵測到的物件數量。有效載荷採用小端格式：

```text
offset  size  content
------  ----  -------
  0      4    uint32  N = valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   trailing bytes are padding and must be ignored
```

每個 `RawBox` 記錄的長度為 24 位元組：

| 偏移量 | 大小 | 類型 | 欄位 | 意義 |
| --- | --- | --- | --- | --- |
| 0 | 4 | `int32` | `x` | 原始影像中左上角的位置（以像素為單位）。|
| 4 | 4 | `int32` | `y` | 來源影像中，左上角 y 座標。 |
| 8 | 4 | `int32` | `w` | 原始影像的寬度（以像素為單位）。|
| 12 | 4 | `int32` | `h` | 原始影像中的高度（以像素為單位）。|
| 16 | 4 | `float32` | `score` | 在 `[0.0, 1.0]` 中，NMS 之後的置信度。|
| 20 | 4 | `int32` | `class_id` | 模型定義的類別 ID。|

對應的 Python `struct` 格式，用於單一記錄，為 `"<iiiifi"`。

如果存在上游預處理的元資料，則座標將以原始影像的像素為單位。它們不會正規化到 `[0, 1]`，也不會以模型內部「黑邊」輸入空間的形式表示。

### 分割 + 姿態的組合酬載

`yolox-seg-pose` 輸出單一緩衝區，其中包含三個區域。所有區域均依相同的槽位數 `top_k` 配置：

| 區域 | 偏移量 | 步幅 | 內容 |
| --- | --- | --- | --- |
| 標頭 | `0` | 4 | `int32` 偵測數 |
| 邊界框 | `4` | 24 | 如上所述的 `BoundingBoxOut` 記錄 |
| 遮罩 | `4 + 24*top_k` | `mask_w * mask_h` | `uint8`，每個槽位一個平面 |
| 姿態 | `4 + (24 + mask_w*mask_h)*top_k` | 204 | 17 x `{uint32 x, uint32 y, float32 visibility}` |

使用 `decode_segmentation_pose(...)` 取得邊界框、遮罩及關鍵點。各張量的第 `i` 列代表同一次偵測。若只需邊界框，請使用 `decode_bbox(...)` 或 `BoxDecodeResults(...)`。

組合解碼器也接受舊式 `BBOX` 標籤。不支援同時繪製遮罩與姿態。
不要將組合緩衝區傳給 `poseyolo`：該繪製器預期方框後緊接關鍵點，
但組合負載在該位置儲存的是遮罩。

每次偵測有 17 個關鍵點槽位。後端會將未使用的槽位及被 `pose_classes` 排除之類別的關鍵點設為零。輔助函式會原樣複製這些值。

由於通道 0 儲存物件性，Core 以類別頭深度減一推算 `num_classes`。例如，30 個通道表示 29 個類別。明確指定的 `num_classes` 必須一致。

### 關鍵點類別

設定具有關鍵點的類別 ID：

```python
options = pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.YoloXSegPose)
options.yolox_seg_pose.pose_classes = [0, 5]
```

`ModelOptions.yolox_seg_pose` 也接受相同設定。

- 清單以外類別的關鍵點座標及可見度會設為零。
- 空的 `BoxDecodeOptions` 清單會繼承模型設定。未設定清單時，每個類別都會產生關鍵點。
- ID 必須唯一且位於 `[0, num_classes)` 範圍內。此選項僅支援 `YoloXSegPose`。

## 當 `model.run` 傳回原始的標題時

某些模型路徑會從 `model.run(...)` 傳回原始的特徵圖頭，而不是傳回已解碼的 `BBOX` 張量。這並不是表示執行失敗。這表示模型已執行，但該路徑在您讀取輸出時並沒有包含「框解碼」步驟。

請使用以下規則：

- `detections=...` 或一個 `BBOX` 張量：解析壓縮後的 BBOX 負載，或使用。
  解碼輔助工具。
- `raw_output_heads=...`: 新增「BoxDecode」階段，或檢查模型路徑。
  使用模型特定的後處理方式來處理原始的張量。

請勿將原始標頭解析為框。原始張量的佈局取決於導出的模型系列和模型封存檔合約。

## 覆寫合約

MPK 提供張量契約與解碼系列資訊。請在 `ModelOptions` 中設定篩選控制項。
對於 `SimaBoxDecode(Model, ...)` 節點，正值篩選引數會覆寫已解析的 Model 控制項；
零值引數則繼承它們。`ModelOptions.score_threshold = 0.0` 是真正的零分數門檻，
不是要求還原封裝內的閾值。

| 執行階段參數 | 傳遞的值 | 行為 |
| --- | --- | --- |
| `decode_type` | 空值 / `Unspecified` | 在支援的情況下，保留模型封存檔或路徑規劃器推論。|
| `decode_type` | 具體類型 | 覆寫本次運行的解碼規則。|
| `original_width` / `original_height` | `0` | 保留封裝後的幾何資訊或上游預處理的元資料。|
| `original_width` / `original_height` | 必須是正整數 | 用於覆寫原始尺寸，以便進行座標映射。|
| 綁定模型的節點 `detection_threshold` | `0.0` | 繼承已解析的 Model 分數門檻。 |
| 綁定模型的節點 `detection_threshold` | `> 0.0` | 覆寫分數門檻。 |
| 綁定模型的節點 `nms_iou_threshold` | `0.0` | 繼承已解析的 Model NMS IoU。 |
| 綁定模型的節點 `nms_iou_threshold` | `> 0.0` | 覆寫 NMS IoU。 |
| 綁定模型的節點 `top_k` | `0` | 繼承已解析的 Model 輸出上限。 |
| 綁定模型的節點 `top_k` | `> 0` | 覆寫保留的最大偵測數量。 |

邊界框解碼需要解析出正值 `top_k`；零不代表要求無界的輸出儲存空間。
對於未綁定 `Model` 的原始輸出頭，請在建置圖前設定 `top_k`。
綁定模型的節點可將自身 `top_k` 保持為零，以繼承正值 Model 上限。
這些控制項不會改變獨立的結果讀取輔助函式：它們的 `top_k=0` 表示對已受限制的
輸出不再加設上限。零分數門檻允許零分數，但不會略過幾何驗證、背景排除、NMS 或輸出上限。
NMS IoU 為零時，YOLO26 略過 NMS。其他啟用 NMS 的 YOLO 與 SSD 設定組合仍會
套用量化 IoU 抑制。正值 YOLO26 閾值會啟用 NMS。

`num_classes` 會比較呼叫端設定的值與從 MPK 張量契約推導出的類別數量：

| 模型系列 | 設定的 `num_classes` | 從 MPK 推導的 `num_classes` | 行為 |
| --- | --- | --- | --- |
| 任何支援的模型 | `0` | 可推導的正值 | 使用從 MPK 推導的類別數量。 |
| 任何支援的模型 | 正整數 | 相同的值 | 使用設定的類別數量。 |
| 類別切分不明確的模型 | 正整數 | 無法取得 | 使用設定的類別數量。當單一類別標頭的切分無法可靠推導時，這是必要的。 |
| YOLOv5 或 YOLO26 | 正整數 | 不同的值 | 在建立管線之前失敗，並回報兩個值。這些原始標頭配置會從張量深度推導類別數量。 |
| SSD 或其他 YOLO26 之前的非姿態 YOLO 系列 | 正整數 | 不同的值 | 套用既有的系列專屬明確覆寫行為。姿態解碼器與 SuperPoint 保留其系列專屬規則。 |

`detection_threshold` 這是 BoxDecode 節點/階段建構函式所使用的名稱。 `ModelOptions.score_threshold` 是模型路由選項，它會將相同的控制訊號傳送至。

## 解碼類型對應表

| API 列舉 | 後端權杖 | 典型的模型系列 |
| --- | --- | --- |
| `BoxDecodeType::Yolo` | `yolo` | 通用 YOLO 樣式的檢測頭 |
| `BoxDecodeType::YoloV5` | `yolov5` | YOLOv5 偵測 |
| `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLOv5 分割 |
| `BoxDecodeType::YoloV7` | `yolov7` | YOLOv7 偵測 |
| `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLOv7 分割 |
| `BoxDecodeType::YoloV8` | `yolov8` | YOLOv8 偵測 |
| `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLOv8 分割 |
| `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLOv8 姿勢 |
| `BoxDecodeType::YoloV9` | `yolov9` | YOLOv9 偵測 |
| `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLOv9 分割 |
| `BoxDecodeType::YoloV10` | `yolov10` | YOLOv10 偵測 |
| `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLOv10 分割 |
| `BoxDecodeType::YoloV26` | `yolo26` | YOLO26 偵測 |
| `BoxDecodeType::YoloV26Pose` | `yolo26-pose` | YOLO26 姿勢 |
| `BoxDecodeType::YoloV26Seg` | `yolo26-seg` | YOLO26 分割 |
| `BoxDecodeType::YoloV26Obb` | `yolo26-obb` | YOLO26 旋轉物件偵測 (`BBOX_OBB_V1`) |
| `BoxDecodeType::YoloV6` | `yolov6` | YOLOv6 偵測 |
| `BoxDecodeType::YoloX` | `yolox` | YOLOX 偵測 |
| `BoxDecodeType::YoloXSegPose` | `yolox-seg-pose` | 同時包含邊界框、遮罩及關鍵點頭的 YOLOX 封裝匯出 |
| `BoxDecodeType::Ssd` | `ssd` | 精確選擇已準備好的 SSD300、SSD-Mobile-300、SSD-Mobile-320 或 SSDlite-Mobile-320 合約，並從已排序的磁碟頭幾何結構中選取。 |
| `BoxDecodeType::SuperPoint` | `superpoint` | SuperPoint 檢測器和描述符後處理 |
| `BoxDecodeType::Detr` | `detr` | DETR 樣式的變壓器檢測 |
| `BoxDecodeType::EffDet` | `effdet` | EfficientDet 偵測 |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | 五個有序 RPN 輸出頭；全域 NMS 後按面積層級套用 top-K |
| `BoxDecodeType::Centernet` | `centernet` | CenterNet 偵測 |

`BoxDecodeType::Unspecified` 是一個未設定的標記，會在執行階段之前就發生錯誤。SSD 處方識別碼是一個內部核心合約（`ssd300-v1`、`ssd-mobile-300-v1`、`ssd-mobile-320-v1` 或 `ssdlite-mobile-320-v1`），而不是另一個公開的解碼類型或後端標記。核心會在降低之前解析它，而已安裝的物件解碼器會繼續接收其支援的 `ssd` 系列標記，並從已經驗證的磁碟頭幾何形狀中選擇對應的固定實作。

## 選擇合適的類型

- 如果您使用的是 SiMa 提供的或 SiMa 編譯的模型套件，請選擇與模型系列相符的 `BoxDecodeType`，並將 `decode_type_option` 保持為 `Auto`。
- 如果您的偵測結果遺漏，或者所有分數都異常偏低，請先確認解碼模型是否與匯出的模型頭相符。YOLOX、YOLOv6 和 YOLO26 使用原始/邏輯樣式的模型頭，不應像僅使用機率值的 YOLO 模型頭一樣進行處理。
- 如果方塊的位置或大小比例顯示不正確，請檢查影像調整大小的策略。僅在您的圖表沒有上游 `Preproc` 階段來寫入調整大小的中繼資料時，才使用 `resize_mode_override`。
- 如果您正在製作自訂的模型套件，請務必確保封存檔準確地描述了檢測頭：張量順序、邏輯形狀、實際儲存方式、dtype/量化、分數範圍、類別數量，以及任何切片輸出。應用程式程式碼不應需要針對這些細節進行調整。

## 形狀與佈局指南

不同的偵測模型會使用不同的輸出層設定。有些模型會針對每個特徵圖層使用一個張量；有些模型則會將邊界框、物件性、類別、關鍵點或遮罩分割成不同的張量。有些模型的輸出是稠密的 HWC 張量；有些則是由編譯器/執行階段進行封裝或切片。

對於模型封裝流程，這由封裝的合約來處理。對於手動連接的張量，關鍵規則是：完全匹配匯出的輸出層格式。不要僅僅根據秩或通道數來選擇解碼類型。

進階張量合約規則：

- 除了 `YoloV5` 偵測以外的 YOLO 系列解碼類型（`Yolo`、`YoloV7`、`YoloV8`、`YoloV9`、`YoloV10`，以及分割／姿態的變體）預期使用分離式標頭，或與模型系列相符的封裝式標頭。
- 打包後的 YOLO 模型，其類別數量和模型深度必須在所有模型中保持一致。
  功能等級。
- `YoloV5` 偵測只接受三個依 P3/P4/P5 排序且尚未解碼的封裝式標頭。其網格必須具備 stride 8/16/32 的幾何結構，且每個邏輯深度都必須是 `3 * (num_classes + 5)`。BoxDecode 會套用 sigmoid、網格與 stride 轉換，以及標準的 YOLOv5 錨點（`{10,13},{16,30},{33,23}`、`{30,61},{62,45},{59,119}`、`{116,90},{156,198},{373,326}`）。自訂的 AutoAnchor 表格與已解碼的六張量框／類別匯出必須使用其他契約。
- `YoloV26` 使用分組的原始左/上/右/下邊界框預測頭，以及類別分數預測頭。
- `Ssd` 並**不是**一個通用的 SSD 解碼器。它僅能解析**四個預先設定的設定檔**。
  從編譯時完整的、依序排列的 loc/conf H/W/C 簽章開始。任何其他標頭集合或順序都會被拒絕，並顯示錯誤訊息，其中包含觀察到的和支援的簽章：
  - **SSD300**`dboxes300_coco`)：300×300 輸入，特徵圖
    `{38,19,10,5,3,1}`，每個單元格的先驗值 `{4,6,6,6,4,4}`，信賴度通道順序。
`class*A + anchor`，透過類別維度上的 **softmax** 函數計算類別分數（包含索引 0 的背景）。
  - **SSD-Mobile-300-v1** (`ssd_anchor_generator`)：300×300 輸入，特徵
    地圖 `{19,10,5,3,2,1}`、每個單元格的先驗值 `{3,6,6,6,6,6}`、置信度通道
    順序 `anchor*C + class`，透過每個類別的 **sigmoid** 函數計算類別分數（忽略背景）。
  - **SSD-Mobile-320-v1** (`ssd_anchor_generator`)：320×320 輸入，特徵
    地圖 `{20,10,5,3,2,1}`、每個單元格的先驗值 `{3,6,6,6,6,6}`、置信度通道
    順序 `anchor*C + class`，透過每個類別的 **sigmoid** 函數計算類別分數（忽略背景）。
  - **SSDlite-Mobile-320-v1**（TorchVision `DefaultBoxGenerator`）：320×320
    輸入、特徵圖 `{20,10,5,3,2,1}`，每個層級的每個單元都有六個先驗框，
    定位順序 `anchor*4 + {dx,dy,dw,dh}`，置信度順序
    `anchor*C + class`，以及通過對所有 91 個類別（包括背景）進行 **softmax** 運算得到的類別分數。

所有配方都使用分組的每層定位頭（深度 = `4 * priors-per-cell`），與類別-置信度頭（深度 = `num_classes * priors-per-cell`）配對，並使用 FasterRcnnBoxCoder 方差縮放（`scale_xy 0.1`、`scale_wh 0.2`）以及 **拉伸**（各向異性）預處理調整大小。分數激活由配方固定（與設備上的解碼器匹配），並且按角色分組的佈局是自動選擇的——將 `decode_type_option` 設置為 `Auto`。如果使用非分組佈局，則會被拒絕。

**模型框架是設定檔的一部分**，而不僅僅是頭幾何形狀。SSD300-v1 和 SSD-Mobile-300-v1 需要 300×300；兩個 320-v1 設定檔都需要 320×320。如果在建置時解析的預處理調整大小目標或模型尺寸覆蓋為任何其他大小，則會被拒絕，因為先驗表和拉伸反投影僅在該框架下有效。

原始/獨立的 `SimaBoxDecode` 建置永遠不會發明調整大小模式。保留上游 `Preproc` 元資料要求，或者使用顯式原始重載來斷言外部執行的 `ResizeMode::Stretch`；Letterbox 和 Crop 將被拒絕。

**`num_classes` 協議。**編碼的類別數量始終從置信度頭的深度（`conf_depth / priors-per-cell`，包括索引 0 處的背景）推導得出。SSD300-v1 允許使用連續的前綴選擇，例如準備好的 81 到 8 的路徑；其他三個設定檔需要準確的編碼計數。如果選擇無效，則會在建置時被拒絕。如果要使用設定檔的預設值，請將其設置為未設定。
- `Detr` 會根據最大頭部深度推斷出類別通道，並且需要一個有效的。
  類別：維度。
- `EffDet` 與 `Centernet` 使用各自模型系列的契約；不要將它們路由至 YOLO 解碼類型。
- `*-seg` 解碼類型會產生帶有框線的輸出，以及特定任務的遮罩資料。

如果自訂模型套件與任何完整的排序簽章都不相符，請準備一個新的、明確支援的設定檔，而不是降低比對器的準確度。

## Python 筆記

在從 Python 設定模型選項時，如果有的話，請使用類型化的列舉值，而不是字串：

```python
opt = pyneat.ModelOptions()
opt.decode_type = pyneat.BoxDecodeType.YoloV8
```

使用與模型任務相符的輔助工具來解析輸出：

```python
outputs = model.run([image])

boxes = pyneat.decode_bbox(outputs)[0].to_numpy()

pose = pyneat.decode_pose(outputs)[0]
pose_boxes = pose.boxes.to_numpy()
keypoints = pose.keypoints.to_numpy()

seg = pyneat.decode_segmentation(outputs)[0]
seg_boxes = seg.boxes.to_numpy()
masks = seg.masks.to_numpy()
```

## 升級

新選項會改變 C++ 物件佈局。Core 使用 ABI 6、`libsima_neat.so.6`。請使用相符的標頭與函式庫，重新建置 C++ 應用程式、外掛程式及 Python 繫結。請勿透過相容性符號連結將 ABI 5 二進位檔連接至 ABI 6。


### 組合 YOLOX 輸入准入

`YoloXSegPose` 接受 BF16 與 INT8 輸出頭，或儲存在連續或正值、不重疊步幅空間中的
原生 FP32 輸出頭（包括平面式 CHW 檢視）。FP32 位址與步幅必須四位元組對齊；
不接受通道分塊 FP32 與 UINT8。不會加入隱含、有損的 FP32 轉換。
原型具有自己的 dtype 與逐張量 INT8 量化；每個偵測輸出頭的角色仍須採用一致 dtype。
Core 保留這些逐埠資訊供 Kernels 准入使用，而不會將第一個輸出頭的 dtype 或量化
套用到所有埠。三個偵測層級在兩軸的步幅分別為 8、16 與 32。
模型寬度與高度必須是 32 的倍數，但不必相等。原型維持 160×160×32，
不會隨模型尺寸暗中調整大小。

偏心填補或裁切座標請使用前處理中繼資料。仿射轉換將**模型映射到原始**座標；
此組合設定不支援剪切、旋轉與 ROI 清單。裁切在準備階段需要權威性的仿射中繼資料。
這是針對外部已前處理張量的解碼器契約，不是替 `Model` 前處理新增裁切產生功能。
少於 17 個關鍵點的模型仍發布固定的 17 槽位公開線路格式，未使用槽位設為零。
