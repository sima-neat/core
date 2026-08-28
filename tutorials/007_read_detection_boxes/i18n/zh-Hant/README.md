# 007 從模型輸出讀取檢測框

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | yolo_v8s |
| Labels | postprocessing, boxdecode, detection |

## Concept

使用 `SimaBoxDecode` 將原始模型輸出解碼為可用的邊界框——包含閾值處理、NMS（非最大值抑制）和座標映射，這些都整合到一個後處理階段中——然後將結果讀取為已解析的框或原始的打包位元組緩衝區。

## Walkthrough

偵測器不會直接輸出框。它的原始輸出是一堆特徵圖，在產生任何意義之前，仍然需要進行閾值處理、非最大值抑制和座標映射。`SimaBoxDecode` 是後處理階段，它將這三個步驟整合到一個最佳化的步驟中，將推理張量轉換為源圖像像素中的最終檢測結果。

本章設定了解碼器——選擇模型系列，使用 `decode_type`，使用分數閾值控制置信度，使用 NMS IoU 閾值抑制重疊，並使用 `top_k` 限制輸出——然後執行模型並讀取傳回的檢測數量。到最後，您將擁有一個設定好的偵測器管線，以及從其輸出讀取的檢測數量，此外（在下面的「實務參考」中），還包含完整的線格式，以便您可以在任何執行階段自行解析框。

### 設定解碼器 {#step-configure-decode}

這些選項同時設定輸入合約和後處理行為。`decode_type`（這裡為 `YoloV8`）選擇模型系列的解碼路徑。置信度閾值會在 NMS 之前剔除較弱的候選者；NMS IoU 閾值控制重疊框的合併程度；`top_k` 限制最終數量，以實現確定性的下游成本；`boxdecode_original_width`/`boxdecode_original_height` 將解碼後的座標映射回源圖像像素。關於這些參數的調整指南，請參閱下面的「實務參考」。

**C++：** `decode_type` 採用 `BoxDecodeType::YoloV8` 列舉。閾值/NMS/`top_k` 值稍後通過 `stages::BoxDecodeOptions` 傳遞，而不是通過 `Model::Options`。

**Python：** `decode_type` 採用 `pyneat.BoxDecodeType.YoloV8` 列舉，並且 `score_threshold`、`nms_iou_threshold`、`top_k` 和 `boxdecode_original_width`/`boxdecode_original_height` 直接設定在 `ModelOptions` 上。（`score_threshold` 和 C++ 中的 `detection_threshold` 名稱相同，控制相同的內容——請參閱「實務參考」中的命名說明。）

### 建置模型 {#step-load-model}

從檔案加上選項建置 `Model`，將解碼設定綁定到模型，以便從中衍生的推理和後處理階段使用上述設定。

### 執行預處理、推理和解碼 {#step-run-decode}

此時，一個幀將通過預處理、MLA 推理和框解碼器，以產生檢測輸出。

**C++：**路徑以分階段的方式明確定義：`stages::Preproc` 產生輸入張量，`stages::Infer` 執行模型，以及一個 `stages::BoxDecodeOptions`（包含 `detection_threshold = 0.55`、`nms_iou_threshold = 0.5`、`top_k = 100`）設定了解碼，然後執行。

**Python：**`model.run([tensor])` 在單次呼叫中執行整個已設定的路徑，並傳回一個 `TensorList`。當 BoxDecode 連接到模型路徑時，第一個張量是封裝後的 `BBOX` 輸出。

### 讀取框 {#step-read-boxes}

最後，將解碼輸出轉換為您可以使用的內容。

**C++：**`stages::BoxDecodeResults(...)` 傳回一個 `BoxDecodeResultList`；前一個結果的 `boxes` 向量已經解析為 `{x1, y1, x2, y2, score, class_id}`，並限制到來源像素，因此 `decoded.boxes.size()` 是檢測計數。

**Python：**結果是一個單一的 `BBOX` `uint8` 張量，位於 `outputs[0]` 中。前四個小端字節是檢測計數（`struct.unpack_from("<I", buf, 0)`）；完整的記錄佈局在「實際操作」中記錄。如果執行階段未將 BoxDecode 連接到 `model.run`，則傳回的 `TensorList` 包含原始的特徵圖標頭。

## In Practice

`SimaBoxDecode` 輸出一個帶有標籤 `BBOX` 的單一輸出張量。該張量攜帶一個封裝的字節緩衝區，執行階段解析器將其解釋為浮點檢測。理解這兩個層級的協定（線路緩衝區與解析後的 `Box` 記錄）是從 Python 或 C++ 讀取輸出的關鍵。

### BBOX 張量

解碼階段為每個輸入幀產生一個 `BBOX` 張量，具有以下內容：

| 欄位 | 值 |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | rank-1：`[N_bytes]`，其中 `N_bytes` 是模型封存檔封裝的緩衝區容量（例如，標準 YOLOv8 封裝中的 `[20160]`） |

張量形狀是一個**字節計數**，而不是檢測計數。封裝的字節包含一個小標頭和一個連續的固定大小框記錄陣列。`N_bytes` 由模型封存檔的 `buffers.input[0].size` 欄位確定（位於框解碼階段的設定 JSON 中），並且限制了解碼器在單個幀中可以輸出的最大檢測數量（有關執行階段維度如何與封裝值交互，請參閱下面的「覆寫協定」）。

### 封裝的線路格式

`uint8` 緩衝區以小端格式佈局：

```
offset  size  content
------  ----  -------
  0      4    uint32  N = number of valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   (trailing bytes up to buffer capacity are padding, ignored)
```

每個 `RawBox` 記錄長度為 24 位元組：

| 記錄中的位移量 | 大小 | 類型 | 欄位 | 意義 |
|---|---|---|---|---|
| 0 | 4 | int32 | `x`     | 來源影像中，左上角 x 座標（以像素為單位）|
| 4 | 4 | int32 | `y`     | 來源影像中，左上角 y 座標（以像素為單位）|
| 8 | 4 | int32 | `w`     | 寬度，以原始像素為單位 |
| 12 | 4 | int32 | `h`     | 原始影像中的高度（以像素為單位）|
| 16 | 4 | float32 | `score` | 非最大值抑制 (NMS) 後的檢測置信度 `[0.0, 1.0]` （數值 `detection_threshold` 閘門開啟) |
| 20 | 4 | int32 | `class_id` | 預測的類別 ID（模型定義；從 0 開始編號；類別名稱對應表位於模型封存檔的元資料中）|

標準的 Python `struct` 格式符合單一記錄的是 `"<iiiifi"`
（小端序，4 個帶號整數，一個浮點數，一個帶號整數）。

執行階段的解析輔助函式（`parse_bbox_bytes` /
`decode_bbox_tensor` 在 `include/pipeline/DetectionTypes.h`,
`tests/unit_testing/unit_detection_types_bbox_test.cpp` 將電線合約固定下來)
擴展每個 `RawBox` 變成 `Box` 用於後續程式碼的結構：

```cpp
struct Box {
  float x1, y1, x2, y2;  // x2 = x + w, y2 = y + h; clamped to [0, img_w|h]
  float score;
  int   class_id;
};
```

### 座標空間

從中解碼出的座標為 `BBOX` 位於「原始影像像素」中，與您傳遞的座標系統相同。 `original_width` / `original_height` （或模型封存檔是與之一起封裝的）。它們並未經過標準化處理。 `[0, 1]`，而且它們並未以模型內部矩形輸入空間的形式呈現。語法分析器會限制。 `(x1, y1, x2, y2)` 到 `[0, original_width]` / `[0, original_height]`
因此，呼叫程式碼可以直接在原始框架上繪製這些內容。

### 實際範例

使用教學的執行階段設定（`original_width = 640`,
`original_height = 640`, `top_k = 100`) 以及預設的 YOLOv8 模型套件`buffers.input[0].size = 20160` 在「boxdecode config」中，單一解碼後的影格會產生：

- `out.kind == SampleKind.Tensor`
- `out.payload_tag == "BBOX"`
- `out.tensor.dtype == UInt8`, `out.tensor.shape == [20160]`
- 位元組 `[0:4]` 給 `N` 以小端格式； `0 <= N <= 100` 因為
  `top_k = 100`。 `N` 的 `0` 表示「此影格中沒有超過閾值的檢測結果」——迴圈執行零次，不產生任何輸出。
- 位元組 `[4 : 4 + 24 * N]` 保留有效的檢測結果；之後的所有內容
  偏移量為零/填充，必須忽略。

在 Python 中讀取一個框，這是一個 `struct.unpack_from`:

```python
import struct
payload = out.tensor.copy_payload_bytes()
count = struct.unpack_from("<I", payload, 0)[0]
for i in range(count):
    x, y, w, h, score, cls = struct.unpack_from("<iiiifi", payload, 4 + 24 * i)
    # (x, y, w, h) in source pixels; x2 = x + w, y2 = y + h
```

在 C++ 中，`stages::BoxDecode` 輔助函式會傳回一個 `BoxDecodeResult`，它已經為您完成了這個解包操作：`result.boxes[i]` 是一個 `Box`，其中 `(x1, y1, x2, y2)` 已經從 `(x, y, x+w, y+h)` 取得，並且已限制在圖像範圍內。

### 覆寫合約：執行階段尺寸與封裝模型封存檔的預設值

`SimaBoxDecode` 是從一個經過訓練的模型封存檔建構而成，該模型封存檔包含 `decode_type`、`detection_threshold`、`nms_iou_threshold`、`top_k`、`original_width` 和 `original_height` 的封裝預設值。 公開建構函式為：

```cpp
SimaBoxDecode(const Model& model,
              const std::string& decode_type = "",
              int original_width = 0, int original_height = 0,
              double detection_threshold = 0.0,
              double nms_iou_threshold = 0.0,
              int top_k = 0);
```

以及其 Python 版本 `pyneat.nodes.sima_box_decode(model, ...)`，每個欄位都使用一個簡單的「正值覆寫，零/空值保留」規則。

> **命名注意事項。** `detection_threshold` 是 `SimaBoxDecode` 建構函式使用的名稱。`ModelOptions.score_threshold`（用於 Python 教程中）會傳遞到同一個參數。這兩個名稱指的是相同的底層控制項。

| 執行階段參數 | 傳遞的值 | 行為 |
|---|---|---|
| `decode_type` | `""`（空） | 保留模型封存檔 / 模型路徑推論 |
| `decode_type` | 非空字串 | 覆寫此執行次數的模型封存檔值 |
| `original_width` / `original_height` | `0` | 保留模型封存檔封裝的維度 |
| `original_width` / `original_height` | 正整數 | 覆寫有效設定中的 `original_width` / `original_height` |
| `detection_threshold` | `0.0` | 保留模型封存檔封裝的閾值 |
| `detection_threshold` | `> 0.0` | 覆寫（同時觸發下面的 YOLOv8 臨界警告） |
| `nms_iou_threshold` | `0.0` | 保留模型封存檔封裝的 NMS IoU |
| `nms_iou_threshold` | `> 0.0` | 覆寫 |
| `top_k` | `0` | 保留模型封存檔封裝的 top-K |
| `top_k` | `> 0` | 覆寫 |

該規則嚴格適用於每個欄位：

- **Python 路徑** — 教程覆寫了每個欄位，因為 `ModelOptions` 設定了正值。
- **C++ 路徑** — `read_detection_boxes.cpp` 傳遞了 `0.55f, 0.5f, 100`（因此 `detection_threshold`、`nms_iou_threshold` 和 `top_k` 均被覆寫）以及 `bgr.cols, bgr.rows`（因此 `original_width` / `original_height` 也被覆寫）。

實際影響：

- 如果您的模型封存檔是針對與您的原始幀不同的解析度進行封裝的，請明確傳遞 `original_width` 和 `original_height`，以便坐標定位在原始像素中。
- 將 `detection_threshold` 和 `nms_iou_threshold` 保持在 `0.0` 是獲得模型封存檔驗證的預設值的最安全方法；只有在您有意重新調整時才覆寫。
- 明智地使用較低的 `detection_threshold`。 閾值越低，通過閾值處理的候選框越多，NMS 的成本隨著存活框的數量呈平方增長——因此，非常低的閾值會顯著增加後處理計算量和延遲。 僅將其降低到您需要檢測到弱檢測的程度；將其與 `top_k` 搭配使用，以限制最壞情況。

### 解碼類型和張量合約

`BoxDecodeType` 是一種類型化的 API（`simaai::neat::BoxDecodeType` / `neat.BoxDecodeType`），並且在解碼階段應始終明確設定。以下執行階段合約來自 `internals/gst_plugins/genericboxdecode_v2/gstneatboxdecode.cpp`（`infer_num_classes`、`infer_yolo_decoupled_classes`、`infer_yolo_packed_classes`、`compute_required_output_size`）。

核心張量合約規則：
- YOLO 系列的解碼類型（`yolo`、`yolov5*`、`yolov7*`、`yolov8*`、`yolov9*`、`yolov10*`）：
  - 分離式標頭：類別標頭的深度必須可重複，且為 `> 4`。
  - 封裝式標頭：每個標頭的深度都必須滿足 `depth = 3 * (num_classes + 5)`，並且在所有標頭中保持一致。
- `yolo26`：具有 4 個通道的原始 l/t/r/b 邊界框張量和可重複的類別標頭深度的分離式分組標頭，深度為 `> 4`。
- `detr`：類別通道是從所有標頭中的最大深度推斷得出，並且必須為 `> 4`。
- 其他非 YOLO 解碼類型（`effdet`、`rcnn-stage1`、`centernet`）：回退類別推斷使用最大深度，並且需要 `> 4`。
- 分段解碼標記（`*-seg`）可在 v2 中啟用類似分段的輸出大小調整（為每個檢測添加遮罩有效載荷）。

| API 列舉 | 後端權杖 | 預期合約 |
|---|---|---| `BoxDecodeType::Yolo` | `yolo` | YOLO 解耦或打包的深度卷積 | `BoxDecodeType::YoloV5` | `yolov5` | YOLO 解耦或打包的深度卷積 | `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLO 深度卷積 + 分割路徑 | `BoxDecodeType::YoloV7` | `yolov7` | YOLO 解耦或打包的深度卷積 | `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLO 深度卷積 + 分割路徑 | `BoxDecodeType::YoloV8` | `yolov8` | YOLO 解耦或打包的深度卷積 | `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLO 深度卷積 + 分割路徑 | `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLO 解耦或打包的深度卷積 | `BoxDecodeType::YoloV9` | `yolov9` | YOLO 解耦或打包的深度卷積 | `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLO 深度卷積 + 分割路徑 | `BoxDecodeType::YoloV10` | `yolov10` | YOLO 解耦或打包的深度卷積 | `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLO 深度卷積 + 分割路徑 | `BoxDecodeType::YoloV26` | `yolo26` | YOLO26 分組的原始長寬高/上下左右邊界框預測結果 + 類別分數預測結果 | `BoxDecodeType::Detr` | `detr` | `num_classes = max(depth)` （必須是 `> 4`) |
| `BoxDecodeType::EffDet` | `effdet` | 備援最大深度推論`> 4`) |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | 備援最大深度推論`> 4`) |
| `BoxDecodeType::Centernet` | `centernet` | 備援最大深度推論`> 4`) |

快速失敗機制：
- `stages::BoxDecodeOptions` 需要以解碼類型明確建構。
- `stages::BoxDecode(...)` 和 `nodes::SimaBoxDecode(...)` 遇到 `BoxDecodeType::Unspecified` 時會快速失敗。

明確設定解碼類型：

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.5;
opt.top_k = 100;
```

```python
opt = neat.ModelOptions()
opt.decode_type = neat.BoxDecodeType.YoloV8
```

## Run

從以下位置執行 **Python** 和 **C++（預先建置）** 指令：Neat 安裝根目錄（包含 `share/` 以及 `lib/`）；從**原始碼**開始執行**建置**指令，指令應從**儲存庫的根目錄**執行。

**Python:**
```bash
python3 share/sima-neat/tutorials/007_read_detection_boxes/read_detection_boxes.py \
  --model /tmp/yolo_v8s.tar.gz --width 640 --height 640
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_007_read_detection_boxes \
  --model /tmp/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_007_read_detection_boxes
./build/tutorials-standalone/tutorial_007_read_detection_boxes \
  --model /tmp/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

預期輸出（方塊數量取決於框架；合成框架會產生零個方塊）：

```text
boxes=0
[OK] 007_read_detection_boxes
```

（Python 建構會輸出 `detections=...`，或者在執行階段如果沒有將 BoxDecode 連接到 `model.run`，則輸出 `raw_output_heads=...`。）若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## 原始程式碼檔案
- C++：`tutorials/007_read_detection_boxes/read_detection_boxes.cpp`
- Python：`tutorials/007_read_detection_boxes/read_detection_boxes.py`
