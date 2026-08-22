# 005 設定模型選項

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | yolo_v8s |
| Labels | model-options, configuration, contracts |

## Concept

`ModelOptions` 是一個結構體，用於定義您的輸入資料、模型的管線階段以及其輸出解碼之間的協定。當您想要超越預設行為時，這是您首先需要設定的地方。

## Walkthrough

第 001 章載入了一個具有合理預設值的模型。實際模型（尤其是像 YOLOv8 這樣的檢測模型）需要您*宣告*它們的合約：輸入的像素格式和大小是什麼，應該如何進行正規化，以及原始網路輸出如何轉換為過濾後的框。`ModelOptions` 將所有這些內容分組到一個單一的結構中，您可以在建構之前填寫該結構。

本章設定了一個端到端的 YOLOv8 模型，然後檢查執行階段從這些選項中解析出的合約。到本章結束時，您將設定輸入、預處理和後處理參數，讀取解析後的 `input_specs()`/`output_specs()`/`metadata`，並通過設定的模型運行一個確定性的幀。

### 宣告輸入和預處理 {#step-set-input-preproc}

第一個區塊描述了幀的外觀以及如何為網路準備它。`format`（這裡為 `BGR`）以及 `input_max_width`/`height`/`depth` 邊界設定了輸入合約，執行階段會驗證這些合約並為其分配緩衝區。正規化欄位提供了模型訓練時使用的每個通道的平均值和標準差，因此原始像素會縮放到網路預期的範圍內。

**C++：** 欄位位於 `opt.preprocess.*` 下：`kind = InputKind::Image`、`color_convert.input_format = PreprocessColorFormat::BGR` 和 `normalize.enable = AutoFlag::On`，其中 `mean`/`stddev` 為 `std::array<float, 3>`。

**Python：** 欄位位於 `opt.preprocess.*` 下：`kind = pyneat.InputKind.Image`、`color_convert.input_format = pyneat.PreprocessColorFormat.BGR` 和 `normalize.enable = pyneat.AutoFlag.On`，其中 `mean`/`stddev` 為列表。

### 宣告後處理 {#step-set-postproc}

第二個區塊定義了檢測器的輸出。`decode_type` 選擇了 YOLOv8 框解碼路徑，`score_threshold`、`nms_iou_threshold` 和 `top_k` 過濾了原始檢測結果——捨棄了低置信度的框，合併了重疊的框，並限制了存留的框的數量。`boxdecode_original_width`/`boxdecode_original_height` 為解碼器提供了源幀幾何資訊，以便將正規化的坐標映射回像素，`name_suffix` 使生成的階段名稱保持穩定，以便在與其他階段組合時，圖（graph）保持可讀性。

**C++：** `decode_type = BoxDecodeType::YoloV8`；幾何欄位為 `boxdecode_original_width`/`boxdecode_original_height`。

**Python：** `decode_type = pyneat.BoxDecodeType.YoloV8`；幾何欄位為 `boxdecode_original_width`/`boxdecode_original_height`。

### 載入並檢查已解析的合約 {#step-load-and-inspect}

使用這些選項建構 `Model`，即可將合約與封存檔進行比對。然後，我們將其讀取回來：`input_specs()` 和 `output_specs()` 報告協商後的張量約束，而 `metadata()` 則顯示封存在封存檔中的關鍵/值合約。在載入後檢查這些內容，可以確認執行階段已接受您的選項，並告訴您將要處理的具體形狀。

**C++：** 規格是 `TensorConstraint` 值；我們印出具體的形狀。

**Python：** 我們印出 `input_specs()[0]` 和 `output_specs()[0]` 中的形狀，以及 `len(model.metadata())`。

### 執行一個影格 {#step-run-inference}

最後，我們合成一個 `640×640` 的 BGR 影格，並將其傳遞到已設定的模型中，以確認整個合約從頭到尾都能正確執行，並印出傳回的輸出數量。

**C++：** 影格是一個 `cv::Mat`；`run()` 傳回一個 `TensorList`，我們印出其 `size()` 作為 `outputs=`。

**Python：** 影格被封裝為一個 `Tensor`，透過 `Tensor.from_numpy(...)`；`run()` 傳回一個 `TensorList`，因此我們印出其長度。

## Run

執行它，您應該會看到已解析的規格形狀、元資料鍵的數量以及輸出總計。從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預建版本）** 命令；從 **程式碼庫根目錄** 執行 **從原始碼建置** 命令。

**Python:**
```bash
python3 share/sima-neat/tutorials/005_configure_model_options/configure_model_options.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_005_configure_model_options
./build/tutorials-standalone/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

預期輸出（形狀和鍵的數量取決於模型封存檔；C++ 建置會印出詳細的規格行和 `outputs=`，Python 建置會印出形狀和 `output_count=`）：

```text
input_specs[0]: shape=[640,640,3]
output_specs[0]: shape=[]
metadata_keys=8
outputs=1
[OK] 005_configure_model_options
```

若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

### 詳細程度預設值

框架建置/運行的訊息輸出，透過 `VerboseOptions` 在 `GraphOptions`、`Model::Options` 和 `Model::RouteOptions` 上進行控制。

目前的開發預設值：`VerboseOptions::debug_all()`。當您希望減少輸出時，請明確調用 `production()` 或 `quiet()`。

| 預設值 | 預期用途 |
|---|---|
| `VerboseOptions::quiet()` | 抑制框架進度與詳細輸出。 |
| `VerboseOptions::production()` | 僅顯示乾淨階段的進度。 |
| `VerboseOptions::debug_plugins()` | 保留生產環境下的使用者體驗，同時也顯示外掛程式和 GStreamer 相關資訊。 |
| `VerboseOptions::debug_all()` | 強制在所有主題上進行完整的詳細程度掃描。 |

若要進行執行階段佇列/吞吐量調整，請參閱 [調整吞吐量和佇列深度](/tutorials/tune-throughput-and-queues)。

## 原始程式碼檔案
- C++：`tutorials/005_configure_model_options/configure_model_options.cpp`
- Python：`tutorials/005_configure_model_options/configure_model_options.py`
