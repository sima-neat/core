# 023 執行 MIPI 相機模型

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | User-provided camera-compatible model |
| Labels | mipi, camera, live-input, model, ev74 |

## Concept

將 Modalix DevKit MIPI 相機連接到一個包含 `CameraInput` 的 `Graph`，將即時的 `NV12` 幀饋送到模型管理的預處理程序中，並提取模型輸出。這是已部署的視覺應用程式的直接相機路徑：相機擁有原始資料，CVU/EV74 處理圖像預處理，MLA 執行推論，應用程式則使用結果。

## Walkthrough

本章假設相機已透過板載疊加層和 libcamera 正常運作。Neat 不會選擇 `.dtbo` 檔案或調整 ISP；它會在 `libcamerasrc` 能夠產生影像時，開始處理這些影像。在執行教學之前，請使用 [硬體 MIPI 指南](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) 和 GStreamer caps 檢查來驗證相機。

將本教學視為第二個關卡。第一個關卡是相機啟動：疊加層、驅動程式、libcamera、ISP 和精確的 caps 設定。第二個關卡是 Neat 圖：將相機影像輸入到 CVU 預處理、MLA 推論、可選的 EV74 BoxDecode，以及輸出提取。

### 設定相機來源 {#step-configure-camera}

`CameraInputOptions` 描述了 Neat 從 `libcamerasrc` 請求的來源 caps：解析度、幀率、格式以及一個可選的 libcamera 相機名稱。對於尚未公開 SiMaAI 零拷貝緩衝區的目前相機堆疊，請設定 `allow_cpu_fallback = true`。當您的 `libcamerasrc` 支援時，仍可透過 `--strict-zero-copy` 使用嚴格的零拷貝。

### 設定模型路徑 {#step-configure-model}

模型將相機影像視為 `NV12` 影像。設定模型管理的預處理，用於色彩轉換、調整大小、正規化、量化和鑲嵌。範例將模型管理的 CVU 預處理固定到 `EV74`，以確保生產圖不會悄悄地變成 CPU 影像管線。使用 `--decode none` 時，路徑將在 MLA 處終止，並傳回原始模型張量。使用 YOLO `--decode` 標記時，BoxDecode 將作為模型管理的 EV74 後處理階段執行。

### 組合來源擁有的圖 {#step-compose-graph}

首先新增 `CameraInput`，然後使用 `include_input = false` 新增模型路徑。由於影像源自正在執行的管線內部，因此沒有公開的 `Input` 節點。`include_output = true` 會保留一個提取端點，用於檢測或張量。

### 提取輸出 {#step-pull-output}

建立圖並提取固定數量的輸出。逾時表示在 `--pull-timeout-ms` 之前，沒有模型輸出到達應用程式；相機可能已停止、caps 可能未協商成功，或者下游階段（例如 BoxDecode）可能受到反壓。列印張量計數和第一個張量的形狀，以便在新增應用程式邏輯之前，確認資料正在傳輸。

## Run

直接在已設定 MIPI 攝影機的 Modalix DevKit 上執行此教學。從 Neat 安裝目錄執行預先建置的指令；從程式碼庫的根目錄執行從原始碼建置的指令。模型封存檔必須與您要求的預處理和選擇性 `--decode` 模式相符。

預設的拉取逾時時間為 15 秒。當您在全新硬體上收集首次執行診斷資訊時，請增加 `--pull-timeout-ms`。

**Python:**
<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/model.tar.gz --frames 5 --decode none
</ShellCommand>

**C++ (prebuilt):**
<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /路徑/到/model.tar.gz --frames 5 --decode none
</ShellCommand>

對於支援「BoxDecode」路徑的 YOLO 樣式模型，請選擇一個解碼標記，例如 `yolov8` 或 `yolov9seg`：

<ShellCommand prompt="devkit">
``
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/yolo.tar.gz --frames 5 --decode yolov8
``
</ShellCommand>

<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /path/to/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

**C++ (build from source):**
<ShellCommand prompt="devkit">
./build.sh --target tutorial_023_run_mipi_camera_model
</ShellCommand>

<ShellCommand prompt="devkit">
./build/tutorials-standalone/tutorial_023_run_mipi_camera_model \
  --model /路徑/到/model.tar.gz --frames 5 --decode none
</ShellCommand>

預期的輸出形狀取決於模型和解碼路徑。原始 MLA 輸出通常包含模型特定的張量：

```text
frame=0 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=1 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=2 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=3 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=4 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
[OK] 023_run_mipi_camera_model
```

如果使用受支援的 BoxDecode 路徑，則輸出會變更為解碼後的檢測或分割張量。請使用張量數量和第一個形狀作為移動檢查，而不是作為通用的合約。

如果您看到 `output_timeout`，請使用 `gst-launch-1.0` 驗證相機，然後使用 `--print-backend` 檢查產生的後端。對於 BoxDecode 路徑，請確認模型封存檔、`--decode` 標記以及閾值是否與模型相符。

## In Practice

當您需要檢查產生的 GStreamer 路徑時，請使用 `--print-backend`。 產生的路徑應包含 `libcamerasrc`、`neatcamerabridge`（當啟用回退時）、`neatprocesscvu`、`neatprocessmla`、可選的 EV74 後處理，以及 `appsink`。 除非您有意新增僅用於除錯的路徑，否則它不應包含 `appsrc`、`ostosima`、`videoconvert` 或 `videoscale`。

## 原始檔案
- C++：`tutorials/023_run_mipi_camera_model/run_mipi_camera_model.cpp`
- Python：`tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py`
