# 026 執行多個模型

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | resnet_50, yolo_v8s |
| Labels | PCIe, queues, concurrency, classification, detection |

## Concept

Modalix PCIe 卡提供 0 到 5 號協同處理佇列。每個啟用的 `Model` 都會擁有一個佇列，因此可以透過將不同的 `ConnectionOptions.queue` 指派給每個實例，來讓獨立的模型同時執行。本教學使用 ResNet-50 於 0 號佇列，以及 YOLOv8s 於 1 號佇列，且不新增全域協調器。

## Walkthrough

這兩個模型有意使用不同的影像：ResNet-50 會分類一張清晰的拉布拉多犬照片，而 YOLOv8s 則會偵測繁忙街道場景中的人和汽車。

### 載入特定模型的影像 {#step-load-assets}

在佔用佇列之前，請驗證兩個模型封存檔，並解碼兩個封裝的資源。將影像分開可以使每個結果都具有意義，並避免將分類人像作為物件偵測工作負載使用。

### 將一個模型指派給每個佇列 {#step-assign-queues}

建立兩個普通的 `Model` 物件。將 ResNet-50 設定為使用 ImageNet 影像預處理於 0 號佇列，並將 YOLOv8s 設定為使用 COCO 影像預處理加上框解碼於 1 號佇列。建置錯誤會識別失敗的佇列和模型；如果第二次建置失敗，則會關閉已建置的模型。

### 同時執行兩個佇列 {#step-run-concurrently}

在個別主機執行緒中，為每個模型啟動一個阻塞影像推論。每個呼叫仍然使用簡單的同步 `run` 行為，但由於它們針對不同的實體佇列，因此這些呼叫會重疊。

### 獨立地解譯每個結果 {#step-read-results}

0 號佇列會傳回一個 FP32 分類張量，並列印其分數最高的 ImageNet 類別。1 號佇列會傳回已解碼的 BBOX 記錄，並列印偵測類別、信賴度以及來源影像座標。關閉任一模型只會釋放其指派的佇列。

## Run

如 [教學設定](/tutorials/before-you-run) 中所述，請安裝 PCIe 主機套件，並下載教學範例。從解壓縮後的 PCIe 額外檔案根目錄中，下載這兩個模型：

```bash
sima-cli models download --id resnet_50 --variant w8a8 --force
unzip -p models/resnet_50/w8a8/resnet_50_w8a8_default.zip resnet_50_w8a8_mpk.tar.gz > resnet_50_mpk.tar.gz
sima-cli models download --id yolov8s --variant w8a8 --force
unzip -p models/yolov8s/w8a8/yolov8s_w8a8_default.zip yolov8s_w8a8_mpk.tar.gz > yolo_v8s_mpk.tar.gz
```

該程式需要此目錄中確切的路徑 `resnet_50_mpk.tar.gz` 和 `yolo_v8s_mpk.tar.gz`。每個 `unzip` 命令會從其下載的 ZIP 中解壓縮其中一個檔案。

在 Windows 上，請依照[在 Windows 上執行應用程式](/getting-started/neat-library/pcie-host/#run-applications-on-windows)的說明設定教學課程並執行程式。請在 PowerShell 中下載模型，並使用 `Expand-Archive` 解壓縮封存檔：

```powershell
sima-cli models download --id resnet_50 --variant w8a8 --force
Expand-Archive models\resnet_50\w8a8\resnet_50_w8a8_default.zip -DestinationPath models\resnet_50\w8a8 -Force
Copy-Item models\resnet_50\w8a8\resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
sima-cli models download --id yolov8s --variant w8a8 --force
Expand-Archive models\yolov8s\w8a8\yolov8s_w8a8_default.zip -DestinationPath models\yolov8s\w8a8 -Force
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

執行 Python：

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/026_run_multiple_models/run_multiple_models.py
```

執行預先建置的 C++ 教程：

```bash
./lib/sima-pcie-host/tutorials/tutorial_026_run_multiple_models
```

或者重新建置：

```bash
./build.sh --target tutorial_026_run_multiple_models
./build/tutorials-standalone/tutorial_026_run_multiple_models
```

在 Windows 上，請在您的教學課程副本中，從 **x64 Native Tools Command Prompt** 使用 CMake 重新建置。將 `C:\opencv\build` 替換為[在 Windows 上執行應用程式](/getting-started/neat-library/pcie-host/#run-applications-on-windows)中所述 OpenCV 套件的 `build` 目錄：

```bat
call "%ProgramFiles%\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_026_run_multiple_models
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_026_run_multiple_models.exe
```

使用提供的模型和素材，兩個版本都會輸出類似的結果：

```text
queue=0 model=resnet_50 output_shape=[1, 1000] top1=208 (Labrador retriever)
queue=1 model=yolo_v8s detections=...
  person score=... box=(...)
[OK] 026_run_multiple_models
```

本教程有意將 ResNet-50 固定到佇列 0，將 YOLOv8s 固定到佇列 1。
僅在使用其他卡時才傳遞 `--card N`。

## In Practice

佇列分配是一種應用程式資源決策：兩個正在運行的模型不能擁有相同的物理佇列。在開始工作之前建置模型，在發生錯誤時報告特定的佇列，並在正常和錯誤路徑中關閉所有成功建置的模型。分離的 `Model` 實例可將結果和錯誤隔離，同時保持易於理解。

對於部署診斷，請繼續使用
[PCIe 模型工作流程](/develop-apps/development-workflow/pcie-model/) 和
[疑難排解指南](/reference/troubleshooting/)。

## 原始檔案

- `run_multiple_models.cpp`
- `run_multiple_models.py`
- `../assets/labrador.jpg`
- `../assets/street-scene.png`
