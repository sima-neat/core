# 024 在 PCIe 上執行您的第一個模型

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, inference, tensor, image, detection |

## Concept

PCIe 主機 API 接受模型準備好的張量或已解碼的圖像像素。
張量模式將預處理保留在主機上。圖像模式會傳送原始像素，並讓 Modalix 卡片調整大小、轉換顏色並對其進行正規化。
添加框解碼會保留圖像輸入，但會將六個原始 YOLO 輸出張量替換為一個緊湊的檢測列表。

## Walkthrough

使用相同的 YOLOv8s 檔案和 640x480 街景執行三個獨立的程式。每個程式都演示一種模式，同步使用佇列 0，並關閉一個模型。這可確保每個範例都足夠簡短，可以單獨複製。

### 執行模型準備好的張量 {#step-tensor-mode}

主機會將圖像調整大小，使其符合模型報告的 `[640, 640, 3]` 輸入，將 BGR 轉換為 RGB，並將像素縮放到 `[0, 1]`。`Model.run()` 會傳送該 FP32 張量，而無需卡片端圖像預處理，並列印所有六個原始 YOLO 輸出路由。

### 將預處理移動到卡片 {#step-image-mode}

將 `preprocess.kind` 設置為 `Image`，將傳入的像素識別為 BGR，並選擇 `COCO_YOLO` 預設設定。主機現在會傳送已解碼的像素，而卡片會執行調整大小、BGR 到 RGB 轉換和正規化。該程式會列印六個原始輸出路由名稱和形狀，以便您可以將其與張量模式進行比較。

### 在卡片上解碼檢測 {#step-decode-boxes}

添加 `BoxDecodeType.YoloV8`、分數閾值、NMS 閾值和輸出限制。傳回的 BBOX 張量以檢測計數開始，然後是固定大小的記錄，其中包含 `(x, y, width, height, score, class_id)`。該範例會解析並列印源圖像座標中的前十條記錄。

### 解析 BBOX 張量 {#step-parse-boxes}

驗證框解碼是否傳回了一個填充的張量，讀取其前導計數，並拒絕超過有效負載的計數。然後，每個剩餘的 24 位元組記錄都會轉換為一個檢測，以便進行列印。

## Run

依照[教學設定](/tutorials/before-you-run)的說明安裝 PCIe 主機套件並下載教學套件包。在解壓縮後的 PCIe extras 根目錄執行下列命令。在 Windows 上，請依照[在 Windows 上執行應用程式](/getting-started/neat-library/pcie-host/#run-applications-on-windows)的說明設定教學課程並執行程式。

這些程式需要此目錄中的 `yolo_v8s_mpk.tar.gz`。請從 [Model Browser](/tools/model-browser/) 下載 YOLOv8s，從下載的 ZIP 檔案中解壓縮封存檔，並以該名稱儲存：

```bash
sima-cli models download --id yolov8s --variant w8a8 --output models --force
unzip -o models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

在 Windows 上，請在 PowerShell 中下載模型，並使用 `Expand-Archive` 解壓縮封存檔：

```powershell
sima-cli models download --id yolov8s --variant w8a8 --output models --force
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8 -Force
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

如果下載的是 `<id>_<variant>_mpk.tar.gz` 檔案而不是 ZIP 檔案（例如 Model Browser 開始以 ZIP 檔案封裝模型之前最後一次建置的模型），請直接從 `models/<id>/<variant>` 複製該檔案，不要解壓縮。

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_tensor_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_boxdecode.py
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_024_run_tensor_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_boxdecode
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_024_run_tensor_mode
./build.sh --target tutorial_024_run_image_mode
./build.sh --target tutorial_024_run_image_boxdecode

./build/tutorials-standalone/tutorial_024_run_tensor_mode
./build/tutorials-standalone/tutorial_024_run_image_mode
./build/tutorials-standalone/tutorial_024_run_image_boxdecode
```

在 Windows 上，請在您的教學課程副本中，從 **x64 Native Tools Command Prompt** 使用 CMake 建置。將 `C:\opencv\build` 替換為[在 Windows 上執行應用程式](/getting-started/neat-library/pcie-host/#run-applications-on-windows)中所述 OpenCV 套件的 `build` 目錄：

```bat
call "%ProgramFiles%\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_024_run_tensor_mode
cmake --build build --config Release --target tutorial_024_run_image_mode
cmake --build build --config Release --target tutorial_024_run_image_boxdecode
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_024_run_tensor_mode.exe
build\Release\tutorial_024_run_image_mode.exe
build\Release\tutorial_024_run_image_boxdecode.exe
```

匹配的 C++ 和 Python 程式會列印張量模式和圖像模式的相同六個原始輸出合約，然後是已解碼的人、汽車或其他可見物件：

```text
Tensor mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_tensor_mode
Image mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_image_mode
Image + boxdecode detections=...
  person score=... box=(...)
[OK] 024_run_image_boxdecode
```

預設值為卡片 0 和佇列 0。僅在要使用其他卡片時，傳遞 `--card N`；其管理位址會自動推導。

## In Practice

當您的應用程式已經產生完全符合 `model.info()` 報告的 dtype、形狀、佈局、顏色順序和數值範圍時，請使用張量模式。當應用程式自然擁有已解碼的像素，並且您希望卡片應用可重複的模型預處理時，請使用圖像模式。啟用框解碼時，應用程式需要檢測而不是原始特徵圖。

每種模式都使用相同的 `pcie::Model`/`pyneatpcie.Model` 生命週期。只有 `ModelOptions` 和提交的有效負載會發生變化。繼續使用 [非同步執行 PCIe 推論](/tutorials/run-pcie-inference-async)，以使用 `push()` 和 `pull()` 重疊提交和完成。

## 原始檔案

- `run_tensor_mode.cpp`
- `run_tensor_mode.py`
- `run_image_mode.cpp`
- `run_image_mode.py`
- `run_image_boxdecode.cpp`
- `run_image_boxdecode.py`
- `../assets/street-scene.png`
