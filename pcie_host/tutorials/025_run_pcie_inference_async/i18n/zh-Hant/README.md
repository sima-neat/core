# 025 執行 PCIe 非同步推論

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, asynchronous, throughput, detection |

## Concept

同步 `run()` 對於第一次推論來說是理想的，但它會等待每個結果，然後再提交下一個影像。一個調用 `push()` 的生產者，同時一個消費者執行 `pull()`，可以讓一個 PCIe 模型保持忙碌狀態。吞吐量必須從完成的結果中計算，而不是僅僅從提供給模型的影像數量來計算——這是在預熱之後。

## Walkthrough

本教學會重新使用 YOLOv8s 影像加框解碼設定和 640x480 街景，這些來自教學 024。它提交一個重複的影像，因此儲存和影像解碼不會扭曲 PCIe 測量。

### 設定一個檢測模型 {#step-configure-model}

一次載入影像，設定卡端 COCO 預處理和 YOLOv8 框解碼，然後在佇列 0 上建立一個 `Model`。缺少的檔案和卡啟動錯誤會在測量開始之前停止程式。

### 預熱管線 {#step-warm-up}

執行幾個完整的檢測，但不計時。預熱可以消除模型啟動和第一個緩衝區效應，從而更準確地報告工作負載。

### 同時提交和檢索 {#step-push-pull}

一個執行緒使用 `push()` 提交影像，而另一個執行緒使用具有有限超時的 `pull()` 檢索 BBOX 輸出。一個小型應用程式擁有的 FIFO 儲存每個有序提交的開始時間。任何拒絕、超時或格式不正確的結果都會關閉模型並喚醒另一個執行緒。

該範例僅依賴於正常的 `Model` 流程控制行為；應用程式中沒有佇列深度調整。

### 報告完成的工作 {#step-report-results}

僅在兩個執行緒都完成並且所有已接受的影像都已檢索後，才停止計時。每秒幀數使用已完成輸出的數量。平均延遲從每次提交嘗試開始，直到其匹配的有序結果到達為止。

## Run

安裝 PCIe 主機套件，並按照 [教學設定](/tutorials/before-you-run) 中所述下載教學套件。從解壓縮後的 PCIe extras 根目錄中，下載 YOLOv8s（如果尚未存在）：

```bash
sima-cli modelzoo get yolo_v8s
```

該程式需要此目錄中的確切路徑 `yolo_v8s_mpk.tar.gz`。如果 Model Zoo 使用了不同的名稱或位置，請將下載的檔案複製到指定位置：

```bash
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

執行 Python：

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/025_run_pcie_inference_async/run_pcie_inference_async.py
```

執行預建的 C++ 教學：

```bash
./lib/sima-pcie-host/tutorials/tutorial_025_run_pcie_inference_async
```

或者重新編譯它：

```bash
./build.sh --target tutorial_025_run_pcie_inference_async
./build/tutorials-standalone/tutorial_025_run_pcie_inference_async
```

確切的計時取決於主機和卡，但兩個程式都使用相同的測量邊界並輸出：

```text
completed=1000
elapsed_seconds=...
throughput_fps=...
average_latency_ms=...
total_detections=...
[OK] 025_run_pcie_inference_async
```

教學程式總是使用五個幀進行預熱，然後測量 1,000 個完成的幀。僅在要使用另一個卡時，才傳遞 `--card N`。

## In Practice

保持提交和檢索的平衡。如果應用程式無限期地推送而不進行拉取，則正常的反壓最終會減慢提交速度。一個專用的消費者也可以使故障變得簡單：有限的超時可以識別出停滯的結果，並且關閉模型即使生產者正在等待，也會釋放佇列 0。

為了獲得具有代表性的基準測試，請將重複的幀替換為固定的影像集，並將磁碟讀取操作放在計時區域之外。繼續執行 [同時執行多個模型](/tutorials/run-multiple-models)，以同時執行兩個不同的模型。

## 原始檔案

- `run_pcie_inference_async.cpp`
- `run_pcie_inference_async.py`
- `../assets/street-scene.png`
