# 002 以非同步方式執行推論

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Model | resnet_50 |
| Labels | async, push-pull, throughput, runtime |

## Concept

從一個產生器執行緒中饋入模型，同時從另一個執行緒中取得預測結果，藉此將輸入和輸出分開，以實現真正的吞吐量。與第 001 章相同的 ResNet 路徑，現在採用非同步方式。

## Walkthrough

第一章執行了一個模型，使用單一同步呼叫：傳入一幀，並阻塞直到結果傳回。這很簡單，但會浪費運算資源——產生輸入的執行緒和處理輸出的執行緒是同一個執行緒，因此它們永遠無法同時執行。本章保留完全相同的 ResNet-50 模型，並將其轉換為以吞吐量為導向的管線，方法是將這兩個工作分開。

其機制是異步 `Run`：你 `build()`  `Graph` 在 `Async` 模式，然後透過兩個獨立的呼叫來驅動它—— `push(...)` 來自一位製作人 `pull(...)` 來自消費者。最後，您將有一個生產者執行緒，以最快的速度將影格傳輸到執行階段，而執行階段會接受這些影格，同時主要執行緒會提取預測結果，以及一個最終的 `pushed=N pulled=N` 這條線證明了沒有任何東西遺失。

### 載入模型 {#step-load-model}

我們從與第 001 章完全相同的方式開始——建置一個 `Model` 從檔案中擷取——但我們在此也聲明一個 `RouteOptions` 搭配 `include_input` 以及 `include_output` 設定。這些旗標會指示模型在組成圖時，公開其自身的輸入和輸出邊界，以便周圍的管線可以將影格推送進來，並將張量提取出來。

### 建立非同步管線 {#step-build-async}

一 `Model` 無法直接透過推/拉方式來驅動； `Run` 是。我們將模型包裝在一個全新的 `Graph` 透過 `graph.add(model.graph(route_opt))`，然後 `build(...)` 將其與一個代表性的框架搭配。傳遞樣本框架可讓 `build()` 事先協商好具體的張量形狀。傳回的 `Run` 是處理器將共用的控制項。

### 從生產者推送框架 {#step-push-frames}

製作者的唯一任務是提供輸入。我們啟動一個執行緒，該執行緒會循環處理準備好的影格，並呼叫。 `push(...)` 對於每一個，然後再呼叫 `close_input()` 以表示不再有更多影格即將傳輸——這個訊號讓接收者知道何時應該停止。由於產生器是獨立運作，因此在傳送下一個影格之前，它不會等待任何結果。

**C++：** `std::thread` 執行迴圈；一個原子 `pushed` 計數器和一個 `producer_done` 旗標會隨著進程更新，因此主要執行緒可以在沒有鎖的情況下觀察進度。

**Python：** `threading.Thread` 命名為 `frame_producer` 執行迴圈；之後，消費者會進行檢查。 `thread.is_alive()` 以檢測完成情況。

### 從消費者端提取結果 {#step-pull-results}

主要執行緒會進行資料消耗。它會循環調用 `pull(timeout_ms=2000)`，該函數會回傳下一個可用的輸出，或者如果在逾時期間沒有任何輸出，則回傳空值。在沒有輸出的情況下，我們會檢查生產者是否已完成——如果是，我們就會停止；否則，我們會繼續等待。每個實際結果都會被縮減為一個 top-1 類別索引，然後進行輸出。在循環結束後，我們會加入生產者，並確認 `pushed == pulled`。

**C++：** `pull()` 會回傳一個 `optional<Sample>`；在讀取位元組之前，使用 `tensors_from_sample(...)` 提取張量。

**Python：** `pull()` 會回傳一個 `Sample` 或 `None`；`sample.tensor.to_numpy()` 會將陣列傳遞給 `argmax`。

## Run

執行它，您應該會看到每個影格都有一行 `top1=` 輸出，後面接著推送/拉取的統計資訊。從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **原始碼儲存庫的根目錄** 執行 **從原始碼建置** 命令。

**Python:**
```bash
python3 share/sima-neat/tutorials/002_run_inference_async/run_inference_async.py \
  --model /tmp/resnet_50.tar.gz --n 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_002_run_inference_async \
  --model /tmp/resnet_50.tar.gz --n 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_002_run_inference_async
./build/tutorials-standalone/tutorial_002_run_inference_async \
  --model /tmp/resnet_50.tar.gz --n 4
```

預期的輸出（確切的索引取決於影像；C++ 建置會新增一個 `pushed=...` 欄位，Python 建置僅輸出 `pulled=...`）：

```text
top1=285
top1=285
top1=285
top1=285
pushed=4 pulled=4
[OK] 002_run_inference_async
```

若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

本章使用非同步推送/拉取介面。若要使用確定性的合成輸入來測量相同的模型，請繼續閱讀 [基準測試您的模型](/tutorials/benchmark-your-model)。對於完整的建置與執行以及同步與非同步模型，以及完整的 `RunOptions` 介面，請參閱 [建置您的第一個圖](/tutorials/build-inference-pipeline)。對於佇列深度、溢位原則以及負載下的測量，請參閱 [調整輸送量和佇列深度](/tutorials/tune-throughput-and-queues)。

## 原始程式碼檔案
- C++：`tutorials/002_run_inference_async/run_inference_async.cpp`
- Python：`tutorials/002_run_inference_async/run_inference_async.py`
