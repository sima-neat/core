# 003 基準測試您的模型

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5-10 minutes |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

執行已編譯的模型，使用確定性的合成張量，並列印 `Model::benchmark()` 傳回的延遲、吞吐量、功耗和能耗等主要指標。

## Walkthrough

第 001 章和第 002 章展示了如何執行模型一次，然後如何以非同步方式驅動它。本章回答了下一個實際問題：「此模型在設備上的執行速度如何？」基準測試 API 故意設計得較小。您載入模型，選擇要測量的樣本數量，呼叫 `benchmark(...)`，並讀取傳回的 `BenchmarkReport`。

基準測試使用模型的 `input_specs()` 來建立確定性的合成輸入。這使得它對於快速的模型初步基準測試以及比較已編譯的模型變體非常有用，但它不是相機基準測試。它不包括相機解碼、實際的預處理變化、動態輸入大小或依賴資料的後處理行為。

### 載入模型 {#step-load-model}

從先前模型教程中使用的相同的已編譯 `.tar.gz` 檔案開始。由於基準測試會根據模型聲明的輸入規格建立合成張量，因此不需要任何圖像。

**C++：** 從檔案路徑建置 `simaai::neat::Model`。

**Python：** 從檔案路徑建置 `pyneat.Model`。

### 執行基準測試 {#step-run-benchmark}

呼叫 `benchmark(samples)`。API 會預熱非同步模型執行器，測量非同步推送/拉取窗口，將摘要列印到標準輸出，並在 `BenchmarkReport` 中傳回相同的主要值。

樣本數量是測量的合成輸入的數量。對於更穩定的吞吐量和功耗數字，請使用較大的數量；當您只想進行快速的初步檢查時，請使用較小的數量。

其路徑以 BoxDecode 結尾的檢測模型也可以使用 `BenchmarkOptions`。設定 `original_width`、`original_height` 和 `resize_mode`，以描述源圖像幾何形狀，BoxDecode 在將檢測映射到模型座標時使用該幾何形狀。合成張量仍然保持模型形狀：

```cpp
simaai::neat::BenchmarkOptions options;
options.num_samples = 100;
options.original_width = 1920;
options.original_height = 1080;
options.resize_mode = simaai::neat::ResizeMode::Letterbox;
auto report = model.benchmark(options);
```

Python 會透過 `pyneat.BenchmarkOptions` 公開相同的欄位。設定原始的兩個維度，或省略這兩個維度；如果省略，基準測試會從已解析的模型路徑推斷幾何形狀。每次執行的基準測試幾何形狀優先於 `ModelOptions` 中已棄用的 BoxDecode 幾何形狀。

### 閱讀報告 {#step-read-report}

傳回的報告僅保留大多數使用者需要的標題欄位：以毫秒為單位的平均端到端延遲、以每秒幀數為單位的吞吐量、如果有的話，以瓦特為單位的平均板載功耗，以及如果有的話，以焦耳為單位的測量能量。

功耗遙測取決於板載支援。如果執行階段無法對當前目標進行功耗軌的取樣，基準測試仍會報告延遲和吞吐量，並將功耗欄位設定為零。

## Run

執行它，您應該會看到 `benchmark()` 輸出的基準測試摘要，後面接著從傳回的報告中輸出的相同值。從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **儲存庫根目錄**執行 **從原始碼建置** 命令。

**Python:**
```bash
python3 share/sima-neat/tutorials/003_benchmark_your_model/benchmark_your_model.py \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_003_benchmark_your_model
./build/tutorials-standalone/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

預期的輸出（確切的數字取決於模型、板載和當前負載；C++ 建置還會輸出尾部的 `[OK]` 行）：

```text
NEAT Benchmark
Input: synthetic
Samples: 100
Latency:      12.4 ms
FPS:          80.6
Power avg:    2.3 W
Energy:       2.8 J
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
[OK] 003_benchmark_your_model
```

若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

當您需要快速了解已編譯的模型封存檔時，請使用此基準測試：它是否能正常執行？測得的非同步吞吐量是多少？以及在此目標平台上，主要的板卡功耗是多少？

對於應用程式效能，也請基準測試實際的管線。合成模型輸入是故意設計為穩定的，因此它不代表相機抖動、編解碼器成本、實際的預處理、主機在負載下的排程，或下游應用程式邏輯。若要針對手動建立的非同步執行來調整佇列深度和反壓，請參閱[調整吞吐量和佇列深度](/tutorials/tune-throughput-and-queues)。

`Model::benchmark()` 需要具體的 `input_specs()` 尺寸。如果輸入形狀是動態或非具體的，基準測試將明確失敗，而不是猜測一個形狀。

## 原始程式碼檔案
- C++：`tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python：`tutorials/003_benchmark_your_model/benchmark_your_model.py`
