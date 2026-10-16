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

呼叫 `benchmark(samples)`。API 會預熱模型，測量單次執行（single-flight）延遲，接著在另一個非同步推送/拉取階段中測量吞吐量，將摘要列印到標準輸出，並在 `BenchmarkReport` 中傳回相同的主要值。

每個階段都會持續執行，直到至少測量了 `samples` 個輸入且已經過 10 秒，因此吞吐量階段的功耗和溫度讀數會涵蓋多個樣本。設定 `BenchmarkOptions::min_duration_ms` 可變更此下限；設為 0 則可進行快速的初步檢查，讓每個階段在測量 `samples` 個輸入後即停止。

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

傳回的報告以大多數使用者需要的標題欄位開頭：以毫秒為單位的平均端到端延遲、以每秒幀數為單位的吞吐量、以瓦特為單位的平均與峰值 SoM 功耗，以及處理 `samples` 個輸入所需的能量（以焦耳為單位）。`throughput_duration_ms` 會回報吞吐量階段執行了多久。

功耗資料來自 Sentinel，也就是 Neat 安裝程式所設定的板子遙測常駐程式。`report.telemetry` 還包含吞吐量階段的電源軌功耗、晶粒與板子溫度、CPU 使用率以及 MLA 記憶體；各欄位說明請參閱[測量模型執行效能](/develop-apps/development-workflow/model#power-and-board-telemetry)。如果 Sentinel 未執行或正在記錄另一個追蹤，基準測試仍會報告延遲和吞吐量，將功耗欄位保持為零，並將 `report.telemetry.status` 設定為原因。

### 進階：層統計資料 {#step-read-layer-stats}

在任一教學命令中加入 `--include-layer-stats`，即可在基準測試報告中包含各層執行時間。此選項預設停用。透過 API 啟用的方式如下：

```cpp
simaai::neat::BenchmarkOptions options;
options.include_layer_stats = true;
auto report = model.benchmark(options);
```

```python
options = pyneat.BenchmarkOptions()
options.include_layer_stats = True
report = model.benchmark(options)
```

從 `report.layer_stats` 讀取每個已編譯 MLA 層的 `name`、從零開始的 `stage_index` 與 `total_execution_time_us`。執行時間為零的 no-op 項目也會保留。各層時間是在獨立的剖析執行中測量，因此總和不等於整體延遲。

### 儲存報告 {#step-save-report}

呼叫 `report.save(path)` 即可將完整報告寫入 YAML 或 JSON，包含已要求的各層統計資料：

```cpp
auto path = report.save("results/model/yolo_benchmark.yaml");
auto json_path = report.save("results/model/yolo_benchmark.json");
```

```python
path = report.save("results/model/yolo_benchmark.yaml")
json_path = report.save("results/model/yolo_benchmark.json")
```

副檔名 `.json` 選擇 JSON；`.yaml` 與其他結尾使用 YAML。檔名保持不變。

請指定包含檔名的路徑。不存在的父目錄會自動建立，既有檔案會被取代，並傳回寫入的路徑。儲存需要明確呼叫；`benchmark()` 不會自動建立報告檔案。

在任一教學命令後加入 `--report results/model/yolo_benchmark.yaml`。搭配 `--include-layer-stats`，即可同時儲存主要測量值與各層執行時間。

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
./build.sh --all --no-doc
./build/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

預期的輸出（確切的數字取決於模型、板載和當前負載；C++ 建置還會輸出尾部的 `[OK]` 行）：

```text
NEAT Benchmark
Input: synthetic
Samples: 100
Compiled batch: 1
Measurement detail: e2e+throughput-only
Latency: 1.97 ms
FPS:     1457.04 inferences/s
Batch throughput: 1457.04 batches/s
Throughput window: 30.09 s, 43849 inputs
Power:         13.19 W avg, 13.81 W peak (SoM)
MLA power:     3.05 W avg, 3.25 W max
Energy:        0.91 J for 100 inputs
Die temp:      46.34 °C avg, 47.56 °C max
CPU usage:     12.38 % avg, 13.22 % max
MLA memory:    29.65 MiB
CMA used:      6035.02 MiB avg, 6044.38 MiB max
Linux memory:  682.88 MiB avg, 697.41 MiB max
Telemetry:     Sentinel, 15 samples, trace 20261014T144839.984Z-neat-benchmark-20261014t144839-905z
report_latency_ms=1.96852
report_fps=1457.04
report_avg_power_watts=13.1875
report_energy_joules=0.905085
report_peak_power_watts=13.8125
report_throughput_duration_ms=30094.5
report_telemetry_status=ok
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
