# 003 測量模型效能

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 分鐘 |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

使用 `model.benchmark()` 測量已編譯模型的延遲、吞吐量、功耗與耗能。基準測試會建立合成輸入，因此不需要影像或資料集。

## Walkthrough

### 載入模型 {#step-load-model}

在 C++ 中使用 `simaai::neat::Model`，或在 Python 中使用 `pyneat.Model`，載入已編譯的 `.tar.gz` 封存檔。模型必須在 `input_specs()` 中宣告明確的輸入尺寸。

### 執行基準測試 {#step-run-benchmark}

呼叫 `model.benchmark()` 使用預設設定，或呼叫 `model.benchmark(100)` 指定樣本數。API 會預熱模型、測量循序執行的延遲與非同步執行的吞吐量、列印摘要，並傳回 `BenchmarkReport`。

增加樣本數可讓吞吐量與功耗測量更穩定。

### 閱讀報告 {#step-read-report}

| 欄位 | 意義 | 單位 |
| --- | --- | --- |
| `latency_ms` | 預熱後的平均延遲 | ms |
| `fps` | 邏輯推論吞吐量 | inferences/s |
| `avg_power_watts` | 測量吞吐量期間的平均板卡功耗 | W |
| `energy_joules` | 測量吞吐量期間的總耗能 | J |

無法取得板卡功耗遙測資料時，功耗與耗能維持為零。

您也可以在 Python 程式碼中儲存報告：

```python
model = pyneat.Model("model.tar.gz")
report = model.benchmark()
report.save_json("benchmark.json")
```

使用 `report.to_json()` 取得 JSON 文字，或使用 `print(report)` 再次顯示報告。JSON 會明確標示無法取得的測量結果，不會將其視為有效的零值。

## Run

請從包含 `share/` 與 `lib/` 的 Neat 安裝根目錄執行 Python 與預先建置的 C++ 範例。從原始碼建置的命令則需在儲存庫根目錄執行。兩個範例都會先列印基準測試摘要，再列印傳回的報告欄位。

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

報告欄位範例，數值僅供示意：

```text
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
```

若要用於自己的 C++ 專案，請參閱[如何執行教學範例](/tutorials#compile-a-copy-yourself)。

## In Practice

請使用相同的模型設定與裝置比較結果。儲存報告時，一併記錄模型封存檔與樣本數。

合成輸入無法反映相機時序或依資料而變的處理行為。請另外測量實際的應用程式管線。非同步佇列的調整方式請參閱[調整吞吐量與佇列深度](/tutorials/tune-throughput-and-queues)。

BoxDecode 來源影像尺寸與其他基準測試選項，請參閱[測量模型執行效能](/develop-apps/development-workflow/model#measure-model-execution)。

## 原始程式碼檔案
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
