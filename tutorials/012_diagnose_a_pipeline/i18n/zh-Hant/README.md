# 012 診斷並分析管線

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | <10 minutes |
| Model | None |
| Labels | diagnostics, debugging, observability |

## Concept

透過三個檢查來初步診斷管線：`graph.validate()`、一次測量的 `run.run()`，以及 `MeasureReport` 診斷，以判斷它是否已正確連接，以及它的效能如何，然後再進行深入的除錯。

## Walkthrough

當管線出現問題時，人們往往傾向於直接進行元件層級的除錯。本章介紹一種更經濟的初步方法：一種可重複的初步檢測流程，它會依序回答三個問題——*圖的合約是否有效？單次執行是否成功？執行階段診斷顯示了什麼？* 它可以在幾秒鐘內捕獲大多數的錯誤設定，防止問題演變成耗時數小時的除錯過程，並且它適用於與第 004 章中相同的最簡輸入 → 輸出圖。

在本章結束時，您將驗證圖的合約，執行單次測量的框架，並列印測量報告，以確定管線是否正常運作。

### 驗證合約 {#step-validate-graph}

`validate()` 是一種合約層級的檢查，它在 `build()` 之前執行。它會測試節點順序、限制和後端解析路徑，而無需串流任何資料，並傳回一份報告，其中包含標準的 `error_code`。一個空的/`ok` 代碼表示圖的結構是健全的；任何其他代碼都會將錯誤分類（請參閱下方的錯誤分類），以便您知道從何處開始查找。首先執行此操作意味著您永遠不會浪費時間在一個從未能夠建立的圖的執行階段行為上進行除錯。

### 執行單個測量的框架 {#step-run-with-measurement}

接下來，在 `start_measurement()` 視窗內建立並執行單個確定性的框架。`output_memory = Owned` 要求提供已擁有的輸出緩衝區，以便在呼叫後結果保持有效。一個框架就足夠了：如果它成功，則表示管線正在運作；如果它拋出異常，則該異常會包含一份結構化的報告，您可以像 `validate()` 一樣對其進行分類。

### 讀取執行階段診斷 {#step-read-diagnostics}

在記錄了一次執行後，`MeasureReport` 會總結管線的健康狀況：計數器（`inputs_enqueued`、`outputs_pulled`、丟棄的資料）、端到端延遲、節點指標、外掛程式/核心計時、邊緣計時以及可選的功耗。`MeasureReport::to_text()` 是您在升級到探測和 DOT 圖（如 [實際應用](#in-practice) 中所述）之前捕獲的基準。

## Run

執行它，您應該會看到驗證程式碼和測量報告列印到標準輸出。從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **儲存庫根目錄** 執行 **從原始碼建置** 命令。本章不需要模型封存檔。

**Python:**
```bash
python3 share/sima-neat/tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_012_diagnose_a_pipeline
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_012_diagnose_a_pipeline
./build/tutorials-standalone/tutorial_012_diagnose_a_pipeline
```

預期的輸出結果（計數器值和摘要字串會因每次執行而異）：

```text
validate.error_code=
measure.inputs_enqueued=1 outputs_pulled=1
measure.text_size=...
[OK] 012_diagnose_a_pipeline
```

（Python 建構會輸出 `validate_error_code=`、`inputs_enqueued=... outputs_pulled=...` 和 `measure_text_size=...`。）若要將本章的 C++ 原始程式碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

結構化的診斷資訊、錯誤分類法、除錯控制項，以及當您執行 `validate()` / `start_measurement()` / `MeasureReport` 時，用於處理問題的外掛程式失敗流程。

### 圖報告 (GraphReport)

`GraphReport` 捕捉結構化的診斷資訊：
- 管線字串（用於重現）
- 標準 `error_code`（機器分流）
- `repro_note`（人類摘要 + 提示）
- 節點報告和所屬元件名稱
- 匯流排訊息和錯誤詳細資訊
- 可選的流程/計時計數器

當發生錯誤時，`NeatError` 會攜帶一個 `GraphReport`，您可以將其記錄或序列化。

### 錯誤分類法

框架錯誤使用穩定的程式碼系列：

| 錯誤程式碼 | 意義 | 一般解決方案 |
|---|---|---|
| `misconfig.pipeline_shape` | 節點順序/形狀合約違規 | 確保對於推動型管線，`Input()` 位於最前面，對於拉動型管線，`Output()` 位於最後面 |
| `misconfig.caps` | 框架的 caps-override 或相鄰節點合約不匹配 | 對齊 `caps_override` 和已宣告的節點合約 |
| `misconfig.media_caps` | 執行階段 GStreamer 媒體協商不匹配 | 對齊格式、解析度、幀率或插入轉換 |
| `misconfig.input_shape` | 輸入張量/幀/樣本形狀/佈局不匹配 | 驗證寬度/高度/深度、佈局、dtype、儲存 |
| `build.plugin_missing` | 必需的 GStreamer 元件或編解碼器不可用 | 安裝/替換它，並使用 `gst-inspect-1.0` 進行驗證 |
| `build.property_invalid` | GStreamer 屬性名稱或值無效 | 使用 `gst-inspect-1.0 <element>` 進行檢查 |
| `build.pipeline_syntax` | 自訂 GStreamer 片段的語法無效 | 修正它，並使用 `gst-launch-1.0` 進行驗證 |
| `runtime.pull` | 拉取操作失敗，但沒有更具體的理由 | 檢查附加的報告和第一個上游錯誤 |
| `io.parse` | 已儲存的圖 JSON 解析/架構失敗 | 驗證 JSON 和必需的節點欄位 |
| `io.open` | 圖的儲存/載入檔案開啟/讀取/寫入失敗 | 檢查路徑是否存在、權限和儲存狀況 |

`PullError.code` 使用相同的分類法（不僅僅是異常路徑）。
這是一個簡短的分流清單。請參閱 [完整的錯誤程式碼目錄](/reference/error-codes)，包括
從先前粗略的執行階段和建置程式碼的遷移。

### 程式化處理

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build(input);
  simaai::neat::Sample out;
  simaai::neat::PullError perr;
  const auto st = run.pull(500, out, &perr);
  if (st == simaai::neat::PullStatus::Error) {
    if (perr.code == simaai::neat::error_codes::kMediaCaps) {
      // Fix the incompatible upstream/downstream media contract.
    } else {
      // Handle another specific code, including future codes, or report it.
    }
  }
} catch (const simaai::neat::NeatError& e) {
  if (e.report().error_code == simaai::neat::error_codes::kPluginMissing) {
    // Install or replace the missing GStreamer component.
  }
}
```

### 偵錯開關（環境）

重要的環境變數（詳見 [架構](/develop-apps/contribute/architecture)）：
- `SIMA_GST_DOT_DIR`：為失敗案例產生 DOT 圖
- `SIMA_GST_BOUNDARY_PROBES`：邊界流程計數器
- `SIMA_GST_ELEMENT_TIMINGS`：每個元件的計時資訊
- `SIMA_GST_FLOW_DEBUG`：每個元件的流程計數器
- `SIMA_GST_ENFORCE_NAMES`：強制執行命名規範

若要進行簡短的執行，並將經過處理的原始 GStreamer 資訊附加到錯誤訊息中，請使用：

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug` 不是 Neat Library 的設定。

### 偵錯工作流程

1. 擷取 `GraphReport.error_code`，並首先根據分類對失敗案例進行分類。
2. 擷取 `GraphReport.repro_note`，以取得具體的上下文資訊和內建提示。
3. 擷取管線文字：`Graph::describe_backend()` 或 `last_pipeline()`。
4. 擷取結構化診斷資訊：`MeasureReport::to_text()` 或 `NeatError::report()`。
5. 檢查 `GraphReport.bus`，以找出第一個終端 `ERROR` 來源 + 詳細資訊。
6. 如果在執行階段發生停頓/逾時，請啟用邊界/元件探測，以找出流程停止的位置。

建議的支援套件：
- `error_code`
- `repro_note`
- 完整的 `pipeline_string`
- 前 3-5 個終端匯流排錯誤 (`GraphReport.bus`)
- 在執行/驗證過程中使用的環境覆寫

### 常見失敗 → 解決方案

| 症狀 | 可能原因 | 解決方案 |
|---|---|---|
| `missing ... plugin` | 找不到 GStreamer 外掛程式 | 檢查 `GST_PLUGIN_PATH`，執行 `gst-inspect-1.0 <plugin>` |
| `appsink 'mysink' not found` | 缺少終端 `Output()` | 確保 `Output` 是執行/建置管線中最後一個節點 |
| `caps_override is set; renegotiation disabled` | 已固定 caps | 移除 `caps_override` 或保持輸入 caps 不變 |
| `tensor caps change not supported` | 執行階段張量形狀/資料類型發生變化 | 保持張量形狀/資料類型穩定（不重新協商） |

### 偵錯外掛程式失敗

當外掛程式失敗時，Neat 會引發一個 `NeatError`，其訊息包含 GStreamer 錯誤和結構化的偵錯字串。使用這些欄位可以快速找到根本原因。

1. **閱讀結構化欄位。** 尋找錯誤文字中的 `debug` 鍵/值欄位：
   - `node`：管線中失敗元件的名稱
   - `config_path`：JSON 設定檔案（如果適用）
   - `model_path`：模型/封包路徑（如果適用）
   - `hint`：可操作的修復指導
   - `detail`：額外的上下文資訊，例如缺少鍵或分配

   請參閱 [錯誤格式參考](/reference/error_format) 以取得完整清單。
2. **確認管線的上下文。** 使用來自 `Graph::last_pipeline()` 或錯誤報告的管線字串：
   - 驗證 `node` 名稱是否出現在管線中。
   - 確認 `config_path` 是否存在且可讀。
   - 對於 caps 錯誤，請檢查與失敗節點連接的上游元件。
3. **應用常見的修正方法。**
   - **設定錯誤**：驗證 JSON 語法、必要鍵和任何模型路徑。
   - **Caps 錯誤**：新增或修正解析器元件（例如，`h264parse`），確保 caps 包含必要的欄位，例如 `parsed=true`、`stream-format=byte-stream`、`alignment=au`。
   - **設定器錯誤**：確保上游元件使用必要的設定器類型（系統與 simaai 記憶體/區段）。
4. **使用上述除錯選項捕捉更多診斷資訊**（`SIMA_GST_DOT_DIR`、`SIMA_GST_FLOW_DEBUG`、`SIMA_GST_ELEMENT_TIMINGS`）。

## 原始檔案
- C++：`tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.cpp`
- Python：`tutorials/012_diagnose_a_pipeline/diagnose_a_pipeline.py`
