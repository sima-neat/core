# 016 調整輸送量和佇列深度

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | performance, tuning, async, queues |

## Concept

調整非同步管線的控制參數，以控制系統在負載下的行為——佇列深度和溢出策略——然後測量實際發生的情況。

## Walkthrough

效能調整只有在您的正確性基準穩定時才會有所幫助；本章假設已達到穩定狀態，並著重於決定非同步管線在工作量以超過其處理能力的速度到達時的行為方式。您將設定佇列深度，選擇在該佇列已滿時發生的情況，以非阻塞方式推送一組確定的影格，排出結果，並讀取測量報告，以了解您是否丟棄了任何內容以及每個影格需要多長時間。

到最後，您將擁有一個用於測量在反壓情況下非同步運行的可用框架：佇列計數、丟棄計數、提取的輸出、平均延遲和推送成本。相同的迴圈是針對真實管線，根據 [實際應用](#in-practice) 中的啟發式方法進行調整的基礎。

### 設定執行選項 {#step-configure-run-options}

`RunOptions` 是決定負載下非同步行為的地方。我們設定 `queue_depth`（執行階段接受的同時進行的樣本數量）、`overflow_policy`（當該佇列已滿時發生的情況——`Block`、`KeepLatest` 或 `DropIncoming`）、`output_memory = Owned`（傳回的張量擁有其資料，因此在提取後仍會保留）。然後，我們以 `Async` 模式 `build()` 圖，這為我們提供了一個具有獨立的生產者和消費者端的運行。

**C++：** 超出限制策略從 `--drop` 解析為 `simaai::neat::OverflowPolicy::{Block,KeepLatest,DropIncoming}`；`graph.build(input, opt)` 傳回運行句柄。

**Python：** 策略使用 `getattr(pyneat.OverflowPolicy, ...)` 解析；`graph.build([tensor], opt)` 傳回運行句柄。

### 推送工作負載並排出 {#step-push-workload}

這是測試佇列策略的地方。我們在緊密迴圈中呼叫 `try_push(...)`——這是一種非阻塞推送，它僅傳回樣本是否被接受，因此在 `DropIncoming`/`KeepLatest` 下，已滿的佇列會顯示為被拒絕的推送，而不是停頓。在推送完一組影格後，我們呼叫 `close_input()` 以指示不再有輸入，然後使用 `pull(...)` 迴圈排出消費者端，直到它傳回空值。將 `try_push` 與 `close_input` 結合，再加上一個排出迴圈，是標準的非阻塞非同步模式。

### 讀取測量報告 {#step-read-measurement}

在執行完畢後，我們停止測量範圍。報告 `counters` 群組會在執行階段提供數值，例如：已排入佇列的輸入、已捨棄的輸入、已產出的輸出等等。 `input` 提供推送端的指標，例如平均推送成本和重新協商的輸入。這些指標可以幫助您判斷，您的佇列深度和溢出策略是否達到預期效果：是否發生封包遺失、延遲是否增加，以及推送路徑的成本是否降低。

## Run

本章不需要模型封存檔。請執行以下指令：**Python** 和 **C++（預先建置）**。Neat 安裝根目錄（包含 `share/` 以及 `lib/`）；從**原始碼**開始執行**建置**指令，指令應從**儲存庫的根目錄**執行。

**Python:**
```bash
python3 share/sima-neat/tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.py \
  --iters 32 --queue 4 --drop block
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_016_tune_throughput_and_queues \
  --iters 32 --queue 4 --drop block
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_016_tune_throughput_and_queues
./build/tutorials-standalone/tutorial_016_tune_throughput_and_queues \
  --iters 32 --queue 4 --drop block
```

預期輸出（確切的數量和時間取決於主機和策略）：

```text
inputs_enqueued=32
inputs_dropped=0
outputs_pulled=32
avg_latency_ms=0.42
avg_push_us=18.0
renegotiations=0
[OK] 016_tune_throughput_and_queues
```

（Python 建構會列印相同的鍵，但不會列印結尾的 `[OK]` 行。)

若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂 `CMakeLists.txt`（不需要額外的資料夾），請參閱登陸頁面上的[如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

關於佇列大小設定、捨棄策略、預設設定和輸出生命週期安全性的實用指南。

### 佇列大小設定 (`queue_depth`)

經驗法則：
- 對於低延遲管線，從 `queue_depth = 4–16` 開始。
- 如果您的產生器是間歇性的，或者下游元件具有可變的延遲（解碼/MLA/後處理），則增加佇列。
- 如果您需要**最新**的影格（例如，即時相機預覽），則保持佇列較小。

### 溢出策略 (`RunOptions::overflow_policy`)

- `Block`：最能保證正確性；當佇列已滿時，產生器會等待。
- `DropIncoming`：保留佇列中的工作，當達到飽和狀態時，捨棄傳入的樣本。
- `KeepLatest`：優先選擇最新的影格，捨棄最舊的佇列樣本。

對於即時串流，`KeepLatest` 通常會產生最低的端到端延遲。

### 預設設定和重新協商

使用 `RunOptions::preset` 來控制延遲/安全性的權衡：
- `Realtime`：最低延遲，積極的新鮮度行為。
- `Balanced`：在可能的情況下，從零拷貝開始，執行啟動探測檢查，如果可靠性出現問題，則回退到拷貝模式。
- `Reliable`：保守的行為和穩定的輸出所有權。

對於動態輸入，輸入形狀重新協商是自動的（上述的 `renegotiations` 計數器報告了它發生的頻率）。

### 輸出生命週期 (`output_memory`)

- `output_memory = Owned`：傳回的 `Tensor` 擁有其資料。
- `output_memory = ZeroCopy`：張量可能引用在提取後重複使用的執行階段緩衝區。
- `output_memory = Auto`：執行階段首先選擇零拷貝，然後在需要可靠性時回退到擁有模式。

如果您需要保留超出目前步驟的張量資料，請呼叫 `clone()` 或 `cpu().contiguous()`。

### 緩衝池安全性

- `RunAdvancedOptions::max_input_bytes` 設定輸入緩衝區分配的硬性上限。
- 如果需要更大的緩衝區，則執行階段會立即失敗，並顯示明確的錯誤。

當輸入大小發生變化時，使用這些設定來保護長時間執行的程序，使其免受無限分配的影響。

## 原始檔案
- C++：`tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.cpp`
- Python：`tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.py`
