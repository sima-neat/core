---
title: 診斷 SSD 元件延遲
description: 保留效能證據，同時維持驗收契約不變
---

# 診斷 SSD 元件延遲

當 `ssd_mobilenet_boxdecode` 未達元件延遲預算時，請使用此程序。
FPS 結果通過不能抵銷元件 p99 失敗。

## 明確保留量測邊界

目前的圖會將數值 BoxDecode 與其線路格式發布轉接器準備為分開的 A65 成員。
Core 擁有輸出線路格式；Kernels 擁有數值操作。Scheduler 透過 Runner 提交已檢查的成員。

| 元件 | 開始 | 結束 | 包含的工作 |
|---|---|---|---|
| `boxdecode_backend_wall_v2` | 呼叫數值解碼 | 相依的線路格式發布呼叫完成 | 解碼、其間的排程與存取工作、線路編碼 |
| `boxdecode_stage_wall_v2` | 完整影格准入，取得輸出之前 | 安全釋放、完成處理與輸入釋放完成 | 集區取得、綁定、全部成員執行與收尾 |

階段與後端計時都不包含該影格本身的下游推送。
後端區間不包含先前的轉換成員；階段區間則包含它們。

這些是實際經過時間的範圍，不是 CPU 使用率或純核心運算時間。它們刻意採用與舊的
`boxdecode_backend` 和 `boxdecode_plugin_exclusive` 不同的身分。不要比較不同區間
來證明數值核心效能退化，也不要移除佇列時間讓結果通過。

已納入版本控制的每週基準要求 1,000 個樣本，後端與階段的絕對 p99 上限分別為
0.7 ms 和 0.93 ms。一般吞吐量容忍度不會放寬這些元件限制。
診斷時請保持限制、測試固定資料、暖機、迭代次數與期限不變。

## 保留現有證據

效能矩陣現在會為每次情境嘗試保存完整擷取的 stdout 與 stderr，涵蓋成功執行、
格式錯誤輸出、略過與監督程式失敗。每個結果的 `run_meta.artifacts` 都會指向
唯一的 `scenario-logs/` 目錄，其中包含：

- `stdout.log` 與 `stderr.log`：完整擷取的串流，不只是末尾內容。
- `invocation.json`：命令、工作目錄、迭代次數、期限及兩個追蹤控制環境值。
  不會傾印整個環境。
- `outcome.json`：比較結果及可取得的程序／監督程式結果。

日誌包含執行檔輸出的所有內容；分享前請先檢閱。監督程式拒絕執行時可能沒有程序輸出。
空白日誌不能證明程序已執行、完成或釋放裝置資源。
請保留其失敗記錄，並遵循正常復原程序。

如需獨立的診斷執行，請使用現有控制項：

- `SIMA_NEAT_METRICS_RETAIN_TRACE=1` 保留 LTTng Common Trace Format (CTF) 檔案。
- `SIMA_NEAT_METRICS_TRACE_DIR` 選擇其父目錄；收集器會為每個工作階段建立唯一子目錄。

請**在 DevKit 上**，於全新且已准入的硬體測試範圍內執行矩陣，並使用正確安裝的提供者
與韌體。在准入前設定這些環境值；不要修改進行中且已固定的驗證範圍。
此受監督的診斷程序透過 `SIMA_HARDWARE_SUPERVISOR_SOCKET` 明確選用常駐監督程式；
標準 CI 不需要它。矩陣本身不會取得板卡擁有權，也不會授權韌體變更。

例如，在選定已安裝的測試目錄且診斷環境獲准後，從目標 Core 原始碼目錄執行：

```bash
: "${CORE_ROOT:?set the target Core checkout path}"
: "${CORE_TESTS:?set the installed target test directory}"
: "${SIMA_HARDWARE_SUPERVISOR_SOCKET:?use an admitted hardware-test scope}"
cd "$CORE_ROOT"
export SIMA_NEAT_METRICS_RETAIN_TRACE=1
export SIMA_NEAT_METRICS_TRACE_DIR="$CORE_ROOT/ssd-diagnostic-traces"
python3 tests/perf/tools/run_perf_matrix.py \
  --repo-root "$CORE_ROOT" \
  --prebuilt-tests-dir "$CORE_TESTS" \
  --include-long \
  --results-dir "$CORE_ROOT/ssd-diagnostic-results"
```

這會執行完整的每週效能矩陣，包括 SSD，不會重新建置提供者。
目標程式請使用 SDK 交叉編譯與 DevKit 啟動程式；絕不可在 x86 主機上執行 AArch64 測試。

若啟用追蹤保留、提供非空的追蹤目錄覆寫，或覆寫迭代次數，結果會記錄
`run_meta.execution_purpose: diagnostic`。比較與失敗碼保持不變；
此標籤不是放寬門檻的許可。若沒有這些覆寫，則記錄 `baseline_comparison`，
但這本身不能證明已完成完整的標準 CI 執行。

要求的追蹤父目錄不是實際工作階段身分。請讀取
`run_meta.measure_report.plugin_latency.trace_dir` 及其收集狀態，
再將該確切目錄與結果及情境日誌一併封存。保留 CTF 記錄中的外掛程式執行個體、執行、
圖、影格與要求身分。遇到缺少事件或追蹤遺失時應據實回報，不要虛構關聯。
重用結果目錄會替換最上層結果 JSON，但保留先前的嘗試日誌目錄；
不同驗證活動應使用新的結果與追蹤根目錄。

## 先歸因成本，再進行最佳化

保留的實際時間區間 CTF 可揭露離群值與事件身分，但無法單獨區分數值運算時間與
解碼到發布之間的排程間隔。不要將獨立計算的百分位數相減。
請關聯相同影格，並在需要時透過另行加入量測程式碼的建置收集下列邊界：

1. 輸出取得與綁定／存取準備。
2. 成員入列、工作執行緒啟動、執行開始與執行結束。
3. 數值解碼結束至發布執行開始。
4. 發布執行、存取安全釋放、完成處理與輸入釋放。
5. 暖機後視窗內的分配、匯入、複製位元組及佇列／喚醒次數。

Runner 與 Scheduler 有私用的 `SIMA_RUNTIME_TEST_TRACE` 測試接點，但正式建置會停用它們。
使用這些接點需要另外建置且相符的提供者，並驗證跨執行緒／成員的關聯；
僅啟用環境變數不會產生這些事件。加入額外量測的結果應與最終標準效能結果分開保存。

若派送成本占主要部分，請先評估一般化、有界的就緒工作接續方式，同時保留每個已准入
工作及其目前的記憶體借用。現有 `PreparedSequence` 接受持續有效的 `BoundOperation`
物件；它不能直接取代 Scheduler 僅於呼叫期間有效的 DMA 借用。
絕不可假裝那些借用持續有效、在完成回呼內遞迴執行相依工作，或為省時而移除存取時期。

若解碼占主要部分，請在 Kernels 中最佳化數值實作，並檢查輸出等價性。
若取得或發布占主要部分，請最佳化實際量測到的路徑，不要加入隱藏的負載複製或弱化擁有權。
任何排程變更都必須測試公平性、有界堆疊／容量、失敗傳遞、重新綁定、安全釋放，
以及正確的量測邊界。

最後，移除診斷覆寫並重新執行未變更的標準矩陣。
保留原始失敗結果；診斷通過不能取代它。
