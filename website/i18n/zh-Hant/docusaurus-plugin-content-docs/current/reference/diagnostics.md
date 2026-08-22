---
title: "診斷與除錯"
description: "收集圖報告診斷資訊、執行階段錯誤碼，以及圖指標成品。"
sidebar_position: 9
---

# 診斷與除錯

## 圖表報告

`GraphReport` 會記錄結構化的診斷資訊：
- 管線字串（用於重現）
- 標準 `error_code`（機器分流）
- `repro_note`（人工摘要 + 提示）
- 節點報告和所屬元件名稱
- 匯流排訊息和錯誤詳細資訊
- 可選的流程/計時器計數器

當發生錯誤時，`NeatError` 會包含一個 `GraphReport`，您可以將其記錄或序列化。

## 錯誤分類法

框架錯誤使用穩定的程式碼系列：

| 錯誤碼 | 意義 | 常見的解決方法 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 節點順序/形狀合約違規 | 確保對於推送管線，`Input()` 位於第一個位置，對於拉取管線，`Output()` 位於最後一個位置 |
| `misconfig.caps` | 框架的 caps-override 或相鄰節點合約不匹配 | 調整 `caps_override` 和宣告的節點合約 |
| `misconfig.input_shape` | 輸入的張量/影格/樣本的形狀或資料類型與模型的要求不符 | 請提供預期的形狀和資料類型，或設定模型預處理 |
| `misconfig.runtime_abi_mismatch` | Neat 和一個執行階段外掛程式使用了不相容的 ABI | 請安裝版本相符的 Neat Library 和執行階段 |
| `misconfig.graph_element_name` | 無法將穩定的節點名稱指派給自訂元素 | 為自訂元素提供穩定且獨一無二的名稱 |
| `misconfig.input_capacity` | 來源圖片超過預處理輸入容量 | 請增加 `input_max_width` / `input_max_height`，或在模型階段之前縮放圖片 |
| `misconfig.media_caps` | 相鄰的 GStreamer 階段需要不相容的媒體功能 | 請調整格式、解析度和幀率，或插入轉換 |
| `misconfig.media_format` | 某個階段接收到不支援的媒體格式 | 請設定支援的格式，或插入格式轉換 |
| `misconfig.tensor_dtype_missing` | 張量合約缺少 dtype/格式 | 在上游合約中宣告一個受支援的張量 dtype |
| `misconfig.option_out_of_range` | 目前的張量設定選項無效 | 請選擇診斷工具中顯示的範圍內的數值 |
| `build.parse_launch` | 一個 `gst_parse_launch` 失敗不再有更具體的分類。 | 請檢查附件中的報告，以了解語法分析器的上下文。 |
| `build.pipeline_syntax` | 自訂 GStreamer 片段語法無效 | 使用 `gst-launch-1.0` 修正並驗證片段 |
| `build.plugin_missing` | 一個必要的 GStreamer 元件或編解碼器外掛程式未安裝。 | 安裝/更換後，請檢查。 `gst-inspect-1.0` |
| `build.property_invalid` | 某個元素的屬性未知或無效 | 請使用 `gst-inspect-1.0` 檢查屬性名稱和值 |
| `runtime.pull` | 由於沒有更明確的根本原因，因此拉取操作失敗。| 請檢查隨附的報告以及第一個上游錯誤。|
| `runtime.element_failed` | 某個階段在執行階段失敗，但沒有更明確的對應關係 | 請修正報告中出現問題的階段及其上游輸入 |
| `runtime.output_timeout` | 在設定的逾時時間內，沒有收到任何輸出 | 請檢查資料流程或增加預期的逾時時間 |
| `runtime.unexpected_eos` | 管線在達到預期結束之前就結束了，導致缺少必要的輸出 | 請檢查是否過早達到來源的結束，並確保提供足夠的輸入 |
| `io.parse` | JSON 或階段設定解析/結構驗證失敗 | 驗證設定語法和必要欄位 |
| `io.open` | 圖形檔案儲存/載入時，開啟/讀取/寫入失敗 | 請檢查路徑是否存在、權限是否正確，以及儲存裝置的健康狀況 |
| `io.file_not_found` | 輸入檔案不存在 | 請檢查檔案路徑，並確認檔案是否存在於 DevKit 中。 |
| `io.permission_denied` | 檔案或裝置無法讀取 | 請檢查檔案或裝置的擁有者/權限 |
| `io.rtsp_connection_failed` | 無法連線到 RTSP 來源 | 請檢查 URL、網路連通性、伺服器和憑證 |
| `io.camera_not_found` | 要求的相機無法使用 | 請選擇可用的相機或使用預設相機 |
| `io.model_not_found` | 要求的模型封存檔不存在 | 請更正模型路徑，並確認已正確安裝 |
| `io.source_ended` | 輸入來源已達到其正常結束點 | 停止處理或提供更多輸入 |
| `codec.invalid_h264_stream` | 輸入內容沒有有效的 H.264 影格 | 請提供完整的 H.264 串流或更正編碼器 |
| `codec.decode_failed` | 解碼器在接收到資料流後失敗 | 請驗證編解碼器和輸入資料的完整性 |
| `codec.encode_failed` | 編碼器無法編碼提供的影格 | 請檢查輸入格式、解析度和編碼器設定 |
| `resource.memory_allocation_failed` | 必要的記憶體設定失敗 | 減少工作負載的記憶體使用量，並釋放其他應用程式或管線所使用的記憶體 |
| `resource.device_memory_exhausted` | 裝置 DMA/CMA 分配失敗 | 減少同時串流、解析度或緩衝區大小 |
| `resource.output_pool_exhausted` | 所有輸出緩衝區仍在使用中 | 釋放零拷貝輸出或使用擁有的副本 |
| `resource.buffer_too_small` | 緩衝區小於其宣告的有效載體大小 | 請修正尺寸/步幅，或設定所需的位元組數 |
| `resource.disk_full` | 由於儲存空間已滿，寫入作業失敗 | 請釋放更多空間或選擇其他目的地 |
| `infra.dispatcher_unavailable` | 無法取得加速器的執行階段 | 請停止其他正在執行的工作負載，並驗證 DevKit 的相容性 |
| `infra.accelerator_execution_failed` | 加速器無法執行模型階段 | 請重新啟動管線，並減少加速器的並行工作量 |
| `DispatcherUnavailable` | 這是 `infra.dispatcher_unavailable` 的舊拼寫方式。| 將處理程式遷移到標準的基礎架構程式碼。|
| `internal.plugin_failure` | 一個外掛程式發生錯誤，但沒有提供使用者可採取行動的分類資訊 | 記錄錯誤報告並聯繫支援團隊 |

`PullError.code` 使用相同的分類法（不僅僅是異常路徑）。
請參閱 [錯誤碼目錄](/reference/error-codes)，以了解 C++ 和 Python 常數名稱，以及針對先前粗略程式碼的應用程式的遷移指南。

生產訊息會故意省略 GStreamer 內部細節。外掛程式除錯詳細程度會新增原始 GError 網域/程式碼、元素工廠、訊息和結構化外掛程式詳細資訊。已識別的憑證和 URL 秘密參數（包括 URI 使用者資訊、`auth`、`playback-token`、`hdnts`、`stream-key` 和 `tkn`）在儲存任何形式的資料之前都會被刪除。面向報告的管線字串、節點片段、重現命令和序列化的 JSON 會被刪除，而不會更改內部保存的可執行管線。

## 程式化的處理方式

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

## 除錯控制項（環境）

重要的環境變數（詳見[架構](/develop-apps/contribute/architecture)，以了解更多資訊）：
- `SIMA_GST_DOT_DIR`：為失敗案例產生 DOT 圖。
- `SIMA_GST_BOUNDARY_PROBES`：邊界流量計數器
- `SIMA_GST_ELEMENT_TIMINGS`：每個元素的時序設定
- `SIMA_GST_FLOW_DEBUG`：每個元素的流程計數器。
- `SIMA_GST_ENFORCE_NAMES`：強制執行命名規則。

為了將經過處理的原始 GStreamer 資訊附加到 `NeatError::what()` 和 `GraphReport.repro_note`，請為執行失敗的指令設定這兩個變數：

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug` 不是 Neat Library 的設定。在正常運作時，請保持關閉詳細輸出；它僅適用於短期的診斷執行，並且可能包含特定於部署的檔案路徑或媒體位址，即使已刪除已識別的憑證欄位。

## 除錯工作流程

1) 首先擷取 `GraphReport.error_code`，然後依據分類法將錯誤事件分組。
2) 擷取 `GraphReport.repro_note`，以便提供具體的背景資訊和內建提示。
3) 擷取管線文字：`Graph::describe_backend()` 或 `last_pipeline()`。
4) 擷取結構化診斷資訊：`MeasureReport::to_text()` 或 `NeatError::report()`。
5) 檢查 `GraphReport.bus`，以找出第一個終端點 `ERROR` 的來源和詳細資訊。
6) 如果程式在執行階段發生停頓或逾時，請啟用邊界/元素探測，以找出程式停止的位置。

建議的支援套件：
- `error_code`
- `repro_note`
- 完整的 `pipeline_string`。
- 前 3-5 個終端公車錯誤 (`GraphReport.bus`)
- 在「執行/驗證」過程中使用的環境覆寫設定。

## 客戶圖表效能成品

對於吞吐量／延遲／功耗報告，建議使用圖表執行結果的 JSON 匯出格式：

```cpp
RunOptions opt;
opt.enable_board_power();        // graph-level power when supported by the board/SOM
Run run = graph.build(opt);

// run your normal push/pull loop inside a measurement window, then:
auto report = run.start_measurement().stop();
std::cout << report.to_text();
```

匯出的內容會明確指定範圍：

- `run.graph_metrics.throughput_fps` 和 `run.graph_metrics.power` 是圖層級的標題。
- `run.node_metrics[]` 僅包含節點/外掛程式的延遲資訊；節點/外掛程式的耗電量資訊則刻意省略。
- `latency_semantics` 和 `aggregation` 告訴您，這些值是代表整個執行期間的資料，還是代表特定時間窗口內的增量變化。
- `plugin_metrics_unattributed[]` 會保留無法明確對應到單一節點的「核心/外掛」資料列。

對於要測量的時間範圍，請使用 `Run::start_measurement()`，並將傳回的 `MeasureReport` 傳遞給
`run_to_json(run, report, ...)` / `save_run_json(run, report, ...)`。由於無法在沒有時間範圍內局部計數器的情況下精確地減去累計的最小值/最大值計數器，因此標記為不可用的測量時間範圍節點為 `min_ms`/`max_ms`。

重要提示：目前的 DVT 板可以驗證選項設定和 JSON 結構，但其功率讀數不被視為數值上可靠。SOM 硬體是預期的用於功率數值驗證的平台。

## 常見問題 → 解決方案

| 症狀 | 可能原因 | 解決方案 |
| --- | --- | --- |
| `missing ... plugin` | GStreamer 外掛程式未找到 | 請檢查 `GST_PLUGIN_PATH`，並執行 `gst-inspect-1.0 <plugin>`。 |
| `appsink 'mysink' not found` | 缺少終端節點 `Output()` | 請確認 `Output` 是執行/建置管線中的最後一個節點 |
| `caps_override is set; renegotiation disabled` | 已固定大小寫 | 移除 `caps_override` 或維持輸入的大小寫設定 |
| `tensor caps change not supported` | 在執行階段變更張量形狀/資料類型 | 維持張量形狀/資料類型穩定（不重新協商） |

如需了解結構化外掛程式錯誤和可操作的提示，請參閱[疑難排解](/reference/troubleshooting)（疑難排解）。
