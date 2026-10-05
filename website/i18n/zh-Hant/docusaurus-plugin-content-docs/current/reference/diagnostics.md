---
title: "診斷與除錯"
description: "收集 GraphReport 診斷資訊、執行階段錯誤碼，以及圖指標成品"
sidebar_position: 9
---

# 診斷與除錯

## GraphReport

`GraphReport` 會擷取結構化的診斷資訊：
- 管線字串（用於重現）
- 標準 `error_code`（供機器分類處理）
- `repro_note`（人類可讀的摘要 + 提示）
- 節點報告與其擁有的元素名稱
- 匯流排訊息與錯誤詳細資料
- 選用的資料流/計時計數器

發生錯誤時，`NeatError` 會攜帶一個 `GraphReport`，供您記錄或
序列化。

## 錯誤分類法

框架錯誤使用穩定的錯誤碼系列：

| 錯誤碼 | 意義 | 常見修正方式 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 違反節點順序/形狀合約 | 推送式管線須以 `Input()` 開頭，拉取式管線須以 `Output()` 結尾 |
| `misconfig.caps` | 框架 caps 覆寫或相鄰節點合約不相符 | 讓 `caps_override` 與宣告的節點合約保持一致 |
| `misconfig.input_shape` | 輸入張量/影格/樣本的形狀或資料型別不符合模型合約 | 提供預期的形狀與資料型別，或設定模型前處理 |
| `misconfig.runtime_abi_mismatch` | Neat 與執行階段外掛程式使用不相容的 ABI | 安裝版本相符的 Neat Library 與執行階段 |
| `misconfig.graph_element_name` | 無法為自訂元素指派穩定的節點名稱 | 為自訂元素提供穩定且唯一的名稱 |
| `misconfig.input_capacity` | 來源影像超過前處理輸入容量 | 增加 `input_max_width` / `input_max_height`，或在模型階段之前縮放 |
| `misconfig.media_caps` | 相鄰的 GStreamer 階段需要不相容的媒體 caps | 讓格式、解析度與影格率保持一致，或插入轉換 |
| `misconfig.media_format` | 某個階段收到不受支援的媒體格式 | 設定受支援的格式，或插入格式轉換 |
| `misconfig.tensor_dtype_missing` | 張量合約沒有 dtype/格式 | 在上游合約中宣告受支援的張量 dtype |
| `misconfig.option_out_of_range` | 某個階段選項對目前的張量無效 | 選擇診斷訊息所示範圍內的值 |
| `build.parse_launch` | `gst_parse_launch` 失敗，且沒有更具體的分類 | 檢查附加的報告以了解剖析器脈絡 |
| `build.pipeline_syntax` | 自訂 GStreamer 片段的語法無效 | 修正該片段，並以 `gst-launch-1.0` 驗證 |
| `build.plugin_missing` | 未安裝必要的 GStreamer 元素或編解碼器外掛程式 | 安裝/替換該元件，並以 `gst-inspect-1.0` 確認 |
| `build.property_invalid` | 元素屬性未知或無效 | 以 `gst-inspect-1.0` 檢查屬性名稱與值 |
| `runtime.pull` | pull 失敗，且沒有更具體的根本原因 | 檢查附加的報告與第一個上游錯誤 |
| `runtime.element_failed` | 某個階段失敗，且沒有更具體的對應 | 修正所回報的階段及其上游輸入 |
| `runtime.output_timeout` | 在設定的逾時之前沒有任何輸出抵達 | 確認來源資料流；若等待屬於預期，則增加逾時 |
| `runtime.unexpected_eos` | 管線在產生必要的輸出之前就到達 EOS | 檢查來源是否過早出現 EOS，並提供足夠的輸入 |
| `io.parse` | JSON 或階段設定的剖析/結構描述失敗 | 驗證設定語法與必要欄位 |
| `io.open` | 圖儲存/載入時的檔案開啟/讀取/寫入失敗 | 檢查路徑是否存在、權限與儲存裝置的健康狀態 |
| `io.file_not_found` | 輸入檔案不存在 | 修正路徑，並確認檔案存在於 DevKit 上 |
| `io.permission_denied` | 檔案或裝置無法讀取 | 修正擁有權/權限 |
| `io.rtsp_connection_failed` | 無法連線至 RTSP 來源 | 確認 URL、網路連通性、伺服器與認證資訊 |
| `io.camera_not_found` | 要求的相機無法使用 | 選取已回報的相機，或使用預設相機 |
| `io.model_not_found` | 要求的模型封存檔不存在 | 修正模型路徑，並確認已安裝該封存檔 |
| `io.source_ended` | 輸入來源到達其正常結尾 | 停止取用該來源，或提供更多輸入 |
| `io.response_too_large` | 有界限的本機通訊協定回應超過其大小上限 | 安裝相符的客戶端與服務版本 |
| `codec.invalid_h264_stream` | 輸入沒有有效的 H.264 影格 | 提供完整的 H.264 串流，或修正編解碼器 |
| `codec.decode_failed` | 解碼器在接受串流後失敗 | 確認編解碼器與輸入的完整性 |
| `codec.encode_failed` | 編碼器無法編碼所提供的影格 | 確認輸入格式、解析度與編碼器設定 |
| `resource.memory_allocation_failed` | 必要的記憶體分配失敗 | 減少工作負載的記憶體用量，並釋放其他應用程式或管線所使用的記憶體 |
| `resource.device_memory_exhausted` | 裝置 DMA/CMA 分配失敗 | 減少同時進行的串流、解析度或緩衝量 |
| `resource.output_pool_exhausted` | 所有輸出緩衝區仍在使用中 | 釋放零複製輸出，或改用自有副本 |
| `resource.buffer_too_small` | 緩衝區小於其宣告的酬載 | 修正尺寸/步幅，或分配所需的位元組數 |
| `resource.disk_full` | 儲存空間已滿，導致寫入失敗 | 釋放空間，或選擇其他目的地 |
| `infra.dispatcher_unavailable` | 無法取得加速器執行階段 | 停止相互競爭的工作負載，並確認 DevKit 相容性 |
| `infra.accelerator_execution_failed` | 加速器無法執行模型階段 | 重新啟動管線，並減少同時執行的加速器工作 |
| `infra.peripheral_daemon_unavailable` | SiMa Sentinel 無法提供周邊裝置目錄 | 以 `sima-cli neat install sentinel` 安裝或更新 Sentinel，或啟動 `simaai-sentinel.service` |
| `infra.peripheral_daemon_timeout` | 周邊裝置目錄要求未及時完成 | 檢查 `simaai-sentinel.service` 及其 journal 日誌，然後重試 |
| `DispatcherUnavailable` | `infra.dispatcher_unavailable` 的舊版拼寫 | 將處理常式遷移至標準的基礎架構錯誤碼 |
| `internal.plugin_failure` | 外掛程式失敗，且沒有使用者可自行處置的分類 | 擷取報告並聯絡支援團隊 |

`PullError.code` 使用相同的分類法（不僅限於例外路徑）。
請參閱 [錯誤碼目錄](/reference/error-codes)，了解 C++ 與 Python 常數名稱，以及
針對曾比對先前粗略錯誤碼之應用程式的遷移指引。

正式環境訊息會刻意省略 GStreamer 內部細節。外掛程式除錯
詳細程度會加入原始 GError 網域/錯誤碼、元素工廠、訊息，以及
結構化的外掛程式詳細資料。可辨識的認證資訊與 URL 機密參數（包括 URI
userinfo、`auth`、`playback-token`、`hdnts`、`stream-key` 與 `tkn`）會在任一
形式儲存之前遮蔽。面向報告的管線字串、節點片段、重現指令，以及
序列化 JSON 都會經過遮蔽，而不會變更內部保存的可執行管線。

## 程式化處理

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

## 除錯控制項（環境變數）

重要的環境變數（詳細資訊請參閱 [架構](/develop-apps/contribute/architecture)）：
- `SIMA_GST_DOT_DIR`：為失敗寫出 DOT 圖
- `SIMA_GST_BOUNDARY_PROBES`：邊界資料流計數器
- `SIMA_GST_ELEMENT_TIMINGS`：各元素的計時
- `SIMA_GST_FLOW_DEBUG`：各元素的資料流計數器
- `SIMA_GST_ENFORCE_NAMES`：強制執行命名合約

若要將經過遮蔽的原始 GStreamer 脈絡附加到 `NeatError::what()` 與
`GraphReport.repro_note`，請為失敗的指令同時設定這兩個變數：

```bash
SIMA_NEAT_VERBOSE_LEVEL=2 \
SIMA_NEAT_VERBOSE_TOPICS=gstreamer \
./your-neat-application
```

`NEAT_LOG_LEVEL=debug` 不是 Neat Library 的設定。正常運作時，請保持詳細輸出
停用；它僅供短時間的診斷執行使用，可能包含部署特定的路徑或
媒體位址，即使可辨識的認證欄位已經遮蔽。

## 除錯工作流程

1) 擷取 `GraphReport.error_code`，並先依分類法將失敗歸類。
2) 擷取 `GraphReport.repro_note`，取得具體脈絡與內建提示。
3) 擷取管線文字：`Graph::describe_backend()` 或 `last_pipeline()`。
4) 擷取結構化診斷資訊：`MeasureReport::to_text()` 或 `NeatError::report()`。
5) 檢查 `GraphReport.bus`，找出第一個終止性 `ERROR` 的來源與詳細資料。
6) 若執行階段停滯或逾時，請啟用邊界/元素探針，以找出資料流停止的位置。

建議的支援資料包：
- `error_code`
- `repro_note`
- 完整的 `pipeline_string`
- 前 3-5 個終止性匯流排錯誤（`GraphReport.bus`）
- 執行/驗證時使用的環境變數覆寫

## 客戶圖效能成品

如需輸送量/延遲/功耗報告，建議使用圖執行的 JSON 匯出：

```cpp
RunOptions opt;
opt.enable_board_power();        // graph-level power when supported by the board/SOM
Run run = graph.build(opt);

// run your normal push/pull loop inside a measurement window, then:
auto report = run.start_measurement().stop();
std::cout << report.to_text();
```

匯出內容會明確區分範圍：

- `run.graph_metrics.throughput_fps` 與 `run.graph_metrics.power` 是圖層級的主要指標。
- `run.node_metrics[]` 只包含節點/外掛程式延遲；刻意不提供節點/外掛程式功耗。
- `latency_semantics` 與 `aggregation` 會告訴您這些值涵蓋整個執行期間，還是量測時間窗內的差值。
- `plugin_metrics_unattributed[]` 會保留無法對應到恰好一個節點的核心/外掛程式資料列。

若要量測特定時間窗，請使用 `Run::start_measurement()`，並將傳回的 `MeasureReport` 傳給
`run_to_json(run, report, ...)` / `save_run_json(run, report, ...)`。量測時間窗內的節點
`min_ms`/`max_ms` 會標示為無法取得，因為若沒有時間窗本地計數器，累計的最小值/最大值計數器無法
精確相減。

功耗附註：目前的 DVT 板可以驗證選項的傳遞路徑與 JSON 結構，但其瓦數
讀數不被視為數值上可靠。SOM 硬體才是用於
功耗數值驗證的預定平台。

## 常見失敗 → 修正方式

| 症狀 | 可能原因 | 修正方式 |
| --- | --- | --- |
| `missing ... plugin` | 找不到 GStreamer 外掛程式 | 檢查 `GST_PLUGIN_PATH`，並執行 `gst-inspect-1.0 <plugin>` |
| `appsink 'mysink' not found` | 缺少終端 `Output()` | 確認 `Output` 是執行/建置管線中的最後一個節點 |
| `caps_override is set; renegotiation disabled` | caps 已固定 | 移除 `caps_override`，或讓輸入 caps 保持固定 |
| `tensor caps change not supported` | 執行階段變更了張量形狀/dtype | 讓張量形狀/dtype 保持穩定（不重新協商） |

如需結構化外掛程式錯誤與可採取行動的提示，請參閱
[疑難排解](/reference/troubleshooting)。

## 平台執行階段復原

在 Platform 3.0.0 上，分派器無法使用是需要調查的錯誤，不是啟動舊版服務的要求。Core 不會因分派器錯誤而初始化 MLA 記憶體或重設遠端處理器。舊版復原程式碼與指令碼已移除；請使用平台核准的復原程序。

若驅動程式回報完成狀態不明，請持續保留其 DMA 緩衝區與原始集區借用權。關閉檔案描述元、停止應用程式或重新啟動服務，都不能證明硬體已停止存取記憶體。請收集失敗報告，並在重試前使用平台核准的復原程序。

### 維持 Core 與 Internals 配對

建置與安裝 Core 時，請使用相符的 B1157 Internals 套件。Core 會分別檢查執行階段設定檔、核心原始碼版本與 SDK sysroot 收據，而不是只看公開 C++ ABI 版本。即使公開 ABI 相同，較舊的套件也不能作為相容替代品。

安裝程式在變更套件前，會檢查套件組中 `neat-runtime` 的設定檔與相符的
`neat-gst-plugins` 版本。平台檢查覆寫不會繞過執行階段配對。若檢查失敗，請取得相符的套件組；不要取代其收據或強制安裝較舊的執行階段。

在板上安裝前，請停止使用 CVU 或硬體編解碼器的應用程式。完整安裝程式會在所有套件安裝完成後啟用已暫存的 EV74 韌體；若這些裝置仍開啟，則拒絕重設 EV74。設定
`NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=OFF` 可讓韌體維持暫存狀態，
稍後再以
`sudo /usr/libexec/sima-neat-firmware/install.sh --activate`
啟用。
