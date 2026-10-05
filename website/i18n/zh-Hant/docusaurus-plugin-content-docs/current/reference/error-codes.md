---
title: "錯誤碼目錄"
description: "穩定的框架錯誤碼、其發生時機，以及如何因應"
sidebar_position: 7
---

# 錯誤碼目錄

Neat 透過 `NeatError` 與 `PullError` 呈現具型別的失敗。每個失敗都會提供穩定的
錯誤碼、人類可讀的訊息，以及（若有的話）帶有結構化脈絡的 `GraphReport`。

請使用錯誤碼進行程式化的分類處理，並將訊息顯示給開發人員。完整的
公開常數集合位於
[`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h)。

## 行為上的破壞性變更與遷移

診斷分類法現在會保留特定的 GStreamer 根本原因。公開方法的簽章
維持不變，但比對確切錯誤字串的程式碼可能需要遷移：

| 先前的比對結果 | 現在傳回的更具體錯誤碼 | 遷移方式 |
| --- | --- | --- |
| 執行階段 GStreamer 協商錯誤時的 `misconfig.caps` | `misconfig.media_caps`；若只有格式不相容，則為 `misconfig.media_format` | 處理媒體錯誤碼。`misconfig.caps` 僅保留給框架驗證 caps 覆寫與相鄰節點合約時使用。 |
| 每次 `gst_parse_launch` 失敗時的 `build.parse_launch` | `build.plugin_missing`、`build.property_invalid` 或 `build.pipeline_syntax` | 處理特定的建置錯誤碼。保留 `build.parse_launch` 作為未分類剖析器失敗的後備。 |
| 傳播而來的匯流排失敗時的 `runtime.pull` | 根本原因錯誤碼，例如 `misconfig.media_caps`、`io.rtsp_connection_failed` 或 `resource.output_pool_exhausted` | 處理根本原因錯誤碼，並保留預設分支。對於沒有特定原因的本機 pull 失敗，`runtime.pull` 仍是後備。 |

請使用 C++ 或 Python 常數，而不要重複撰寫字串常值。務必為較新的 Neat Library 建置所引入的錯誤碼
保留預設路徑。

## 公開常數

兩種語言的 API 都提供相同的值：

| 錯誤碼 | C++ | Python |
| --- | --- | --- |
| `misconfig.pipeline_shape` | `error_codes::kPipelineShape` | `pyneat.ERROR_PIPELINE_SHAPE` |
| `misconfig.caps` | `error_codes::kCaps` | `pyneat.ERROR_CAPS` |
| `misconfig.input_shape` | `error_codes::kInputShape` | `pyneat.ERROR_INPUT_SHAPE` |
| `misconfig.runtime_abi_mismatch` | `error_codes::kRuntimeAbiMismatch` | `pyneat.ERROR_RUNTIME_ABI_MISMATCH` |
| `misconfig.graph_element_name` | `error_codes::kGraphElementName` | `pyneat.ERROR_GRAPH_ELEMENT_NAME` |
| `misconfig.media_caps` | `error_codes::kMediaCaps` | `pyneat.ERROR_MEDIA_CAPS` |
| `misconfig.media_format` | `error_codes::kMediaFormat` | `pyneat.ERROR_MEDIA_FORMAT` |
| `misconfig.input_capacity` | `error_codes::kInputCapacity` | `pyneat.ERROR_INPUT_CAPACITY` |
| `misconfig.tensor_dtype_missing` | `error_codes::kTensorDtypeMissing` | `pyneat.ERROR_TENSOR_DTYPE_MISSING` |
| `misconfig.option_out_of_range` | `error_codes::kOptionOutOfRange` | `pyneat.ERROR_OPTION_OUT_OF_RANGE` |
| `build.parse_launch` | `error_codes::kParseLaunch` | `pyneat.ERROR_PARSE_LAUNCH` |
| `build.pipeline_syntax` | `error_codes::kPipelineSyntax` | `pyneat.ERROR_PIPELINE_SYNTAX` |
| `build.plugin_missing` | `error_codes::kPluginMissing` | `pyneat.ERROR_PLUGIN_MISSING` |
| `build.property_invalid` | `error_codes::kPropertyInvalid` | `pyneat.ERROR_PROPERTY_INVALID` |
| `runtime.pull` | `error_codes::kRuntimePull` | `pyneat.ERROR_RUNTIME_PULL` |
| `runtime.element_failed` | `error_codes::kRuntimeElementFailed` | `pyneat.ERROR_RUNTIME_ELEMENT_FAILED` |
| `runtime.output_timeout` | `error_codes::kOutputTimeout` | `pyneat.ERROR_OUTPUT_TIMEOUT` |
| `runtime.unexpected_eos` | `error_codes::kUnexpectedEos` | `pyneat.ERROR_UNEXPECTED_EOS` |
| `io.parse` | `error_codes::kIoParse` | `pyneat.ERROR_IO_PARSE` |
| `io.open` | `error_codes::kIoOpen` | `pyneat.ERROR_IO_OPEN` |
| `io.file_not_found` | `error_codes::kFileNotFound` | `pyneat.ERROR_FILE_NOT_FOUND` |
| `io.permission_denied` | `error_codes::kPermissionDenied` | `pyneat.ERROR_PERMISSION_DENIED` |
| `io.rtsp_connection_failed` | `error_codes::kRtspConnectionFailed` | `pyneat.ERROR_RTSP_CONNECTION_FAILED` |
| `io.camera_not_found` | `error_codes::kCameraNotFound` | `pyneat.ERROR_CAMERA_NOT_FOUND` |
| `io.model_not_found` | `error_codes::kModelNotFound` | `pyneat.ERROR_MODEL_NOT_FOUND` |
| `io.source_ended` | `error_codes::kSourceEnded` | `pyneat.ERROR_SOURCE_ENDED` |
| `io.response_too_large` | `error_codes::kResponseTooLarge` | `pyneat.ERROR_RESPONSE_TOO_LARGE` |
| `codec.invalid_h264_stream` | `error_codes::kInvalidH264Stream` | `pyneat.ERROR_INVALID_H264_STREAM` |
| `codec.decode_failed` | `error_codes::kDecodeFailed` | `pyneat.ERROR_DECODE_FAILED` |
| `codec.encode_failed` | `error_codes::kEncodeFailed` | `pyneat.ERROR_ENCODE_FAILED` |
| `resource.memory_allocation_failed` | `error_codes::kMemoryAllocationFailed` | `pyneat.ERROR_MEMORY_ALLOCATION_FAILED` |
| `resource.device_memory_exhausted` | `error_codes::kDeviceMemoryExhausted` | `pyneat.ERROR_DEVICE_MEMORY_EXHAUSTED` |
| `resource.output_pool_exhausted` | `error_codes::kOutputPoolExhausted` | `pyneat.ERROR_OUTPUT_POOL_EXHAUSTED` |
| `resource.buffer_too_small` | `error_codes::kBufferTooSmall` | `pyneat.ERROR_BUFFER_TOO_SMALL` |
| `resource.disk_full` | `error_codes::kDiskFull` | `pyneat.ERROR_DISK_FULL` |
| `infra.dispatcher_unavailable` | `error_codes::kDispatcherUnavailable` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE` |
| `infra.accelerator_execution_failed` | `error_codes::kAcceleratorExecutionFailed` | `pyneat.ERROR_ACCELERATOR_EXECUTION_FAILED` |
| `infra.peripheral_daemon_unavailable` | `error_codes::kPeripheralDaemonUnavailable` | `pyneat.ERROR_PERIPHERAL_DAEMON_UNAVAILABLE` |
| `infra.peripheral_daemon_timeout` | `error_codes::kPeripheralDaemonTimeout` | `pyneat.ERROR_PERIPHERAL_DAEMON_TIMEOUT` |
| `DispatcherUnavailable`（舊版） | `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## 設定錯誤

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 圖的拓撲無效，或缺少輸入/輸出邊界。 | 修正圖的連接，以及必要的 `Input` 或 `Output` 節點。 |
| `misconfig.caps` | 在框架驗證期間，caps 覆寫或相鄰節點合約不相容。 | 讓宣告的格式、尺寸、速率與相鄰節點合約保持一致。 |
| `misconfig.input_shape` | 輸入張量不符合預期的形狀或資料型別。 | 提供預期的輸入，或透過模型選項設定模型前處理。 |
| `misconfig.runtime_abi_mismatch` | Neat 與已安裝的執行階段外掛程式使用不相容的 ABI。 | 安裝相符的 Neat Library 與執行階段外掛程式建置。 |
| `misconfig.graph_element_name` | 自訂片段包含無法指派穩定節點名稱的元素。 | 為自訂元素提供穩定且唯一的名稱。 |
| `misconfig.media_caps` | 相連的 GStreamer 階段需要不相容的媒體 caps。 | 讓各階段保持一致，或插入所需的轉換、縮放或速率轉換節點。 |
| `misconfig.media_format` | 相連的階段需要不相容的媒體格式。 | 設定共同的格式，或加入明確的格式轉換。 |
| `misconfig.input_capacity` | 來源影像超過已設定的前處理輸入容量。 | 增加 `input_max_width` 與 `input_max_height`，或在模型階段之前縮放來源。 |
| `misconfig.tensor_dtype_missing` | 張量合約省略了其資料型別或格式。 | 在上游張量合約中宣告受支援的資料型別。 |
| `misconfig.option_out_of_range` | 某個選項對目前的輸入合約無效。 | 將該選項設為診斷訊息所示範圍內的值。 |

## 建置失敗

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `build.parse_launch` | GStreamer 無法建置所產生的管線。 | 檢查自訂片段、元素屬性，以及外掛程式是否可用。 |
| `build.pipeline_syntax` | 自訂 GStreamer 片段的語法無效。 | 修正該片段，並以 `gst-launch-1.0` 驗證。 |
| `build.plugin_missing` | 缺少必要的 GStreamer 元素或編解碼器外掛程式。 | 安裝或替換該元件，然後以 `gst-inspect-1.0` 確認。 |
| `build.property_invalid` | 元素屬性的名稱或值無效。 | 以 `gst-inspect-1.0 <element>` 檢查該屬性。 |

## 執行階段失敗

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `runtime.pull` | pull 作業失敗，且沒有更具體的錯誤碼。 | 檢查附加的報告與第一個上游錯誤。 |
| `runtime.element_failed` | 管線階段停止，且沒有更具體的分類。 | 修正所回報階段的設定及其上游輸入。 |
| `runtime.output_timeout` | 在設定的等待時間到期前沒有任何輸出抵達。 | 確認來源資料流與背壓；若預期需要等待，則調整逾時。 |
| `runtime.unexpected_eos` | 管線在產生必要的輸出之前就到達 EOS。 | 檢查輸入是否過早出現 EOS，並確認已提供足夠的輸入。 |

## I/O 失敗

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `io.parse` | Neat 無法剖析 JSON、模型合約或階段設定。 | 驗證設定的語法、結構描述與必要欄位。 |
| `io.open` | Neat 無法開啟檔案、裝置或遠端資源。 | 確認路徑或位址、權限與資源可用性。 |
| `io.file_not_found` | 輸入檔案不存在。 | 修正路徑，並確認檔案存在於 DevKit 上。 |
| `io.permission_denied` | 無法以所需的存取權限開啟檔案或裝置。 | 修正所回報資源的擁有權或權限。 |
| `io.rtsp_connection_failed` | Neat 無法連線至 RTSP 來源。 | 確認 URL、伺服器、網路連通性與認證資訊。 |
| `io.camera_not_found` | 要求的相機無法使用。 | 選取可用的相機，或使用預設相機。 |
| `io.model_not_found` | 要求的模型封存檔不存在。 | 修正模型路徑，並確認已安裝該封存檔。 |
| `io.source_ended` | 輸入來源到達其正常結尾。 | 停止取用該來源；若應用程式預期還有更多資料，則提供額外的輸入。 |
| `io.response_too_large` | 有界限的本機通訊協定回應超過其明定的大小上限。 | 縮減目錄大小，或安裝相符的客戶端與服務版本。 |

## 管線具體化失敗

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 管線拓撲無效，或在 GStreamer 建構之後，最終的元素名稱重複、模稜兩可或遺失。 | 為每個明確的元素在其具體化的區段內指定唯一的簡短名稱。讓 `name=` 宣告與具名 pad 參照保持同步。 |
| `build.parse_launch` | GStreamer 因語法、外掛程式或屬性無效，而無法剖析或建構最終的 launch 字串。 | 檢查 `GraphReport::pipeline_string`；以 `gst-launch-1.0` 驗證片段，並以 `gst-inspect-1.0` 驗證外掛程式。 |

這些檢查會在 `Graph::build()` 期間自動執行。對於依輸入而定的相連區段，當第一個輸入使該區段具體化時，
也可能出現相同的錯誤碼與 `GraphReport`。

## 編解碼器失敗

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `codec.invalid_h264_stream` | 輸入不包含有效的 H.264 影格。 | 提供完整的 H.264 串流，並確認所設定的編解碼器。 |
| `codec.decode_failed` | 解碼器無法解碼已接受的串流。 | 確認編解碼器，並檢查編碼後的輸入是否完整且未損毀。 |
| `codec.encode_failed` | 編碼器無法編碼所提供的影格。 | 確認輸入格式、解析度與編碼器設定。 |

## 資源失敗

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `resource.memory_allocation_failed` | 必要的記憶體分配失敗，且沒有裝置特定的原因。 | 減少串流數量、解析度或緩衝量，並釋放其他工作負載所使用的記憶體。 |
| `resource.device_memory_exhausted` | 裝置的連續 DMA/CMA 記憶體已耗盡。 | 減少同時進行的串流、輸入解析度或緩衝區深度。 |
| `resource.output_pool_exhausted` | 所有輸出緩衝區仍在使用中。 | 及早釋放零複製輸出，或改用自有副本。 |
| `resource.buffer_too_small` | 緩衝區小於其宣告的影格或張量酬載。 | 修正上游的尺寸與步幅，或分配所需的位元組數。 |
| `resource.disk_full` | 寫入失敗，因為目的地的可用空間不足。 | 釋放空間，或選擇其他目的地。 |

## 基礎架構失敗

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat 無法取得加速器執行階段。 | 確認 DevKit 相容性，並停止獨占加速器的工作負載。 |
| `infra.accelerator_execution_failed` | 加速器無法執行模型階段。 | 重新啟動管線，並減少同時執行的加速器工作負載。 |
| `infra.peripheral_daemon_unavailable` | 本機 SiMa Sentinel API socket 不存在或拒絕連線、Sentinel 的周邊裝置探索已停用或停止、已安裝的 Sentinel 版本過舊而無法提供周邊裝置目錄，或 Sentinel 傳回了其他非預期的 HTTP 狀態。 | 以 `sima-cli neat install sentinel` 安裝或更新 Sentinel，或啟動 `simaai-sentinel.service`，然後檢查其 journal 日誌。 |
| `infra.peripheral_daemon_timeout` | 有時限的周邊裝置目錄要求未完成。 | 檢查 `simaai-sentinel.service` 與提供者的健康狀態，然後重試。 |

## 內部失敗

| 錯誤碼 | 發生時機 | 處理方式 |
| --- | --- | --- |
| `internal.plugin_failure` | Neat 外掛程式失敗，且沒有使用者可自行處置的分類。 | 擷取附加的 `GraphReport`，並向支援團隊回報此失敗。 |

`DispatcherUnavailable` 是為了相容性而接受的舊版拼寫。新的應用程式應
使用 `infra.dispatcher_unavailable` 與 `error_codes::kDispatcherUnavailable` 常數。

## 以程式化方式處理錯誤

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build();
  // Push and pull application data.
} catch (const simaai::neat::NeatError& error) {
  if (error.report().error_code == simaai::neat::error_codes::kInputShape) {
    handle_input_contract_error(error.report());
  } else {
    throw;
  }
}
```

`PullError.code` 使用相同的常數。請勿剖析 `what()`，也不要比對人類可讀的文字。

## 延伸閱讀

- [診斷與除錯](/reference/diagnostics) — 正式環境訊息、除錯詳細資料，以及
  `GraphReport` 的收集。
- [外掛程式錯誤格式](/reference/error_format) — GStreamer 外掛程式
  錯誤的結構化合約。
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — 具型別的例外。
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — 結構化的錯誤脈絡。
