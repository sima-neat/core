---
title: "錯誤碼目錄"
description: "穩定的框架錯誤碼、它們發生的時機，以及如何處理。"
sidebar_position: 7
---

# 錯誤碼目錄

Neat 透過 `NeatError` 和 `PullError` 呈現類型錯誤。 每次錯誤都會提供一個穩定的錯誤碼、一個人類可讀的訊息，以及（如果有的話）一個帶有結構化上下文的 `GraphReport`。

使用錯誤碼進行程式化的錯誤分類。 將訊息顯示給開發人員。 所有公開常數都位於 [`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h) 中。

## 行為上的重大變更與遷移

目前的診斷分類法會保留特定的 GStreamer 根本原因。公開方法的簽名沒有改變，但比較精確錯誤字串的程式碼可能需要進行遷移：

| 上一次的匹配結果 | 現在傳回更精確的程式碼 | 遷移 |
| --- | --- | --- |
| `misconfig.caps` 會導致在執行階段發生的 GStreamer 協商錯誤 | `misconfig.media_caps`，或者僅當格式不相容時，會發生 `misconfig.media_format` | 處理媒體程式碼。僅將 `misconfig.caps` 保留用於框架驗證，以驗證 caps 覆寫和相鄰節點合約。|
| 每個 `gst_parse_launch` 失敗都匹配 `build.parse_launch` | `build.plugin_missing`、`build.property_invalid` 或 `build.pipeline_syntax` | 處理特定的建構錯誤碼。將 `build.parse_launch` 作為未分類的解析器失敗的後備方案。 |
| 對於傳播的匯流排故障，執行 `runtime.pull` |。根本原因代碼，例如 `misconfig.media_caps`、`io.rtsp_connection_failed` 或 `resource.output_pool_exhausted` |。處理這些根本原因代碼，並保留一個預設分支。`runtime.pull` 仍然是針對沒有特定原因的本地拉取失敗的後備方案。|

請使用 C++ 或 Python 的常數，而不是重複使用字串字面量。務必保留一個預設路徑，以供較新版本的 Neat Library 提供的程式碼使用。

## 公開常數

這兩種語言 API 都提供相同的數值：

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
| `DispatcherUnavailable`（舊版）| `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## 設定錯誤

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 此圖的拓撲結構無效，或缺少輸入/輸出邊界。| 請更正圖的連接，以及必要的 `Input` 或 `Output` 節點。|
| 在框架驗證期間，覆蓋大小寫或相鄰節點合約是不相容的。`misconfig.caps` | 請調整宣告的格式、尺寸、速率和相鄰節點合約。| |
| `misconfig.input_shape` | 輸入的張量與預期的形狀或資料類型不符。| 請提供預期的輸入，或透過模型選項來設定模型預處理。|
| `misconfig.runtime_abi_mismatch` | Neat 和已安裝的執行階段外掛程式使用不相容的 ABI。| 請安裝相符的 Neat Library 和執行階段外掛程式版本。|
| `misconfig.graph_element_name` | 自訂片段包含一個無法指定穩定節點名稱的元素。| 請為自訂元素指定穩定且獨一無二的名稱。|
| `misconfig.media_caps` | 已連接的 GStreamer 階段需要不相容的媒體參數。| 請調整這些階段，或插入所需的轉換、縮放或幀率轉換節點。|
| `misconfig.media_format` | 連接的階段需要不相容的媒體格式。| 請設定一個通用的格式，或新增明確的格式轉換。|
| `misconfig.input_capacity` | 來源影像超過已設定的預處理輸入容量。| 請增加 `input_max_width` 和 `input_max_height`，或在模型階段之前縮放來源影像。|
| `misconfig.tensor_dtype_missing` | 張量合約缺少其資料類型或格式。| 在上游張量合約中宣告一個受支援的資料類型。|
| `misconfig.option_out_of_range` | 某個選項對於目前的輸入合約而言無效。| 請將該選項設定為診斷訊息中顯示的範圍內的值。|

## 建構失敗

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `build.parse_launch` | GStreamer 無法建構產生的管線。| 請檢查自訂片段、元素屬性和外掛程式的可用性。|
| `build.pipeline_syntax` | 自訂的 GStreamer 片段語法無效。| 請修正該片段，並使用 `gst-launch-1.0` 進行驗證。|
| `build.plugin_missing` | 缺少必要的 GStreamer 元素或編解碼器外掛程式。| 請安裝或替換該元件，然後使用 `gst-inspect-1.0` 進行驗證。|
| `build.property_invalid` | 元素屬性名稱或值無效。| 請使用 `gst-inspect-1.0 <element>` 檢查該屬性。|

## 執行階段錯誤

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `runtime.pull` | 由於缺少更具體的錯誤碼，拉取操作失敗。| 請檢查隨附的報告以及第一個上游錯誤。|
| `runtime.element_failed` | 管線的某個階段在沒有更明確分類的情況下停止。| 請修正已報告的階段設定及其上游輸入。|
| 在設定的等待時間結束之前，沒有任何輸出產生。`runtime.output_timeout`。| 請檢查資料流和反壓機制，或者在預期需要更長等待時間時調整逾時設定。|。|
| `runtime.unexpected_eos` | 管線在產生必要輸出之前就已達到資料流結束 (EOS)。| 請檢查輸入資料，確認是否過早達到資料流結束，並確認已提供足夠的輸入資料。|

## I/O 錯誤

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `io.parse` | Neat 無法解析 JSON、模型合約或階段設定。 | 驗證設定語法、結構描述和必要欄位。 |
| `io.open` | Neat 無法開啟檔案、裝置或遠端資源。| 請檢查路徑或位址、權限以及資源是否可用。|
| `io.file_not_found` | 輸入檔案不存在。| 請檢查檔案路徑，並確認檔案是否存在於 DevKit 中。|
| 由於權限不足，無法開啟檔案或裝置。`io.permission_denied` | 請檢查並更正相關資源的擁有者或權限。| |
| `io.rtsp_connection_failed` | Neat 無法連接到 RTSP 來源。| 請檢查 URL、伺服器、網路可達性以及憑證。|
| `io.camera_not_found` | 要求的相機目前無法使用。| 請選擇可用的相機，或使用預設相機。|
| `io.model_not_found` | 要求的模型封存檔不存在。| 請更正模型路徑，並確認已安裝該封存檔。|
| `io.source_ended` | 一個輸入來源已達到其正常結束點。| 停止從該來源讀取資料，或者如果應用程式需要更多資料，請提供額外的輸入。|

## 管線實體化失敗

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `misconfig.pipeline_shape` | 管線拓撲結構無效，或者在 GStreamer 建構之後，最終元件的名稱重複、不明確或遺失。| 為每個明確元件賦予一個獨一無二的簡短名稱，使其位於其已實體的區段中。確保 `name=` 宣告和具名連接埠參考保持同步。|
| `build.parse_launch` | GStreamer 無法解析或建構最終的管線字串，因為語法、外掛程式或屬性無效。| 檢查 `GraphReport::pipeline_string`；使用 `gst-launch-1.0` 驗證片段，並使用 `gst-inspect-1.0` 驗證外掛程式。|

這些檢查會在 `Graph::build()` 過程中自動進行。對於取決於輸入的連通片段，當第一個輸入產生該片段時，相同的程式碼和 `GraphReport` 就可以顯示出來。

## 編碼器發生錯誤。

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `codec.invalid_h264_stream` | 輸入內容不包含有效的 H.264 影格。| 請提供完整的 H.264 串流，並確認已設定的編解碼器。|
| `codec.decode_failed` | 解碼器無法解碼接收到的資料流。| 請確認使用的編碼器，並檢查編碼後的輸入資料是否完整且未損毀。|
| `codec.encode_failed` | 編碼器無法編碼提供的影格。| 請檢查輸入格式、解析度和編碼器設定。|

## 資源失敗

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `resource.memory_allocation_failed` | 必要的記憶體設定失敗，且沒有特定裝置原因。| 減少串流數量、解析度或緩衝區大小，並釋放其他工作負載所使用的記憶體。|
| `resource.device_memory_exhausted` | 裝置的連續 DMA/CMA 記憶體已用盡。| 減少同時處理的串流、輸入解析度或緩衝區深度。|
| `resource.output_pool_exhausted` | 所有輸出緩衝區仍在使用中。| 請盡快釋放零拷貝輸出，或使用擁有權的副本。|
| `resource.buffer_too_small` | 緩衝區小於其宣告的框架或張量有效載荷。| 請修正上游的維度和步幅，或設定所需的位元組數。|
| `resource.disk_full` | 寫入作業失敗，因為目的地沒有足夠的可用空間。| 請釋放可用空間或選擇另一個目的地。|

## 基礎設施故障

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat 無法取得加速器的執行階段。| 請確認 DevKit 的相容性，並停止獨佔使用加速器的工作負載。|
| `infra.accelerator_execution_failed` | 加速器無法執行模型階段。| 請重新啟動管線，並減少同時執行的加速器工作負載。|

## 內部錯誤

| 程式碼 | 觸發時 | 處理方式 |
| --- | --- | --- |
| `internal.plugin_failure` | 一個 Neat 外掛程式在沒有使用者可採取任何行動的情況下發生錯誤。| 擷取附加的 `GraphReport` 圖，並將錯誤回報給支援團隊。|

`DispatcherUnavailable` 是一個為了相容性而保留的舊拼寫。新的應用程式應該使用 `infra.dispatcher_unavailable`，以及 `error_codes::kDispatcherUnavailable` 常數。

## 以程式化的方式處理錯誤。

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

`PullError.code` 使用相同的常數。請勿剖析 `what()` 或比對人類可讀的文字。

## 更多資訊

- [診斷與除錯](/reference/diagnostics) — 產品訊息、除錯詳細資料等。
  `GraphReport` 集合。
- [外掛程式錯誤格式](/reference/error_format) — 這是 GStreamer 外掛程式的結構化錯誤訊息格式。
  錯誤。
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror)——這是已輸入的例外狀況。
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — 結構化的錯誤資訊。
