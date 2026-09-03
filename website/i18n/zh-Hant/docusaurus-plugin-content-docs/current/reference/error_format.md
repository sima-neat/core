---
title: "外掛程式錯誤格式"
description: "來自外掛程式失敗的結構化錯誤欄位"
sidebar_position: 8
---

# 外掛程式錯誤格式

當一個外掛程式遇到致命狀況時，它會在 GStreamer 匯流排上發布一個 `GST_MESSAGE_ERROR`。Neat 將這個錯誤升級為 `NeatError`，並保留支援的結構化詳細資訊，以便進行分類和呈現。

## 錯誤領域與錯誤碼

以下是各個外掛程式中建議使用的網域/代碼：

- 設定解析/驗證：`GST_RESOURCE_ERROR_SETTINGS`
- 缺少檔案：`GST_RESOURCE_ERROR_NOT_FOUND`
- 調度員目前無法使用：`GST_RESOURCE_ERROR_BUSY`；請使用。
  `GST_RESOURCE_ERROR_NOT_FOUND` 僅限於搭配特定分派器的診斷 ID，或結構化的分派器欄位。
- 設定失敗：`GST_RESOURCE_ERROR_NO_SPACE_LEFT`
- 大小寫/協商錯誤：`GST_STREAM_ERROR_FORMAT`
- 執行階段處理失敗：`GST_STREAM_ERROR_FAILED`

## 具有版本控制的結構化詳細資訊

新的 Neat 外掛程式錯誤會附加一個名為 `simaai-neat-error` 的 `GstStructure`。版本 1 包含一個無符號整數欄位 `neat-schema-version=1`。核心會從版本 1 讀取結構化欄位，如果版本未知或遺失，則會回退到普通的 GStreamer 域、程式碼、訊息和偵錯字串。這可防止未來的結構描述被誤用舊的假設來解讀。

常見欄位：
- `neat-schema-version`
- `neat-diagnostic-id`
- `neat-reason`
- `plugin`
- `node`
- `stage`
- `graph-id`
- `frame-id`
- `stream-id`
- `input-caps`
- `output-caps`
- `allocator`
- `dispatcher-error`

輸入容量錯誤也會提供 `actual-width`、`actual-height`、`actual-stride`、
`maximum-width`、`maximum-height`、`maximum-stride`、`resize-width`、`resize-height`、
`required-bytes`、`allocated-bytes` 和 `input-format`。

輸入合約錯誤也會提供 `input-name`、`segment-name`、`required-bytes`、`actual-bytes`、
`expected-shape`、`expected-layout`、`expected-dtype`、`received-shape`、`received-layout` 和
`received-dtype`。版面設定欄位可區分諸如 `[3, 224, 224]`（`CHW`）和
`[224, 224, 3]`（`HWC`）等形狀。

較舊的外掛程式可能會在調試字串中放置一個以空格分隔的 `key='value'` 列表。核心繼續使用這些欄位作為相容性的後備方案。

## 範例

```text
simaai-neat-error, neat-schema-version=(uint)1,
neat-diagnostic-id=(string)neatprocesscvu.input_contract_mismatch,
plugin=(string)neatprocesscvu, node=(string)model_0,
expected-shape=(string)"[3, 224, 224]", expected-layout=(string)CHW,
expected-dtype=(string)Float32, received-shape=(string)"[224, 224, 3]",
received-layout=(string)HWC, received-dtype=(string)UInt8;
```

## 備註

- 預設情況下，`NeatError::what()` 包含標準化的錯誤碼，以及使用者可讀的錯誤訊息。
  上下文、修正措施和診斷 ID。它省略了原始的 GStreamer 訊息和除錯字串。
- 將 `SIMA_NEAT_VERBOSE_LEVEL=2` 和 `SIMA_NEAT_VERBOSE_TOPICS=gstreamer` 設定為一個較短的值。
  診斷執行。這會將經過處理的技術細節附加到 `NeatError::what()` 和 `GraphReport.repro_note`。`NeatError::report()` 仍然是診斷的結構化介面。
- 「`NEAT_LOG_LEVEL=debug`」並非「Neat Library」的設定。
- URI 中的使用者資訊和已識別的憑證欄位，包括 `auth`、`playback-token`、`hdnts`。
  `stream-key` 和 `tkn`——會在用於報告的管線字串、重現命令、結構化詳細資訊和 JSON 中被遮蔽。在分享支援套件之前，請先檢查其中是否包含特定部署的檔案路徑和媒體位址。
