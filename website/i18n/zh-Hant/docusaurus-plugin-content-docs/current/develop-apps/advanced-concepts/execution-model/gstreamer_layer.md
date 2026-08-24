---
title: "在 GStreamer 之下"
description: "這個框架如何抽象化 GStreamer，哪些部分會直接呈現，以及何時應該使用原始的 GStreamer。"
sidebar_position: 3
slug: /develop-apps/advanced-concepts/gstreamer_layer
---

# GStreamer 之下

Neat 框架的管線在 GStreamer 上運行。應用程式可見的幾乎所有內容——`Graph`、`Run`、`Node`——都是對 GStreamer 概念的類型化封裝。本頁說明了分層結構：哪些內容被隱藏，哪些內容沒有被隱藏，以及何時直接使用 GStreamer 才是正確的做法。

## 框架抽象化的內容

| GStreamer 概念 | 框架抽象化 |
|---|---|
| `gst-launch` 文字片段 | `Node::backend_fragment(int node_index)` |
| 元素名稱（`name=…`） | 從 `Node::element_names()` 獲得的確定性 `n<idx>_<role>` |
| 管線字串（連接的片段） | `Graph::add()` 會建立並連接 |
| Caps 協商 | `Graph::build()` 透過 `NodeCapsBehavior` 驗證 caps |
| `gst_pipeline_set_state()` | `Graph::run()` / `Run::start()` |
| 匯流排訊息 | `GraphReport::bus_messages` |
| `appsrc` 推送 API | `Run::push()`（僅適用於具有 `InputRole::Push` 的節點） |
| `appsink` 拉取 API | `Run::pull()` |
| 每個元素的定時 | 從 `Run::start_measurement()` 獲得的 `MeasureReport` |

應用程式永遠不會直接編寫啟動字串，永遠不會直接命名元素，也不會觸及 GStreamer C API。所有內容都透過節點進行。

## 哪些內容會暴露出來

框架不會（也不應該）隱藏一些 GStreamer 概念：

- **Caps 語義**——影片/音訊 cap 攜帶哪些欄位。應用程式程式碼可以讀取 [`FormatTag`](/reference/cppapi/files/include-pipeline-formatspec-h) 並檢查 `Sample` 中繼資料，這反映了相關的 cap 欄位。
- **緩衝區旗標**——不連續性、EOS、間隙。框架會在 `Sample` 上傳播這些旗標，以便應用程式程式碼可以對資料流邊界做出反應。
- **事件順序**——GStreamer 確保事件（caps、區段、EOS）與緩衝區一起按順序流動。框架在拉取端保留這一點。

如果您需要知道已建立圖的確切 GStreamer 啟動字串，請呼叫 `Graph::describe()`——它會產生確定性的 `gst-launch` 重現器，該重現器會逐字節地重新建立管線。

## 何時使用原始 GStreamer

在正常的應用程式程式碼中，您不需要使用它。以下是一些適用情況：

- **自訂 GStreamer 外掛程式**——如果您想要一個框架未作為節點提供的 GStreamer 元素，請編寫一個節點子類，該子類封裝您的外掛程式並發出正確的 `backend_fragment()`。請參閱設計深入探討（§0.10）中的「建立自訂節點」。
- **診斷工具**——`repro_gst_launch` 重現器來自 `Graph::describe()`，它完全是 GStreamer 將使用的啟動字串；您可以將其貼到 `gst-launch-1.0` 中，以進行離線偵錯。
- **外掛程式作者**——SiMa 自己的 GStreamer 外掛程式（`sima*` 系列）記錄在外掛程式 manifest ABI 中（參見 [`gst/SimaPluginStaticManifestAbi.h`](/reference/cppapi/files/include-gst-simapluginstaticmanifestabi-h)），並且由框架自動載入。

## 確定性保證

這個框架的元件命名是確定性的——相同的節點列表，搭配相同的選項，總是會產生相同的 `gst-launch` 字串。這使得：

- `repro_gst_launch` 欄位實際上可以重現。
- 測試快照在不同執行次數之間保持穩定。
- 元件識別（例如，用於測量歸因）對機器來說更易於處理。

慣例是 `n<node_index>_<role>`，其中 `role` 是節點作者選擇的一個簡短且穩定的識別碼。請參閱 [節點 API 群組](/reference/cppapi/groups/nodes)，以了解參與此命名方案的公開節點包裝函式。

## 更多資訊

- “GStreamer 抽象化”——設計深入探討的第 0.8 節。
- [節點 API](/reference/cppapi/groups/nodes)——產生確定性後端片段的具體節點包裝函式。
- [`Graph::describe()`](/reference/cppapi/classes/simaai-neat-graph)——列印啟動字串。
- “SiMa 外掛程式資訊檔”——設計深入探討的第 51 節和第 95 節。
