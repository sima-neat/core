---
title: "每幀屬性"
description: "擷取選定的多部分 HTTP 標頭，並從解碼後的框架中讀取這些標頭（這些標頭是與框架一起傳送的）。"
sidebar_position: 4
slug: /develop-apps/advanced-concepts/frame_attributes
---

# 每幀屬性

有些攝影機會在傳輸過程中描述每一幀：序列號、擷取時間戳、頻道名稱。Neat 可以將選定的這些值傳遞到解碼過程中，並將它們傳回對應的 `Sample`，即該值所屬的幀。

`Sample::attributes` 是一個簡單的字串到字串的映射。Neat 會複製它，並將其與其幀關聯，然後在重新使用目標緩衝區時清除它。它永遠不會解析、合併或重新解釋任何值。鍵和值必須是有效的 UTF-8，並且不能包含嵌入的 NULL 位元組，因為 GStreamer 字串表示形式無法保留它們。

## 保證內容

> 選定的多部分內容標頭將始終與其解碼後的幀關聯，透過預設的 `HttpMjpegDecodedInput` 路徑、佇列/分支以及核心樣本到 GStreamer 的具體邊界。

這是本次發布的全部保證。任何超出此範圍的內容都列在 [不支援的路徑](#unsupported-paths) 下，並且 Neat 會導致圖的建構失敗，而不是靜默地丟棄屬性。

## 啟用擷取

預設情況下，擷取功能已關閉。指定您想要使用的標頭會啟用它。

```cpp
#include "nodes/groups/HttpMjpegDecodedInput.h"

simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions opt;
opt.url = "http://camera.local/stream";
opt.header_capture.headers = {"Image-Index", "Image-Time"};

auto source = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
```

重新閱讀它們：

```cpp
simaai::neat::Sample sample;
if (run.pull(1000, sample) == simaai::neat::PullStatus::Ok) {
  const auto it = sample.attributes.find("image-index");
  if (it != sample.attributes.end()) {
    // it->second is the value this frame was sent with.
  }
}
```

在 Python 中，相同的表面概念如下，其中 `attributes` 是一個動態映射——項目指派會到達底層的 `Sample`，而指派一個字典會取代其內容：

```python
import pyneat

opt = pyneat.HttpMjpegDecodedInputOptions()
opt.url = "http://camera.local/stream"
opt.header_capture.headers = ["Image-Index", "Image-Time"]
source = pyneat.groups.http_mjpeg_decoded_input(opt)

# ... later, on a pulled sample:
index = sample.attributes.get("image-index")

sample.attributes["image-index"] = "42"     # reaches the Sample
sample.attributes = {"image-time": "..."}   # replaces the whole map
```

## 標頭規則

已設定的清單是一個**允許清單**。一個空的清單會完全停用擷取功能，並保持現有的拓撲結構和行為不變。

| 規則 | 行為 |
|---|---|
| 大小寫 | 已設定的名稱和輸出的鍵會正規化為 ASCII 小寫；匹配時不區分大小寫。以小寫鍵讀取屬性。 |
| 允許清單中的重複項 | 在正規化後合併。 |
| 標頭在單個部分中重複 | 最後一個值優先。 |
| 標頭在某個部分中不存在 | 鍵會被省略。它永遠不會從先前的框架繼承。 |
| 標頭存在但為空 | 保留為一個空字串。 |
| 空白字元 | 僅修剪周圍的 SP/HTAB。否則，值不會重新解釋。 |
| MIME 類型 | 必須存在一個 `Content-Type`，且必須為 `image/jpeg`（允許參數）。如果不存在，JPEG 有效載荷檢查將確定部分類型。 |
| JPEG 有效載荷 | 一個部分必須包含精確一個完整的 JPEG，從 SOI 到 EOI。截斷、空白或串連的圖像會導致資料流失敗。 |
| 無效輸入 | 無效的標頭名稱、摺疊的標頭行以及 CR/LF/NUL 注入都會被拒絕——資料流會產生錯誤，而不是正規化為看似安全的形式。 |

使用 `count()` / `get()` 來區分「不存在」和「空白」，而不是通過測試空字串。

### 限制

當以下任何一個限制超過時，解析會失敗，而不是截斷：

| 限制 | 值 |
|---|---|
| `kMultipartHeaderCaptureMaxHeaders` | 64 個選定的標頭名稱 |
| `kMultipartHeaderCaptureMaxNameBytes` | 每一個名稱 128 位元組 |
| `kMultipartHeaderCaptureMaxLineBytes` | 每一個標頭行 8 KiB |
| `kMultipartHeaderCaptureMaxBlockBytes` | 每一個部分標頭區塊 64 KiB |
| 多部分 JPEG 主體 | 每一個 MIME 部分 64 MiB |

一個格式不正確的允許清單會在建構時被拒絕，並產生 `std::invalid_argument`。

## 不支援的路徑

在啟用擷取功能時，`HttpMjpegDecodedInput` 不會建立包含 `use_videoconvert`、`use_videoscale`、`use_videorate` 或 `extra_fragment` 的圖。通過這些元素進行的保留尚未得到驗證，並且明確的建構錯誤比元資料在資料流中悄悄消失要好。

對於從多個輸入建立新的邏輯樣本的節點（模型、連接、聚合器），也不會定義屬性。這些節點不會合併屬性。

## 如何保持關聯

啟用擷取功能的圖使用一個私有的進程內元素，該元素解析部分邊界**和**部分標頭，並使用一個狀態機器，因此一個部分的標頭會附加到攜帶其位元組的緩衝區中；沒有任何副通道可能導致漂移。該元素會輸出完整的已解析 JPEG 框架，因此 `jpegparse` 不會插入到啟用擷取功能的路徑中。如果附加選定的屬性失敗，則該框架不會被傳遞，並且資料流會報告錯誤。

透過解碼，外掛程式會擷取每個已接受的已編碼圖片的屬性，並將其還原到解碼後的輸出中，解碼器會將其與原始圖片關聯起來——而不是與下一個輸出的圖片關聯。每個已接受的圖片都會產生精確的一個終端結果，因此重新排序、丟棄和輸出緩存區的重複使用都無法將值移動到不同的影格中。此機制與編解碼器和傳輸方式無關，這使得它可以在稍後擴展到其他已編碼來源，而無需重新設計。

## 相容性

`Sample` 和來源選項結構新增了附加欄位。使用欄位名稱或聚合初始化來使用這些結構的原始碼仍然可以編譯。

這些公開結構的二進位佈局已更改，因此**已建置的元件必須重新建置**。Neat ABI/SOVERSION 保持在 **4**：由於 0.4.0 版本尚未發布，因此所有 ABI-4 元件都會一起重新建置並發布，而不是更新 ABI。

## 稍後新增另一個來源

解碼器路徑是通用的。新的已編碼來源只需要將巢狀屬性結構附加到它傳遞給解碼器的緩衝區中；解碼器或範例邊界中沒有任何內容是特定於傳輸方式的。每個新的來源仍然擁有的內容是它自己的提取規則以及它保證的圖形形狀。
