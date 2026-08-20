---
title: "記憶體模型"
description: "零拷貝緩衝區、區段，以及 (緩衝區 ID、實體位址、虛擬位址) 這個三元組，還有快取一致性。"
sidebar_position: 3
slug: /develop-apps/advanced-concepts/memory_model
---

# 記憶體模型

Neat 框架的執行階段會處理大量的位元組——包括編碼後的視訊幀、解碼後的 YUV 平面、FP32 輸入張量、INT8 量化後的圖塊，以及 MLA 暫存區圖像。如果沒有明確的記憶體模型，就必須在每個階段的邊界進行複製。本頁說明了該框架如何避免這些複製操作。

## 緩衝區三元組：`(buffer_id, paddr, vaddr)`

框架移動的每個緩衝區都由以下三個要素來識別：

- **`buffer_id`** — 一個穩定的整數，用於追蹤緩衝區的生命週期（參考計數、區段所有權），這是執行階段所使用的。
- **`paddr`** — 實體位址，這是 IOMMU 對緩衝區的視圖。MLA / EV74 / DMA 硬體會看到這個位址。
- **`vaddr`** — 虛擬位址，這是應用程式所使用的位址。CPU 程式碼會對此位址進行解參。

這個三重緩衝區機制讓緩衝區可以由任何一方（CPU 或加速器）直接存取，而無需複製。單次設定會同時出現在核心頁表（因此軟體可以讀取它）和 IOMMU 頁表（因此硬體可以將資料直接寫入其中）。

階段間的資料傳遞是傳遞三重緩衝區，而不是位元組。

## 片段

緩衝區來自具名稱的「區段」。區段是指記憶體中的一塊連續區域，由特定的設定器（例如 DMA-BUF、CMA、ION 或一般堆積）提供支援，並附加有元資料，說明哪些元件可以存取它：僅限 CPU、僅限 MLA、兩者皆可，等等。在執行階段，系統會根據每個緩衝區將要觸及的階段，選擇合適的區段。

範例：

- 一個 `nv12_decode` 區段包含從 H.264 解碼後的 YUV 資料——CPU 可以讀取這些資料以進行診斷，IOMMU 可以讀取這些資料以供調整大小節點使用。
- 一個 `mla_input` 區段會儲存傳遞給 MLA 的經過鑲嵌處理的張量——只有 MLA 硬體會讀取它；CPU 存取需要明確的映射。
- 一個 `model_output` 區段在進行去網格化處理後會儲存 FP32 張量——這些張量是 CPU 可以讀取的，因此應用程式可以將它們提取出來。

一個 `Tensor` 會將其片段與三元組一起傳遞，因此框架可以判斷從 CPU 程式碼發起的讀取/寫入操作是否有效。

## 快取一致性

MLA、EV74 和 CPU 都各自擁有自己的快取記憶體。當一個元件將資料寫入緩衝區，而另一個元件從該緩衝區讀取資料時，框架會在邊界處插入快取清除/無效化指令。應用程式程式碼無需考慮這一點——這會在緩衝區跨越不同階段時，於分段層級進行處理。

應用程式程式碼需要考慮的唯一情況是：當透過 `Mapping` 將 `TensorBuffer` 對應到可以直接由 CPU 讀取或寫入時。框架會在取消對應時插入適當的無效化（讀取對應）或清除（寫入對應）指令。請參閱 [`MapMode`](/reference/cppapi/namespaces/simaai-neat) 和 [`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer)。

## 實際應用中的零拷貝技術

典型的推論管線：

```
file → demux → H.264 decode → resize → preproc → MLA → postproc → app
```

如果沒有採用零拷貝技術，就需要進行七次拷貝。透過緩衝區三重緩衝和分段，就可以實現零拷貝——每個階段都會傳遞 `(buffer_id, paddr, vaddr)`，而下一個階段則會在原地進行操作。

框架的規劃器負責選擇分段，以便相鄰的階段可以共享資源。當兩個相鄰的階段具有不相容的分段需求時，規劃器會插入一個 `Transfer` `ConversionKind`，並將其記錄在任何作用中的 `ConversionTraceCollector` 中。請注意這些——它們是實際資料在執行階段移動的唯一位置。

## 相機來源與自適應記憶體

即時相機畫面透過平台相機堆疊進入，因此其記憶體類型取決於已安裝的內核、驅動程式和 `libcamerasrc` 路徑。`CameraInput` 首先請求裝置/SiMaAI 的零拷貝記憶體。其私有橋接器透過 `GST_QUERY_ALLOCATION` 提出一個標準池；該池會分配來自單一封裝 SiMaAI 分配的已驗證平面，並將其匯出為 DMA-BUFs。`libcamerasrc` 將這些 DMA-BUFs 導入 ISP，橋接器會解開相同的封裝分配，以供後續的 CVU/MLA 階段使用。

當相機堆疊僅提供 OS/libcamera 緩衝區，且 `allow_cpu_fallback` 已啟用時，Neat 會插入一個私有相機記憶體橋接器。橋接器會將每個畫面複製到池化的 SiMaAI 緩衝區中，標記預期的中繼資料，然後將該緩衝區傳遞給模型管理的 CVU 預處理。此複製操作是相容性橋接器；調整大小、色彩轉換、正規化、量化和鑲嵌等操作仍應在 CVU/EV74 上執行。

不要僅僅為了使 MIPI 相機輸入正常運作而新增一個公用的 `OsToSima`、`videoconvert` 或 `videoscale` 階段。使用 [`CameraInput`](/reference/nodes/camera-input)，並讓來源路徑負責記憶體調整。

## 相關類型

- [`TensorBuffer`](/reference/cppapi/structs/simaai-neat-tensorbuffer)——緩衝區三重容器。
- [`Segment`](/reference/cppapi/structs/simaai-neat-segment) — 分段控制項。
- [`Mapping`](/reference/cppapi/structs/simaai-neat-mapping) — RAII 映射處理常式，用於直接存取 CPU。
- [`MemoryContract`](/reference/cppapi/files/include-contracts-contracttypes-h)——說明一個節點偏好的記憶體設定方式。
- [`ConversionKind::Transfer`](/reference/cppapi/files/include-pipeline-tensorconversion-h)——這是唯一一種會複製跨片段內容的轉換類型。

## 更多閱讀資料

- 「張量與緩衝區」——設計深度探討的第 0.10 節、第 18 節、第 19 節和第 20 節。
- 「張量緩衝區應用程式二進位介面 (ABI)」——設計深入探討的第 20 節。
