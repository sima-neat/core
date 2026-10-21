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

對於已綁定儲存空間的密集數值 `Tensor`，形狀 `{}` 表示秩為零的純量，其密集資料大小為一個 `dtype` 元素。這不同於 `{0,256}` 之類含有零長度維度的形狀，後者的密集資料大小為零。緊密位元組複製與 `clone()` 會檢查檢視偏移量和儲存空間邊界，同時保留純量值。預設且未綁定的 Tensor 透過 `dense_bytes_tight()` 傳回零，且不允許緊密位元組複製或建立副本。明確標示為編碼、位元組串流、鑲嵌或文字的承載緩衝區仍保留原有的承載位元組語意。公開 `map()` 傳回的位元組範圍不變。

## 快取一致性

Scheduler 透過 Runner 的共用記憶體權威準備 CPU／裝置存取。Runner 負責配置身分、DMA-BUF 保留、CPU／裝置互斥、經檢查的範圍與基於世代的可見性。MLA 與 EV 發布已完成的裝置寫入，不會提前執行 CPU READ 存取期間。必要的 READ 快取維護在 CPU 映射或 A65 工作執行緒取得經檢查的存取時，於**讀取之前**執行，而不是在解除映射時或後端完成回呼中執行。僅裝置的後續階段不做 CPU 可見性工作；獨立影格仍可重疊執行。

呼叫 [`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer) 時，請搭配適當的 [`MapMode`](/reference/cppapi/namespaces/simaai-neat) 使用 `Mapping`。Tensor 映射會保留映射防護與目前儲存空間／產生端借用，直到 CPU 存取結束。包含寫入清理的必要 END 同步會在這些擁有者釋放前完成。部分寫入會保留相鄰快取列位元組。同步失敗會隔離受影響的配置，而非讓它正常重用。

每個公開 DMA 映射保留自己的 Scheduler 租約。相容讀取共用 Runner 存取權威，
包括重疊檢視；寫入仍受互斥限制。不同映射的虛擬位址不必相同。
釋放一個讀取端不能使另一個失效，只有最終擁有者才執行必要 END 轉換。

自有且可快取的 CMA 需要準備時的平台與匯出端准入；失敗需要處理，
不是退回整個緩衝區同步的許可。匯入的 DMA-BUF 與其他 heap 透過相同擁有權引擎
保留其匯出端契約。

私用全 A65 配方可保留明確父批次；子項需要存活的父能力、確切權限與範圍涵蓋。
最終父項釋放會在所有子項安全結束後清理寫入。混合裝置階段不取得隱含批次成員資格。
不加入隱藏暫存複製。原始 FD 寫入端與自訂回呼仍須遵守擁有權契約；
移除圖允許清單不能證明未追蹤的存取安全。

傳輸邊界保留底層配置身分、經檢查的偏移與長度，以及原始集區緩衝區擁有權。僅含中繼資料的封套會共用記憶體並保留集區父項。EV/CVU 與 MLA 匯入會保留產生端直到完成。完成狀態不明時，不得回收可能仍在使用的儲存空間。

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
