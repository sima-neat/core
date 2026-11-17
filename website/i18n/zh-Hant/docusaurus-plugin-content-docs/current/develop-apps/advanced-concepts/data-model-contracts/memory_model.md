---
title: "記憶體模型"
description: 配置身分、FrameLayout 放置、零複製擁有權與快取可見性。
sidebar_position: 3
slug: /develop-apps/advanced-concepts/memory_model
---

# 記憶體模型

Core 規劃儲存空間；Scheduler 透過 Runner 準備緩衝池。Runner 管理配置身分、DMA-BUF 映射與 CPU／裝置存取。

## 配置與檢視

每個階段只有一個不可變的 `FrameLayout`，包含 `FrameAllocation` 清單、配置內相對範圍 `FrameSpan`、輸入路由與輸出擁有權。範圍指定配置、位移與預留容量。邏輯張量形狀、步幅與發布位元組數是獨立資訊；MLA 硬體填補必須容納於預留容量內。

檢視共用根載體。配置索引屬於緩衝群組；不同群組中位移皆為零，不代表緩衝區重疊。匯入的儲存空間保留匯出者、已檢查位移、平面幾何與產生端借用。

對於已綁定儲存空間的密集數值 `Tensor`，形狀 `{}` 表示秩為零的純量，其密集資料大小為一個 `dtype` 元素。這不同於 `{0,256}` 之類含有零長度維度的形狀，後者的密集資料大小為零。緊密位元組複製與 `clone()` 會檢查檢視偏移量和儲存空間邊界，同時保留純量值。預設且未綁定的 Tensor 透過 `dense_bytes_tight()` 傳回零，且不允許緊密位元組複製或建立副本。明確標示為編碼、位元組串流、鑲嵌或文字的承載緩衝區仍保留原有的承載位元組語意。公開 `map()` 傳回的位元組範圍不變。

## 放置

Core 選擇路由並解析執行端，編譯已檢查配方、解析載體與擁有權，再依存取與生命週期證據配置。階段佈局與發布資訊只確定一次。

- 任何 EV74 存取都需要 CMA。
- 自有且僅供 MLA／A65 使用的載體預設使用 DMS。
- 符合條件的新 A65 公開終端可以使用 HostCpu 儲存空間。
- 匯入儲存空間維持固定。分支納入所有使用端；檢視保留載體原則。

`AllocationPolicy::PreferDmsThenCma` 明確允許符合條件的自有 DMS 配置在容量不足時退回 CMA，不允許退回一般 Linux 記憶體。Queue 邊界、獨立模型呼叫與保留的輸出維持擁有權及生命週期限制。

對於 EV74 Preproc → MLA → A65 BoxDecode，推論輸入使用 CMA，MLA 輸出使用獨立的 DMS 配置。實際放置取決於所有使用端已解析的執行端。

## 擁有權與 CPU 映射

應用程式持有零複製輸出時，輸出會保留儲存空間與產生端借用。Scheduler 和 Runner 將裝置輸入保留至確定完成；僅有逾時或取消，不代表可以重用。

使用 [`TensorBuffer::map()`](/reference/cppapi/structs/simaai-neat-tensorbuffer) 與適當的 [`MapMode`](/reference/cppapi/namespaces/simaai-neat)。`Mapping` 保留已檢查存取及儲存空間，直到關閉。相容讀取可以共用，寫入仍互斥。必要快取維護在 CPU 讀取前及寫入後完成，並保留相鄰快取行的位元組。同步失敗的儲存空間會被隔離。

自有的快取 CMA 與 DMS 使用已接納的 SimaHeap 範圍 API。匯入緩衝區保留匯出者合約。僅供裝置使用的後續階段不會開啟 CPU 讀取時期；不會從節點名稱或圖中位置推測 CPU 存取。內部 CPU 批次保留父物件，直到所有子存取退役。

中繼資料封套可共用記憶體而不複製承載資料，但型別轉換、明確傳輸與 Owned 輸出仍可能複製位元組。零複製不代表每個階段都原地執行。

## 相機緩衝區

[`CameraInput`](/reference/nodes/camera-input) 檢查擷取的 DMA-BUF 佈局，並在下游使用端使用期間保留相機父緩衝區。相容緩衝區保留原有儲存空間與平面幾何。若佈局需要調整且啟用 `allow_cpu_fallback`，橋接器會明確複製到 CMA。來源調整與模型前處理各自獨立。

## 相關型別

- [`TensorBuffer`](/reference/cppapi/structs/simaai-neat-tensorbuffer)：儲存空間與檢視。
- [`Segment`](/reference/cppapi/structs/simaai-neat-segment)：保留的配置範圍。
- [`Mapping`](/reference/cppapi/structs/simaai-neat-mapping)：作用域內的 CPU 存取。
- [`MemoryContract`](/reference/cppapi/files/include-contracts-contracttypes-h)：節點宣告的記憶體需求。
