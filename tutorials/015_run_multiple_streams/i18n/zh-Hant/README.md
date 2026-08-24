# 015 在單一圖中執行多個串流

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | None |
| Labels | graph, multistream, scheduler, join |

## Concept

透過單一的公共圖 `Graph` 執行多個邏輯串流，並將兩個命名的輸入合併為一個確定性的輸出束——這是多攝影機或多來源系統背後的模式，其中相關的輸入在進行後續處理之前必須對齊。

## Walkthrough

前幾章介紹了單一輸入和單一輸出。真正的多攝影機和並行分支系統則更複雜：多個串流獨立進行，它們的結果必須在任何後續處理之前正確地*重新合併*。本章將展示用於實現此目的的合併基本元件——一個具有兩個命名輸入和一個命名輸出的合併圖，該圖僅在雙方都產生匹配的影格時才會輸出一個組合。

您推送的每個樣本都帶有一個 `stream_id` 和一個 `frame_id`。合併策略 `ByFrame` 會等待，直到兩個命名輸入（`left` 和 `right`）都傳遞了具有相同 `frame_id` 的樣本，然後輸出精確一個組合。最後，您將建立一個合併圖，通過其兩個輸入將確定性的每個串流/每個影格的工作負載分發出去，並將合併的組合拉回——驗證輸出計數，並驗證每個組合都包含兩個欄位。

### 建立合併圖 {#step-build-combine-graph}

`graphs::Combine` (C++) / `graphs.combine` (Python) 傳回一個普通的公共 `Graph` 片段——除了其形狀之外，它沒有什麼特別之處：兩個命名輸入、一個命名輸出和一個合併策略。我們將 `["left", "right"]` 作為輸入名稱，將 `"combined"` 作為輸出名稱，並將 `CombinePolicy.ByFrame` 傳遞給它，以選擇影格 ID 匹配。列印 `describe()` 會顯示生成的拓撲結構，而 `build()` 會將描述轉換為可執行的句柄。該圖預設以非同步方式運行，因此每個串流都可以獨立地進行。

輸出佇列是有限制的。與為整個工作負載分配足夠的佇列空間不同，此範例在推送下一對之前，會先拉取每個合併的組合。生產者和消費者同步進行，因此隨著影格計數的增加，記憶體使用量保持在有限的範圍內。

`CombinePolicy.ByFrame` 根據 `Sample.frame_id` 進行匹配；`CombinePolicy.ByPts` 是另一種替代方案，它根據呈現時間戳 (`Sample.pts_ns`) 進行匹配，當影格不共享乾淨的影格索引時使用。

### 推送串流 {#step-push-streams}

現在我們驅動工作負載。對於每個影格和每個串流，我們都會合成一個小的確定性 RGB 樣本，並標記其 `stream_id` 和一個唯一的 `frame_id`，然後將其推送到*兩個*命名輸入中。由於 ID 是確定性地計算的 (`frame * streams + sid`)，因此合併具有明確的配對關係——`left` 影格 N 始終具有匹配的 `right` 影格 N。在匹配的 `right` 推送之後，我們會在移動到下一對之前，先清空該對的合併輸出。

**C++：**每個樣本都是明確地建構而成，作為一個 `Sample`，它包含一個 `Tensor`（HWC、UInt8、RGB），並設定了 `frame_id` 和 `stream_id`；`run.push("left", sample)` 會傳回一個布林值，您應該將其與 `run.last_error()` 進行比較。

**Python：**`make_rgb_sample(...)` 透過 `Tensor.from_numpy(...)` 從 NumPy 陣列建構 `Sample`；`run.push("left", [sample])` 接受一個樣本列表。

### 提取每個已合併的 bundle {#step-pull-bundles}

在每次配對成功推送後，我們會從指定的輸出 `"combined"` 提取一次。每次成功的提取都會傳回執行階段在兩個輸入都傳遞了該幀之後發出的 bundle。在產生資料的同時進行提取，可以防止受限的輸出佇列填滿，並將反壓傳播到輸入端。兩個範例都會驗證每個 bundle 是否包含兩個已合併的欄位，然後呼叫 `close()` 以乾淨地結束執行。預期的 bundle 數量等於 `streams * frames`，證明沒有遺漏任何配對。

**C++：**`run.pull("combined", timeout_ms)` 傳回一個可選的 bundle；我們讀取 `bundle.stream_id` 和 `bundle.fields.size()`，並驗證每個 bundle 是否具有兩個欄位。

**Python：**`run.pull("combined", 2000)` 傳回 bundle 或 `None`；如果發生逾時，該範例會立即失敗，並驗證每個 bundle 的欄位數量。

## Run

本章不需要模型封存檔。從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **程式碼庫根目錄** 執行 **從原始碼建置** 命令。

**Python:**
```bash
python3 share/sima-neat/tutorials/015_run_multiple_streams/run_multiple_streams.py \
  --streams 8 --frames 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_015_run_multiple_streams \
  --streams 8 --frames 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_015_run_multiple_streams
./build/tutorials-standalone/tutorial_015_run_multiple_streams \
  --streams 8 --frames 4
```

預期的輸出（C++ 建置也會列印圖的描述；兩個建置都會列印前幾個 bundle）：

```text
received=32 fields=2
[OK] 015_run_multiple_streams
```

若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（不需要額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## 原始程式碼檔案
- C++：`tutorials/015_run_multiple_streams/run_multiple_streams.cpp`
- Python：`tutorials/015_run_multiple_streams/run_multiple_streams.py`
