# 013 建立自訂資料圖

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | graph, traversal, metadata |

## Concept

建立一個最小且實用的公共 Neat `Graph` ——一個*名為* `Input` 已連接到一個「已命名」的 `Output` ——然後，依據名稱推送一個範例，並驗證該範例的元資料是否能在傳輸過程中保留。

## Walkthrough

第三章建立了一個匿名的輸入 → 輸出圖，並使用位置參數 `run()` 進行驅動。真正的協調 — 扇出、扇入、每個串流的路由 — 需要通過*名稱*來處理端點，而不是通過位置。本章介紹了在盡可能小的圖上，具有命名端點的表面，以便您可以在多串流和嵌入式模型章節建立在它的基礎上之前，單獨查看命名和連接機制。

公共的 `Graph` 是應用程式組合的表面：您 `add(...)` 節點，`connect(...)` 命名的端點，`build()` 一次，形成一個可重複使用的 `Run`，然後通過名稱 `push("image", ...)` 和 `pull("out", ...)`。到最後，您將會將一個張量 `Sample` 推送到一個命名的圖中，並確認它的 `stream_id`、`frame_id` 和 `pts_ns` 保持不變 — 這證明了執行階段可以從端到端地保留元資料。

### 組合圖 {#step-compose-graph}

新增兩個節點。`Input("image")` 宣告一個名為 `image` 的推送端點；`Output("out")` 宣告一個名為 `out` 的拉取端點。這些名稱就是合約 — 這些是您稍後將傳遞給 `push(...)` 和 `pull(...)` 的確切字串。命名端點（而不是依賴於新增順序）是使具有多個輸入或輸出的較大圖在驅動時不產生歧義的原因。

**C++：** 節點來自 `simaai::neat::nodes::Input("image")` 和 `nodes::Output("out")`。

**Python：** 節點來自 `pyneat.nodes.input("image")` 和 `pyneat.nodes.output("out")`。

### 連接端點 {#step-connect-endpoints}

`connect("image", "out")` 宣告邊緣：推送到 `image` 的幀會流向 `out`。只有兩個節點，這就是整個拓撲，但 `connect(...)` 是您將用於在更大的圖中建立分支和合併的相同呼叫。然後，我們列印 `graph.describe()` 以轉儲組合的拓撲 — 這是一個快速的合理性檢查，以確保在建立之前，圖的連接方式符合您的意圖。

### 建立並推送一個範例 {#step-build-and-push}

`build()`（這裡不需要任何啟動範例）將描述轉換為可執行的 `Run`。然後，我們建置一個確定性的張量 `Sample` — 一個 8×8×3 的 RGB 圖像，其中包含已知的 `stream_id`、`frame_id` 和 `pts_ns` — 並通過名稱將其 `push(...)` 到 `image` 端點。範例的元資料是我們將在另一端檢查的內容。

**C++：** `push(...)` 傳回一個布林值；如果發生錯誤，我們會顯示 `run.last_error()`。範例由 `make_sample()` 建立。

**Python：** `push("image", [sample])` 接受一個範例列表。範例由 `make_rgb_sample()` 建立。

### 提取輸出並驗證中繼資料 {#step-pull-and-verify}

`pull("out", ...)` 從指定的輸出端點提取結果，並設定逾時時間，之後我們將`close()` 該執行。由於輸入和輸出之間沒有轉換，正確的管線會傳回相同的邏輯樣本——因此，讀取 `stream_id`、`frame_id` 和 `pts_ns`，並確認我們推送的值，可以驗證執行階段是否在遍歷過程中保留了每個樣本的中繼資料。這個保證是讓後續階段可以信任幀識別和時間戳的原因。

## Run

執行它，您應該會看到圖的描述，然後是來回傳輸的中繼資料。從 Neat 安裝根目錄（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **儲存庫根目錄** 執行 **從原始碼建置** 命令。本章不需要模型封存檔。

**Python:**
```bash
python3 share/sima-neat/tutorials/013_build_a_custom_data_graph/build_a_custom_data_graph.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_013_build_a_custom_data_graph
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_013_build_a_custom_data_graph
./build/tutorials-standalone/tutorial_013_build_a_custom_data_graph
```

預期的輸出（`graph.describe()` 轉儲的前面）：

```text
stream=graph frame=42 pts_ns=123456789
[OK] 013_build_a_custom_data_graph
```

（Python 建置會列印 `stream_id=graph frame_id=42 pts_ns=123456789`。）若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（不需要額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## 原始程式碼檔案
- C++：`tutorials/013_build_a_custom_data_graph/build_a_custom_data_graph.cpp`
- Python：`tutorials/013_build_a_custom_data_graph/build_a_custom_data_graph.py`
