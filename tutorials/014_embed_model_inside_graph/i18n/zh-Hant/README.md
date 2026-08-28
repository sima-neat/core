# 014 將模型嵌入到圖中

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | yolo_v8s |
| Labels | graph, hybrid, model, mpk |

## Concept

將已編譯的模型放入一個公開的 `Graph` 中，並使用 `graph.add(model)`，這樣您就可以獲得圖層級的協調（路由、排程、額外的輸入/輸出），並將其包裝在模型執行周圍，而無需進入內部執行階段圖。

## Walkthrough

第三章建立了一個由基本輸入/輸出節點組成的圖；第一章將一個模型作為獨立物件執行。本章將這兩者結合起來：一個 `Model` 本身就是一個與圖相容的節點，因此您可以像其他任何階段一樣，將其組合成一個公開的 `Graph`。這就是生產系統在需要圖層級控制時（多個輸入、具名輸出、自訂路由）所採用的橋接模式，同時仍然將模型執行視為一個可重複使用的片段。

關鍵思想是，您永遠不會觸及低階執行階段圖、`StageModelExecutorOptions` 或內部節點 ID。您將模型傳遞給 `graph.add(...)`，NEAT 會在建立時將該片段（根據需要進行預處理/推理/後處理）降低到正確的內部執行計畫中。最後，您將會將一個模型組合成一個公開的圖，列印組成的拓撲，並讀取模型的輸出基數。

### 載入模型 {#step-load-model}

建構會載入編譯的檔案，並為執行做好準備，就像在第一章中一樣。在這裡，我們只獲取路徑（沒有選項物件），因為本章是關於組合，而不是預處理。結果的 `Model` 現在是一個圖層可以理解的物件。

### 將模型組合成圖 {#step-compose-graph}

這是本章的重點。一個新的 `Graph` 會按順序新增三個節點：一個具名的輸入邊界、模型本身以及一個具名的輸出邊界。由於 `Model` 與圖相容，因此 `add(model)` 會將整個模型路徑作為單一片段新增——沒有特殊的 API，也沒有進入執行階段。列印 `graph.describe()` 會顯示組成的拓撲，以便您可以確認模型是否已插入到具名邊界之間。

**C++：**邊界來自 `simaai::neat::nodes::Input("image")` 和 `nodes::Output("result")`；模型直接傳遞給 `graph.add(model)`。

**Python：**邊界來自 `pyneat.nodes.input("image")` 和 `pyneat.nodes.output("result")`；模型直接傳遞給 `graph.add(model)`。

### 檢查模型 {#step-inspect-model}

最後，我們讀取模型片段實際貢獻的內容。這確認模型已正確載入，並讓您可以看到圖將在下游產生的輸出拓撲。

**C++：**`model.info()` 傳回一個資訊結構；我們列印 `model_name` 以及 `output_topology.physical_outputs` 和 `logical_outputs`，以便模型輸出的接線是明確的。

**Python：**本章的程式碼只是會印出一行確認訊息，表示模型片段已新增到公開的圖中。

## Run

本章需要一個模型封存檔（`yolo_v8s`）。從 **Neat 安裝目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 指令；從 **原始碼儲存庫目錄** 執行 **從原始碼建置** 指令。

**Python:**
```bash
python3 share/sima-neat/tutorials/014_embed_model_inside_graph/embed_model_inside_graph.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_014_embed_model_inside_graph \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_014_embed_model_inside_graph
./build/tutorials-standalone/tutorial_014_embed_model_inside_graph \
  --model /tmp/yolo_v8s.tar.gz
```

預期的輸出結果（C++ 建置也會先印出建置後的圖描述）：

```text
model=yolo_v8s physical_outputs=1 logical_outputs=1
[OK] 014_embed_model_inside_graph
```

（Python 建置會先印出圖描述，然後印出 `model fragment added to public Graph`。）

若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（不需要額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## 原始碼檔案
- C++：`tutorials/014_embed_model_inside_graph/embed_model_inside_graph.cpp`
- Python：`tutorials/014_embed_model_inside_graph/embed_model_inside_graph.py`
