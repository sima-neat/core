# 008 將模型整合到您的管線中

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | graph, composition, patterns |

## Concept

將模型放入一個包含 `Graph`、`model.graph()` 和 `model.graph(options)` 的圖中——這兩種組合模式的不同之處在於您對接線的控制程度，因此您可以知道哪一種更適合快速執行，以及哪一種更適合多攝影機部署。

## Walkthrough

第 003 章逐步建立圖。這是一種最明確的流程，但一旦您擁有一個 `Model`，您通常不希望手動設定其內部元件。`model.graph(...)` 將模型的管線作為一個群組提供，您可以將其放入一個 `Graph` 中，只需使用一個 `add(...)` 即可。有趣的問題是，這個群組會帶來多少邊界資訊——而這正是 `ModelRouteOptions` 控制的。

本章將針對同一個模型，對比兩種路由設定：一種是包含自身公共輸入/輸出邊界的獨立可執行圖，另一種是省略輸入的附加圖，以便它可以連接到上游來源（例如，一個攝影機），並使用明確的名稱。到本章結束時，您將會組合這兩種設定，並列印每個設定的後端 GStreamer 字串，以便您可以準確地看到接線方式的差異。

### 組合一個可執行的模型圖 {#step-model-graph}

第一種模式要求模型提供一個完全可執行的圖。在路由選項中設定 `include_input = true` 和 `include_output = true`，會指示 `model.graph(opts)` 在模型群組周圍插入明確的公共輸入和輸出邊界，因此，生成的 `Graph` 可以獨立建立和執行，而無需附加任何其他元件。`graph.add(model.graph(opts))` 是整個組合過程——這個單一的 `add` 是所有模式的基礎。列印 `describe_backend()` 會顯示生成的 GStreamer 管線字串。

**C++：**路由選項是 `Model::RouteOptions`；圖是 `simaai::neat::Graph`。

**Python：**路由選項是 `pyneat.ModelRouteOptions`；圖是 `pyneat.Graph`。

### 設定附加時的路由選項 {#step-route-options}

第二種模式將模型附加到上游來源，而不是為其提供自身的輸入。在這裡，`include_input = false` 會移除公共輸入邊界（幀將來自其他地方），`include_output = true` 會保留輸出，而 `upstream_name`、`name_suffix` 和 `buffer_name` 會使接線和元件名稱明確化。像這樣一致的命名方式，有助於在多攝影機或多模型部署中，使後端圖更易於閱讀和診斷。

### 附加模型群組 {#step-attached-graph}

設定了這些選項後，`graph.add(model.graph(opts))` 會注入相同的模型群組，現在它會連接到指定的上游，而不是攜帶自己的來源。這與第一個模式中的 `add` 呼叫完全相同——只有路由選項發生了變化——這就是重點：組合是一個操作，而 `ModelRouteOptions` 則是決定該群組帶入哪些邊界的控制項。

**C++：**每個變體都會列印其 `describe_backend()`，以便您可以比較這兩個後端字串；然後，該檔案還會建立並執行一個手動連接的直接 `Input -> Output` 圖，以確認端到端的路徑，並列印 `direct_rank=`。

**Python：**連接的變體會列印 `attached_graph_built=True`，以確認組合已成功。

## Run

從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **儲存庫根目錄**執行 **從原始碼建置** 命令。

**Python:**
```bash
python3 share/sima-neat/tutorials/008_plug_model_into_pipeline/plug_model_into_pipeline.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_008_plug_model_into_pipeline \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_008_plug_model_into_pipeline
./build/tutorials-standalone/tutorial_008_plug_model_into_pipeline \
  --model /tmp/yolo_v8s.tar.gz
```

預期的輸出（C++ 建置會列印每個後端圖字串，然後列印直接圖的等級）：

```text
model_graph_backend=
...
attached_graph_backend=
...
direct_rank=3
[OK] 008_plug_model_into_pipeline
```

（Python 建置會先列印 `direct_graph_backend=`，然後列印後端字串，最後列印 `attached_graph_built=True`。）若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（無需額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## 原始檔案
- C++：`tutorials/008_plug_model_into_pipeline/plug_model_into_pipeline.cpp`
- Python：`tutorials/008_plug_model_into_pipeline/plug_model_into_pipeline.py`
