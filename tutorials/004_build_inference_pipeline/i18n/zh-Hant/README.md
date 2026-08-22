# 004 建立您的第一個圖

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | None |
| Labels | graph, build, run, pipeline |

## Concept

手動建立一個 `Graph`，包含輸入節點、輸出節點，但不包含模型，然後讓一個框架通過它。在將模型添加到圖中之前，先查看管線的基本元件。

## Walkthrough

第 001 章使用三行程式碼執行一個模型。這種便利性隱藏了一個兩部分的生命週期，每個非trivial的 Neat 程式都會直接使用：首先，您會**描述**一個管線，作為一個 `Graph`，然後將該描述**建構**成一個可執行的 `Run`。本章通過建構盡可能小的管線（一個輸入節點連接到一個輸出節點，中間沒有模型），並將單個幀推送到其中，使這個生命週期可見。

其好處在於概念上：一個 `Graph` 是一個*可重複使用的定義*，您只需建構一次，然後多次執行，而不是一次性的呼叫。到本章結束時，您將建立一個圖，將其轉換為可執行的管線，並讀取輸出張量的等級——證明該幀已成功通過。

### 描述輸入 {#step-configure-input}

在連接節點之前，先宣告幀的外觀。`InputOptions` 就是這個合約：像素 `format`、`width`/`height`、通道 `depth`，以及執行階段是否會為每個緩衝區添加時間戳。從這些選項建構的輸入節點會驗證傳入的幀是否符合管線預期的形狀。

**C++：** C++ 額外設定 `is_live = false`，以標記這是一個非即時（檔案/張量）來源。

### 建構圖 {#step-compose-graph}

現在建構結構。一個新的 `Graph` 是一個空的建構表面，而 `add()` 會按順序附加節點。我們添加了兩個節點——一個輸入節點（如上所述設定）和一個空白的輸出節點。這就是整個拓撲：幀從輸入端進入，從輸出端離開，中間沒有任何內容。這是稍後章節中，模型或預處理階段將插入的位置。

**C++：** 節點來自 `simaai::neat::nodes::Input(...)` 和 `nodes::Output()`。

**Python：** 節點來自 `pyneat.nodes.input(...)` 和 `pyneat.nodes.output()`。

### 建構管線 {#step-build-pipeline}

`build()` 是從*描述*到*可執行*的轉換。它將添加的節點解析為一個具體的管線，根據實際樣本驗證輸入/輸出合約，並建立一個可重複使用的 `Run` 句柄。我們傳遞一個代表性的幀，以便 `build()` 可以鎖定協商後的張量形狀；下一步使用 `Run::run(...)` 進行確定性的逐一呼叫。

**C++：** 樣本幀是一個 `cv::Mat`，並且 `run_opt.output_memory = Owned` 要求執行階段傳回擁有的輸出緩衝區。

**Python：** 我們首先將幀轉換為一個 `Tensor`，來自一個 NumPy 陣列，使用 `Tensor.from_numpy(...)`，然後使用它進行建構。

### 執行一個幀並讀取結果 {#step-run-frame}

手邊有一個 `Run`，`run()` 會同步推送一個框架並拉取一個結果。由於沒有模型，因此輸出會反映輸入合約——因此，只需讀取張量的 *rank*，即可確認一個框架已完成整個流程。在實際的管線中，這個相同的 `run()`/push/pull 介面就是驅動推論的方式。

**C++：** `run()` 會傳回一個 `TensorList`；讀取 `sample.front().shape.size()`。

**Python：** 使用張量輸入的 `run()` 會傳回一個 `TensorList`；讀取 `len(outputs[0].shape)`。

## Run

執行它，您應該會看到輸出張量的 rank 輸出到 stdout。從 **Neat 安裝根目錄**（包含 `share/` 和 `lib/` 的目錄）執行 **Python** 和 **C++（預先建置）** 命令；從 **儲存庫根目錄**執行 **從原始碼建置** 命令。本章不需要模型封存檔。

**Python:**
```bash
python3 share/sima-neat/tutorials/004_build_inference_pipeline/build_inference_pipeline.py \
  --width 320 --height 240
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_004_build_inference_pipeline \
  --width 320 --height 240
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_004_build_inference_pipeline
./build/tutorials-standalone/tutorial_004_build_inference_pipeline \
  --width 320 --height 240
```

預期輸出：

```text
tensor_rank=3
[OK] 004_build_inference_pipeline
```

（Python 建置會輸出 `output_rank=...`。）若要將本章的 C++ 原始碼整合到您自己的專案中，並使用自訂的 `CMakeLists.txt`（不需要額外的資料夾），請參閱登陸頁面上的 [如何執行教學](/tutorials#compile-a-copy-yourself)。

## In Practice

如何將 `build`/`run`、執行模式、推/拉介面，以及 `RunOptions` 整合在一起，一旦您超越單次同步呼叫。

### 建構與執行

- `Graph::build(...)` 建構管線，並傳回一個 `Run` 控點，用於推/拉控制。
- `Graph::run(...)` 是一個同步的便捷路徑：它會建構（如果需要），推入一個輸入，並拉出一個輸出。

### 同步與非同步

- 對於簡單的一次性呼叫，請使用 `Graph::run(...)`。
- 當您想要一個可重複使用的執行器，以及明確的 `push(...)` / `pull(...)` 控制時，請使用 `Graph::build(...)` — 請參閱 [非同步執行推論](/tutorials/run-inference-async)。

### 推/拉 API

`Run` 公開：
- `push(...)` / `try_push(...)` 用於輸入（`cv::Mat`、`Tensor` 或 `Sample`）。
- `pull(...)`、`pull_tensor(...)`、`pull_tensor_or_throw(...)` 用於輸出。

如果您需要輸出中繼資料（時間戳、串流 ID），請使用 `pull()` 以取得 `Sample`。如果您只需要張量有效載荷，請使用 `pull_tensor()`。

### RunOptions（簡單 API）

常見的設定：
- `preset`：延遲/安全性設定檔（`Realtime`、`Balanced`、`Reliable`）。
- `queue_depth`：執行階段佇列深度。
- `overflow_policy`：佇列溢出行為（`Block`、`KeepLatest`、`DropIncoming`）。
- `output_memory`：輸出擁有權原則（`Auto`、`ZeroCopy`、`Owned`）。
- `on_input_drop`：用於已捨棄輸入事件的回呼掛鉤。

對於負載下的佇列深度、溢出和測量，請參閱 [調整輸送量和佇列深度](/tutorials/tune-throughput-and-queues)。

### RunAdvancedOptions（專家 API）

進階設定可在 `RunOptions::advanced` 下選擇性啟用：
- `advanced.max_input_bytes`：限制輸入緩衝區的增長。
- `advanced.copy_input`：強制進行防禦性輸入複製。

使用 `Run::start_measurement()` 來檢查延遲、輸送量、輸入計數器、外掛程式/邊緣計時，以及在單個測量視窗中的可選板 PMIC 電源遙測。

若要包含板載電源，請在程式碼中啟用它（無需環境變數），並從測量報告中讀取：

```cpp
simaai::neat::RunOptions run_opt;
run_opt.enable_board_power(); // default 100 ms sampling, auto-detects built-in profile
auto run = graph.build(inputs, run_opt);
auto scope = run.start_measurement();
run.push(inputs);
(void)run.pull_tensors(5000);
auto report = scope.stop();
```

```python
run_opt = neat.RunOptions()
run_opt.enable_board_power()  # default 100 ms sampling, auto-detects built-in profile
run = graph.build(tensor, run_opt)
scope = run.start_measurement()
run.push(tensor)
_ = run.pull_tensors(5000)
report = scope.stop()
```

`Model::build(run_opt)`、`Model::build(route_opt, run_opt)` 和 `Graph::build(run_opt)` 將相同的執行階段選項傳遞給底層的 `Run`，因此使用一個圖層級的板子電源監測器，而不是為每個管線重複採樣。如果您需要強制使用特定的內建設定檔，則仍可使用板子特定的輔助工具：`enable_modalix_som_power()`、`enable_modalix_dvt_power()`。

## 原始檔案
- C++：`tutorials/004_build_inference_pipeline/build_inference_pipeline.cpp`
- Python：`tutorials/004_build_inference_pipeline/build_inference_pipeline.py`
