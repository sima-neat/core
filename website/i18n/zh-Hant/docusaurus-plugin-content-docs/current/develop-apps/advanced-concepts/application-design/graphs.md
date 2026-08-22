---
title: "使用圖來建置應用程式"
description: "如何使用公開的圖形 API 來建構模型、節點、具名稱的輸入/輸出、分支、組合以及執行流程。"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/graphs
---

# 使用圖建置應用程式

當您只想載入並執行單一已編譯的模型封存檔時，請使用 `Model`。當您想要圍繞模型和節點建置應用程式時，請使用 `Graph`：新增公開的輸入和輸出，連接可重複使用的片段，分支資料流，合併資料流，驗證應用程式，並儲存或視覺化實際執行的內容。

其概念模型刻意保持簡潔：

| 概念 | 意義 |
|---|---|
| `Model` | 從磁碟載入的已編譯模型封存檔，例如 `resnet50.tar.gz` 或 `yolov8.tar.gz`。 |
| `Node` | 一個處理步驟：輸入、輸出、轉換、來源、接收器、模型階段或輔助階段。 |
| `Graph` | 應用程式的連接計畫：存在哪些節點/片段，以及資料如何在它們之間流動。 |
| `Run` | 由 `Graph::build()` 傳回的即時執行句柄：推送輸入、提取輸出、收集指標、停止。 |

簡而言之：

```text
Graph = what to run
Run   = the running instance
```

當您需要更短的路徑時，請從任務頁面開始：

- [圖](/develop-apps/development-workflow/graph) 講解圖的建立。
- [執行圖形](/develop-apps/development-workflow/pipeline) 講解執行階段生命週期、佇列、測量和吞吐量。
- [節點](/develop-apps/development-workflow/node) 映射常見節點和群組。
- [張量與樣本](/develop-apps/development-workflow/core_types) 解釋有效載荷和中繼資料。

大多數應用程式程式碼應使用公開的 `simaai::neat::Graph` 和 `simaai::neat::Run`。 請勿使用較低層級的實作命名空間來建立應用程式；這些不是客戶 API。

## 何時需要一個圖？

| 目標 | 建議的 API |
|---|---|
| 在單一輸入上執行一個模型 | `Model::run(...)` 或 `Model::build(...)` |
| 在模型周圍新增應用程式輸入/輸出邊界 | `Graph` |
| 使用自訂處理節點來組合一個模型 | `Graph::add(...)` |
| 在多個應用程式中重複使用 Graph 片段 | 傳回或傳遞一個 `Graph` 片段 |
| 路由多個輸入或輸出 | 命名 `nodes::Input(...)` / `nodes::Output(...)`，再加上 `connect(...)` |
| 將一個資料流分支到多個消費者 | `graphs::Branch(...)` |
| 將多個資料流組合為一個邏輯輸出 | 使用 `graphs::Combine(...)` 和一個 `CombinePolicy` |
| 儲存或視覺化執行的拓撲和指標 | `save_run_json(run, ...)` |

## 第一個圖：一個輸入、一個模型、一個輸出

這是最小的完整應用程式樣式圖：

```cpp
#include <neat.h>

#include <iostream>

namespace neat = simaai::neat;

int main() {
  neat::Model model("resnet50.tar.gz");

  neat::Graph app;
  app.add(neat::nodes::Input("image"));
  app.add(model);
  app.add(neat::nodes::Output("classes"));

  neat::Run run = app.build();

  neat::Tensor image = /* create or load an image tensor */;
  run.push("image", neat::TensorList{image});

  std::optional<neat::Sample> result = run.pull("classes", /*timeout_ms=*/1000);
  if (result) {
    // Consume result->tensors, result->detections, or other Sample metadata.
  }

  run.stop();
}
```

逐行說明：

- `nodes::Input("image")` 宣告一個名為 `image` 的公開輸入節點。
- `app.add(model)` 將模型的選定路徑插入到圖中。
- `nodes::Output("classes")` 宣告一個名為 `classes` 的公開輸出節點。
- `app.build()` 驗證並編譯整個圖，然後傳回一個 `Run`。
- `run.push("image", ...)` 將資料傳送到指定的輸入節點。
- `run.pull("classes", ...)` 從指定的輸出節點接收資料。

與 Python 程式碼相同的結構：

```python
import pyneat

model = pyneat.Model("resnet50.tar.gz")

app = pyneat.Graph()
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

run = app.build()

image = ...  # Create or load a tensor-compatible object.
run.push("image", [image])

result = run.pull("classes", timeout_ms=1000)
run.stop()
```

Python 的 `Run.push(...)` 函式需要一個類似批次（batch）的序列。請傳遞 `[tensor]` 或 `[sample]`，而不是單獨的張量/樣本物件。

## 執行圖

一個建構好的 `Run` 接受與 Neat 中其他位置使用的相同類型的公用負載：

| 負載 | 何時使用 |
|---|---|
| `TensorList` | 當您傳遞張量且不需要額外的樣本中繼資料時。 |
| `Sample` | 當您需要時間戳、`frame_id`、`stream_id`、文字/音訊/視訊中繼資料、檢測結果或 EOS（序列結束）時。 |
| `std::vector<cv::Mat>` | 當您想要從 OpenCV 取得方便的影像輸入時。 |

常見的 C++ 呼叫：

```cpp
run.push(neat::TensorList{image});
run.push("image", neat::TensorList{image});

run.push(sample);
run.push("image", sample);

auto out = run.pull(/*timeout_ms=*/1000);
auto named = run.pull("classes", /*timeout_ms=*/1000);

neat::TensorList tensors = run.pull_tensors("classes", 1000);
neat::Sample sample_out = run.pull_samples("classes", 1000);
```

當逾時或關閉時，應回傳一個空的 `std::optional`，此時請使用 `pull(...)`。當您需要一個帶有類型且在逾時或發生錯誤時會引發異常的便利輔助函式時，請使用 `pull_tensors(...)` 或 `pull_samples(...)`。

對於有限的應用程式推送串流，在收集最終指標之前，請先關閉輸入並清空緩衝：

```cpp
run.close_input();
while (auto out = run.pull("classes", 1000)) {
  // Drain remaining output.
}
run.stop();
```

對於以任務為中心的執行階段設定指南，包括佇列策略、輸出所有權、資料丟棄遙測和多串流測量，請參閱 [執行圖形](/develop-apps/development-workflow/pipeline)。

## `build()` 與 `build(first_input)`

大多數圖都可以無需輸入樣本的情況下建構：

```cpp
neat::Run run = app.build();
```

當圖已經宣告足夠的形狀/邊界資訊時，或者當圖擁有其來源節點時（例如 RTSP/檔案/靜態影像輸入），請使用此設定。

「已設定初始值」建置會在建置期間為 Neat 提供第一個輸入：

```cpp
neat::Run run = app.build(neat::TensorList{first_image});
```

當第一個輸入應該在開始串流之前，先設定形狀/格式的調整時，請使用此設定。預設情況下，已啟用「已設定的建置預先檢查」，因此 Neat 可以一次推送/提取設定，以在建置過程中捕捉第一個樣本的錯誤，而不是傳回一個 `Run`，該設定會在稍後立即導致失敗。

為了獲得更好的輸送量、延遲和功耗資料，請在實際工作負載執行後儲存指標，而不是在建置後立即儲存。

## 圖的名稱不是端點名稱

:::warning
`Graph("name")` 是一個用於診斷、儲存圖形檔案和視覺化的標籤。它**不**會宣告一個名為 `name` 的公開輸入或輸出。
:::

錯誤的心理模型：

```cpp
neat::Graph camera("image");
// This does not make run.push("image", ...) valid by itself.
```

正確的端點宣告：

```cpp
neat::Graph camera("camera_route");
camera.add(neat::nodes::Input("image"));
```

至於輸出結果：

```cpp
neat::Graph classifier("classifier");
classifier.add(neat::nodes::Output("classes"));
```

將 `Input("image")` 和 `Output("classes")` 視為圖的片段中公開的入口。圖的名稱就像建築物上的標誌。

## 檢查端點名稱，而不是猜測

在建構之前，請檢查圖中宣告的邏輯公開端點：

```cpp
for (const auto& name : app.inputs()) {
  std::cout << "graph input: " << name << "\n";
}
for (const auto& name : app.outputs()) {
  std::cout << "graph output: " << name << "\n";
}
```

建置完成後，檢查 `Run` 實際上接受哪些內容：

```cpp
for (const auto& name : run.input_names()) {
  std::cout << "run input: " << name << "\n";
}
for (const auto& name : run.output_names()) {
  std::cout << "run output: " << name << "\n";
}
```

將此用於模型路徑以及任何多輸入/多輸出應用程式。端點匹配必須精確：
`Input("image_l")` 可以繫結到名為 `image_l` 的模型輸入；`Input("my_random_name")` 則不行。

## 未命名的便利 API

對於單輸入/單輸出圖，您可以省略端點名稱：

```cpp
neat::Graph app;
app.add(neat::nodes::Input());
app.add(model);
app.add(neat::nodes::Output());

neat::Run run = app.build();
run.push(neat::TensorList{image});
auto result = run.pull(1000);
```

這對於快速的腳本和測試非常方便。對於非簡單的應用程式，建議使用名稱。

如果一個圖有許多可能的輸入或輸出，未命名的 `push(...)` 或 `pull()` 會導致錯誤，並
回報可用的名稱。這種錯誤是故意的：Neat 不應該猜測您指的是哪個攝影機、張量或輸出層。

## 模型是圖的片段

一個 `Model` 可以直接新增到一個圖中：

```cpp
neat::Model yolo("yolov8.tar.gz");

neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

`Graph::add(model)` 會將從檔案和模型選項中選取的模型路徑插入。該路徑可能包含預處理、MLA 推論、後處理、張量轉換和檢測解碼階段。
對於常見的線性情況，您不必手動呼叫 `model.graph()`。

對於更進階的組合，您可以檢查或重複使用該路徑作為 `Graph` 的一部分：

```cpp
neat::Graph route = yolo.graph();

auto model_inputs = route.inputs();
auto model_outputs = route.outputs();
```

### 多輸入模型

對於多輸入模型，請勿猜測名稱。請詢問路線：

```cpp
neat::Graph route = model.graph();

for (const auto& name : route.inputs()) {
  std::cout << "model expects input: " << name << "\n";
}
```

然後，為您的上游片段命名，使其與模型的輸入名稱相符：

```cpp
neat::Graph left_camera;
left_camera.add(neat::nodes::Input("image_l"));

neat::Graph uv_camera;
uv_camera.add(neat::nodes::Input("image_uv"));

neat::Graph app;
app.connect(left_camera, route);  // Binds image_l -> model image_l.
app.connect(uv_camera, route);    // Binds image_uv -> model image_uv.
```

如果宣告了 `left_camera`，並且指定了 `Input("a_new_name_image_l")`，它就不會與 `image_l` 綁定。請新增一個小型適配器圖，並使用正確的端點名稱，而不是依賴隱含的重新命名。

### 獨立模型圖

預設情況下，`model.graph()` 會傳回一個可重複使用的模型片段，其中包含開放的命名端點。如果您希望傳回的圖可以獨立執行，請要求明確的公開輸入/輸出節點：

```cpp
neat::Model::RouteOptions route_opt;
route_opt.include_input = true;
route_opt.include_output = true;

neat::Graph standalone = model.graph(route_opt);
neat::Run run = standalone.build();
```

為了進階或除錯用途，模型路徑可以公開個別的實體輸出：

```cpp
route_opt.expose_all_outputs = true;
```

除非您明確需要個別的實體輸出緩衝區，否則請將此選項停用。預設模型的行為是公開路由合約預期的邏輯模型輸出。如果模型只有一個實體輸出，`expose_all_outputs = true` 仍然只會公開一個輸出。

## `add()` 與 `connect()`

有兩種組合工具：

| API | 意義 | 何時使用 |
|---|---|---|
| `add(x)` | 將內容附加或插入到目前的線性鏈中。 | 您指的是「同一管線中的下一個步驟」。 |
| `connect(a, b)` | 透過命名的端點連接兩個圖的片段。 | 您正在組合可重複使用的片段或建置拓撲結構。 |
| `connect("a", "b")` | 連接已在同一圖中宣告的兩個端點。 | 您正在建置一個小型輔助片段。 |

線性組合：

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
```

片段組成：

```cpp
neat::Graph app;
app.connect(camera, model_route);
app.connect(model_route, output_sink);
```

輔助片段內部的內部端點接線：

```cpp
neat::Graph pass_through("pass_through");
pass_through.add(neat::nodes::Input("in"));
pass_through.add(neat::nodes::Output("out"));
pass_through.connect("in", "out");
```

主要規則：`add()` 表示一個線性鏈。`connect()` 表示圖的拓撲結構。

## 可重複使用的 Graph 片段

函式可以傳回可重複使用的 Graph 片段：

```cpp
neat::Graph make_classifier(neat::Model& model) {
  neat::Graph g("classifier");
  g.add(neat::nodes::Input("image"));
  g.add(model);
  g.add(neat::nodes::Output("classes"));
  return g;
}
```

以線性方式使用可重複使用的程式碼片段：

```cpp
neat::Graph classifier = make_classifier(model);

neat::Graph app;
app.add(classifier);
```

或者明確地指定線段片段：

```cpp
neat::Graph app;
app.connect(camera, classifier);
app.connect(classifier, class_sink);
```

如果將節點 `add()` 附加到一個分支之後會產生歧義，則 Neat 會失敗，並提示您改用 `connect(...)`。
這樣比默默地附加到錯誤的分支上更好。

## 分支單一資料流

當單一輸入資料流應該導向多個具名輸出時，請使用 `graphs::Branch`。

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});
```

意義：

```text
image -> preview
      -> model_input
```

範例：

```cpp
neat::Graph camera;
camera.add(neat::nodes::Input("image"));

neat::Graph preview;
preview.add(neat::nodes::Output("preview"));

neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});

neat::Graph app;
app.connect(camera, branch);
app.connect(branch, preview);
```

在將分支連接到模型時，請選擇與模型輸入名稱相符的分支輸出名稱：

```cpp
neat::Graph route = model.graph();
for (const auto& name : route.inputs()) {
  std::cout << "choose a branch output matching: " << name << "\n";
}
```

分支是明確的，因為它會影響佇列和反壓機制。如果某個分支速度較慢，根據輸出選項和下游圖，它可能會減慢速度或導致資料遺失，相對於其他分支而言。

Python：

```python
branch = pyneat.graphs.branch("image", ["preview", "model_input"])
```

## 結合多個資料流

當需要將多個輸入資料流合併為單一邏輯輸出時，請使用 `graphs::Combine`。

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);
```

意義：

```text
left  --\
        +--> stereo
right --/
```

策略：

| 策略 | 意義 |
|---|---|
| `CombinePolicy::None` | 不自動合併。多個來源指向單一輸出時，會導致輸出失敗。 |
| `CombinePolicy::ByFrame` | 將樣本與完全相同的 `Sample::frame_id` 進行匹配。如果缺少畫面 ID，則會導致失敗；沒有 PTS 備用方案。 |
| `CombinePolicy::ByPts` | 將樣本與完全相同的 `Sample::pts_ns` 呈現時間戳進行匹配。如果缺少 PTS，則會導致失敗；沒有畫面 ID 備用方案。 |

簡潔的說明：

- `ByFrame` 表示「給我具有相同畫面編號的左右樣本」。
- `ByPts` 表示「給我具有相同媒體時間戳的樣本」。

範例：

```cpp
neat::Graph left;
left.add(neat::nodes::Input("left"));

neat::Graph right;
right.add(neat::nodes::Input("right"));

neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);

neat::Graph app;
app.connect(left, pair);
app.connect(right, pair);

neat::Run run = app.build();
run.push("left", left_sample_with_frame_id_42);
run.push("right", right_sample_with_frame_id_42);
auto stereo = run.pull("stereo", 1000);
```

Python：

```python
pair = pyneat.graphs.combine(["left", "right"], "stereo", pyneat.CombinePolicy.ByFrame)
```

如果樣本不包含所需的關鍵資訊，合併階段將會失敗，並顯示診斷訊息，而不是進行推測。

## 資料來源與資料匯出

資料進入圖的途徑有兩種，資料離開圖的途徑也有兩種。

### 應用程式推送的輸入

當應用程式程式碼會推送資料時，請使用 `nodes::Input(...)`。

```cpp
app.add(neat::nodes::Input("image"));
run.push("image", neat::TensorList{image});
```

### 圖所擁有的輸入來源

當圖擁有資料來源時，請使用來源節點或來源片段：

```cpp
app.add(neat::nodes::RTSPInput("rtsp://camera/stream"));
```

或是一個可重複使用的已解碼 RTSP 片段：

```cpp
neat::nodes::groups::RtspDecodedInputOptions opt;
opt.url = "rtsp://camera/stream";

app.add(neat::nodes::groups::RtspDecodedInput(opt));
```

當一個圖擁有其資料來源時，您通常會呼叫 `build()`，然後提取輸出；您不應該從應用程式程式碼中將資料推送到該資料來源。

### 應用程式提取的輸出

當應用程式程式碼應該提取結果時，請使用 `nodes::Output(...)`。

```cpp
app.add(neat::nodes::Output("detections"));
auto out = run.pull("detections", 1000);
```

### 圖所擁有的輸出節點

當圖需要自行輸出結果時，請使用輸出節點或輸出節點群組：

```cpp
neat::UdpOutputOptions udp;
udp.host = "192.0.2.10";
udp.port = 5000;

app.add(neat::nodes::UdpOutput(udp));
```

對於針對該模式而建構的圖，也提供類似伺服器的 RTSP 輸出：

```cpp
neat::RtspServerHandle server = app.run_rtsp(rtsp_options);
```

## 驗證與診斷

如果您希望在開始執行階段資源之前，先進行驗證，以便獲得結構化的報告，請在建置之前進行驗證：

```cpp
neat::GraphReport report = app.validate();
if (!report.error_code.empty()) {
  std::cerr << report.repro_note << "\n";
}
```

在「建置/執行/推送/拉取」操作期間，捕捉 `NeatError`：

```cpp
try {
  neat::Run run = app.build();
} catch (const neat::NeatError& e) {
  std::cerr << e.what() << "\n";

  const neat::GraphReport& report = e.report();
  std::cerr << "error_code: " << report.error_code << "\n";
  std::cerr << "hint: " << report.repro_note << "\n";
}
```

實用的除錯輔助工具：

```cpp
std::cout << app.describe() << "\n";
std::cout << app.describe_backend() << "\n";
```

- `describe()` 會列印公開圖的摘要資訊：端點、片段和拓撲結構。
- `describe_backend()` 會列印較低層級的後端詳細資訊，這在偵錯產生的管線字串或執行階段路由時很有用。

如需錯誤碼分類和問題處理流程，請參閱 [錯誤代碼](/reference/error-codes/)。

## 儲存和載入圖的組成

`Graph::save(path)` 會儲存公開圖的組成：節點、端點名稱、明確的端點邊緣、輸出選項、合併原則和模型路由來源。

```cpp
app.save("app.graph.json");

neat::Graph loaded = neat::Graph::load("app.graph.json");
neat::Run run = loaded.build();
```

這會儲存圖的計畫，而不是正在執行的管線，也不是執行階段指標。對於執行階段指標，請使用「執行 JSON 匯出」。

模型路由的來源非常重要。模型片段不僅僅是一個後端程式碼片段的列表：它還包含從模型封存檔、路由選項以及針對多輸入模型的輸入路由處理器中繼資料中衍生的輸入/輸出名稱。如果儲存的圖中包含模型片段，Neat 會儲存模型封存檔的路徑和重新載入模型時所需的路由選項。如果載入時缺少封存檔，Neat 會顯示可操作的錯誤，而不是默默地建立不完整的路由。

## 匯出並視覺化已執行的內容

`Run` 既知道公開的圖形結構，也知道簡化的執行階段結構。它可以匯出為版本化的 JSON 成品，用於 CI、除錯、支援工單或離線視覺化。

### 建置時的拓撲快照

當您希望在圖建立後立即取得成品時，請使用建置時匯出：

```cpp
neat::RunOptions opt;
opt.run_export.path = "/tmp/startup.graph_run.json";
opt.run_export.label = "startup";

neat::Run run = app.build(opt);
```

這是一個初始的網路拓撲快照。由於目前尚未執行任何樣本，因此可能不包含任何吞吐量/延遲計數器。

Python：

```python
opt = pyneat.RunOptions()
opt.run_export.path = "/tmp/startup.graph_run.json"
opt.run_export.label = "startup"

run = app.build(opt)
```

### 執行完成後擷取快照並顯示指標

在工作負載執行或完成後，使用「執行完成後匯出」功能：

```cpp
neat::Run run = app.build();
run.push("image", neat::TensorList{image});
auto out = run.pull("classes", 1000);

neat::RunExportOptions export_opt;
export_opt.label = "after_smoke_test";
export_opt.metadata = {{"test_name", "smoke"}};

std::string err;
if (!neat::save_run_json(run, "/tmp/final.graph_run.json", export_opt, &err)) {
  throw std::runtime_error(err);
}
```

Python：

```python
run = app.build()
run.push("image", [image])
out = run.pull("classes", timeout_ms=1000)

export_opt = pyneat.RunExportOptions()
export_opt.label = "after_smoke_test"
export_opt.metadata = {"test_name": "smoke"}

run.save_json("/tmp/final.graph_run.json", export_opt)
```

匯出者會擷取目前執行的狀態快照；它不會停止執行。如果您需要有限工作負載的最終數值，請在儲存之前呼叫 `run.close_input()` 並清空輸出，或呼叫 `run.stop()`。

若要包含板卡功耗遙測資料：

```cpp
neat::RunOptions opt;
opt.enable_board_power(/*sample_interval_ms=*/100);

neat::Run run = app.build(opt);
```

JSON 結構的 `sima.neat.graph_run` 版本為 `1`。該結構位於 `schemas/graph_run_v1.schema.json`，而 CI 驗證器則位於 `tests/perf/tools/graph_run_schema.py`。

在沒有網路連線的情況下呈現圖形成品：

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json -o /tmp/final.graph_run.html
```

選擇要呈現的檢視方式：

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view public
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view lowered
```

- `public` 顯示使用者建立的圖：包含命名的輸入、輸出、片段，以及 `connect(...)` 連接邊。
- `lowered` 顯示 Neat 在內部執行的內容：管線片段、產生的分支/合併階段、佇列和執行階段邊。

## 常見模式

### 影像分類

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(resnet);
app.add(neat::nodes::Output("classes"));
```

### 物體偵測

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### RTSP 攝影機到模型再到應用程式拉取輸出

```cpp
neat::nodes::groups::RtspDecodedInputOptions source_opt;
source_opt.url = "rtsp://camera/stream";

neat::Graph app;
app.add(neat::nodes::groups::RtspDecodedInput(source_opt));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### 應用程式輸入至由圖所擁有的 UDP 輸出

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::UdpOutput(udp_options));
```

### 分支預覽與模型路徑

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_image"});
```

將名稱變更為 `model_image`，以符合模型路徑輸入，或插入明確的配接器片段。

### 結合左側/右側串流

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "pair",
                                         neat::CombinePolicy::ByPts);
```

使用 `ByPts` 當媒體時間戳記是同步的關鍵時，請使用 `ByFrame` 當畫面 ID 為同步金鑰時。

### GenAI 和其他階段片段

GenAI 和其他非線性/階段型功能仍應以公開方式進入應用程式碼。
`Graph` 片段並透過以下方式執行： `Graph::build() -> Run`:

```cpp
neat::Graph app;
app.add(genai_fragment);

neat::Run run = app.build();
run.push("prompt", prompt_sample);
auto token = run.pull("tokens", 1000);
```

精確的 GenAI 程式碼片段工廠和範例輔助函式的名稱取決於已安裝的 GenAI 套件。圖的規則相同：新增或連接公用程式碼片段，然後使用具名稱的 `Run::push(...)` 和 `Run::pull(...)`。

## 應避免的模式和常見錯誤

### 不要將圖標籤用作端點

錯誤：

```cpp
neat::Graph image("image");
run.push("image", neat::TensorList{tensor}); // Graph label is not an endpoint.
```

正確：

```cpp
neat::Graph image;
image.add(neat::nodes::Input("image"));
```

### 請勿猜測模型輸入的名稱

錯誤：

```cpp
left.add(neat::nodes::Input("my_left"));
app.connect(left, model);
```

正確：

```cpp
for (const auto& name : model.graph().inputs()) {
  std::cout << name << "\n";
}
```

然後，為上游端點命名，使其相符。

### 請勿在具有多個端點的圖中使用未命名的推送/拉取操作

錯誤範例：

```cpp
run.push(neat::TensorList{left});
run.push(neat::TensorList{right});
```

正確：

```cpp
run.push("left", neat::TensorList{left});
run.push("right", neat::TensorList{right});
```

### 請勿在沒有設定 CombinePolicy 的情況下意外地啟動扇出作業

錯誤範例：

```cpp
neat::Graph bundle;
bundle.add(neat::nodes::Output("bundle"));

app.connect(left, bundle);
app.connect(right, bundle); // Ambiguous: how should left/right be synchronized?
```

正確：

```cpp
neat::Graph bundle = neat::graphs::Combine({"left", "right"},
                                           "bundle",
                                           neat::CombinePolicy::ByFrame);
```

### 除非您指的是片段邊界，否則請勿在中間插入「輸入/輸出」

`Input` 和 `Output` 是公開的邊界宣告。在可重複使用的片段中，這正是您想要的。在純粹的線性應用程式中，如果在中間新增一個額外的 `Output`，可能會產生一個實際的可拉取匯流點和反壓，除非該邊界被另一個 `connect(...)` 邊緣所消耗。

### 請勿在應用程式程式碼中使用較低層級的執行階段圖 API

請避免使用較低層級的執行階段圖 API 來編寫或教學應用程式程式碼。

請改用公開的應用程式介面：

```cpp
neat::Graph
neat::Run
app.build()
```

## 進階說明：邊界實體化

命名為 `Input` 和 `Output` 的節點是片段的公開合約宣告。它們比用於移動緩衝區的執行階段物件的層級更高。

在可執行管線建置之前，`Graph::build()` 會正規化邊界：

| 邊界宣告 | 何時實體化 | 何時省略 |
|---|---|---|
| `nodes::Input("name")` | 沒有上游圖與其連接，因此它必須是公開的 `Run::push("name", ...)` 端點 | 有上游圖向其提供資料，因此它僅僅是一個內部片段參數 |
| `nodes::Output("name")` | 沒有下游圖使用它，因此它必須是公開的 `Run::pull("name")` 端點 | 有下游圖使用它，因此它僅僅是一個內部片段的回傳值 |

省略並不意味著被遺忘。編譯器會保留來源資訊，因此 `describe()`、驗證錯誤、
指標和執行階段匯出 JSON 仍然可以參考使用者介面的端點名稱。

這可以防止可重複使用的片段在應用程式中間建立隱藏的 appsrc/appsink 樣式的執行階段 I/O。例如：

```cpp
neat::Graph app;
app.connect(camera, route);
app.connect(route, display);
```

可執行資料的路徑是 `camera -> route body -> display`，而不是 `camera -> route.Input -> route.Output -> display`，中間還有額外的實體接收器/發送器。

## API 快速參考

### C++

```cpp
// Composition
neat::Graph app("debug_label");
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
app.connect(fragment_a, fragment_b);
app.connect("from_endpoint", "to_endpoint");

// Endpoint inspection
auto graph_inputs = app.inputs();
auto graph_outputs = app.outputs();

// Build/run
neat::Run run = app.build();
run.push("image", neat::TensorList{image});
auto out = run.pull("classes", 1000);

// Runtime endpoint inspection
auto run_inputs = run.input_names();
auto run_outputs = run.output_names();

// Validation/debug/export
neat::GraphReport report = app.validate();
std::cout << app.describe() << "\n";
app.save("app.graph.json");
neat::save_run_json(run, "/tmp/app.graph_run.json");
```

### Python

```python
app = pyneat.Graph("debug_label")
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

print(app.inputs())
print(app.outputs())

run = app.build()
run.push("image", [image])
out = run.pull("classes", timeout_ms=1000)

print(run.input_names())
print(run.output_names())

app.save("app.graph.json")
run.save_json("/tmp/app.graph_run.json")
```

## 更多閱讀資料

- [模型程式設計模型](/develop-apps/development-workflow/model)
- [節點程式設計模型：群組與邊界](/develop-apps/development-workflow/node#boundary-nodes)
- [張量和範例程式設計模型](/develop-apps/development-workflow/core_types)
- [執行階段調整（教學 016）](/tutorials/tune-throughput-and-queues)
- [診斷 (教學 012)](/tutorials/diagnose-a-pipeline)
- [GStreamer 層](/develop-apps/advanced-concepts/gstreamer_layer)
