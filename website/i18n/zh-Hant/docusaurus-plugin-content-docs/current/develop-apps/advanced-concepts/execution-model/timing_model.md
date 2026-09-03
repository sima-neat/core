---
title: "非同步與同步計時模型"
description: "`Graph.run(...)` 和 `Graph.build(...)`/ `Run` 之間的關聯是什麼——也就是說，工作何時開始，結果何時產生。"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/timing_model
---

# 非同步與同步計時模型

`Graph` 和 `Run` 提供兩種驅動工作的方式：

- **單次執行**：`Graph.run(input, ...)` 將輸入推送進去，並在單次呼叫中等待輸出。不帶輸入的 `Graph.run()` 會執行一個由來源擁有的圖，直到達到 EOS（End of Stream，串流結束）。
- **可重複使用的執行**：`Graph.build(...)` 會傳回一個即時的 `Run`。您的應用程式會推送輸入、提取輸出、關閉輸入、清空、測量，並停止執行。

這兩種模式都使用相同的 `Graph` 計劃和相同的硬體。不同之處在於誰擁有迴圈。

## 單次執行模式

當您只有一個輸入，並且想要最短的正確路徑時，請使用單次執行模式。

```cpp
simaai::neat::Graph graph("classifier");
graph.add(simaai::neat::nodes::Input("image"));
graph.add(model);
graph.add(simaai::neat::nodes::Output("classes"));

simaai::neat::TensorList out = graph.run(
    simaai::neat::TensorList{image_tensor});
```

對於由來源擁有的圖，請不帶任何輸入參數地呼叫 `Graph.run()`：

```cpp
simaai::neat::Graph graph("file_job");
graph.add(source_fragment);
graph.add(model);
graph.add(sink_fragment);

graph.run();  // Blocks until EOS or error.
```

針對以下情況，請使用單次執行模式：

- 煙霧測試；
- 短時間的驗證執行；
- 屬於原始碼的作業，這些作業應該完整執行；
- 程式碼不應管理一個長時間存在的執行階段控點。

## 可重複使用的執行模式

當您的應用程式擁有迴圈時，請使用 `Graph.build(...)`。

```cpp
auto run = graph.build();

while (have_more_inputs()) {
  run.push("image", simaai::neat::TensorList{next_tensor()});

  if (auto out = run.pull("classes", /*timeout_ms=*/0)) {
    consume(*out);
  }
}

run.close_input();
while (auto out = run.pull("classes", /*timeout_ms=*/1000)) {
  consume(*out);
}
run.close();
```

針對以下情況，請使用可重複使用的模式：

- 即時影片、RTSP 或相機輸入；
- 串流處理，應用程式控制節奏；
- 多輸入或多輸出圖；
- 吞吐量測試；
- 測量、匯出、清空和停止控制。

## 推送時機

`Run::push(...)` 在輸入於圖的邊界被接受後傳回。它不會等待該輸入遍歷每個節點。

當輸入佇列已滿時：

- `OverflowPolicy::Block` 對產生者施加反壓；
- `OverflowPolicy::DropIncoming` 拒絕新的輸入；
- `OverflowPolicy::KeepLatest` 捨棄較舊的佇列輸入，以保持即時路徑的新鮮度；
- `try_push(...)` 傳回 `false`，而不是阻塞。

選擇與來源相符的策略。檔案和批次作業通常需要 `Block`。即時串流通常需要新鮮度策略。

## 拉取時機

`Run::pull(...)` 從輸出邊界傳回下一個可用的 `Sample`。

當逾時和 EOS 可以共享相同的「無樣本」路徑時，請使用方便的超載：

```cpp
if (auto sample = run.pull("classes", /*timeout_ms=*/1000)) {
  consume(*sample);
}
```

當發生逾時、關閉或錯誤時，請使用結構化的狀態超載處理，以便對不同情況進行不同的處理：

```cpp
simaai::neat::Sample sample;
simaai::neat::PullError error;

switch (run.pull("classes", /*timeout_ms=*/1000, sample, &error)) {
case simaai::neat::PullStatus::Ok:
  consume(sample);
  break;
case simaai::neat::PullStatus::Timeout:
  break;
case simaai::neat::PullStatus::Closed:
  break;
case simaai::neat::PullStatus::Error:
  throw std::runtime_error(error.message);
}
```

沒有隱藏的後備機制：多輸出圖需要一個具名稱的 `pull("output", ...)`，除非只有一個明確的輸出。

## 遙測：測量您擁有的迴圈

在您的應用程式擁有的工作負載周圍使用 `Run::start_measurement(...)`。傳回的 `MeasureReport` 是公開的計時介面，包含：

- 端到端推送至輸出延遲和吞吐量；
- 推送、拉取和丟棄樣本的執行階段計數器；
- 在 `MeasureOptions` 中請求時，外掛程式/核心和邊緣計時；
- 當執行期間啟用電源監測時，可選的電源遙測。

除非問題是端到端應用程式成本，否則將設定、檔案下載、每幀記錄和報告匯出放在測量的熱迴圈之外。

## 更多資訊

- [執行圖形](/develop-apps/development-workflow/pipeline) — 生命週期、選項、反壓、測量和吞吐量。
- [`Graph::run()`](/reference/cppapi/classes/simaai-neat-graph) — 單次執行和來源擁有的進入點。
- [`Graph::build()`](/reference/cppapi/classes/simaai-neat-graph) — 可重複使用的執行進入點。
- [`Run`](/reference/cppapi/classes/simaai-neat-run) — 推送、拉取、關閉、清空、停止和測量。
