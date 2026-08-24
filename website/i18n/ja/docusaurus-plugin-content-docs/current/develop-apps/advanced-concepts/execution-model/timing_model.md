---
title: "非同期と同期のタイミングモデル"
description: "`Graph.run(...)`と`Graph.build(...)`/`Run`がどのように関連しているか — 作業がいつ行われ、結果がいつ返されるか。"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/timing_model
---

# 非同期と同期のタイミングモデル

`Graph`と`Run`は、処理を実行するための2つの方法を提供します。

- **1回の実行**: `Graph.run(input, ...)`は、入力を送信し、1回の呼び出しで出力が完了するのを待ちます。入力なしの`Graph.run()`は、EOS（End of Stream）に達するまで、ソースが所有するグラフを実行します。
- **再利用可能な実行**: `Graph.build(...)`は、実行可能な`Run`を返します。アプリケーションは、入力を送信し、出力を取得し、入力を閉じ、残りの処理を実行し、測定を行い、実行を停止します。

どちらのモードも、同じ`Graph`プランと、同じハードウェアを使用します。違いは、ループを誰が所有するかです。

## 1回の実行モード

1つの入力があり、最短の正しいパスを求める場合は、1回の実行モードを使用します。

```cpp
simaai::neat::Graph graph("classifier");
graph.add(simaai::neat::nodes::Input("image"));
graph.add(model);
graph.add(simaai::neat::nodes::Output("classes"));

simaai::neat::TensorList out = graph.run(
    simaai::neat::TensorList{image_tensor});
```

ソースが所有するグラフの場合、引数なしで`Graph.run()`を呼び出します。

```cpp
simaai::neat::Graph graph("file_job");
graph.add(source_fragment);
graph.add(model);
graph.add(sink_fragment);

graph.run();  // Blocks until EOS or error.
```

以下の用途には、ワンショットモードを使用してください。

- スモークテスト
- 短時間の検証実行
- 完了まで実行されるべき、ソースコードが所有するジョブ
- 長期間にわたってランタイムハンドルを管理する必要がないコード

## 再利用可能な実行モード

アプリケーションがループを所有している場合は、`Graph.build(...)` を使用してください。

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

以下の用途には、再利用可能なモードを使用します。

- ライブビデオ、RTSP、またはカメラ入力
- アプリケーションが処理速度を制御するストリーム処理
- 複数の入力または出力を持つグラフ
- スループットテスト
- 測定、エクスポート、ドレイン、および停止制御

## プッシュタイミング

`Run::push(...)` は、入力がグラフの境界で受け入れられた後に処理を終了します。入力がすべてのノードを通過するまで待機することはありません。

入力キューがいっぱいの場合：

- `OverflowPolicy::Block` は、プロデューサーにバックプレッシャーを適用します。
- `OverflowPolicy::DropIncoming` は、新しい入力を拒否します。
- `OverflowPolicy::KeepLatest` は、古いキューに入れられた入力を破棄し、ライブパスを最新の状態に保ちます。
- `try_push(...)` は、ブロックする代わりに `false` を返します。

ソースに合ったポリシーを選択します。ファイルおよびバッチジョブでは通常、`Block` が必要です。ライブストリームでは通常、最新性を保つポリシーが必要です。

## プルタイミング

`Run::pull(...)` は、出力境界から次の利用可能な `Sample` を返します。

タイムアウトと EOS が同じ「サンプルなし」パスを共有できる場合は、便利なオーバーロードを使用します。

```cpp
if (auto sample = run.pull("classes", /*timeout_ms=*/1000)) {
  consume(*sample);
}
```

タイムアウト、クローズ、エラーが発生した場合に、それぞれ異なる処理を行う必要がある場合は、構造化されたステータスオーバーロードを使用してください。

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

隠れたフォールバック処理は行わない：複数の出力を持つグラフには、明確な単一の出力がない限り、名前付きの`pull("output", ...)`が必要。

## テレメトリー：担当するループを計測する

アプリケーションが担当するワークロードの周囲に`Run::start_measurement(...)`を使用する。返される`MeasureReport`は、公開されているタイミング情報であり、以下が含まれる：

- エンドツーエンドのプッシュから出力までの遅延とスループット。
- プッシュ、プル、およびドロップされたサンプルのランタイムカウンタ。
- `MeasureOptions`で要求された場合のプラグイン/カーネルおよびエッジのタイミング。
- パワー監視が有効になっている場合のオプションの電力テレメトリー。

セットアップ、ファイルダウンロード、フレームごとのロギング、およびレポートのエクスポートは、計測対象のホットループの外で行う。ただし、質問がエンドツーエンドのアプリケーションコストである場合は除く。

## 関連資料

- [グラフを実行する](/develop-apps/development-workflow/pipeline) — ライフサイクル、オプション、バックプレッシャー、計測、およびスループット。
- [`Graph::run()`](/reference/cppapi/classes/simaai-neat-graph) — 1回限りの実行と、ソースが所有するエントリポイント。
- [`Graph::build()`](/reference/cppapi/classes/simaai-neat-graph) — 再利用可能な実行エントリポイント。
- [`Run`](/reference/cppapi/classes/simaai-neat-run) — プッシュ、プル、クローズ、ドレイン、ストップ、および計測。
