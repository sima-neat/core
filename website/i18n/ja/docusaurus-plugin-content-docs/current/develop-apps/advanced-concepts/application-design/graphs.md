---
title: "グラフを使ってアプリケーションを構築する"
description: "公開されているグラフ API を使用して、モデル、ノード、名前付きの入力/出力、分岐、結合、および実行をどのように構成するか。"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/graphs
---

# グラフを使用したアプリケーションの構築

コンパイル済みのモデルアーカイブを1つだけロードして実行したい場合は、`Model` を使用します。モデルとノードを中心にアプリケーションを構築したい場合は、`Graph` を使用します。公開入力と出力、再利用可能なフラグメントの接続、ストリームの分岐、ストリームの結合、アプリケーションの検証、および実際に実行された内容の保存または可視化を行います。

このモデルは、意図的にシンプルに設計されています。

| 概念 | 意味 |
|---|---|
| `Model` | ディスクからロードされたコンパイル済みのモデルアーカイブ。例：`resnet50.tar.gz` または `yolov8.tar.gz`。 |
| `Node` | 1つの処理ステップ：入力、出力、変換、ソース、シンク、モデルステージ、またはヘルパーステージ。 |
| `Graph` | アプリケーションの配線計画：存在するノード/フラグメントと、それらの間のデータフロー。 |
| `Run` | `Graph::build()` によって返されるライブ実行ハンドル：入力のプッシュ、出力のプル、メトリクスの収集、停止。 |

要するに：

```text
Graph = what to run
Run   = the running instance
```

より短いパスが必要な場合は、タスクページから始めます。

- [グラフ](/develop-apps/development-workflow/graph) は、グラフの作成方法を説明します。
- [グラフを実行する](/develop-apps/development-workflow/pipeline) は、ランタイムのライフサイクル、キュー、測定、およびスループットを説明します。
- [ノード](/develop-apps/development-workflow/node) は、一般的なノードとグループをマッピングします。
- [テンソルとサンプル](/develop-apps/development-workflow/core_types) は、ペイロードとメタデータを説明します。

ほとんどのアプリケーションコードでは、公開されている `simaai::neat::Graph` と `simaai::neat::Run` を使用する必要があります。より低レベルの実装ネームスペースを使用してアプリケーションを構築しないでください。これらは顧客向けのAPIではありません。

## グラフが必要なのはいつですか？

| 目的                                  | 推奨されるAPI                               |
| ------------------------------------- | --------------------------------------------- |
| 1つの入力に対して1つのモデルを実行する | `Model::run(...)` または `Model::build(...)` |
| モデルの周囲にアプリケーションの入出力境界を追加する | `Graph`                     |
| カスタム処理ノードを使用してモデルを構成する | `Graph::add(...)`                   |
| 複数のアプリケーションでグラフの一部を再利用する | `Graph` の一部を返す/渡す |
| 複数の入力または出力をルーティングする | 名前付きの `nodes::Input(...)` / `nodes::Output(...)` と `connect(...)` |
| 1つのストリームを複数のコンシューマーに分岐する | `graphs::Branch(...)`                 |
| 複数のストリームを1つの論理出力に結合する | `graphs::Combine(...)` と `CombinePolicy` |
| 実行されたトポロジーとメトリックを保存または視覚化する | `save_run_json(run, ...)`             |

## 最初のグラフ：1つの入力、1つのモデル、1つの出力

これは、最も小さい完全なアプリケーションスタイルのグラフです。

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

行ごとに説明します。

- `nodes::Input("image")` は、`image` という名前のパブリックな入力ポートを宣言します。
- `app.add(model)` は、モデルによって選択されたルートをグラフに挿入します。
- `nodes::Output("classes")` は、`classes` という名前のパブリックな出力ポートを宣言します。
- `app.build()` は、グラフ全体を検証およびコンパイルし、`Run` を返します。
- `run.push("image", ...)` は、指定された入力にデータを送信します。
- `run.pull("classes", ...)` は、指定された出力からデータを受信します。

Python で記述した場合も同じ構造になります。

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

Pythonの`Run.push(...)`は、バッチのようなシーケンスを想定しています。単一のテンソル/サンプルオブジェクトではなく、`[tensor]`または`[sample]`を渡してください。

## グラフの実行

Neatの他の場所で使用されているものと同じパブリックペイロード型を、組み込みの`Run`が受け入れます。

| ペイロード | 使用する場面 |
|---|---|
| `TensorList` | テンソルを渡し、追加のサンプルメタデータが必要ない場合。 |
| `Sample` | タイムスタンプ、`frame_id`、`stream_id`、テキスト/オーディオ/ビデオメタデータ、検出結果、またはEOSが必要な場合。 |
| `std::vector<cv::Mat>` | OpenCVから画像データを簡単に入力したい場合。 |

一般的なC++呼び出し：

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

タイムアウトまたは接続が閉じられた場合に空の `std::optional` を返す必要がある場合は、`pull(...)` を使用します。タイムアウトまたはエラー時に例外をスローする、型付きの便利なヘルパーが必要な場合は、`pull_tensors(...)` または `pull_samples(...)` を使用します。

有限のアプリプッシュストリームの場合、最終的なメトリックを収集する前に、入力ストリームを閉じて、バッファを空にします。

```cpp
run.close_input();
while (auto out = run.pull("classes", 1000)) {
  // Drain remaining output.
}
run.stop();
```

タスクに焦点を当てたランタイムのプレイブック（キューポリシー、出力の所有権、ドロップテレメトリ、マルチストリーム測定などを含む）については、[グラフを実行する](/develop-apps/development-workflow/pipeline) を参照してください。

## `build()` と `build(first_input)`

ほとんどのグラフは、入力サンプルなしで構築できます。

```cpp
neat::Run run = app.build();
```

グラフがすでに十分な形状/キャプチャ情報を持っている場合、またはグラフが自身のソースノード（RTSP/ファイル/静止画入力など）を所有している場合に、これを使用します。

シードビルドでは、Neat はビルド中に最初の入力を受け取ります。

```cpp
neat::Run run = app.build(neat::TensorList{first_image});
```

ストリーミングが開始される前に、最初の入力が形状/フォーマットの調整の初期値として使用されるようにする場合にこれを使用します。初期値を使用したビルド前のチェックはデフォルトで有効になっているため、Neat は、ビルド中に最初のサンプルでエラーが発生した場合に、すぐにエラーが発生して終了する `Run` を返す代わりに、初期値を一度プッシュ/プルしてエラーを捕捉することができます。

スループット、遅延、および電力に関する数値については、実際のワークロードが実行された後にメトリックを保存し、ビルド直後には保存しないでください。

## グラフ名はエンドポイント名ではありません

:::warning
`Graph("name")` は、診断、保存されたグラフファイル、および可視化のためのラベルです。これは、`name` という名前のパブリックな入力または出力を宣言するものではありません。
:::

誤った思考モデル：

```cpp
neat::Graph camera("image");
// This does not make run.push("image", ...) valid by itself.
```

正しいエンドポイントの宣言：

```cpp
neat::Graph camera("camera_route");
camera.add(neat::nodes::Input("image"));
```

そして、出力例は次のとおりです。

```cpp
neat::Graph classifier("classifier");
classifier.add(neat::nodes::Output("classes"));
```

`Input("image")`と`Output("classes")`を、グラフの一部における公開されたインターフェースと捉えましょう。グラフの名前は、単にそのインターフェースを示す標識に過ぎません。

## 推測する代わりに、エンドポイント名を確認する

ビルド前に、グラフによって宣言された論理的な公開エンドポイントを確認してください。

```cpp
for (const auto& name : app.inputs()) {
  std::cout << "graph input: " << name << "\n";
}
for (const auto& name : app.outputs()) {
  std::cout << "graph output: " << name << "\n";
}
```

ビルド後、`Run` が実際にどのような入力を受け付けるかを確認してください。

```cpp
for (const auto& name : run.input_names()) {
  std::cout << "run input: " << name << "\n";
}
for (const auto& name : run.output_names()) {
  std::cout << "run output: " << name << "\n";
}
```

この機能を、モデルのルートや、複数の入力と複数の出力を伴うアプリケーションに使用してください。エンドポイントのマッチングは厳密に行われます。

`Input("image_l")` は、`image_l` という名前のモデル入力にバインドできますが、`Input("my_random_name")` はバインドできません。

## 名前が不要な便利な API

1つの入力と1つの出力を持つグラフの場合、エンドポイント名を省略できます。

```cpp
neat::Graph app;
app.add(neat::nodes::Input());
app.add(model);
app.add(neat::nodes::Output());

neat::Run run = app.build();
run.push(neat::TensorList{image});
auto result = run.pull(1000);
```

これは、簡単なスクリプトやテストに便利です。より複雑なアプリケーションの場合は、名前を使用することをお勧めします。

グラフに複数の入力または出力の可能性がある場合、名前のない `push(...)` または `pull()` はエラーを発生させ、利用可能な名前を報告します。このエラーは意図的なものです。Neat は、どのカメラ、テンソル、または出力ヘッドを意図しているのかを推測すべきではありません。

## モデルはグラフの一部

`Model` は、グラフに直接追加できます。

```cpp
neat::Model yolo("yolov8.tar.gz");

neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

`Graph::add(model)` は、アーカイブとモデルオプションから選択されたモデルのルートを挿入します。そのルートには、前処理、MLA推論、後処理、テンソルの変換、および検出デコードの段階が含まれる場合があります。一般的な線形の場合、`model.graph()` を手動で呼び出す必要はありません。

より高度な構成を行うには、ルートを調べて、`Graph` の一部として再利用してください。

```cpp
neat::Graph route = yolo.graph();

auto model_inputs = route.inputs();
auto model_outputs = route.outputs();
```

### 複数入力モデル

複数入力モデルの場合、名前を推測しないでください。経路を尋ねてください。

```cpp
neat::Graph route = model.graph();

for (const auto& name : route.inputs()) {
  std::cout << "model expects input: " << name << "\n";
}
```

次に、モデルの入力名に合わせて、上流のフラグメントに名前を付けます。

```cpp
neat::Graph left_camera;
left_camera.add(neat::nodes::Input("image_l"));

neat::Graph uv_camera;
uv_camera.add(neat::nodes::Input("image_uv"));

neat::Graph app;
app.connect(left_camera, route);  // Binds image_l -> model image_l.
app.connect(uv_camera, route);    // Binds image_uv -> model image_uv.
```

`left_camera` が `Input("a_new_name_image_l")` を宣言した場合、`image_l` にバインドされません。暗黙的な名前変更に頼るのではなく、正しいエンドポイント名を持つ小さなアダプターグラフを追加してください。

### 単独のモデルグラフ

デフォルトでは、`model.graph()` は、名前付きのエンドポイントが公開された再利用可能なモデルフラグメントを返します。返されたグラフを単独で実行可能にしたい場合は、明示的なパブリック入力/出力ノードを要求してください。

```cpp
neat::Model::RouteOptions route_opt;
route_opt.include_input = true;
route_opt.include_output = true;

neat::Graph standalone = model.graph(route_opt);
neat::Run run = standalone.build();
```

高度な使用やデバッグのために、モデルのルーティングによって個々の物理的な出力にアクセスできるようにすることができます。

```cpp
route_opt.expose_all_outputs = true;
```

特別な理由がない限り、この機能を無効のままにしておきます。デフォルトのモデルの動作は、ルートコントラクトで予期される論理モデル出力を公開することです。モデルに物理出力が1つしかない場合でも、`expose_all_outputs = true` は依然として1つの出力のみを公開します。

## `add()` と `connect()`

2つの合成ツールがあります。

| API | 意味 | 使用する場面 |
|---|---|---|
| `add(x)` | 現在の線形チェーンに連結または挿入します。 | 「同じパイプラインの次のステップ」を意味する場合。 |
| `connect(a, b)` | 名前付きのエンドポイントを使用して、2つのグラフ断片を接続します。 | 再利用可能な断片を合成する場合、またはトポロジーを構築する場合。 |
| `connect("a", "b")` | 同じグラフ内にすでに宣言されている2つのエンドポイントを接続します。 | 小さなヘルパー断片を構築する場合。 |

線形合成：

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
```

フラグメントの構成：

```cpp
neat::Graph app;
app.connect(camera, model_route);
app.connect(model_route, output_sink);
```

ヘルパーフラグメント内の内部エンドポイントの配線：

```cpp
neat::Graph pass_through("pass_through");
pass_through.add(neat::nodes::Input("in"));
pass_through.add(neat::nodes::Output("out"));
pass_through.connect("in", "out");
```

重要なルール：`add()` は線形のチェーンを意味します。`connect()` はグラフのトポロジーを意味します。

## 再利用可能なグラフフラグメント

関数は、再利用可能なグラフフラグメントを返せます。

```cpp
neat::Graph make_classifier(neat::Model& model) {
  neat::Graph g("classifier");
  g.add(neat::nodes::Input("image"));
  g.add(model);
  g.add(neat::nodes::Output("classes"));
  return g;
}
```

再利用可能なフラグメントを直線的に使用します。

```cpp
neat::Graph classifier = make_classifier(model);

neat::Graph app;
app.add(classifier);
```

または、ワイヤーの断片を明示的に指定してください。

```cpp
neat::Graph app;
app.connect(camera, classifier);
app.connect(classifier, class_sink);
```

分岐の後に`add()` を追加すると意味が曖昧になる場合、Neat は処理を中断し、代わりに `connect(...)` を使用するように指示します。
それは、間違った分岐に黙って追加するよりも優れています。

## 1つのストリームを分岐させる

1つの入力ストリームを複数の名前付き出力に送る必要がある場合は、`graphs::Branch` を使用します。

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});
```

意味：

```text
image -> preview
      -> model_input
```

例：

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

ブランチをモデルに接続する際に、モデルの入力名と一致するように、ブランチの出力名を選択してください。

```cpp
neat::Graph route = model.graph();
for (const auto& name : route.inputs()) {
  std::cout << "choose a branch output matching: " << name << "\n";
}
```

分岐は明示的であるため、キューとバックプレッシャーに影響を与えます。ある分岐の処理が遅い場合、出力オプションと下流のグラフによって、他の分岐と比較して処理速度が低下したり、データが破棄されたりする可能性があります。

Python：

```python
branch = pyneat.graphs.branch("image", ["preview", "model_input"])
```

## 複数のストリームを結合する

複数の入力ストリームを 1 つの論理的な出力にまとめる必要がある場合は、`graphs::Combine` を使用します。

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);
```

意味：

```text
left  --\
        +--> stereo
right --/
```

ポリシー：

| ポリシー | 意味 |
|---|---|
| `CombinePolicy::None` | 自動的に結合しない。複数のプロデューサーからの出力を1つの出力にまとめようとすると、処理は失敗する。 |
| `CombinePolicy::ByFrame` | 完全に同じ`Sample::frame_id`を持つサンプルを一致させる。フレームIDが欠損している場合、処理は失敗する。PTSによる代替処理は行われない。 |
| `CombinePolicy::ByPts` | 完全に同じ`Sample::pts_ns`のプレゼンテーションタイムスタンプを持つサンプルを一致させる。PTSが欠損している場合、処理は失敗する。フレームIDによる代替処理は行われない。 |

平易な表現：

- `ByFrame`は、「同じフレーム番号を持つ左右のサンプルをください」という意味です。
- `ByPts`は、「同じメディアタイムスタンプを持つサンプルをください」という意味です。

例：

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

サンプルに必須のキーが含まれていない場合、推測する代わりに、エラーメッセージが表示されて結合処理は失敗します。

## データソースとデータシンク

グラフにデータが入力される方法は2つ、グラフからデータが出力される方法も2つあります。

### アプリケーションからプッシュされる入力

アプリケーションコードがデータをプッシュする場合、`nodes::Input(...)` を使用します。

```cpp
app.add(neat::nodes::Input("image"));
run.push("image", neat::TensorList{image});
```

### グラフが所有する入力ソース

グラフがデータソースを所有している場合は、ソースノードまたはソースフラグメントを使用します。

```cpp
app.add(neat::nodes::RTSPInput("rtsp://camera/stream"));
```

または、再利用可能なデコード済みの RTSP フラグメント：

```cpp
neat::nodes::groups::RtspDecodedInputOptions opt;
opt.url = "rtsp://camera/stream";

app.add(neat::nodes::groups::RtspDecodedInput(opt));
```

グラフが自身のソースを所有する場合、通常は`build()`を呼び出し、その後で出力を取得します。アプリケーションコードからそのソースにデータを書き込むことはありません。

### アプリケーションから取得する出力

アプリケーションコードが出力を取得する必要がある場合は、`nodes::Output(...)`を使用します。

```cpp
app.add(neat::nodes::Output("detections"));
auto out = run.pull("detections", 1000);
```

### グラフが所有する出力先

グラフが結果を自身で書き出す必要がある場合は、出力先ノードまたはグループを使用します。

```cpp
neat::UdpOutputOptions udp;
udp.host = "192.0.2.10";
udp.port = 5000;

app.add(neat::nodes::UdpOutput(udp));
```

そのモード用に構築されたグラフに対しては、サーバー形式の RTSP 出力も利用できます。

```cpp
neat::RtspServerHandle server = app.run_rtsp(rtsp_options);
```

## 検証と診断

ランタイムリソースを起動する前に、構造化されたレポートが必要な場合は、ビルド前に検証を実行してください。

```cpp
neat::GraphReport report = app.validate();
if (!report.error_code.empty()) {
  std::cerr << report.repro_note << "\n";
}
```

捕まえる `NeatError` build/run/push/pull コマンドの実行時:

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

便利なデバッグ用ツール：

```cpp
std::cout << app.describe() << "\n";
std::cout << app.describe_backend() << "\n";
```

- `describe()` は、公開されているグラフの概要（エンドポイント、フラグメント、およびトポロジー）を出力します。
- `describe_backend()` は、より詳細なバックエンド情報を出力します。これは、生成されたパイプライン文字列またはランタイムルーティングのデバッグ時に役立ちます。

エラーコードの分類とトリアージのワークフローについては、[エラーコード](/reference/error-codes/) を参照してください。

## グラフ構成の保存と読み込み

`Graph::save(path)` は、公開されているグラフ構成（ノード、エンドポイント名、明示的なエンドポイントエッジ、出力オプション、結合ポリシー、およびモデルルートの出所）を書き込みます。

```cpp
app.save("app.graph.json");

neat::Graph loaded = neat::Graph::load("app.graph.json");
neat::Run run = loaded.build();
```

これは、実行中のパイプラインやランタイムメトリクスではなく、グラフプランを保存します。ランタイムメトリクスについては、Run JSON エクスポートを使用してください。

モデルのルーティングの系統が重要です。モデルの断片は、バックエンドのスニペットのリスト以上のものです。モデルアーカイブから派生した入力/出力名、ルーティングオプション、および複数入力モデルの入力ルーティングプロセッサメタデータを含んでいます。保存されたグラフにモデルの断片が含まれている場合、Neat は、それを再構築するために必要なモデルアーカイブのパスとルーティングオプションを保存します。アーカイブがロード時に見つからない場合、Neat は、不完全なルーティングをサイレントに構築するのではなく、対処可能なエラーを表示して処理を停止します。

## 実行された内容をエクスポートして可視化する

`Run` は、公開されているグラフの形状と、最適化されたランタイムの形状の両方を把握しています。これは、CI、デバッグ、サポートチケット、またはオフラインでの可視化のために、バージョン管理された JSON アーティファクトとしてエクスポートできます。

### ビルド時のトポロジーのスナップショット

グラフの構築が完了した直後にアーティファクトが必要な場合は、ビルド時のエクスポートを使用してください。

```cpp
neat::RunOptions opt;
opt.run_export.path = "/tmp/startup.graph_run.json";
opt.run_export.label = "startup";

neat::Run run = app.build(opt);
```

これは、初期のトポロジーのスナップショットです。まだサンプルが実行されていないため、スループット/遅延カウンターがゼロになっている場合があります。

Python：

```python
opt = pyneat.RunOptions()
opt.run_export.path = "/tmp/startup.graph_run.json"
opt.run_export.label = "startup"

run = app.build(opt)
```

### 実行後のスナップショット（メトリクス付き）

ワークロードの実行後または終了後に、実行後のエクスポート機能を使用します。

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

エクスポート機能は、現在の実行状況のスナップショットを記録します。実行を停止することはありません。有限のワークロードの最終的な数値が必要な場合は、保存する前に、`run.close_input()` を呼び出して出力を処理するか、`run.stop()` を呼び出してください。

ボードの電力に関するテレメトリデータを含めるには：

```cpp
neat::RunOptions opt;
opt.enable_board_power(/*sample_interval_ms=*/100);

neat::Run run = app.build(opt);
```

JSONスキーマは、バージョン`1`の`sima.neat.graph_run`です。スキーマは`schemas/graph_run_v1.schema.json`に、CIバリデーターは`tests/perf/tools/graph_run_schema.py`にあります。

インターネットに接続せずに、グラフのアーティファクトをレンダリングします。

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json -o /tmp/final.graph_run.html
```

表示するビューを選択してください。

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view public
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view lowered
```

- `public` は、ユーザーが作成したグラフを表示します。具体的には、名前付きの入力、出力、フラグメント、および `connect(...)` エッジが表示されます。
- `lowered` は、Neat が内部で実行した内容を表示します。具体的には、パイプラインのセグメント、生成された分岐/結合ステージ、キュー、およびランタイムエッジが表示されます。

## 一般的なパターン

### 画像分類

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(resnet);
app.add(neat::nodes::Output("classes"));
```

### オブジェクト検出

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### RTSPカメラからモデルへ、そしてアプリが引き出す出力

```cpp
neat::nodes::groups::RtspDecodedInputOptions source_opt;
source_opt.url = "rtsp://camera/stream";

neat::Graph app;
app.add(neat::nodes::groups::RtspDecodedInput(source_opt));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### アプリケーションからの入力と、グラフが所有する UDP 出力

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::UdpOutput(udp_options));
```

### ブランチのプレビューとモデルのパス

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_image"});
```

モデルのルート入力と一致するように、ファイル名を `model_image` に変更するか、明示的なアダプターフラグメントを挿入します。

### 左/右ストリームを結合

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "pair",
                                         neat::CombinePolicy::ByPts);
```

メディアのタイムスタンプが同期のキーである場合は、`ByPts` を使用します。フレーム ID が同期のキーである場合は、`ByFrame` を使用します。

### GenAI およびその他のステージフラグメント

GenAI およびその他の非線形/ステージベースの機能は、引き続きアプリケーションコードにパブリックな `Graph` フラグメントとして組み込まれ、`Graph::build() -> Run` を通じて実行される必要があります。

```cpp
neat::Graph app;
app.add(genai_fragment);

neat::Run run = app.build();
run.push("prompt", prompt_sample);
auto token = run.pull("tokens", 1000);
```

正確な GenAI フラグメントファクトリとサンプルヘルパーの名前は、インストールされている GenAI パッケージによって異なります。
グラフルールは同じです。パブリックフラグメントを追加または接続し、次に名前付きの `Run::push(...)` と `Run::pull(...)` を使用します。

## 避けるべきパターンと注意点

### グラフラベルをエンドポイントとして使用しない

誤り：

```cpp
neat::Graph image("image");
run.push("image", neat::TensorList{tensor}); // Graph label is not an endpoint.
```

正しい：

```cpp
neat::Graph image;
image.add(neat::nodes::Input("image"));
```

### モデルへの入力項目の名前を推測しないでください

誤り：

```cpp
left.add(neat::nodes::Input("my_left"));
app.connect(left, model);
```

正しい：

```cpp
for (const auto& name : model.graph().inputs()) {
  std::cout << name << "\n";
}
```

次に、上流のエンドポイントに適切な名前を付けてください。

### 複数のエンドポイントを持つグラフで、名前のないプッシュ/プル操作は使用しないでください。

誤った例：

```cpp
run.push(neat::TensorList{left});
run.push(neat::TensorList{right});
```

正しい：

```cpp
run.push("left", neat::TensorList{left});
run.push("right", neat::TensorList{right});
```

### CombinePolicy を設定せずに誤ってファンインしないようにしてください。

誤った例：

```cpp
neat::Graph bundle;
bundle.add(neat::nodes::Output("bundle"));

app.connect(left, bundle);
app.connect(right, bundle); // Ambiguous: how should left/right be synchronized?
```

正しい：

```cpp
neat::Graph bundle = neat::graphs::Combine({"left", "right"},
                                           "bundle",
                                           neat::CombinePolicy::ByFrame);
```

### 入力/出力を途中に挿入しないこと（ただし、フラグメントの境界を示す場合を除く）

`Input` と `Output` は、公開されている境界の宣言です。再利用可能なフラグメントにおいては、まさにそれが必要なものです。純粋に線形なアプリケーションにおいて、途中に余分な `Output` を追加すると、別の `connect(...)` エッジによって消費されない限り、実際にデータを引き出すことができるシンクやバックプレッシャーが生じる可能性があります。

### アプリケーションコードで、より低レベルのランタイムグラフAPIを使用しないこと

より低レベルのランタイムグラフAPIを使用して、アプリケーションコードを記述したり、その方法を教えたりすることは避けてください。

代わりに、公開されているアプリケーションのインターフェースを使用してください。

```cpp
neat::Graph
neat::Run
app.build()
```

## 補足：境界の具現化

名前付きの`Input`および`Output`ノードは、フラグメントのパブリックコントラクトの宣言です。これらは、バッファーの移動に使用されるランタイムオブジェクトよりも上位レベルにあります。

実行可能なパイプラインの構築前に、`Graph::build()`は境界を正規化します。

| 境界の宣言 | 具現化されるのは… | 省略されるのは… |
|---|---|---|
| `nodes::Input("name")` | 上流のグラフが接続されていない場合。つまり、パブリックな`Run::push("name", ...)`エンドポイントでなければならない | 上流のグラフからデータが供給される場合。つまり、内部フラグメントのパラメータにすぎない |
| `nodes::Output("name")` | 下流のグラフが消費しない場合。つまり、パブリックな`Run::pull("name")`エンドポイントでなければならない | 下流のグラフが消費する場合。つまり、内部フラグメントの戻り値にすぎない |

省略とは、完全に忘れ去ることではありません。コンパイラはプロビナンスを保持するため、`describe()`、検証エラー、メトリクス、およびランタイムエクスポートJSONは、依然としてユーザー向けのエンドポイント名を参照できます。

これにより、再利用可能なフラグメントが、アプリケーションの中央で隠れたappsrc/appsinkスタイルのランタイムI/Oを作成することが防止されます。たとえば：

```cpp
neat::Graph app;
app.connect(camera, route);
app.connect(route, display);
```

実行可能データのパスは`camera -> route body -> display`であり、`camera -> route.Input -> route.Output -> display`ではありません。途中に追加の物理的な終端/供給ポイントがあります。

## APIクイックリファレンス

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

## 参考資料

- [モデルプログラミングモデル](/develop-apps/development-workflow/model)
- [ノードのプログラミングモデル：グループと境界](/develop-apps/development-workflow/node#boundary-nodes)
- [テンソルとサンプルを使用したプログラミングモデル](/develop-apps/development-workflow/core_types)
- [ランタイムの調整（チュートリアル 016）](/tutorials/tune-throughput-and-queues)
- [診断（チュートリアル 012）](/tutorials/diagnose-a-pipeline)
- [GStreamer レイヤー](/develop-apps/advanced-concepts/gstreamer_layer)
