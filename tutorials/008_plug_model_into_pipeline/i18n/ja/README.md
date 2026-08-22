# 008 モデルをパイプラインに組み込む

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | graph, composition, patterns |

## Concept

モデルを、`Graph`、`model.graph()`、および`model.graph(options)`を備えたグラフに組み込みます。これらは、配線制御の度合いが異なる2つの構成パターンであり、簡単な実行と複数カメラによるデプロイのどちらに適しているかを判断できます。

## Walkthrough

第3章では、グラフをノードごとに構築します。これは最も明確な方法ですが、一度`Model`を作成すると、その内部をすべて手動で接続する必要はほとんどありません。`model.graph(...)`は、モデルのパイプラインをグループとして提供し、それを`Graph`に1つの`add(...)`で追加します。重要な点は、そのグループがどの程度の境界線（インターフェース）を一緒に持ち込むかであり、それは`ModelRouteOptions`によって制御されます。

この章では、同じモデルに対して2つの異なるルーティング設定を比較します。1つは、独自のパブリックな入力/出力境界を含む、スタンドアロンで実行可能なグラフであり、もう1つは入力を省略して、上流のソース（たとえばカメラ）に明示的な名前で接続できるグラフです。この章の終わりまでに、両方を構成し、それぞれのバックエンドのGStreamer文字列を出力することで、配線の違いを正確に確認できます。

### 実行可能なモデルグラフを構成する{#step-model-graph}

最初のパターンでは、モデルに完全に実行可能なグラフを要求します。ルーティングオプションで`include_input = true`と`include_output = true`を設定すると、`model.graph(opts)`は、モデルグループの周囲に明示的なパブリックな入力と出力の境界を挿入します。これにより、結果として得られる`Graph`は、他の要素を接続することなく、単独で構築および実行できます。`graph.add(model.graph(opts))`が全体の構成であり、この単一の`add`が、すべてのパターンで最終的に行われる処理です。`describe_backend()`を出力すると、生成されたGStreamerパイプライン文字列が表示されます。

**C++:** ルーティングオプションは`Model::RouteOptions`であり、グラフは`simaai::neat::Graph`です。

**Python:** ルーティングオプションは`pyneat.ModelRouteOptions`であり、グラフは`pyneat.Graph`です。

### 接続時のルーティングオプションを構成する{#step-route-options}

2番目のパターンでは、モデルを独自の入力を持たずに、上流のソースに接続します。ここでは、`include_input = false`によってパブリックな入力境界を削除します（フレームは他の場所から取得されます）、`include_output = true`によって出力を保持し、`upstream_name`、`name_suffix`、および`buffer_name`によって配線と要素の名前を明示的にします。このように一貫した命名を行うことで、バックエンドグラフが読みやすく、複数のカメラまたは複数のモデルを使用したデプロイメントで診断しやすくなります。

### モデルグループを接続する{#step-attached-graph}

これらのオプションを設定すると、`graph.add(model.graph(opts))` は同じモデルグループを注入し、今度は独自のソースを保持するのではなく、名前付きの上流に接続するように設定します。これは、最初のパターンと同じ `add` 呼び出しであり、変更されたのはルートオプションだけです。つまり、合成は単一の操作であり、`ModelRouteOptions` は、グループがどの境界を伴って処理されるかを決定する設定項目です。

**C++:** 各バリアントは、`describe_backend()` を出力するため、2つのバックエンド文字列を比較できます。その後、ファイルは手動で作成した直接的な `Input -> Output` グラフを構築および実行して、エンドツーエンドのパスを確認し、`direct_rank=` を出力します。

**Python:** 添付されたバリアントは、合成が成功したことを確認するために `attached_graph_built=True` を出力します。

## Run

**Neat インストールルート**（`share/` と `lib/` を含むディレクトリ）から、**Python** および **C++（事前にビルドされたもの）** コマンドを実行します。**ソースからビルド** コマンドは、**リポジトリルート**から実行します。

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

予想される出力（C++ ビルドは、各バックエンドグラフ文字列、次に直接グラフのランクを出力します）。

```text
model_graph_backend=
...
attached_graph_backend=
...
direct_rank=3
[OK] 008_plug_model_into_pipeline
```

（Python ビルドは、`direct_graph_backend=` の後にバックエンド文字列、次に `attached_graph_built=True` を出力します。）この章の C++ ソースを、カスタムの `CMakeLists.txt` を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある [チュートリアルの実行方法](/tutorials#compile-a-copy-yourself) を参照してください。

## ソースファイル
- C++: `tutorials/008_plug_model_into_pipeline/plug_model_into_pipeline.cpp`
- Python: `tutorials/008_plug_model_into_pipeline/plug_model_into_pipeline.py`
