# 014 グラフ内にモデルを埋め込む

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | yolo_v8s |
| Labels | graph, hybrid, model, mpk |

## Concept

コンパイルされたモデルを、`graph.add(model)` を使用して、パブリックな `Graph` にドラッグ＆ドロップします。これにより、モデルの実行の周囲にグラフレベルのオーケストレーション（ルーティング、スケジューリング、追加の入力/出力）が適用され、内部のランタイムグラフに直接アクセスすることなく実現できます。

## Walkthrough

第3章では、単純な入力/出力ノードからグラフを構築しました。第1章では、モデルをスタンドアロンオブジェクトとして実行しました。この章では、これら2つを組み合わせます。`Model`自体がグラフ互換のノードであるため、他のステージと同様に、それを公開`Graph`に組み込むことができます。これは、グラフレベルの制御（複数の入力、名前付き出力、カスタムルーティングなど）が必要でありながら、モデルの実行を再利用可能な単一のフラグメントとして扱う場合に、ブリッジパターンを使用するプロダクションシステムが採用する手法です。

重要な考え方は、低レベルのランタイムグラフ、`StageModelExecutorOptions`、または内部ノードIDに決して触れないことです。モデルを`graph.add(...)`に渡し、NEATがそのフラグメント（必要に応じて、前処理/推論/後処理）をビルド時に適切な内部実行プランに変換します。最終的に、モデルを公開グラフに組み込み、組み込まれたトポロジを出力し、モデルの出力カーディナリティを読み取ることができます。

### モデルのロード {#step-load-model}

構築では、コンパイルされたアーカイブをロードし、第1章と同様に、実行に備えて準備します。ここでは、オプションオブジェクトは渡さず、パスのみを渡します。なぜなら、この章は前処理ではなく、構成について扱うからです。結果として得られる`Model`は、グラフレイヤーが理解できるオブジェクトになります。

### モデルをグラフに組み込む {#step-compose-graph}

これがこの章の主な目的です。新しい`Graph`に、名前付き入力境界、モデル自体、名前付き出力境界の順に3つのノードが追加されます。`Model`はグラフ互換であるため、`add(model)`は、モデル全体のルートを単一のフラグメントとして追加します。特別なAPIはなく、ランタイムに直接アクセスすることはありません。`graph.describe()`を出力すると、組み込まれたトポロジが表示され、モデルが名前付き境界の間に正しく組み込まれていることを確認できます。

**C++:** 境界は、`simaai::neat::nodes::Input("image")`と`nodes::Output("result")`から取得され、モデルは直接`graph.add(model)`に渡されます。

**Python:** 境界は、`pyneat.nodes.input("image")`と`pyneat.nodes.output("result")`から取得され、モデルは直接`graph.add(model)`に渡されます。

### モデルの確認 {#step-inspect-model}

最後に、モデルフラグメントが実際にどのようなものを提供するかを読み取ります。これにより、モデルが正しくロードされたことを確認し、グラフが下流で生成する出力トポロジを確認できます。

**C++:** `model.info()`は、情報構造体を返します。`model_name`と`output_topology.physical_outputs`および`logical_outputs`を出力することで、モデルの出力の配線が明確になります。

**Python:** この章のバインディングは、モデルフラグメントがパブリックグラフに追加されたことを確認するメッセージを単純に出力します。

## Run

この章では、モデルアーカイブ（`yolo_v8s`）が必要です。**Neatのインストールルート**（`share/`と`lib/`が含まれるディレクトリ）から、**Python**と**C++（事前にビルドされたもの）**のコマンドを実行します。**ソースからビルド**するには、**リポジトリのルート**からコマンドを実行します。

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

期待される出力（C++ビルドでは、まず合成されたグラフの説明が出力されます）：

```text
model=yolo_v8s physical_outputs=1 logical_outputs=1
[OK] 014_embed_model_inside_graph
```

（Pythonビルドでは、グラフの説明の後に`model fragment added to public Graph`が出力されます。）

この章のC++ソースを、カスタムの`CMakeLists.txt`を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## ソースファイル
- C++: `tutorials/014_embed_model_inside_graph/embed_model_inside_graph.cpp`
- Python: `tutorials/014_embed_model_inside_graph/embed_model_inside_graph.py`
