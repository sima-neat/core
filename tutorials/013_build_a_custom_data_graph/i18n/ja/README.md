# 013 カスタムデータグラフを作成する

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | graph, traversal, metadata |

## Concept

最も小さく、実用的な公共のものを構築する。 Neat `Graph` —1つは「～と名付けられた」 `Input` 名前が付けられたものに接続されています `Output` —次に、サンプルを名前で指定して送り込み、サンプルのメタデータが処理を通じて保持されることを確認します。

## Walkthrough

第 3 章では、匿名の入力 → 出力グラフを構築し、位置情報を用いて実行しました。 `run()` 呼び出し。実際のオーケストレーション（ファンアウト、ファンイン、ストリームごとのルーティング）では、位置ではなく*名前*でエンドポイントを扱う必要があります。この章では、可能な限り小さなグラフで、名前付きエンドポイントの概念を紹介します。これにより、マルチストリームおよび埋め込みモデルの章でそれらを拡張する前に、名前付けと接続の仕組みを分離して確認できます。

公開 `Graph` アプリケーションの構成面は次のとおりです。 `add(...)` ノード、 `connect(...)` 名前付きのエンドポイント、 `build()` 一度、再利用可能なものに `Run`そして `push("image", ...)` そして `pull("out", ...)` 名前で。最終的には、1つのテンソルをプッシュすることになります。 `Sample` 名前付きのグラフを通じて、その内容を確認しました。 `stream_id`, `frame_id`および `pts_ns` 変更なしで出力されました。これは、ランタイムがメタデータを最初から最後まで完全に保持することの証明です。

### グラフを構成します {#step-compose-graph}

ノードを2つ追加します。 `Input("image")` 「push」という名前の終点（エンドポイント）を宣言します。 `image`; `Output("out")` 「pull」という名前の終端点を宣言します。 `out`名前は契約内容を表します。これは、まさにあなたが渡す文字列そのものです。 `push(...)` そして `pull(...)` 後で。エンドポイントに名前を付ける（追加順に依存するのではなく）ことで、複数の入力または出力を持つ大規模なグラフを操作する際に、曖昧さをなくすことができます。

**C++:** ノードは `simaai::neat::nodes::Input("image")` と `nodes::Output("out")` から取得します。

**Python:** ノードは `pyneat.nodes.input("image")` と `pyneat.nodes.output("out")` から生成します。

### エンドポイントを接続する {#step-connect-endpoints}

`connect("image", "out")` エッジを宣言します：フレームがプッシュされるのは `image` 流れ `out`2つのノードだけで、これがネットワーク全体の構成となりますが。 `connect(...)` より大きなグラフでブランチやマージを構築するために使用するのと同じ関数です。次に、以下を出力します。 `graph.describe()` 構成されたトポロジーをダンプする — グラフが意図したとおりに接続されているかをすばやく確認し、構築を開始する前に検証します。

### サンプルを構築してプッシュ {#step-build-and-push}

`build()` （ここでは初期サンプルは不要です）記述を実際に実行可能なものに変換します。 `Run`次に、決定的なテンソルを1つ構築します。 `Sample` — 既知の情報を格納した8×8×3のRGB画像 `stream_id`, `frame_id`および `pts_ns` — そして `push(...)` それを `image` 名前でエンドポイントを指定します。サンプルに含まれるメタデータは、後で確認する内容です。

**C++:** `push(...)` は bool 型の値を返します。失敗した場合は `run.last_error()` を表示します。サンプルは `make_sample()` で作成します。

**Python:** `push("image", [sample])` はサンプルの一覧を受け取ります。サンプルは `make_rgb_sample()` で構築します。

### 出力を取得し、メタデータを検証します。 {#step-pull-and-verify}

`pull("out", ...)` 指定された出力エンドポイントから結果を取得し、タイムアウト時間を過ぎると、 `close()` 実行時。入力と出力の間に変換処理がないため、正しいパイプラインは同じ論理的なサンプルを返します。したがって、読み出し時には `stream_id`, `frame_id`および `pts_ns` そして、私たちが送信した値を確認することで、ランタイムがサンプルごとのメタデータをトラバーサルを通じて保持していることがわかります。この保証こそが、後続の処理段階でフレームの識別子とタイムスタンプを信頼できるようにするものです。

## Run

実行すると、グラフの説明の後に、往復処理されたメタデータが表示されるはずです。**Python**と**C++（事前にビルドされたもの）**のコマンドを**から実行してください。Neat root をインストールします（ディレクトリには、次のものが含まれます）。 `share/` そして `lib/`); **ソースコードからビルドする**ためのコマンドを**リポジトリのルートディレクトリ**から実行します。この章ではモデルアーカイブは必要ありません。

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

期待される出力（以下に続く） `graph.describe()` ダンプ：

```text
stream=graph frame=42 pts_ns=123456789
[OK] 013_build_a_custom_data_graph
```

（Pythonのビルドでは、`stream_id=graph frame_id=42 pts_ns=123456789` が出力されます。）この章のC++ソースコードをカスタムの `CMakeLists.txt` を使って独自のプロジェクトに組み込むには（追加のフォルダーは不要です）、ランディングページの[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## ソースファイル
- C++: `tutorials/013_build_a_custom_data_graph/build_a_custom_data_graph.cpp`
- Python： `tutorials/013_build_a_custom_data_graph/build_a_custom_data_graph.py`
