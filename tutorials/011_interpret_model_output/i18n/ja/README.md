# 010 モデルの出力の読み込みと解釈

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | None |
| Labels | output, patterns, sink |

## Concept

`run.pull()` または `model.run()` から安全にデータを読み込みます。各実行結果は `Sample` を返します。これは、テンソルであるか、名前付きフィールドの集合であるか、またはその両方である可能性のある、小さな合計型です。そのため、ペイロードにアクセスする前に、その種類を分類します。

## Walkthrough

スループットを最適化したり、複雑なグラフロジックを追加したりする前に、実行によって返されるものを読み取るための、安定した防御的な方法が必要です。出力は常に`Sample`ですが、その形状は異なります。単一のテンソルである場合もあれば、名前付きフィールドのバンドルである場合もあります（第009章参照）。バンドルから`.tensor`にアクセスしたり、存在しない形状を想定したりすることは、この章で回避することを学ぶバグです。

以前と同じ最小限の同期グラフを構築し、1フレームを実行し、次に結果を体系的に*検査*します。その`kind`、テンソルが存在するかどうか、フィールドの数、およびテンソルのランクを確認します。最終的には、ランタイムが提供するあらゆるモデルに対して機能する、再利用可能な出力読み取りパターンが得られます。

### 入力の構成 {#step-configure-input}

入力コントラクト（ピクセルの`format`、`width`、`height`、`depth`）を宣言し、プッシュするフレームと一致させます。これは、これらの章全体で使用されるのと同じ境界コントラクトです。

### グラフの作成と構築 {#step-compose-graph}

入力ノードを出力ノードに接続し、`build()`して同期`Run`に変換し、フレームを渡して、`build()`が具体的な形状をネゴシエートできるようにします。間にモデルがないため、出力は入力とミラーリングされます。これは、出力構造を研究するのに最適な場所である理由です。

### 1フレームの実行 {#step-run-frame}

1フレームをプッシュし、1つの結果を同期的に取得します。単一の`run(...)`呼び出しは、1フレームのショートカットです。その戻り値は、ここで分析するオブジェクトです。

**C++:** `run.run(...)`は`TensorList`を返します。単一のテンソル出力の場合、これは1つのエントリを意味し、次のステップで`out.size()`と`out.front()`を使用して検査します。

**Python:** `run.run(...)`は`Sample`を返し、`.kind`、`.tensor`、`.tensors`、および`.fields`を直接公開します。

### サンプルの検査 {#step-inspect-sample}

これが教訓です。ペイロードの前に構造を読み取ります。まず、存在と種類を確認し、次にテンソルの`shape`からランクを導き出します。各ステップ（空でない出力、空でない形状）を保護することで、形状を制御できないモデルに対して堅牢な出力リーダーを作成できます。

**C++:** `out.size()`とテンソルの存在を報告し、空の場合、または`out.front().shape`が空の場合に例外をスローし、次に`rank`を`shape.size()`から印刷します。（`fields=0`行はプレースホルダーです。`TensorList`は、Pythonの`Sample`が持つバンドルフィールド構造を運びません。）

**Python:** `sample.kind`、`sample.tensor is not None`、`len(sample.fields)`、および最初のテンソルのランクを出力します。これらはすべて、1つの場所にまとめて表示されます。テンソル型の結果の場合、`.tensor` が存在します。テンソルセットの結果の場合、`.tensors` を読み取ります。バンドルブランチの場合、`.kind` を確認し、`.fields` を読み取ります。

## Run

**Python** および **C++（事前にビルドされたもの）** コマンドを、**Neat のインストールルート**（`share/` と `lib/` を含むディレクトリ）から実行します。**ソースコードからビルド** コマンドは、**リポジトリのルート**から実行します。この章では、モデルアーカイブは必要ありません。

**Python:**
```bash
python3 share/sima-neat/tutorials/011_interpret_model_output/interpret_model_output.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_011_interpret_model_output
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_011_interpret_model_output
./build/tutorials-standalone/tutorial_011_interpret_model_output
```

予想される出力（C++）：

```text
outputs=1 has_tensor=yes fields=0
rank=3
[OK] 011_interpret_model_output
```

Python ビルドは、`Sample` 経由で同じ情報を出力します。

```text
sample_kind=SampleKind.TensorSet
has_tensor=False
num_fields=0
output_rank=3
```

この章の C++ ソースコードを、カスタムの `CMakeLists.txt` を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある [チュートリアルの実行方法](/tutorials#compile-a-copy-yourself) を参照してください。

## In Practice

モデルの出力を読み取るための防御的なチェックリスト。

### 読み取り前に分類する

- まず、`kind` を確認します。単一のテンソル結果は `SampleKind.Tensor`、複数のフィールドを持つ結果は `SampleKind.Bundle` です。
- テンソル型の場合、`tensor` が存在し、`fields` は空です。バンドル型の場合、`fields` を読み取り、`tensor` が存在することを前提としないでください。

### 契約を検証する

- テンソルを参照する前に、テンソルが存在することを確認します。
- ランクを計算または次元をインデックス化する前に、`shape` が空でないことを確認します。
- コンシューマーが特定の要素型を期待する場合、`tensor.dtype` を検査します。

## ソースファイル
- C++: `tutorials/011_interpret_model_output/interpret_model_output.cpp`
- Python: `tutorials/011_interpret_model_output/interpret_model_output.py`
