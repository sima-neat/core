# 015 1つのグラフで複数のストリームを実行する

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | None |
| Labels | graph, multistream, scheduler, join |

## Concept

1つのパブリックな`Graph`を通じて複数の論理ストリームを実行し、2つの名前付き入力を1つの決定的なバンドル出力に結合します。これは、関連する入力が後続の処理の前にアラインメントされる必要がある、マルチカメラまたはマルチソースシステムで使用されるパターンです。

## Walkthrough

前の章では、1つの入力をプッシュし、1つの出力をプルしました。実際のマルチカメラおよび並列ブランチシステムはより複雑です。複数のストリームが独立して進行し、それらの結果は、下流のシステムがそれを使用する前に、正しく*結合*される必要があります。この章では、それを決定的に行うための結合プリミティブ、つまり、2つの名前付き入力と1つの名前付き出力を持つ結合グラフを示します。このグラフは、両方の側が一致するフレームを生成した場合にのみ、バンドルを出力します。

プッシュするすべてのサンプルには、`stream_id`と`frame_id`が含まれます。結合ポリシー`ByFrame`は、名前付きの入力（`left`と`right`）の両方が、同じ`frame_id`を持つサンプルを配信するまで待ち、その後、正確に1つの結合されたバンドルを出力します。最終的には、結合グラフを構築し、2つの入力を通じて決定的なストリーム/フレームごとのワークロードを分散させ、結合されたバンドルをプルして、出力数と各バンドルが2つのフィールドを持つことを検証します。

### 結合グラフの構築 {#step-build-combine-graph}

`graphs::Combine`（C++）/ `graphs.combine`（Python）は、通常のパブリック`Graph`フラグメントを返します。その形状（2つの名前付き入力、1つの名前付き出力、および結合ポリシー）以外には、特別な点は何もありません。入力名として`["left", "right"]`、出力名として`"combined"`を渡し、フレームIDの一致を選択するために`CombinePolicy.ByFrame`を渡します。`describe()`を出力すると、結果のトポロジーが表示され、`build()`は、その説明を実行可能なハンドルに変換します。グラフはデフォルトで非同期に実行されるため、各ストリームは独立して進捗できます。

出力キューは制限されています。ワークロード全体に必要なキュー領域を割り当てる代わりに、この例では、次のペアをプッシュする前に、各結合されたバンドルをプルします。プロデューサーとコンシューマーは一緒に進むため、フレーム数が増加しても、メモリ使用量は制限されたままになります。

`CombinePolicy.ByFrame`は、`Sample.frame_id`に基づいて一致します。`CombinePolicy.ByPts`は、フレームが明確なフレームインデックスを共有しない場合に、プレゼンテーションタイムスタンプ（`Sample.pts_ns`）に基づいて一致させる代替手段です。

### ストリームのプッシュ {#step-push-streams}

次に、ワークロードを実行します。各フレームと各ストリームについて、その`stream_id`と一意の`frame_id`がタグ付けされた小さな決定的なRGBサンプルを合成し、それを*両方*の名前付き入力にプッシュします。IDは決定的に計算されるため（`frame * streams + sid`）、結合は、一意のペアを見つけることができます。`left`フレームNには、常に一致する`right`フレームNがあります。一致する`right`をプッシュした後、そのペアの結合された出力をドレインしてから、次のペアに進みます。

**C++:** 各サンプルは、`frame_id`と`stream_id`が設定された`Tensor`（HWC、UInt8、RGB）をラップした`Sample`として明示的に構築されます。`run.push("left", sample)`は、`run.last_error()`に対して確認すべきboolを返します。

**Python:** `make_rgb_sample(...)`は、`Tensor.from_numpy(...)`を介してNumPy配列から`Sample`を構築します。`run.push("left", [sample])`は、サンプルのリストを受け取ります。

### 各結合されたバンドルをプルする {#step-pull-bundles}

各一致するペアがプッシュされる直後に、名前付きの出力`"combined"`から一度プルします。各成功したプルは、ランタイムが両方の入力がそのフレームを送信した後に送信したバンドルを返します。生成と同時に処理することで、バッファリングされた出力キューがいっぱいになり、入力側にバックプレッシャーが伝播するのを防ぎます。両方の例では、すべてのバンドルに2つの結合されたフィールドが含まれていることを確認し、その後`close()`を呼び出して、実行をクリーンに終了します。予想されるバンドルの数は`streams * frames`と等しく、ペアリングが削除されなかったことを証明します。

**C++:** `run.pull("combined", timeout_ms)`は、オプションのバンドルを返します。`bundle.stream_id`と`bundle.fields.size()`を読み取り、各バンドルに2つのフィールドがあることを確認します。

**Python:** `run.pull("combined", 2000)`は、バンドルまたは`None`を返します。この例では、タイムアウトが発生するとすぐに失敗し、各バンドルのフィールド数を検証します。

## Run

この章では、モデルアーカイブは必要ありません。**Neatのインストールルート**（`share/`と`lib/`が含まれるディレクトリ）から、**Python**および**C++（事前にビルドされたもの）**コマンドを実行します。**ソースからビルドする**コマンドは、**リポジトリのルート**から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/015_run_multiple_streams/run_multiple_streams.py \
  --streams 8 --frames 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_015_run_multiple_streams \
  --streams 8 --frames 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_015_run_multiple_streams
./build/tutorials-standalone/tutorial_015_run_multiple_streams \
  --streams 8 --frames 4
```

予想される出力（C++ビルドではグラフの説明も出力されます。両方のビルドでは、最初のいくつかのバンドルが出力されます）。

```text
received=32 fields=2
[OK] 015_run_multiple_streams
```

カスタムの`CMakeLists.txt`（追加のフォルダーは不要）を使用して、この章のC++ソースを独自のプロジェクトに統合する方法については、ランディングページにある[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## ソースファイル
- C++: `tutorials/015_run_multiple_streams/run_multiple_streams.cpp`
- Python: `tutorials/015_run_multiple_streams/run_multiple_streams.py`
