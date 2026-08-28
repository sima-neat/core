# 004 最初のグラフを作成する

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | None |
| Labels | graph, build, run, pipeline |

## Concept

手動で`Graph`を構成します。入力ノード、出力ノードを設定し、モデルは使用せずに、1フレームだけパイプラインを実行します。モデルを追加する前に、パイプラインの基本的な要素を個別に確認します。

## Walkthrough

第001章では、3行のコードでモデルを実行します。この簡潔さの裏には、すべての実用的な Neat プログラムが直接使用する2段階のライフサイクルが隠されています。まず、`Graph` としてパイプラインを**記述**し、次にその記述を、実行可能な `Run` に**構築**します。この章では、可能な限り小さなパイプライン（1つの入力ノードが1つの出力ノードに接続され、間にモデルがない）を構成し、単一のフレームをそのパイプラインに通すことで、このライフサイクルを可視化します。

その結果、概念的に理解しやすくなります。`Graph` は、一度構築して何度も実行できる*再利用可能な定義*であり、一度限りの呼び出しではありません。この章の終わりには、グラフを作成し、それを実行可能なパイプラインに変換し、出力テンソルのランクを読み出して、フレームが正常に処理されたことを証明します。

### 入力 {#step-configure-input} を記述する

ノードを接続する前に、フレームがどのようなものか宣言します。`InputOptions` は、そのための契約です。ピクセルの `format`、`width` / `height`、チャンネルの `depth`、およびランタイムが各バッファにタイムスタンプを付与するかどうかを指定します。これらのオプションから構築された入力ノードは、受信フレームがパイプラインが期待する形状と一致するかどうかを検証します。

**C++:** C++ では、さらに `is_live = false` を設定して、これを非リアルタイム（ファイル/テンソル）ソースとしてマークします。

### グラフを構成する {#step-compose-graph}

次に、構造を構築します。新しい `Graph` は、空の構成サーフェスであり、`add()` を使用してノードを順番に追加します。正確に2つのノード（上記で構成された入力ノードと、単なる出力ノード）を追加します。これが全体のトポロジーです。フレームは入力から入り、出力から出て、その間に何もありません。これは、後続の章でモデルまたは前処理段階が挿入される場所です。

**C++:** ノードは、`simaai::neat::nodes::Input(...)` と `nodes::Output()` から取得されます。

**Python:** ノードは、`pyneat.nodes.input(...)` と `pyneat.nodes.output()` から取得されます。

### パイプラインを構築する {#step-build-pipeline}

`build()` は、*記述* から *実行可能* への移行です。追加されたノードを具体的なパイプラインに解決し、実際のサンプルに対して入力/出力の契約を検証し、再利用可能な `Run` ハンドルを作成します。代表的なフレームを渡すことで、`build()` がネゴシエートされたテンソルの形状を固定できるようにします。次のステップでは、`Run::run(...)` を使用して、決定的な1つずつ呼び出しを行います。

**C++:** サンプルフレームは `cv::Mat` であり、`run_opt.output_memory = Owned` は、ランタイムに所有権のある出力バッファを返すように要求します。

**Python:** まず、NumPy配列から `Tensor` を `Tensor.from_numpy(...)` を使用して作成し、それを使用して構築します。

### フレームを実行し、結果を読み取る {#step-run-frame}

`Run` を実行すると、`run()` は 1 つのフレームをプッシュし、1 つの結果を同期的に取得します。モデルがないため、出力は入力の契約を反映します。したがって、テンソルの *ランク* を読み取るだけで、フレームがラウンドトリップを完了したことを確認できます。実際のパイプラインでは、この同じ `run()`/push/pull のインターフェースを使用して推論を実行します。

**C++:** `run()` は `TensorList` を返します。`sample.front().shape.size()` を読み取ります。

**Python:** テンソル入力を伴う `run()` は `TensorList` を返します。`len(outputs[0].shape)` を読み取ります。

## Run

実行すると、出力テンソルのランクが標準出力に出力されるはずです。**Neat のインストールルート**（`share/` と `lib/` を含むディレクトリ）から、**Python** および **C++（事前にビルドされたもの）** コマンドを実行します。**ソースコードからビルドする** コマンドは、**リポジトリのルート**から実行します。この章では、モデルアーカイブは必要ありません。

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

期待される出力：

```text
tensor_rank=3
[OK] 004_build_inference_pipeline
```

（Python ビルドは `output_rank=...` を出力します。）この章の C++ ソースコードを、カスタムの `CMakeLists.txt` を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある [チュートリアルの実行方法](/tutorials#compile-a-copy-yourself) を参照してください。

## In Practice

`build`と`run`、実行モード、プッシュ/プルインターフェース、および`RunOptions`が、単一の同期呼び出しを超えてどのように連携するか。

### ビルドと実行

- `Graph::build(...)`は、パイプラインを構築し、プッシュ/プル制御用の`Run`ハンドルを返します。
- `Graph::run(...)`は、同期処理を行うための便利な方法です。必要に応じてグラフを構築し、1つの入力をプッシュし、1つの出力をプルします。

### 同期と非同期

- 単純なワンショット呼び出しには、`Graph::run(...)`を使用します。
- 再利用可能なランナーと明示的な`push(...)` / `pull(...)`制御が必要な場合は、`Graph::build(...)`を使用します。詳細については、[非同期で推論を実行する](/tutorials/run-inference-async)を参照してください。

### プッシュ/プルAPI

`Run`は、以下を公開します。
- 入力用：`push(...)` / `try_push(...)`（`cv::Mat`、`Tensor`、または`Sample`）。
- 出力用：`pull(...)`、`pull_tensor(...)`、`pull_tensor_or_throw(...)`。

出力メタデータ（タイムスタンプ、ストリームID）が必要な場合は、`pull()`を使用して`Sample`を取得します。テンソルペイロードのみが必要な場合は、`pull_tensor()`を使用します。

### RunOptions（シンプルなAPI）

一般的な設定項目：
- `preset`：レイテンシ/安全性のプロファイル（`Realtime`、`Balanced`、`Reliable`）。
- `queue_depth`：ランタイムキューの深さ。
- `overflow_policy`：キューのオーバーフロー動作（`Block`、`KeepLatest`、`DropIncoming`）。
- `output_memory`：出力の所有権ポリシー（`Auto`、`ZeroCopy`、`Owned`）。
- `on_input_drop`：ドロップされた入力イベントに対するコールバックフック。

キューの深さ、オーバーフロー、および負荷下での測定については、[スループットとキューの深さを調整する](/tutorials/tune-throughput-and-queues)を参照してください。

### RunAdvancedOptions（高度なAPI）

高度な設定項目は、`RunOptions::advanced`の下でオプションとして有効になります。
- `advanced.max_input_bytes`：入力バッファーの成長を制限します。
- `advanced.copy_input`：防御的な入力コピーを強制します。

`Run::start_measurement()`を使用して、レイテンシ、スループット、入力カウンター、プラグイン/エッジのタイミング、およびオプションのボードPMIC電力テレメトリを、1つの測定ウィンドウで検査します。

ボードの電力を含めるには、コードで有効にします（環境変数は不要）次に、測定レポートから読み取ります。

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

`Model::build(run_opt)`、`Model::build(route_opt, run_opt)`、および`Graph::build(run_opt)`は、同じランタイムオプションを基盤となる`Run`に渡すため、パイプラインごとに個別にレールサンプリングを行うのではなく、グラフレベルのボード電源モニターを1つ使用します。特定の組み込みプロファイルを強制する必要がある場合は、ボード固有のヘルパーが引き続き利用可能です：`enable_modalix_som_power()`、`enable_modalix_dvt_power()`。

## ソースファイル
- C++: `tutorials/004_build_inference_pipeline/build_inference_pipeline.cpp`
- Python: `tutorials/004_build_inference_pipeline/build_inference_pipeline.py`
