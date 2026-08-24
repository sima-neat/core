# 002 非同期で推論を実行

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Model | resnet_50 |
| Labels | async, push-pull, throughput, runtime |

## Concept

あるスレッドからモデルにデータを供給し、別のスレッドから予測結果を取得することで、入力と出力を分離し、実際の処理能力を向上させます。第001章と同じResNetのパスを使用しますが、今回は非同期処理を行います。

## Walkthrough

第001章では、単一の同期呼び出しでモデルを実行しました。つまり、1つのフレームを渡し、結果が返ってくるまで処理をブロックします。これは単純ですが、計算リソースが無駄になります。入力データを生成するスレッドと、出力データを処理するスレッドは同じスレッドであるため、並行して処理することはできません。この章では、同じResNet-50モデルを使用し、それらの2つのジョブを分割することで、スループットを重視したパイプラインに変換します。

そのメカニズムは、非同期の`Run`です。モデルを`Async`モードで`Graph`に`build()`し、次に、プロデューサーからの2つの独立した呼び出し（`push(...)`）とコンシューマーからの呼び出し（`pull(...)`）で駆動します。最終的には、プロデューサースレッドがランタイムが受け入れる速度でフレームを供給し、メインスレッドが予測結果を抽出するようになり、最後に`pushed=N pulled=N`行が表示され、データが失われていないことが確認されます。

### モデルのロード {#step-load-model}

第001章と同様に、アーカイブから`Model`を構築することから始めますが、ここでは`include_input`と`include_output`を設定した`RouteOptions`も宣言します。これらのフラグは、モデルがグラフに組み込まれたときに、独自の入力と出力の境界を公開するように指示します。これにより、周囲のパイプラインはフレームをプッシュインし、テンソルをプルアウトできます。

### 非同期パイプラインの構築 {#step-build-async}

`Model`は、プッシュ/プルで直接駆動することはできません。`Run`を使用します。モデルを`graph.add(model.graph(route_opt))`を介して新しい`Graph`でラップし、次に代表的なフレームを使用して`build(...)`します。サンプルフレームを渡すことで、`build()`は事前に具体的なテンソルの形状を決定できます。返された`Run`は、両方のスレッドが共有するハンドルです。

### プロデューサーからのフレームのプッシュ {#step-push-frames}

プロデューサーの唯一の仕事は、入力を供給することです。準備されたフレームをループ処理し、各フレームに対して`push(...)`を呼び出し、次に`close_input()`を呼び出して、これ以上フレームが来ないことを通知します。このシグナルは、コンシューマーがいつ停止するかを判断するために使用されます。プロデューサーは独立して実行されるため、次のフレームを送信する前に結果を待つ必要はありません。

**C++:** `std::thread`がループを実行します。アトミックな`pushed`カウンターと`producer_done`フラグが更新され、メインスレッドはロックなしで進捗状況を監視できます。

**Python:** `threading.Thread`（名前は`frame_producer`）がループを実行します。コンシューマーは後で`thread.is_alive()`をチェックして、完了を検出します。

### コンシューマーでの結果のプル {#step-pull-results}

メインスレッドが処理を行います。`pull(timeout_ms=2000)` を呼び出すループがあり、これは次の利用可能な出力を返します。タイムアウト内にデータが到着しない場合は何も返しません。データが空の場合、プロデューサーが処理を終了したかどうかを確認します。終了している場合は停止し、そうでない場合は待機を続けます。各実際の結果は、上位1つのクラスのインデックスに集約され、出力されます。ループの後に、プロデューサーと結合し、`pushed == pulled` が確認されます。

**C++:** `pull()` は `optional<Sample>` を返します。バイトを読み取る前に、`tensors_from_sample(...)` を使用してテンソルを抽出します。

**Python:** `pull()` は `Sample` または `None` を返します。`sample.tensor.to_numpy()` は、`argmax` に渡す配列を提供します。

## Run

実行すると、各フレームに対して1つの `top1=` 行が表示され、その後にプッシュ/プル集計が表示されます。**Neat のインストールルート**（`share/` と `lib/` を含むディレクトリ）から、**Python** および **C++（事前にビルドされたもの）** コマンドを実行します。**ソースからビルドする** コマンドは、**リポジトリのルート**から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/002_run_inference_async/run_inference_async.py \
  --model /tmp/resnet_50.tar.gz --n 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_002_run_inference_async \
  --model /tmp/resnet_50.tar.gz --n 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_002_run_inference_async
./build/tutorials-standalone/tutorial_002_run_inference_async \
  --model /tmp/resnet_50.tar.gz --n 4
```

予想される出力（正確なインデックスは画像によって異なります。C++ ビルドには `pushed=...` フィールドが追加され、Python ビルドは `pulled=...` のみを表示します）。

```text
top1=285
top1=285
top1=285
top1=285
pushed=4 pulled=4
[OK] 002_run_inference_async
```

この章の C++ ソースを、カスタムの `CMakeLists.txt` を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある [チュートリアルの実行方法](/tutorials#compile-a-copy-yourself) を参照してください。

## In Practice

この章では、非同期プッシュ/プルサーフェスを使用します。同じモデルを決定的な合成入力で測定するには、[モデルのベンチマーク](/tutorials/benchmark-your-model) を参照してください。完全なビルドと実行、および同期と非同期のモデル、さらに完全な `RunOptions` サーフェスについては、[最初のグラフの構築](/tutorials/build-inference-pipeline) を参照してください。キューの深さ、オーバーフローポリシー、および負荷下での測定については、[スループットとキューの深さの調整](/tutorials/tune-throughput-and-queues) を参照してください。

## ソースファイル
- C++: `tutorials/002_run_inference_async/run_inference_async.cpp`
- Python: `tutorials/002_run_inference_async/run_inference_async.py`
