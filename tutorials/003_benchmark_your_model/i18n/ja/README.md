# 003 モデルのベンチマーク

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5-10 minutes |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

コンパイルされたモデルに、決定的な合成テンソルを入力として与え、`Model::benchmark()`によって返される、レイテンシ、スループット、電力、エネルギーの主要な数値を表示します。

## Walkthrough

第001章と第002章では、モデルを一度実行する方法、および非同期で実行する方法を示しました。この章では、次の実践的な質問に答えます。「このモデルはデバイス上でどのくらいの速度で実行されるか？」ベンチマークAPIは、意図的に小さく設計されています。モデルをロードし、測定するサンプル数を指定し、`benchmark(...)`を呼び出し、返された`BenchmarkReport`を読み取ります。

ベンチマークでは、モデルの`input_specs()`を使用して、決定的な合成入力を生成します。これにより、モデルの簡単なベンチマークテストや、コンパイルされたモデルのバリアントを比較するのに役立ちますが、カメラベンチマークではありません。カメラのデコード、実際のプリプロセスにおける変動、動的な入力サイズ、またはデータに依存する後処理の動作は含まれません。

### モデルのロード {#step-load-model}

以前のモデルチュートリアルで使用したのと同じコンパイルされた`.tar.gz`アーカイブから開始します。ベンチマークでは、モデルによって宣言された入力仕様から合成テンソルを作成するため、画像は必要ありません。

**C++:** アーカイブパスから`simaai::neat::Model`を構築します。

**Python:** アーカイブパスから`pyneat.Model`を構築します。

### ベンチマークの実行 {#step-run-benchmark}

`benchmark(samples)`を呼び出します。このAPIは、モデルをウォームアップし、`samples`個の入力を1つずつ実行してレイテンシを測定した後、別の非同期プッシュ/プルのパスでスループットを測定し、概要を標準出力に出力し、同じ主要な値を`BenchmarkReport`として返します。

スループット測定パスは、少なくとも`samples`個の入力を測定し、かつ30秒が経過するまで続くため、電力と温度の値が安定します。この下限を変更するには`BenchmarkOptions::min_duration_ms`を設定します。0に設定すると、`samples`個の入力の後で停止する簡単なスモークテストになります。

BoxDecodeで終わる検出モデルは、`BenchmarkOptions`も使用できます。`original_width`、`original_height`、および`resize_mode`を設定して、BoxDecodeがモデル座標から検出をマッピングするときに使用するソース画像のジオメトリを記述します。合成テンソルはモデルの形状のままです。

```cpp
simaai::neat::BenchmarkOptions options;
options.num_samples = 100;
options.original_width = 1920;
options.original_height = 1080;
options.resize_mode = simaai::neat::ResizeMode::Letterbox;
auto report = model.benchmark(options);
```

Pythonは、`pyneat.BenchmarkOptions`を通じて、同じフィールドを公開します。元の両方の次元を設定するか、両方を省略します。省略した場合、ベンチマークは解決されたモデルのルートからジオメトリを推測します。各実行のベンチマークジオメトリは、`ModelOptions`内の非推奨のBoxDecodeジオメトリよりも優先されます。

### レポートを読む {#step-read-report}

返されるレポートは、ほとんどのユーザーが必要とする主要なフィールドから始まります。これには、ミリ秒単位の平均エンドツーエンドのレイテンシ、1秒あたりのフレーム数で表されるスループット、ワット単位のSoMの平均電力とピーク電力、および`samples`個の入力の処理に必要なジュール単位のエネルギーが含まれます。`throughput_duration_ms`は、スループット測定パスの実行時間を示します。

電力は、NeatのインストーラーがセットアップするボードのテレメトリデーモンであるSentinelから取得されます。`report.telemetry`には、スループット測定パスのレール電力、ダイとボードの温度、CPU使用率、MLAメモリも含まれます。各フィールドについては、[モデルの実行を測定する](/develop-apps/development-workflow/model#power-and-board-telemetry)を参照してください。Sentinelが実行されていない場合や別のトレースを記録中の場合でも、ベンチマークはレイテンシとスループットを報告し、電力フィールドをゼロのままにして、`report.telemetry.status`にその理由を設定します。

### 上級: レイヤー統計 {#step-read-layer-stats}

レイヤーのプロファイリングは明示的に有効にする上級オプションです。`BenchmarkOptions.include_layer_stats` の既定値は `false` です。通常のベンチマークは従来の主要な測定を行い、レイヤーのプロファイリングは実行しません。チュートリアルのコマンドに `--include-layer-stats` を追加するか、`BenchmarkOptions` で有効にします。

```cpp
simaai::neat::BenchmarkOptions options;
options.include_layer_stats = true;
auto report = model.benchmark(options);
```

```python
options = pyneat.BenchmarkOptions()
options.include_layer_stats = True
report = model.benchmark(options)
```

`report.layer_stats` を反復処理して、各コンパイル済みレイヤーの `stage_index`、`name`、`total_execution_time_us` を表示します。コンパイラの順序と実行時間がゼロの no-op エントリを保持します。`stage_index` は A65 を含む実行ステージのゼロ始まりの順序です。

ランナーを閉じた後、展開済み MPK のマニフェストから MLA ELF を選択し、`<elf_stem>_stats.yaml` と組み合わせます。YAML を変更せず一時ディレクトリにコピーして `mla-rt --connect mip,ma -vs -N --per-layer-stats <copied_yaml> <elf>` を実行し、生成された `_output.yaml` を検証してレポートに読み込みます。一時ファイルは削除します。MLA ステージごとのタイムアウトは 120 秒で、mla-rt のデフォルト入力を使用します。他の MLA ワークロードが動作していない DevKit と、`PATH` から利用できる `mla-rt` が必要です。既定の `false` では `report.layer_stats` は空になり、主要な値のみを表示します。有効時の出力例：

```text
Layer stats (execution time in microseconds):
  stage_index=0 name=MLA_0/conv total_execution_time_us=161.95
  stage_index=0 name=MLA_0/noop total_execution_time_us=0
  stage_index=2 name=MLA_1/conv total_execution_time_us=42.5
```

MIP からカーネルへの引き継ぎ修正を含む MLA-RT が必要です。eLxr B2121 の MLA-RT `2.2.0~git202610140145.603a8c6-2121`（コミット `603a8c6`）で検証済みです。この修正があれば、`benchmark()` が戻った直後に通常の推論を再開でき、手動リセットや待機は不要です。古いビルドでは、後続のカーネル経由の推論が失敗する状態がハードウェアに残る場合があります。

## Run

実行すると、`benchmark()`によって出力されるベンチマークの概要が表示され、その後に返されたレポートから同じ値が出力されます。**Neatのインストールルート**（`share/`と`lib/`が含まれるディレクトリ）から、**Python**と**C++（事前にビルドされたもの）**コマンドを実行します。**ソースからビルドする**コマンドは、**リポジトリのルート**から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/003_benchmark_your_model/benchmark_your_model.py \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (build from source):**
```bash
./build.sh --all --no-doc
./build/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

予想される出力（正確な数値は、モデル、ボード、および現在の負荷によって異なります。C++ビルドでは、末尾に`[OK]`行も出力されます）。

```text
NEAT Benchmark
Input: synthetic
Samples: 100
Compiled batch: 1
Measurement detail: e2e+throughput-only
Latency: 1.97 ms
FPS:     1457.04 inferences/s
Batch throughput: 1457.04 batches/s
Throughput window: 30.09 s, 43849 inputs
Power:         13.19 W avg, 13.81 W peak (SoM)
MLA power:     3.05 W avg, 3.25 W max
Energy:        0.91 J for 100 inputs
Die temp:      46.34 °C avg, 47.56 °C max
CPU usage:     12.38 % avg, 13.22 % max
MLA memory:    29.65 MiB
CMA used:      6035.02 MiB avg, 6044.38 MiB max
Linux memory:  682.88 MiB avg, 697.41 MiB max
Telemetry:     Sentinel, 15 samples, trace 20261014T144839.984Z-neat-benchmark-20261014t144839-905z
report_latency_ms=1.96852
report_fps=1457.04
report_avg_power_watts=13.1875
report_energy_joules=0.905085
report_peak_power_watts=13.8125
report_throughput_duration_ms=30094.5
report_telemetry_status=ok
[OK] 003_benchmark_your_model
```

この章のC++ソースを、カスタムの`CMakeLists.txt`を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## In Practice

コンパイルされたモデルアーカイブが動作するかどうか、測定された非同期スループットはどの程度か、そしてこのターゲットにおける主要なボードの消費電力はどの程度か、といった簡単な答えが必要な場合は、このベンチマークを使用してください。

アプリケーションのパフォーマンスについては、実際のパイプラインもベンチマークしてください。合成モデルの入力は意図的に安定しているため、カメラの揺れ、コーデックのコスト、実際のプリプロセス、負荷時のホストのスケジューリング、または下流のアプリケーションロジックを表すものではありません。手動で構築した非同期実行によるキューの深さとバックプレッシャーの調整については、[スループットとキューの深さの調整](/tutorials/tune-throughput-and-queues)を参照してください。

`Model::benchmark()`には、具体的な`input_specs()`の次元が必要です。入力の形状が動的であるか、または具体的なものでない場合、ベンチマークは形状を推測するのではなく、明確に失敗します。

## レポートの保存

C++ または Python で `report.save("results/model/yolo_benchmark.yaml")` を呼び出すと、そのファイル名で YAML を保存できます。存在しない親ディレクトリは作成され、既存のファイルはアトミックに置き換えられます。指定したパスが返されます。チュートリアルでは `--report` で同じパスを指定できます。保存は明示的に行います。`benchmark()` はレポートファイルを自動作成しません。

保存されるレポートには、レイテンシのパーセンタイル、スループット測定期間のメタデータ、Sentinel のテレメトリも含まれます。

YAML には `latency_ms`、`fps`、`avg_power_watts`、`energy_joules`、`layer_stats` が含まれます。各 `LayerStats` レコードには `name`、`stage_index`、`total_execution_time_us`（マイクロ秒単位のハードウェア `run_time`）があります。コンパイラが生成する MLA 名は、元の ONNX 演算子の結合や分割を表す場合があります。実行時間がゼロの no-op エントリも保持されます。レイヤーのプロファイリングと主要な測定は別の実行と入力を使用するため、時間を同一視できません。 `stage_index` は MLA と A65 の実行ステージを含む、モデル実行順のゼロ始まりの整数です。アーカイブ内のファイル順ではありません。単一ステージでは `0`、MLA–A65–MLA では MLA のレコードに `0` と `2` を使用します。

ファイルの欠落、プロセス失敗、タイムアウト、不正な統計結果は例外を発生させます。元の入力 YAML は保持され、成功時も失敗時も一時ファイルを削除します。

## ソースファイル
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
