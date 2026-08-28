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

`benchmark(samples)`を呼び出します。このAPIは、非同期モデルランナーをウォームアップし、非同期プッシュ/プルウィンドウを測定し、概要を標準出力に出力し、同じ主要な値を`BenchmarkReport`として返します。

サンプル数は、測定する合成入力の数です。より安定したスループットと電力の数値を得るには、より大きな数を使用します。簡単なテストを実行したい場合は、より小さな数を使用します。

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

返されるレポートには、ほとんどのユーザーが必要とする主要なフィールドのみが含まれます。これには、ミリ秒単位の平均エンドツーエンドのレイテンシ、1秒あたりのフレーム数で表されるスループット、利用可能な場合のワット単位の平均ボード電力、および利用可能な場合のジュール単位の測定エネルギーが含まれます。

電力テレメトリは、ボードのサポートに依存します。ランタイムが現在のターゲットで電力レールをサンプリングできない場合、ベンチマークは引き続きレイテンシとスループットを報告し、電力フィールドをゼロのままにします。

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
./build.sh --target tutorial_003_benchmark_your_model
./build/tutorials-standalone/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

予想される出力（正確な数値は、モデル、ボード、および現在の負荷によって異なります。C++ビルドでは、末尾に`[OK]`行も出力されます）。

```text
NEAT Benchmark
Input: synthetic
Samples: 100
Latency:      12.4 ms
FPS:          80.6
Power avg:    2.3 W
Energy:       2.8 J
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
[OK] 003_benchmark_your_model
```

この章のC++ソースを、カスタムの`CMakeLists.txt`を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## In Practice

コンパイルされたモデルアーカイブが動作するかどうか、測定された非同期スループットはどの程度か、そしてこのターゲットにおける主要なボードの消費電力はどの程度か、といった簡単な答えが必要な場合は、このベンチマークを使用してください。

アプリケーションのパフォーマンスについては、実際のパイプラインもベンチマークしてください。合成モデルの入力は意図的に安定しているため、カメラの揺れ、コーデックのコスト、実際のプリプロセス、負荷時のホストのスケジューリング、または下流のアプリケーションロジックを表すものではありません。手動で構築した非同期実行によるキューの深さとバックプレッシャーの調整については、[スループットとキューの深さの調整](/tutorials/tune-throughput-and-queues)を参照してください。

`Model::benchmark()`には、具体的な`input_specs()`の次元が必要です。入力の形状が動的であるか、または具体的なものでない場合、ベンチマークは形状を推測するのではなく、明確に失敗します。

## ソースファイル
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
