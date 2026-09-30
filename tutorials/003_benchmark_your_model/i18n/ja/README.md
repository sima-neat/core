# 003 モデルをベンチマークする

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 分 |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

`model.benchmark()` で、コンパイル済みモデルのレイテンシー、スループット、電力、エネルギーを測定します。ベンチマークが合成入力を生成するため、画像やデータセットは不要です。

## Walkthrough

### モデルを読み込む {#step-load-model}

C++ では `simaai::neat::Model`、Python では `pyneat.Model` を使って、コンパイル済みの `.tar.gz` アーカイブを読み込みます。モデルの `input_specs()` には、具体的な入力寸法が宣言されている必要があります。

### ベンチマークを実行する {#step-run-benchmark}

デフォルト設定では `model.benchmark()` を呼び出します。サンプル数を指定する場合は `model.benchmark(100)` を使います。API はモデルをウォームアップし、逐次実行のレイテンシーと非同期実行のスループットを測定して、概要を表示し、`BenchmarkReport` を返します。

サンプル数を増やすと、スループットと電力の測定値がより安定する場合があります。

### レポートを確認する {#step-read-report}

| フィールド | 意味 | 単位 |
| --- | --- | --- |
| `latency_ms` | ウォームアップ後の平均レイテンシー | ms |
| `fps` | 論理推論のスループット | inferences/s |
| `avg_power_watts` | スループット測定中の平均ボード電力 | W |
| `energy_joules` | スループット測定中の総消費エネルギー | J |

ボードの電力テレメトリを利用できない場合、電力とエネルギーの値はゼロのままです。

Python コードでは、レポートを保存することもできます。

```python
model = pyneat.Model("model.tar.gz")
report = model.benchmark()
report.save_json("benchmark.json")
```

JSON テキストを取得するには `report.to_json()`、レポートを再表示するには `print(report)` を使います。JSON では、利用できない測定値を有効なゼロとして扱わず、利用不可であることを明示します。

## Run

Python とビルド済み C++ のサンプルは、`share/` と `lib/` がある Neat のインストールルートから実行します。ソースからのビルドコマンドは、リポジトリのルートから実行してください。どちらのサンプルも、ベンチマークの概要に続いて、返されたレポートのフィールドを表示します。

**Python:**
```bash
python3 share/sima-neat/tutorials/003_benchmark_your_model/benchmark_your_model.py \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++（ビルド済み）:**
```bash
./lib/sima-neat/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++（ソースからビルド）:**
```bash
./build.sh --target tutorial_003_benchmark_your_model
./build/tutorials-standalone/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

以下はレポートのフィールドの例です。数値は説明用です。

```text
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
```

独自の C++ プロジェクトについては、[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## In Practice

同じモデル設定とデバイスで結果を比較してください。保存したレポートとともに、モデルアーカイブとサンプル数も記録します。

合成入力は、カメラのタイミングやデータに依存する処理を再現しません。実際のアプリケーションのパイプラインは別途測定してください。非同期キューの調整については、[スループットとキュー深度の調整](/tutorials/tune-throughput-and-queues)を参照してください。

BoxDecode の元画像の寸法やその他のベンチマークオプションについては、[モデルの実行を測定する](/develop-apps/development-workflow/model#measure-model-execution)を参照してください。

## ソースファイル
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
