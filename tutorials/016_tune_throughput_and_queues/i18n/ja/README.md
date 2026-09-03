# 016 スループットとキューの深さを調整する

## Metadata
| Field | Value |
| --- | --- |
| Category | Graphs & Pipelines |
| Difficulty | Advanced |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | performance, tuning, async, queues |

## Concept

負荷時の動作を制御する非同期パイプラインの調整可能なパラメータ（キューの深さとオーバーフローポリシー）を調整し、実際に何が起こったかを測定します。

## Walkthrough

パフォーマンスの調整は、正確性の基準が安定した後にのみ効果があります。この章では、それが前提として扱われ、非同期パイプラインが、処理能力を超える速度でデータが到着した場合にどのように動作するかを決定する設定に焦点を当てます。キューの深さを設定し、キューがいっぱいになった場合の動作を選択し、決定的な量のフレームをノンブロッキングでプッシュし、結果を処理し、データが破棄されたかどうか、および各フレームの処理にかかった時間を知らせる測定レポートを読み取ります。

最終的には、バックプレッシャー下での非同期実行を測定するための動作するフレームワークが完成します。これには、エンキュー数、ドロップ数、取り出された出力、平均レイテンシー、およびプッシュコストが含まれます。同じループが、[In Practice](#in-practice)に記載されているヒューリスティックに基づいて、実際のパイプラインを調整するための基礎となります。

### 実行オプションを設定する {#step-configure-run-options}

`RunOptions`は、負荷下での非同期動作を決定する場所です。`queue_depth`（ランタイムが受け入れる同時処理可能なサンプルの数）、`overflow_policy`（キューがいっぱいになった場合の動作：`Block`、`KeepLatest`、または`DropIncoming`）、`output_memory = Owned`（返されるテンソルは自身のデータを所有するため、取り出し後もデータは保持されます）を設定します。次に、`build()`を使用して、`Async`モードでグラフを構築します。これにより、独立したプロデューサーとコンシューマーを持つ実行が可能になります。

**C++:** オーバーフローポリシーは、`--drop`から`simaai::neat::OverflowPolicy::{Block,KeepLatest,DropIncoming}`に解析されます。`graph.build(input, opt)`は、実行ハンドルを返します。

**Python:** ポリシーは、`getattr(pyneat.OverflowPolicy, ...)`を使用して解決されます。`graph.build([tensor], opt)`は、実行ハンドルを返します。

### ワークロードをプッシュして処理する {#step-push-workload}

ここでは、キューポリシーが実際に適用されます。`try_push(...)`をタイトなループで呼び出します。これはノンブロッキングプッシュであり、サンプルが受け入れられたかどうかを単純に返します。したがって、`DropIncoming`/`KeepLatest`の下でキューがいっぱいになると、処理が停止するのではなく、拒否されたプッシュとして表示されます。バーストの後、`close_input()`を呼び出して、これ以上の入力がないことを通知し、次に、`pull(...)`ループを使用してコンシューマー側を処理し、空になるまで処理します。`try_push`と`close_input`を組み合わせ、さらにドレインループを追加することは、標準的なノンブロッキング非同期パターンです。

### 測定レポートを読み取る {#step-read-measurement}

実行が終了したら、測定範囲を停止します。レポートの「`counters`」グループは、ランタイム側の数値（キューに追加された入力、破棄された入力、出力されたデータ）を示し、「`input`」は、平均プッシュコストや入力の再ネゴシエーションなど、プッシュ側の数値を示します。これらを組み合わせることで、キューの深さとオーバーフローポリシーが意図したとおりに機能したかどうか（フレームがドロップされたか、遅延が増加したか、プッシュパスが効率的だったか）を判断できます。

## Run

この章では、モデルアーカイブは必要ありません。**Neatのインストールルート**（`share/`と`lib/`が含まれるディレクトリ）から、**Python**と**C++（事前にビルドされたもの）**のコマンドを実行します。**ソースコードからビルドする**コマンドは、**リポジトリのルート**から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.py \
  --iters 32 --queue 4 --drop block
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_016_tune_throughput_and_queues \
  --iters 32 --queue 4 --drop block
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_016_tune_throughput_and_queues
./build/tutorials-standalone/tutorial_016_tune_throughput_and_queues \
  --iters 32 --queue 4 --drop block
```

予想される出力（正確なカウントとタイミングは、ホストとポリシーによって異なります）：

```text
inputs_enqueued=32
inputs_dropped=0
outputs_pulled=32
avg_latency_ms=0.42
avg_push_us=18.0
renegotiations=0
[OK] 016_tune_throughput_and_queues
```

（Pythonでビルドした場合、末尾の「`[OK]`」行は表示されません。）

この章のC++ソースコードをカスタムの `CMakeLists.txt` を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページの[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## In Practice

キューのサイズ調整、ドロップポリシー、プリセット、および出力のライフサイクルに関する実践的なガイダンス。

### キューのサイズ調整 (`queue_depth`)

ヒューリスティック：
- 低レイテンシーのパイプラインの場合は、`queue_depth = 4–16` から開始します。
- プロデューサーがバースト的にデータを生成する場合、または下流の要素のレイテンシーが変動する場合（デコード/MLA/後処理）は、キューを増やします。
- **最新の**フレームが必要な場合は（例：ライブカメラプレビュー）、キューを小さく保ちます。

### オーバーフローポリシー (`RunOptions::overflow_policy`)

- `Block`：正確性を最優先とするため、キューがいっぱいになるとプロデューサーは待機します。
- `DropIncoming`：キューに格納されたデータを保持し、飽和状態になると受信サンプルを破棄します。
- `KeepLatest`：最新のフレームを優先し、最も古いキューに格納されたサンプルを破棄します。

ライブフィードの場合、`KeepLatest` を使用すると、通常、エンドツーエンドのレイテンシーが最も低くなります。

### プリセットと再ネゴシエーション

`RunOptions::preset` を使用して、レイテンシーと安全性のトレードオフを制御します。
- `Realtime`：最低レイテンシー、積極的な最新性維持動作。
- `Balanced`：可能な場合はゼロコピーから開始し、起動時のプローブチェックを実行し、信頼性が損なわれた場合はコピーモードにフォールバックします。
- `Reliable`：保守的な動作と安定した出力所有権。

動的な入力の場合、入力形状の再ネゴシエーションは自動的に行われます（上記の `renegotiations` カウンターは、その発生回数を報告します）。

### 出力のライフサイクル (`output_memory`)

- `output_memory = Owned`：返された `Tensor` は、自身のデータを所有します。
- `output_memory = ZeroCopy`：テンソルは、プル後に再利用されるランタイムバッファーを参照する場合があります。
- `output_memory = Auto`：ランタイムは、まずゼロコピーを選択し、信頼性が求められる場合は所有権のあるモードにフォールバックします。

テンソルのデータを現在のステップを超えて保持する必要がある場合は、`clone()` または `cpu().contiguous()` を呼び出します。

### バッファープールの安全性

- `RunAdvancedOptions::max_input_bytes` は、入力バッファーの割り当てに対する上限を設定します。
- より大きなバッファーが必要な場合、ランタイムは明示的なエラーで迅速に失敗します。

これらを使用して、入力のサイズが変更された場合に、長期間実行されるプロセスが無限の割り当てから保護されるようにします。

## ソースファイル
- C++：`tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.cpp`
- Python：`tutorials/016_tune_throughput_and_queues/tune_throughput_and_queues.py`
