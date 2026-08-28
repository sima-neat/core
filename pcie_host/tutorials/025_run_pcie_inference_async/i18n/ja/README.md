# 025 PCIe非同期推論の実行

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, asynchronous, throughput, detection |

## Concept

同期的な`run()`は、最初の推論に最適ですが、次の画像を送信する前に、各結果の完了を待ちます。プロデューサーが`push()`を呼び出し、コンシューマーが`pull()`を実行している場合、1つのPCIeモデルが常に動作状態になります。スループットは、ウォームアップ後の完了した結果から計算する必要があります。モデルに送信された画像だけでは不十分です。

## Walkthrough

このチュートリアルでは、チュートリアル024で使用されたYOLOv8sの画像とボックスデコードの構成、および640x480のストリートシーンを再利用します。ストレージと画像デコードがPCIeの測定に影響を与えないように、同じ画像を繰り返し送信します。

### 1つの検出モデルを構成する{#step-configure-model}

画像を1回ロードし、カード側のCOCO前処理とYOLOv8ボックスデコードを構成し、キュー0に1つの`Model`を構築します。ファイルが見つからない場合や、カードの起動に失敗した場合、測定が開始される前にプログラムが停止します。

### パイプラインをウォームアップする{#step-warm-up}

タイミングを計測せずに、いくつかの完全な検出を実行します。ウォームアップは、モデルの起動と最初のバッファーの効果を、報告されたワークロードから取り除きます。

### 同時に送信および取得する{#step-push-pull}

1つのスレッドが`push()`を使用して画像を送信し、別のスレッドが有限のタイムアウトを持つ`pull()`を使用してBBOX出力を取得します。小さなアプリケーション専用のFIFOは、各順序付けされた送信の開始時刻を保存します。拒否、タイムアウト、または不正な結果が発生した場合、モデルは閉じられ、別のスレッドが起動します。

この例では、通常の`Model`のフロー制御動作のみに依存しており、アプリケーションにはキューの深さの調整はありません。

### 完了した作業を報告する{#step-report-results}

両方のスレッドが完了し、すべての受け入れられた画像が取得された後にのみ、タイミングの計測を停止します。1秒あたりのフレーム数は、完了した出力の数を使用します。平均レイテンシーは、各送信試行から、対応する順序付けされた結果が到着するまでの時間を測定します。

## Run

PCIeホストパッケージをインストールし、[チュートリアルの設定](/tutorials/before-you-run)で説明されているように、チュートリアルのバンドルをダウンロードします。抽出されたPCIeエクストラルのルートから、YOLOv8sがまだ存在しない場合は、ダウンロードします。

```bash
sima-cli modelzoo get yolo_v8s
```

プログラムには、このディレクトリ内の正確なパス`yolo_v8s_mpk.tar.gz`が必要です。Model Zooが別の名前または場所を使用した場合は、ダウンロードしたアーカイブを適切な場所にコピーします。

```bash
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

Pythonを実行します。

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/025_run_pcie_inference_async/run_pcie_inference_async.py
```

事前に構築されたC++チュートリアルを実行します。

```bash
./lib/sima-pcie-host/tutorials/tutorial_025_run_pcie_inference_async
```

または、再構築します。

```bash
./build.sh --target tutorial_025_run_pcie_inference_async
./build/tutorials-standalone/tutorial_025_run_pcie_inference_async
```

正確なタイミングは、ホストとカードによって異なりますが、両方のプログラムは同じ測定境界を使用し、以下を出力します。

```text
completed=1000
elapsed_seconds=...
throughput_fps=...
average_latency_ms=...
total_detections=...
[OK] 025_run_pcie_inference_async
```

チュートリアルでは、常に5つのフレームでウォームアップし、その後、1,000の完了したフレームを測定します。別のカードを使用する場合は、`--card N`を渡します。

## In Practice

送信と取得のバランスを保ちます。アプリケーションが、取得せずに画像を無期限に送信し続けると、通常のバックプレッシャーによって送信が最終的に遅くなります。専用のコンシューマーを使用すると、障害も簡単に処理できます。有限のタイムアウトにより、停止した結果が特定され、モデルを閉じることで、プロデューサーが待機している場合でも、キュー0が解放されます。

代表的なベンチマークを作成するには、繰り返しのフレームを固定された画像セットに置き換え、ディスクからの読み取りをタイミング計測の対象外にします。[複数のモデルを同時に実行](/tutorials/run-multiple-models)に進み、2つの異なるモデルを同時に実行します。

## ソースファイル

- `run_pcie_inference_async.cpp`
- `run_pcie_inference_async.py`
- `../assets/street-scene.png`
