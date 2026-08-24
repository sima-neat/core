# 026 複数のモデルを実行

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | resnet_50, yolo_v8s |
| Labels | PCIe, queues, concurrency, classification, detection |

## Concept

Modalix PCIeカードは、キュー0から3までを公開します。アクティブな`Model`はそれぞれ1つのキューを所有するため、異なる`ConnectionOptions.queue`を各インスタンスに割り当てることで、独立したモデルを同時に実行できます。このチュートリアルでは、グローバルコーディネーターを追加せずに、ResNet-50をキュー0に、YOLOv8sをキュー1に割り当てます。

## Walkthrough

2つのモデルは、意図的に異なる画像を使用します。ResNet-50は、鮮明なラブラドール犬の写真の分類を行い、YOLOv8sは、賑やかな街のシーンで人や車を検出します。

### モデル固有の画像をロードする {#step-load-assets}

両方のモデルアーカイブを検証し、パッケージ化されたアセットをデコードしてから、キューを占有します。画像を分離することで、各結果の意味が明確になり、分類ポートレートをオブジェクト検出のワークロードとして使用することを回避できます。

### 各キューに1つのモデルを割り当てる {#step-assign-queues}

2つの通常の`Model`オブジェクトを作成します。ResNet-50を、ImageNet画像の前処理とともにキュー0に、YOLOv8sを、COCO画像の前処理とボックスデコードとともにキュー1に設定します。ビルドエラーが発生した場合、エラーが発生したキューとモデルが特定されます。すでにビルドされたモデルは、2回目のビルドが失敗した場合に閉じられます。

### 両方のキューを同時に実行する {#step-run-concurrently}

各モデルに対して、ホストの別々のスレッドで1つのブロッキング画像推論を開始します。各呼び出しは、単純な同期`run`動作を使用しますが、呼び出しは異なる物理キューをターゲットとするため、オーバーラップします。

### 各結果を個別に解釈する {#step-read-results}

キュー0は、1つのFP32分類テンソルを返し、最も高いスコアのImageNetクラスを出力します。キュー1は、デコードされたBBOXレコードを返し、検出クラス、信頼度、およびソース画像の座標を出力します。いずれかのモデルを閉じると、割り当てられたキューのみが解放されます。

## Run

PCIeホストパッケージをインストールし、[チュートリアルの設定](/tutorials/before-you-run)で説明されているように、チュートリアルバンドルをダウンロードします。抽出されたPCIeエクストラルのルートから、両方のモデルをダウンロードします。

```bash
sima-cli modelzoo get resnet_50
sima-cli modelzoo get yolo_v8s
```

このプログラムでは、このディレクトリ内の正確なパス`resnet_50_mpk.tar.gz`と`yolo_v8s_mpk.tar.gz`が必要です。Model Zooが他の名前または場所を使用した場合は、ダウンロードしたアーカイブを適切な場所にコピーし、それらを検証します。

```bash
cp /absolute/path/to/downloaded-resnet-archive.tar.gz resnet_50_mpk.tar.gz
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f resnet_50_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

Pythonを実行します。

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/026_run_multiple_models/run_multiple_models.py
```

事前にビルドされたC++チュートリアルを実行します。

```bash
./lib/sima-pcie-host/tutorials/tutorial_026_run_multiple_models
```

または、再ビルドします。

```bash
./build.sh --target tutorial_026_run_multiple_models
./build/tutorials-standalone/tutorial_026_run_multiple_models
```

ドキュメント化されたモデルとアセットを使用すると、両方のバージョンで次のような出力が表示されます。

```text
queue=0 model=resnet_50 output_shape=[1, 1000] top1=208 (Labrador retriever)
queue=1 model=yolo_v8s detections=...
  person score=... box=(...)
[OK] 026_run_multiple_models
```

このチュートリアルでは、ResNet-50をキュー0に、YOLOv8sをキュー1に固定します。別のカードを使用する場合は、`--card N`を渡します。

## In Practice

キューの割り当ては、アプリケーションのリソースに関する決定です。2つのアクティブなモデルは、同じ物理キューを所有することはできません。モデルは、作業を開始する前にビルドし、失敗時に特定のキューを報告し、正常にビルドされたすべてのモデルを、正常な場合とエラーの場合の両方で閉じます。個別の`Model`インスタンスは、結果とエラーを分離したまま、理解しやすい状態を維持します。

デプロイメントの診断については、[PCIeモデルのワークフロー](/develop-apps/development-workflow/pcie-model/)と[トラブルシューティングガイド](/reference/troubleshooting/)を参照してください。

## ソースファイル

- `run_multiple_models.cpp`
- `run_multiple_models.py`
- `../assets/labrador.jpg`
- `../assets/street-scene.png`
