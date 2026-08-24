# 023 MIPIカメラモデルを実行する

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | User-provided camera-compatible model |
| Labels | mipi, camera, live-input, model, ev74 |

## Concept

Modalix DevKit MIPIカメラを、`CameraInput`を備えた`Graph`に接続し、リアルタイムの`NV12`フレームをモデル管理のプリプロセスに送り込み、モデルの出力を取得します。これは、デプロイされたビジョンアプリケーション向けの直接的なカメラパスです。カメラがソースを所有し、CVU/EV74が画像の前処理を行い、MLAが推論を実行し、アプリケーションが結果を消費します。

## Walkthrough

この章では、カメラがすでにボードのオーバーレイと libcamera を通じて動作していることを前提としています。Neat は、`.dtbo` ファイルを選択したり、ISP を調整したりしません。代わりに、`libcamerasrc` がフレームを生成できるようになった時点で、それらのフレームを使用します。チュートリアルを実行する前に、[ハードウェア MIPI ガイド](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) と GStreamer の caps チェックを使用して、カメラを検証してください。

このチュートリアルをゲート 2 と考えてください。ゲート 1 は、カメラの初期設定です。具体的には、オーバーレイ、ドライバー、libcamera、ISP、および正確な caps の設定を行います。ゲート 2 は、Neat グラフです。カメラフレームを CVU プリプロセス、MLA 推論、オプションの EV74 BoxDecode に渡し、最後に結果を出力します。

### カメラソースを構成する {#step-configure-camera}

`CameraInputOptions` は、Neat が `libcamerasrc` から要求するソース caps を記述します。これには、解像度、フレームレート、フォーマット、およびオプションの libcamera カメラ名が含まれます。SiMaAI ゼロコピーバッファーがまだ公開されていない現在のカメラスタックでは、`allow_cpu_fallback = true` を設定します。`libcamerasrc` がサポートしている場合は、`--strict-zero-copy` を使用して、厳密なゼロコピーを有効にすることもできます。

### モデルのルートを構成する {#step-configure-model}

モデルは、カメラフレームを `NV12` 形式の画像として認識します。カラー変換、リサイズ、正規化、量子化、およびテッセレーションのために、モデル管理のプリプロセスを設定します。例では、モデル管理の CVU プリプロセスを `EV74` に固定しています。これにより、本番環境のグラフが静かに CPU イメージパイプラインに変わるのを防ぎます。`--decode none` を使用すると、ルートは MLA で終了し、生のモデルテンソルが返されます。YOLO `--decode` トークンを使用すると、BoxDecode がモデル管理の EV74 ポストプロセスステージとして実行されます。

### ソースが所有するグラフを構成する {#step-compose-graph}

最初に `CameraInput` を追加し、次に `include_input = false` を使用してモデルのルートを追加します。フレームは実行中のパイプライン内で生成されるため、パブリックな `Input` ノードはありません。`include_output = true` を使用すると、検出結果またはテンソル用のプルエンドポイントを保持できます。

### 出力をプルする {#step-pull-output}

グラフを構築し、固定数の出力をプルします。タイムアウトが発生した場合、`--pull-timeout-ms` の時間内にモデル出力がアプリケーションに到達しなかったことを意味します。これは、カメラが停止したか、caps がネゴシエートされなかったか、または BoxDecode などの下流ステージでバックプレッシャーが発生した可能性があります。テンソルの数と最初のテンソルの形状を出力して、アプリケーションロジックを追加する前にデータが移動していることを確認します。

## Run

MIPIカメラが設定されたModalix DevKit上で、このチュートリアルを直接実行します。Neatのインストールルートから、事前に作成されたコマンドを実行します。リポジトリのルートから、ソースコードからビルドするコマンドを実行します。モデルアーカイブは、要求する前処理およびオプションの`--decode`モードと一致する必要があります。

デフォルトのプルタイムアウトは15秒です。起動時の診断情報を収集する際に、`--pull-timeout-ms`の値を大きくしてください（特に、起動直後のボードの場合）。

**Python:**
<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /モデルへのパス/model.tar.gz --frames 5 --decode none
</ShellCommand>

**C++ (prebuilt):**
<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model
--model /path/to/model.tar.gz --frames 5 --decode none
</ShellCommand>

サポートされている BoxDecode ルートを持つ YOLO スタイルのモデルの場合、`yolov8` や `yolov9seg` などのデコードトークンを選択してください。

<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model
--model /path/to/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

**C++ (build from source):**
<ShellCommand prompt="devkit">
./build.sh --target tutorial_023_run_mipi_camera_model
</ShellCommand>

<ShellCommand prompt="devkit">
./build/tutorials-standalone/tutorial_023_run_mipi_camera_model \
  --model /モデルへのパス/model.tar.gz --frames 5 --decode none
</ShellCommand>

期待される出力形状は、モデルとデコード経路によって異なります。生のMLA出力には通常、モデル固有のテンソルが含まれます。

```text
frame=0 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=1 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=2 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=3 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=4 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
[OK] 023_run_mipi_camera_model
```

サポートされているBoxDecode経路を使用すると、出力はデコードされた検出またはセグメンテーションテンソルに変わります。テンソルの数と最初の形状を、普遍的な契約としてではなく、移動の確認として使用してください。

`output_timeout`が表示された場合は、`gst-launch-1.0`を使用してカメラを検証し、次に`--print-backend`を使用して生成されたバックエンドを検査します。BoxDecode経路の場合、モデルアーカイブ、`--decode`トークン、および閾値がモデルと一致することを確認してください。

## In Practice

生成された GStreamer パスを調べたい場合は、`--print-backend` を使用してください。フォールバックが有効になっている場合、実行パスには `libcamerasrc`、`neatcamerabridge`、`neatprocesscvu`、`neatprocessmla`、オプションの EV74 ポストプロセス、および `appsink` が含まれている必要があります。デバッグ専用のパスを意図的に追加した場合を除き、`appsrc`、`ostosima`、`videoconvert`、または `videoscale` は含まれてはなりません。

## ソースファイル
- C++: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.cpp`
- Python: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py`
