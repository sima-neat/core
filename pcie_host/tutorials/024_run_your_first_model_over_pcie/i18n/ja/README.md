# 024 PCIe を使用した最初のモデルを実行する

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, inference, tensor, image, detection |

## Concept

PCIe ホスト API は、モデルで使用できるテンソルまたはデコードされた画像ピクセルを受け入れます。テンソルモードでは、前処理をホスト側で行います。画像モードでは、元のピクセルを送信し、Modalix カードでリサイズ、カラー変換、および正規化を行います。ボックスデコードを追加すると、画像入力は維持されますが、6 つの生の YOLO 出力テンソルが、1 つのコンパクトな検出リストに置き換えられます。

## Walkthrough

同じ YOLOv8s アーカイブと 640x480 のストリートシーンを使用して、3 つの独立したプログラムを実行します。各プログラムは 1 つのモードをデモンストレーションし、キュー 0 を同期的に使用し、1 つのモデルを閉じます。これにより、すべての例が十分に短く、個別にコピーできるようになります。

### モデルで使用できるテンソルを実行する {#step-tensor-mode}

ホストは、画像をモデルによって報告された `[640, 640, 3]` 入力に合わせてレターボックス処理し、BGR を RGB に変換し、ピクセルを `[0, 1]` にスケーリングします。`Model.run()` は、その FP32 テンソルをカード側の画像前処理なしで送信し、6 つすべての生の YOLO 出力ルートを出力します。

### 前処理をカードに移動する {#step-image-mode}

`preprocess.kind` を `Image` に設定し、受信ピクセルを BGR として識別し、`COCO_YOLO` プリセットを選択します。ホストは現在、デコードされたピクセルを送信し、カードはレターボックスリサイズ、BGR から RGB への変換、および正規化を実行します。プログラムは、6 つの生の出力ルート名と形状を出力するため、テンソルモードと比較できます。

### カードで検出をデコードする {#step-decode-boxes}

`BoxDecodeType.YoloV8`、スコアしきい値、NMS しきい値、および出力制限を追加します。返される BBOX テンソルは、検出数の後に、固定サイズのレコードが続き、`(x, y, width, height, score, class_id)` が含まれます。例では、最初の 10 個のレコードを解析して、ソース画像座標で出力します。

### BBOX テンソルを解析する {#step-parse-boxes}

ボックスデコードが 1 つの有効なテンソルを返したことを確認し、先頭のカウントを読み取り、ペイロードを超えるカウントは拒否します。残りの各 24 バイトのレコードは、次に 1 つの検出に変換されて出力されます。

## Run

PCIeホストパッケージをインストールし、[チュートリアルの設定](/tutorials/before-you-run)で説明されているように、チュートリアルバンドルをダウンロードします。抽出したPCIeエクストラのルートで次のコマンドを実行します。Windowsでは、[Windowsでアプリケーションを実行する](/getting-started/neat-library/pcie-host/#run-applications-on-windows)の説明に従ってチュートリアルを設定し、プログラムを実行します。

プログラムには、このディレクトリに `yolo_v8s_mpk.tar.gz` が必要です。[Model Browser](/tools/model-browser/)からYOLOv8sをダウンロードし、ダウンロードしたZIPファイルからアーカイブを展開して、その名前で保存します。

```bash
sima-cli models download --id yolov8s --variant w8a8 --output models
unzip models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

Windowsでは、PowerShellでモデルをダウンロードし、`Expand-Archive`でアーカイブを展開します。

```powershell
sima-cli models download --id yolov8s --variant w8a8 --output models
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_tensor_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_boxdecode.py
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_024_run_tensor_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_boxdecode
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_024_run_tensor_mode
./build.sh --target tutorial_024_run_image_mode
./build.sh --target tutorial_024_run_image_boxdecode

./build/tutorials-standalone/tutorial_024_run_tensor_mode
./build/tutorials-standalone/tutorial_024_run_image_mode
./build/tutorials-standalone/tutorial_024_run_image_boxdecode
```

Windowsでは、チュートリアルのコピー先で、**x64 Native Tools Command Prompt** からCMakeでビルドします。`C:\opencv\build` は、[Windowsでアプリケーションを実行する](/getting-started/neat-library/pcie-host/#run-applications-on-windows)で説明しているOpenCVパッケージの`build`ディレクトリに置き換えてください。

```bat
call "%ProgramFiles%\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_024_run_tensor_mode
cmake --build build --config Release --target tutorial_024_run_image_mode
cmake --build build --config Release --target tutorial_024_run_image_boxdecode
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_024_run_tensor_mode.exe
build\Release\tutorial_024_run_image_mode.exe
build\Release\tutorial_024_run_image_boxdecode.exe
```

対応する C++ および Python プログラムは、テンソルモードと画像モードの両方で同じ 6 つの生の出力コントラクトを出力し、その後、デコードされた人、車、またはその他の見えるオブジェクトを出力します。

```text
Tensor mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_tensor_mode
Image mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_image_mode
Image + boxdecode detections=...
  person score=... box=(...)
[OK] 024_run_image_boxdecode
```

デフォルトはカード 0 とキュー 0 です。別のカードを使用する場合にのみ、`--card N` を渡します。その管理アドレスは自動的に導き出されます。

## In Practice

アプリケーションが正確に `model.info()` によって報告された dtype、形状、レイアウト、カラー順序、および数値範囲を生成する場合に、テンソルモードを使用します。アプリケーションが自然にデコードされたピクセルを所有し、カードに反復可能なモデル前処理を適用させたい場合は、画像モードを使用します。アプリケーションが生の特徴マップではなく、検出を必要とする場合は、ボックスデコードを有効にします。

すべてのモードは、同じ `pcie::Model`/`pyneatpcie.Model` ライフサイクルを使用します。`ModelOptions` と送信されたペイロードのみが変更されます。[PCIe推論を非同期で実行](/tutorials/run-pcie-inference-async) を使用して、`push()` と `pull()` を使用して、送信と完了をオーバーラップさせます。

## ソースファイル

- `run_tensor_mode.cpp`
- `run_tensor_mode.py`
- `run_image_mode.cpp`
- `run_image_mode.py`
- `run_image_boxdecode.cpp`
- `run_image_boxdecode.py`
- `../assets/street-scene.png`
