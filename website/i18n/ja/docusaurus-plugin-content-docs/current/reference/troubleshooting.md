---
title: "トラブルシューティング"
description: "新たに発生したエラーに対する、症状に基づいた修正 Neat ユーザーが最も多くアクセスする"
sidebar_position: 5
---

# トラブルシューティング

各項目は「症状 → 原因 → 解決策」という形式で記載されています。症状の見出しは、正確なエラーメッセージです。このページ内で（Ctrl-F）検索して、表示されているメッセージに対応する項目を探してください。すべての項目は、現在のソースコードに対して検証済み、またはDevKit上で再現されています。

どこから始めればよいかわからない場合は、[行き詰まったときは：診断機能](#when-youre-stuck-diagnostics)に移動してください。

## インストールと環境設定

### 1. `pyneat is not importable. Either Neat is not installed, or the venv is not activated.`

:::info 原因
`pyneat` 仮想環境がアクティブになっていないか、または wheel パッケージが実行中の環境にインストールされていません。
:::

:::tip 修正
Pythonスクリプトを実行する前に、DevKit 環境を有効にしてください。
```bash
source ~/pyneat/bin/activate
```
:::

### 2. GSTプラグインの読み込みに失敗しました：`undefined symbol: _ZN16simaaidispatcher14DispatcherBase14submitPrepared...`

:::info 原因
Neat ランタイムの共有ライブラリが動的ローダーのパスに含まれていないため、GStreamer プラグインは、ロード時にランタイムシンボルを解決できません。
:::

:::tip 修正
起動する前に、ランタイムディレクトリを `LD_LIBRARY_PATH` に設定してください。
```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/neat/runtime:$LD_LIBRARY_PATH
```
:::

### 3. モデルアーカイブが見つかりません — `sima-cli modelzoo` がまだ実行されていません。

:::info 原因
コード（または`SIMA_YOLO_TAR` / `SIMA_RESNET50_TAR` / `SIMA_MODEL_TAR`）で参照されている`.tar.gz`というモデルアーカイブがディスク上に存在しません。
:::

:::tip 修正
Model Zooからダウンロードしてください。
```bash
sima-cli modelzoo get yolo_v8s     # or resnet_50, etc.
```
:::

## ビルド

### 4. `find_package(SimaNeat CONFIG)` がパッケージを見つけられません。

:::info 原因
CMakeが`SimaNeatConfig.cmake`（`lib/cmake/SimaNeat/`にインストールされています）を見つけられません。ネイティブなDevKitへのインストールでは、デフォルトのシステムプレフィックスにありますが、SDKクロスビルドでは、sysrootが`CMAKE_PREFIX_PATH`に含まれていません。
:::

:::tip 修正
`SYSROOT` をエクスポートし、`CMakeLists` で、それをプレフィックスパスに追加するようにします（[こんにちは、Neat テンプレート。](/develop-apps/hello-neat/minimal) はこれを行います）。
```cmake
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu")
endif()
find_package(SimaNeat REQUIRED CONFIG)
```
:::

## モデルの読み込みと設定

### 5. `failed to read image: <path>`

:::info 原因
OpenCV（`cv2.imread` / `cv::imread`）がnullを返しました。これは、ファイルが存在しないか、読み取り可能でないか、またはデコード可能な画像ではないことを意味します。
:::

:::tip 修正
入力テンソルを構築する前に、パスとファイルが有効な JPEG/PNG ファイルであることを確認してください。
:::

### 6. `reason=topk must be > 0`（`boxdecode`より）

:::info 原因
検出モデルの`ModelOptions.top_k`は`0`に設定されたままになっており、ボックスデコード段階では正の値の上限が必要です。
:::

:::tip 修正
肯定的な`top_k`を設定します（チュートリアルでは`100`を使用します）。
```python
opt.top_k = 100
```
（このメッセージは、EV74のボックスデコードプラグインから送信されました。）
:::

### 7. `preproc_upsample_not_supported`

:::info 原因
元の画像は、モデルの入力解像度よりも小さいため、前処理の段階で**アップスケール**する必要があります。しかし、古いEV74の前処理ファームウェアではアップスケールに対応しておらず、ダウンスケールのみが可能です。
:::

:::tip 修正
モデルの入力サイズ以上（例：YOLOv8の場合は640×640以上）のサイズの画像をソースとして入力するか、アップサンプリングカーネルを搭載したビルドに`neat-ev74-firmware`を更新してください。
（このメッセージは、EV74のプリプロセスプラグイン/ファームウェアから送信されます。）
:::

### 8. 低い`score_threshold` → 後処理における遅延の急増

:::info 原因
検出閾値を低く設定するほど、閾値処理を通過する候補ボックスの数が増え、NMSの計算コストは、通過するボックスの数のおおよその**2乗**に比例して増加します。
:::

:::tip 修正
弱い検出結果を確実に捉えられるように、閾値を必要な範囲で下げ、最悪の場合に備えて`top_k`で上限を設定します。[検出ボックスを読み取る](/tutorials/read-detection-boxes)を参照してください。
:::

## 推論を実行中

### 9. `misconfig.media_caps … Internal data stream error … reason not-negotiated (-4)`

:::info 原因
生の画像を入力として使用する場合、前処理段階が有効になっていなかった、または入力の種類が宣言されていなかったため、appsrcと最初の段階の間でcaps（機能記述子）のネゴシエーションを行うことができません。
:::

:::tip 修正
`ModelOptions`で、画像入力と前処理プリセットを宣言します。
```python
opt.preprocess.kind = pyneat.InputKind.Image
opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
```
:::

### 10. `No channel available (all candidate channel opens failed)`

:::info 原因
EV74ディスパッチャーは、ロードされたファームウェアが実装していないカーネルのスケジュールを試みました。これは通常、`neat-runtime`と`neat-ev74-firmware`が**同じビルドではない**（内部ハッシュが一致しない）ためです。たとえば、部分的なアップデートが原因で発生します。
:::

:::tip 修正
一致する`neat-*`セット（同じハッシュ値）をまとめてインストールし、ランタイムとファームウェアが同じハッシュ値を報告することを確認してください。[互換性 → バージョンが一致するセット](/getting-started/compatibility#the-version-matched-set-firmware--runtime)を参照してください。
*(このメッセージはEV74ディスパッチャーから送信されます。)*
:::

### 11. `frame=N rtsp_timeout`

:::info 原因
RTSPプルのタイムアウトが発生しました。URLが間違っているか、ストリームがフレームを送信していません。
:::

:::tip 修正
RTSP URLにアクセス可能であり、ストリームが正常に再生されていることを確認してください。また、使用されているトランスポートプロトコル（TCPまたはUDP）も確認してください。[RTSPストリームを再生する](/tutorials/consume-rtsp-stream)を参照してください。
:::

### 12. `CameraInput strict zero-copy requires external-buffer-mode`

:::info 原因
`CameraInputOptions::allow_cpu_fallback` はデフォルトで false に設定されているため、Neat は、最初から最後まで SiMaAI/デバイスのゼロコピーサポートを必要とします。`libcamerasrc` が汎用的な `external-buffer-mode` プロパティを公開していないか、インストールされているメモリライブラリが、その割り当てを DMA-BUF としてエクスポートできない可能性があります。
:::

:::tip 修正
一貫性のあるカメラとメモリパッケージがインストールされている場合は、厳密なゼロコピーを維持してください。DMA-BUFエクスポートのないカメラスタックで実行する必要がある場合は、互換ブリッジを明示的に選択してください。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::CameraInputOptions camera;
camera.allow_cpu_fallback = true;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera = pyneat.CameraInputOptions()
camera.allow_cpu_fallback = True
```

</CodeTab>
</CodeTabs>

アダプティブモードでは、下流のCVU/MLAステージにSiMaAIメモリが引き続き割り当てられます。データは、アップストリームカメラバッファがEV74によってまだ使用されていない場合にのみ、カメラブリッジでコピーされます。
:::

### 13. `misconfig.media_caps … libcamerasrc … not-negotiated (-4)`

:::info 原因
要求されたカメラの機能設定は、カメラスタックがサポートするモードと一致しないか、またはボードのオーバーレイ/ドライバーがカメラを正しく認識していません。
:::

:::tip 修正
Neatの外で、同じ形式、解像度、およびフレームレートであることを確認してください。

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

もしそれがうまくいかない場合は、まずオーバーレイ、ケーブル、センサー ドライバー、またはカメラ モードを修正してください。[Modalix DevKit MIPIカメラインターフェースガイド](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) を使用して、`.dtbo` と libcamera の検証パスを確認します。検証に合格した場合は、キャプチャ設定を現在の `CameraInputOptions` と比較してください。
:::

### 14. カメラで撮影した映像が緑色、紫色、または色調が強く表示される。

:::info 原因
フレームが誤ったピクセルフォーマットまたはカラー変換で処理されています。最も一般的な間違いは、`NV12` カメラフレームを RGB/BGR バイトとして扱うことです。もし、Neat の処理よりも前に同様の色合いが現れる場合、問題はおそらくカメラの ISP 調整か、libcamera パイプラインにあると考えられます。
:::

:::tip 修正
カメラのキャプションとモデルの事前処理形式を一致させてください。

- 推奨されるモデルのパスをリクエストします。`camera.format = "NV12"`
- `preprocess.color_convert.input_format = PreprocessColorFormat::NV12` を設定します。
- 本番環境で使用するモデルでは、CPUによる`videoconvert`/`videoscale`処理を避けてください。
- 短い`gst-launch-1.0 libcamerasrc ... ! videoconvert ! jpegenc`の簡易テストを実行し、Neatを実行する前に、色調が存在するかどうかを確認します。
:::

### 15. MIPIカメラのチュートリアルにある`frame=N output_timeout`。

:::info 原因
チュートリアルのタイムアウト時間内に、どの出力サンプルもアプリケーションに到達しませんでした。カメラからモデルへのグラフの場合、これはカメラがフレームを送信しなかった、Capsネゴシエーションが失敗した、モデルの処理がまだ開始されていない、またはBoxDecodeなどの後続の段階で出力が生成されなかったことを意味する可能性があります。
:::

:::tip 修正
まず、カメラのみを使用する経路を検証します。次に、より長いタイムアウトとバックエンドの出力機能を有効にして、チュートリアルを再度実行します。

<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/model.tar.gz --frames 2 --decode none \
  --pull-timeout-ms 15000 --print-backend
</ShellCommand>

フォールバックが有効になっている場合、プロセスのパスには、`libcamerasrc`、`neatcamerabridge`、`neatprocesscvu`、`neatprocessmla`、および`appsink`を含める必要があります。BoxDecodeルートの場合、`--decode`トークンと閾値がモデルアーカイブと一致することを確認してください。
:::

### 16. グラフの処理速度が遅い、またはライブフレームが途中で失われる。

:::info 原因
グラフにバックプレッシャーがかかっています。一般的な原因としては、処理速度に追いつけないプルループ、出力サンプルが長期間保持されている、ホットパスでのフレームごとのロギング、ソースに合わないキューポリシー、または明示的なドロップ/鮮度ポリシーがないライブストリームなどが挙げられます。
:::

:::tip 修正
再利用可能な`Run`を使用し、ランタイムポリシーを明示的にします。

- リアルタイムで入力されるデータなど、最新の情報が必要な場合は、`RunPreset::Realtime` / `pyneat.RunPreset.Realtime` を使用してください。
- すべての入力が重要な、バッチ処理やファイル処理には、`RunPreset::Reliable` / `pyneat.RunPreset.Reliable` を使用してください。
- キューがいっぱいになったときに、アプリの処理をブロックさせたくない場合は、`try_push(...)` を使用してください。
- `on_input_drop` を設定して、`stream_id`、`frame_id`、`port_name`、および理由別にドロップ数をカウントします。
- 継続的にデータを取得してください。出力キューが一杯になると、グラフ全体の処理速度が低下する可能性があります。
- アプリがランタイムによって管理されるバッファーを保持している可能性がある場合は、プッシュする前に、出力のリリースまたはコピーを行ってください。

マルチストリームのグラフの場合、`stream_id`と`frame_id`を保持し、ストリームごとの出力数をチェックします。集計されたFPSは、パフォーマンスが著しく低いストリームを隠してしまう可能性があります。詳細は、[グラフを実行 → 実際の値を偽らずにスループットを調整する](/develop-apps/development-workflow/pipeline#tune-throughput-without-lying-to-yourself)を参照してください。
:::

### 17. `unknown input/output name`、`no unambiguous default input`、または`no unambiguous default output`

:::info 原因
グラフには名前付きのエンドポイントがあり、アプリが誤った名前をプッシュまたはプルしたか、複数のエンドポイントが可能なグラフで名前のない`push(...)` / `pull(...)`を使用した。
:::

:::tip 修正
プッシュまたはプルする前に、名前を確認してください。

```python
run = graph.build()
print("inputs:", run.input_names())
print("outputs:", run.output_names())
```

次に、正確なエンドポイント名を使用してください。

```python
run.push("image", [tensor])
sample = run.pull("detections", timeout_ms=2000)
```

`Graph("name")` は診断ラベルです。エンドポイントを作成するものではありません。エンドポイントは、`nodes.input("name")` と `nodes.output("name")` から生成されます。
:::

### 18. `pull(...)` は、タイムアウトになるまで何も出力しません。

:::info 原因
タイムアウトになるまでに、サンプルが要求された出力に到達しませんでした。グラフの処理がまだ続いているか、出力名が間違っているか、入力にバックプレッシャーがかかっているか、グラフが閉じられたか、またはランタイムエラーが発生した可能性があります。
:::

:::tip 修正
タイムアウト、接続の切断、エラーを区別します。C++では、構造化されたプルオーバーロードを使用してください。

```cpp
simaai::neat::Sample sample;
simaai::neat::PullError error;

switch (run.pull("detections", /*timeout_ms=*/1000, sample, &error)) {
case simaai::neat::PullStatus::Ok:
  break;
case simaai::neat::PullStatus::Timeout:
  // Keep waiting, push more input, or report timeout.
  break;
case simaai::neat::PullStatus::Closed:
  // End of stream.
  break;
case simaai::neat::PullStatus::Error:
  std::cerr << error.code << ": " << error.message << "\n";
  break;
}
```

また、`run.last_error()`、エンドポイント名、入力のデータ型/レイアウト/形式、およびアプリケーションがすべての出力ブランチから継続的にデータを取得しているかどうかを確認してください。
:::

### 19. 古いスニペットが、`push_timeout_ms`、`pull_or_throw`、ルートレベルの`input_max_*`、または`boxdecode_original_*`によってエラーを起こす。

:::info 原因
このコードスニペットは、古いオプション設定またはプライベート/内部パスに対して記述されています。現在のアプリケーションコードでは、公開されている `ModelOptions`、`RunOptions`、および `Run` API を使用する必要があります。
:::

:::tip 修正
現在の公開名を使用してください。

- 入力の負荷を調整するために、`RunOptions.queue_depth`、`overflow_policy`、および`try_push(...)`を使用してください。
- `pull_or_throw` の代わりに、`pull(...)` または構造化された `PullStatus` のオーバーロードを使用してください。
- 古いスニペットでルートレベルの `input_max_*` フィールドが設定されている場合は、動的な入力制限を `ModelOptions.preprocess.input_max_width`、`input_max_height`、および `input_max_depth` の下に移動し、必要に応じてのみそれらの値を設定します。
- BoxDecodeの座標マッピングについては、事前に処理されたメタデータを使用することを推奨します。新しいサンプルでは、非推奨の元のサイズフィールドを設定しないでください。

コピーしたページに古いスペルがまだ表示されている場合は、そのドキュメントを古いものとして扱い、ドキュメントのバグを報告してください。そうすることで、次の読者が同じ間違いに遭遇することを防ぐことができます。
:::

## テンソルと Python の相互運用性

### 20. `… expects a TensorList; pass [tensor] instead of a single Tensor`

:::info 原因
単一の`Tensor`（または`Sample`）が、`run` / `push` / `build`に渡されました。APIでは明示的なリストが必要であり、これは意図的なものであり、バグではありません。
:::

:::tip 修正
以下にまとめます：`model.run([tensor])`、`run.push([tensor])`、`graph.build([tensor])`。
:::

### 21. `image-mode Tensor input requires explicit image format metadata`

:::info 原因
画像を入力とするモデルが、ピクセル形式を持たないテンソルを受け取ったため、Neat はバイトのレイアウトを解釈できません。
:::

:::tip 修正
明示的な形式でテンソルを構築します: `pyneat.Tensor.from_numpy(arr, image_format=pyneat.PixelFormat.RGB)`。
:::

### 22. `byte_format tensors cannot also specify image_format`

:::info 原因
テンソルは、`byte_format=`（不透明なバイトデータ）と`image_format=`（ピクセルデータ）の両方を使用して構築されました。これらは互いに排他的です。
:::

:::tip 修正
どちらか一方のみを合格すればよい。両方合格する必要はない。
:::

## 別のスタックから来た

- **「私の`.engine` / `.blob` / `.dlc` / `.hef`はどこにあるの？」** — Neatは、`.tar.gz`形式のモデルアーカイブを読み込みます。これは、コンパイルされたアーティファクトに相当します。
- **「CUDAストリームまたはOpenCLキューにタスクを固定するにはどうすればよいですか？」** — 固定する必要はありません。非同期の`push`/`pull`を使用して、プロデューサーとコンシューマーを分離し、代わりに`RunOptions`を調整してください。
- **「なぜスループットが公称のTOPSを下回るのか？」**—通常は、ホストのオーバーヘッド、キューの枯渇、出力のバックプレッシャー、またはドロップポリシーが原因であり、アクセラレータ自体が原因ではありません。[グラフを実行します。](/develop-apps/development-workflow/pipeline)を参照してください。

## 行き詰まったとき：診断機能

推測する前に、まずこれらを試してください。

**パイプライン/実行を検査する（PythonとC++）：**
- `graph.validate()` は、グラフを構築する前に、組み込みの契約に基づいて配線を検証する `GraphReport` です。その `error_code` を確認してください。
- `graph.describe()` → 解決されたパイプラインをテキスト形式（ノード名＋キャップチェーン）で表示します。
- `run.input_names()` / `run.output_names()` → ランタイムのプッシュ/プル呼び出しで受け入れられる名前。
- `run.start_measurement()` / `MeasureReport` → カウンター、遅延、入力ストリームのテレメトリ、プラグイン/エッジのタイミング、およびオプションの電力。
- `run.json(...)` / `run.save_json(...)`、または C++ `save_run_json(...)` → サンプルが移動した後で、実行結果を記録する。
- `NeatError::report()` → 実行時にエラーが発生した場合に、構造化されたエラーの詳細を報告します。


### サポート資料を収集する

もし別の開発者またはSiMa.aiサポートからの支援が必要な場合は、別の開発者が再現できるように、証拠となる情報を送信してください。以下を含めてください。

- Neat のバージョン/ビルド情報：Python `pyneat.build_info()` または C++ `sima_neat_version()`、`sima_neat_platform_version()`、および `sima_neat_abi_version()`。
- モデルのアーティファクト名、モデルのパス、およびその生成方法。
- 問題を再現する最小限の実行可能なコードスニペット。
- 入力の形状、データ型、レイアウト、ピクセル形式、ペイロードファミリー、およびグラフがアプリによってプッシュされるか、ソースによって所有されるか。
- `run.input_names()`と`run.output_names()`からのエンドポイント名。
- `GraphReport` は、`graph.validate()` または `NeatError::report()` から取得された JSON データです。
- 実行後、サンプルが処理されたら、`run.save_json(...)` または C++ の `save_run_json(...)` を使用して JSON 形式でデータをエクスポートします。
- 遅延、スループット、パケットロス、または電力消費が問題である場合に、測定結果を出力します。

複数のストリームに関する問題の場合、各ストリームの入力数、受理数、出力数、およびドロップ数も合わせて含めます。集計された FPS では、特定のストリームでデータが不足している状態が隠れてしまう可能性があります。

`GraphReport` を収集する際は、何が起こったかを説明するフィールドを保持してください。

- `error_code`と`repro_note`。
- `pipeline_string`;
- `bus`;
- `repro_gst_launch`と`repro_env`。
- `dot_paths`と`caps_dump`。
- 境界プローブが存在する場合、`boundaries` / `BoundaryFlowStats` が表示されます。
- シードされた`build(input, ...)`の失敗に対する`build_adaptation`。
- 実行後のカウンターとメトリクスについて、JSON形式でエクスポートを実行します。

**フレームワークのデバッグ出力を有効にする**には、`SIMA_DEBUG_PROFILE` を使用します。これは、トレースするコンポーネントをカンマで区切ったリストです。すべてをトレースする場合は `all` を使用するか、範囲を絞ってください。
```bash
export SIMA_DEBUG_PROFILE=all                 # everything
export SIMA_DEBUG_PROFILE=graph,gst,pipeline  # just these areas
```
既知のコンポーネント：`pipeline`、`graph`、`gst`、`appsink`、`inputstream`、`tensor`。デフォルトでは無効（デバッグ出力なし）。

**GStreamerのグラフをダンプして、capsがどこで問題を引き起こしているかを視覚的に確認します。**
```bash
export SIMA_GST_DOT_DIR=/tmp     # writes .dot graphs on build/failure; default: off
```

## エラーコード

`NeatError`（および`GraphReport::error_code`/ `PullError::code`）は、`domain.reason`コードを報告します。フレームワークでは、これらのコードが正確に定義されています。コードを有効にし、詳細についてはメッセージを参照してください。

| コード | 発生条件 |
|---|---|
| `io.open` | ファイルまたはデバイスのパスを開けませんでした。ファイルが存在しない、アクセス権がない、またはカーネルデバイスが存在しない（例：`/dev/rpmsg*`）可能性があります。|
| `io.parse` | JSON/設定ファイルの解析エラー。通常は、MPKコントラクトまたはステージごとの設定に問題がある場合に発生します。|
| `misconfig.pipeline_shape` | パイプラインの形状または最終的な名前の整合性に問題があります。たとえば、シンクの数が不適切、サイクルが存在する、終端の`Output`が欠落している、または要素名が重複しているなどが考えられます。|
| `misconfig.caps` | ストリーミング前に、大文字のオーバーライドまたは隣接するノードのコントラクトが、フレームワークの検証に失敗しました。|
| `misconfig.media_caps` | ランタイム GStreamer において、隣接するメディアステージ間のネゴシエーションが失敗しました。|
| `misconfig.input_shape` | 入力テンソルが、モデルの要件（ランク、空間次元、チャネル数）を満たしていません。|
| `misconfig.runtime_abi_mismatch` | フレームワーク/ランタイムプラグインのABI（アプリケーションバイナリインターフェース）の不一致。通常は、`pyneat`とランタイムのアーティファクトが混在している場合に発生します。|
| `build.plugin_missing` | 必要な GStreamer 要素またはコーデックプラグインが利用できません。|
| `build.property_invalid` | A GStreamer 要素のプロパティ名または値が無効です。 |
| `build.pipeline_syntax` | カスタム GStreamer フラグメントに無効な構文が含まれています。 |
| `build.parse_launch` | `gst_parse_launch` の失敗は、より詳細に分類できませんでした。|
| `runtime.pull` | より具体的な原因や根本原因を示すコードなしに、プル操作が失敗しました。 |
| `infra.dispatcher_unavailable` | MLA/EV74/A65ディスパッチャーの取得に失敗しました。ファームウェアがロードされていない、ライセンスが不足している、またはハードウェアに障害が発生している可能性があります。CPUによる代替処理もできません。|

これは簡単なトラブルシューティングのガイドです。すべてのコードとC++/Pythonの定数名については、[エラーコードの完全なカタログ](/reference/error-codes)を参照してください。
