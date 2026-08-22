---
title: "MIPIカメラを使用してください。"
description: "CameraInput、モデルの前処理、MLA推論、オプションのEV74 BoxDecode、および出力処理を追加して、オーバーレイに対応したMIPIカメラをNeat Graphに組み込みます。"
sidebar_position: 4
slug: /develop-apps/advanced-concepts/mipi-camera-input
---

# MIPIカメラを使用してください。

Modalix DevKit 上の MIPI カメラからフレームを直接読み込む必要がある場合、`CameraInput` を使用します。`CameraInput` は、検証済みの libcamera ストリームと、アクセラレータを優先したモデルのグラフの間の Neat の境界です。

```text
CameraInput -> model-managed CVU preproc -> MLA -> Output
CameraInput -> model-managed CVU preproc -> MLA -> EV74 BoxDecode -> Output
```

`appsrc` は使用できません。ユーザーコードに `ostosima` は含めないでください。また、ご自身で追加しない限り、本番環境の処理フローに CPU を使用した `videoconvert` や `videoscale` を含めないでください。

## 2つのゲート：まずブリングアップを行い、その後Neat。

MIPI CSI-2はカメラ接続インターフェースです。すべてのセンサーをプラグアンドプレイに対応させるわけではありません。動作するカメラパスを実現するには、適切なボードのオーバーレイ、センサードライバー、libcameraパイプライン、ISPの動作、および機能が必要です。

MIPIカメラの作業を、以下の2つの段階に分けて考えます。

1. **ボードの初期設定:** Modalix DevKit を使用すると、センサーと libcamera が要求されたモードでストリーミングできることを確認できます。
2. **Neat グラフの表示:** `CameraInput` は、取得したフレームをモデルによって管理される CVU/MLA 段階に送ります。

Neat はゲート 2 から開始されます。`.dtbo` オーバーレイを選択したり、センサー ドライバーをロードしたり、ISP を調整したりすることはありません。オーバーレイの設定、サポートされているオーバーレイ名、および `cam` の検証については、[Modalix DevKit MIPIカメラインターフェースガイド](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) を参照してください。

## Neat がサポートする機能

`CameraInput` は、プラットフォームのカメラスタックによってすでに起動されているカメラをサポートします。

- カメラは、ボードの電源を切った状態で、Modalix DevKit MIPIポートに接続されます。
- 正しいボードオーバーレイが有効になっています。
- カーネルドライバーとlibcameraパイプラインは、カメラを公開します。
- `libcamerasrc` は、要求された `video/x-raw` 形式（通常は `NV12`）のビデオストリームを処理できます。
- ここで設定するキャプションは、Neat で設定するモデルの事前処理と一致します。

もし、まだそれらの条件が満たされていない場合は、まずカメラのスタックを修正してください。Neatのグラフは正確な結果を出すことができますが、接続されていないセンサーをストリームに変換することはできません。

## カメラストリームを検証します。

libcamera/ でカメラの動作確認を行います。GStreamer グラフを作成する前に、このレイヤーを試してください。このレイヤーが機能しない場合、 Neat グラフでは修正できません。

その上 DevKit、確認してください。 `libcamerasrc` 存在します：

<ShellCommand prompt="devkit">
gst-inspect-1.0 libcamerasrc
</ShellCommand>

`cam` が利用可能な場合は、カメラの一覧と検査モードを表示します。

<ShellCommand prompt="devkit">
cam -l
cam -c 1 -I
</ShellCommand>

次に、Neatから注文する予定のキャップを実際に試してみてください。

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

視覚的な簡易テストとして、いくつかのフレームをJPEG形式でエンコードします。

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! videoconvert ! jpegenc ! \
  multifilesink location=/tmp/mipi-frame-%03d.jpg
</ShellCommand>

`videoconvert`と`jpegenc`は、今回の単発のデバッグチェックには問題ありません。ただし、処理速度が重要な場合は、これらの処理をモデルのパイプラインから除外してください。

## 生のMLA煙のグラフを作成する。

まず、生の MLA ルートから始めます。これにより、モデル固有のポスト処理を追加する前に、カメラのフレームが CVU 前処理と MLA 推論に到達することがわかります。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::CameraInputOptions camera;
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.format = "NV12";
camera.buffer_name = "camera0";
camera.allow_cpu_fallback = true;

neat::Model::Options model_options;
model_options.preprocess.kind = neat::InputKind::Image;
model_options.preprocess.input_max_width = static_cast<int>(camera.width);
model_options.preprocess.input_max_height = static_cast<int>(camera.height);
model_options.preprocess.input_max_depth = 3;
model_options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
model_options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
model_options.preprocess.resize.enable = neat::AutoFlag::On;
model_options.preprocess.resize.width = 640;
model_options.preprocess.resize.height = 640;
model_options.preprocess.resize.mode = neat::ResizeMode::Letterbox;
model_options.preprocess.resize.pad_value = 114;
model_options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
model_options.advanced_execution.preprocess_target = "EV74";
model_options.inference_terminal.mla_only = true;

neat::Model model("/models/yolo.tar.gz", model_options);

neat::Model::RouteOptions route;
route.include_input = false;
route.include_output = true;
route.upstream_name = camera.buffer_name;
route.buffer_name = camera.buffer_name;
route.name_suffix = "_camera0";
route.advanced_execution.preprocess_target = "EV74";

neat::Graph graph("camera_mla_smoke");
graph.add(neat::nodes::CameraInput(camera));
graph.add(model.graph(route));

neat::Run run = graph.build();
std::optional<neat::Sample> output = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.CameraInputOptions()
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.format = "NV12"
camera.buffer_name = "camera0"
camera.allow_cpu_fallback = True

model_options = pyneat.ModelOptions()
model_options.preprocess.kind = pyneat.InputKind.Image
model_options.preprocess.input_max_width = int(camera.width)
model_options.preprocess.input_max_height = int(camera.height)
model_options.preprocess.input_max_depth = 3
model_options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
model_options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
model_options.preprocess.resize.enable = pyneat.AutoFlag.On
model_options.preprocess.resize.width = 640
model_options.preprocess.resize.height = 640
model_options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
model_options.preprocess.resize.pad_value = 114
model_options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
model_options.advanced_execution.preprocess_target = "EV74"
model_options.inference_terminal.mla_only = True

model = pyneat.Model("/models/yolo.tar.gz", model_options)

route = pyneat.ModelRouteOptions()
route.include_input = False
route.include_output = True
route.upstream_name = camera.buffer_name
route.buffer_name = camera.buffer_name
route.name_suffix = "_camera0"
route.advanced_execution.preprocess_target = "EV74"

graph = pyneat.Graph("camera_mla_smoke")
graph.add(pyneat.nodes.camera_input(camera))
graph.add(model.graph(route))

run = graph.build()
output = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

`inference_terminal.mla_only = true` は意図的なものです。これにより、煙の軌跡が `CameraInput -> CVU preproc -> MLA -> Output` に維持されるため、検出のデコードをデバッグする前に、カメラと推論の動きをデバッグできます。

## モデルが必要とする場合に、EV74 BoxDecodeを追加します。

YOLOスタイルの検出モデルの場合、BoxDecodeを明示的に有効にし、後処理をEV74上で実行します。デコードトークンは、MPKの出力形式と一致している必要があります。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
model_options.inference_terminal.mla_only = false;
model_options.decode_type = neat::BoxDecodeType::YoloV9Seg;
model_options.advanced_execution.postprocess_target = "EV74";
model_options.score_threshold = 0.25f;
model_options.nms_iou_threshold = 0.45f;
model_options.top_k = 100;

route.advanced_execution.postprocess_target = "EV74";
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
model_options.inference_terminal.mla_only = False
model_options.decode_type = pyneat.BoxDecodeType.YoloV9Seg
model_options.advanced_execution.postprocess_target = "EV74"
model_options.score_threshold = 0.25
model_options.nms_iou_threshold = 0.45
model_options.top_k = 100

route.advanced_execution.postprocess_target = "EV74"
```

</CodeTab>
</CodeTabs>

`decode_type` は設定せず、生の MLA テンソルが必要な場合は、`mla_only = true` を維持してください。モデルの処理経路でデコードされた検出結果またはセグメンテーションデータを生成する必要がある場合にのみ、`decode_type` を設定します。

## 適切なメモリモードを選択してください。

`CameraInput` には、次の 2 つのモードがあります。

| モード | 使用する際は | 行動 |
| --- | --- | --- |
| 厳格なゼロコピー | お客様の `libcamerasrc` は `external-buffer-mode` を公開し、メモリライブラリは DMA-BUF エクスポートをサポートします。 | Neat は、ソースから直接データを取得し、下流の DMA-BUF プールに格納する必要があります。そうでない場合、処理は失敗します。 |
| 適応型フォールバック（明示的なオプトイン） | デバイスバッファーをエクスポートできないカメラスタックとの互換性が必要です。 | Neat は、OS/libcameraのバッファーを受け入れ、それらをプールされたSiMaAIメモリにコピーして、CVU/MLAへの引き渡しを行います。また、ソースがすでにバッファーを提供している場合は、SiMaAIバッファーをそのまま転送します。 |

厳密なゼロコピーがフレームワークのデフォルト設定です。これには、 `libcamerasrc` 一般的なものを明らかにする `external-buffer-mode` プロパティと、DMA-BUFのエクスポートをサポートするメモリライブラリ。設定 `camera.allow_cpu_fallback = true` 明示的な互換性維持のための手段としてのみ。フォールバックコピーは、アクセラレーターパイプラインへのブリッジであり、ホットパスにCPUによる色変換やスケーリングを追加することを許可するものではありません。

両方のモードにおいて、 Neat カメラキャップの直後、かつ、キューの前に、プライベートなメモリブリッジを配置します。このブリッジは、標準のバッファープールを提案します。 `GST_QUERY_ALLOCATION`そのプールは、検証済みのプレーンレイアウトを、1つのパックされたSiMaAI割り当てから割り当て、各プレーンに対して1つのDMA-BUFをエクスポートします。
`libcamerasrc` DMAバッファをISPキャプチャキューにインポートします。キャプチャ後、ブリッジは同じパックされた割り当てをアンラップし、後続の処理に使用します。これは通常の処理経路であり、フォールバックコピー経路ではありません。

アプリケーションは、ISP出力の保持ポリシーを管理します。
`capture_buffer_count` 引数として `nodes::CameraInputWithCaptureBuffers()` または
`pyneat.nodes.camera_input()` において `0` カメラのデフォルト設定を使用するか、時間的なエンコーダーまたは非同期の機械学習グラフがフレームをより長く保持する場合に、より大きな最小値を要求します。アクティブなカメラパイプラインは、自身の制限を検証します。 Neatのプロバイダーは最大128まで対応します。この数は、プライベートなCSI-to-ISP RAWトランジットリングとは別にカウントされます。 `queue_depth`で制御される Neatのライブ GStreamer キューを使用します。現在のフレームが完全性よりも重要な場合は、フレームを一部だけ保持するキューを使用します。すべてのフレームを保持する必要がある場合は、下流のバックプレッシャーを使用します。互換性コピーモードでは、ブリッジのプライベートプールは必要に応じて拡張されるため、フレームを一部だけ保持するキューが古いフレームを破棄する前に処理がブロックされることはありません。

## CVU/EV74に対する前処理を継続してください。

モデルのパイプラインの場合、モデルによって管理される前処理を推奨します。

- リサイズ、色変換、正規化、量子化、およびテッセレーションのために、`Model::Options::preprocess` または `pyneat.ModelOptions.preprocess` を設定します。
- モデルがそれらをサポートしている場合、モデルによって管理される CVU の事前/事後ターゲットを `EV74` に保持します。
- デバッグ専用のグラフを作成する場合を除き、モデルの前に `VideoConvert`、`VideoScale`、または GStreamer `videoconvert`/`videoscale` を挿入することは避けてください。

カメラはフレームを提供します。CVU（コンピュータービジョンユニット）がフレームの処理を行うべきです。CPUを意図しない画像処理エンジンとして使用すべきではありません。

## トラブルシューティングのクイックマップ

| 症状 | まず確認してください。 |
| --- | --- |
| カメラが検出されません。 | 選択した`.dtbo`、ケーブルの向き、電源の再起動、およびカーネル/libcameraのログを確認してください。 |
| `libcamerasrc` が見つかりません。 | DevKit のビルド用に、対応する Neat/ランタイム カメラ画像またはカメラパッケージをインストールしてください。 |
| `misconfig.media_caps`、または`not-negotiated`。 | 正確な`format,width,height,framerate`を、`gst-launch-1.0`を使用して検証してください。`NV12 1920x1080@30`など、サポートされていることがわかっているモードを試してください。 |
| 厳格なゼロコピー処理の失敗 | `allow_cpu_fallback = true` を設定するか、SiMaAI のゼロコピー機能を活用するカメラスタックを使用してください。 |
| 出力される色がおかしい | フレームが RGB/BGR ではなく、`NV12` として解釈されていることを確認してください。`libcamerasrc` から直接生成された JPEG 画像も誤っている場合は、Neat のデバッグを行う前に、カメラの ISP/調整をデバッグしてください。 |
| 処理能力が低い。 | CPUによるビデオ変換／スケーリングを停止し、継続的にデータを取得し、最新のデータに優先順位を置くライブソースキューポリシーを使用します。 |

その他の症状については、[トラブルシューティング](/reference/troubleshooting)（トラブルシューティング）を参照してください。
