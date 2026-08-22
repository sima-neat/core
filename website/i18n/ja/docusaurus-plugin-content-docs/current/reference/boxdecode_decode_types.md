---
title: "BoxDecode デコードの種類"
description: "オブジェクト検出のポスト処理に適切な BoxDecodeType を選択してください。"
sidebar_position: 6
---

# BoxDecode デコードの種類

`nodes::SimaBoxDecode` は、生の検出ヘッドのテンソルを検出結果に変換します。これはモデル推論の後に実行され、選択されたモデルファミリーのデコード計算を適用し、信頼度の低いボックスをフィルタリングし、NMS（Non-Maximum Suppression）を実行し、デコードされたボックスから始まるテンソルペイロードを出力します。検出モデルは、そのペイロードをボックスとして解析できます。姿勢およびセグメンテーションモデルも、ボックスの後に続くキーポイントまたはマスクを解析できます。

通常のモデルパックの使用では、`Model` を認識するコンストラクタを使用することを推奨します。モデルアーカイブは、デコーダーに必要なテンソルの順序、レイアウト、量子化、クラス数、リサイズメタデータ、およびスコアドメインのヒントを提供します。通常、アプリケーションはデコードファミリーとフィルタリングの閾値を選択するだけです。

## クイックスタート

```cpp
using namespace simaai::neat;

Model model("/path/to/yolov8_model.tar.gz");

auto boxdecode = nodes::SimaBoxDecode(
    model,
    BoxDecodeType::YoloV8,
    /* detection_threshold */ 0.25,
    /* nms_iou_threshold */ 0.45,
    /* top_k */ 100);
```

スタンドアロンのステージで使用する場合：

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.45;
opt.top_k = 100;
```

## 引数

| 引数 | 意味 |
| --- | --- |
| `decode_type` | モデルファミリー/ヘッド形式。例：`BoxDecodeType::YoloV8` または `BoxDecodeType::YoloX`。必須。|
| `detection_threshold` | 検出結果を維持するために必要な最小スコア。`0.25`などの、モデルに適した値を設定してください。|
| `nms_iou_threshold` | 非最大値抑制で使用されるIoU（Intersection over Union）の閾値。|
| `top_k` | 維持する検出結果の最大数。`0` は、バックエンド/モデルのデフォルト値を使用します。|
| `original_width` x `original_height` | は、生のジオメトリコンストラクタを使用する際に、座標マッピングに使用するソース画像のサイズです。|
| `model_width`, `model_height` | モデルの入力サイズを上書きします。これを使用すると、 `Model` コンストラクタは、パッケージ化されたテンソルの契約ではなく、空間デコードのパラメータを変更します。 |
| `resize_mode_override` | は、アップストリームの`Preproc`ステージでリサイズメタデータが書き込まれない場合にのみ使用し、必要に応じて、拡大/レターボックス/トリミングの動作を明示的に指定します。|
| `decode_type_option` | 高度なサブレイアウトセレクター。モデルパックを使用する場合は、`Auto` のままにしておきます。エクスポートされたヘッドレイアウトを把握している場合にのみ変更してください。|

## 入力と出力

**入力:** モデルからの生の検出テンソル。期待されるテンソルの形状は、モデルの種類によって異なります。MPK/モデルアーカイブを使用する場合、Neat は、パッケージ化されたコントラクトからこれらの詳細を読み取ります。

**出力:** デコードされた検出を含む 1 つの BoxDecode テンソル。検出モデルは、標準の `BBOX` ペイロードを使用します。ポーズおよびセグメンテーションモデルは、先頭のボックスを同じままにして、タスク固有のペイロードを追加します。

| モデルタスク | C++ヘルパー | Pythonヘルパー | デコードされたテンソル |
| --- | --- | --- | --- |
| 検出 | `decode_bbox(...)` | `pyneat.decode_bbox(...)` | `[N, 6]` float32 boxes: `x1, y1, x2, y2, score, class_id` |
| ポーズ | `decode_pose(...)` | `pyneat.decode_pose(...)` | ボックス `[N, 6]` とキーポイント `[N, 17, 3]` float32: `x, y, visibility` |
| セグメンテーション | `decode_segmentation(...)` | `pyneat.decode_segmentation(...)` | ボックス `[N, 6]` float32、マスク `[N, 160, 160]` uint8 |
| SuperPoint | `decode_superpoint(...)` | `pyneat.decode_superpoint(...)` | キーポイント `[N,2]`、スコア `[N]`、記述子 `[N,D]` |

検出結果を表示するグラフは、その結果を`SimaRender`に渡すことができます。ボックスのみが必要なアプリケーションコードは、引き続きBoxDecodeの出力に対して`decode_bbox(...)`を使用できます。

## スーパーポイント

SuperPointは、BoxDecode製品の一部として引き続き機能しますが、ボックスであるかのように扱うのではなく、特徴点を出力します。最小限のA65デフォルト設定は次のとおりです。

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.descriptor_output_dtype = TensorDType::Float32;

auto decoder = nodes::SimaBoxDecode(model, options);
```

Pythonでも同じデフォルト値が使用されます。

```python
options = pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.SuperPoint)
options.superpoint.descriptor_output_dtype = pyneat.TensorDType.Float32

decoder = pyneat.nodes.sima_box_decode(model, options=options)
```

`A65V1` はデフォルトのプロファイルです。モデルが異なる数値演算を必要とする場合は、別のプロファイルを明示的に選択してください。Neat は、テンソルの形状や値から動作を推測しません。

| プロファイル | 選択するタイミング | 生産状況 |
|---|---|---|
| `LightGlueV1` | LightGlue互換の検出器、NMS、座標、および記述子の動作 | サポート対象 |
| `MagicLeapDemoV1` | 固定されたMagic Leapデモの動作 | 対応状況 |
| `A65V1` | 以前のA65 SuperPointデコーダーとの互換性 | サポート済み。デフォルト設定。|
| `PaperBicubicV1` | 将来の完全な双三次補間ポリシーのために予約された数値ID | プロダクションで定義されるまで却下 |

数値の処理方法と出力エンコーディングは独立しています。たとえば、デフォルトのV1出力でA65の数値処理方法を選択できます。

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::FeaturePointsV1;
```

従来のバイトレイアウトは、オプションで利用でき、いくつかの追加の制約があります。

```cpp
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::LegacyA65InterleavedV0;
options.superpoint.descriptor_output_dtype = TensorDType::Int8;
```

`SuperPointProfile::Auto` は、まず信頼できるMPKの`superpoint.profile`メタデータを使用します。API（`Model::Options.superpoint.profile`）またはMPKのいずれからもプロファイルが提供されない場合、`A65V1`に解決されます。Neatは、テンソルの形状、値、ファイル名、または下流ノードからプロファイルを推測することはありません。

`detection_threshold=0.0`、`top_k=0`、`nms_radius=-1`、および`border_margin=-1`の公開されたデフォルト値が変更されていない場合、それらは選択されたプロファイルから解決されます。`A65V1`は、閾値`0.1`、Top-K`600`、NMS半径`4`、および境界マージン`0`に解決されます。LightGlueV1とMagicLeapDemoV1は、それぞれ閾値`0.0005`と`0.015`を使用します。どちらもTop-K`600`、NMS半径`4`、および境界マージン`4`を使用します。

`nms_iou_threshold`はSuperPointには適用されません。代わりに、ピクセル半径である`superpoint.nms_radius`を使用してください。デフォルトの出力は、バージョン管理された`FEATURE_POINTS_V1`構造化配列ペイロードです。`LegacyA65InterleavedV0`は、明示的な移行形式であり、256次元のINT8記述子が必要です。`decode_bbox`または`BoxDecodeResults`ではなく、`decode_superpoint`を使用してください。

バージョン管理されたMPK`superpoint`スキーマv1レコードは、エラーが発生した場合に安全に処理されます。これらには、プロファイル名、異なる検出器と記述子テンソルID、64桁の16進数を持つ`sha256:`フィンガープリント、およびサポートされている入力表現`raw-logits-65`と`coarse-pre-l2`が含まれている必要があります。スキーマ0は、移行/手動レコードとしてのみ受け入れられます。省略されたスキーマ0表現フィールドは、これらの2つの生の入力表現に標準化され、診断にデフォルトとして記録されます。不明なスキーマバージョンまたは表現トークンは、コンパイル時にエラーが発生します。APIプロファイルのオーバーライドが、別のMPKプロファイルに対してスタンプされたフィンガープリントと競合する場合、選択されたプロファイルに対してMPKを再スタンプします。Neatは、そのプロビナンスを破棄または再解釈しません。

## BBOXワイヤーペイロード

検出デコード処理は、入力フレームごとに1つのテンソルを出力し、そのテンソルには`BBOX`というタグが付けられます。このテンソルは、ランク1の`UInt8`型のバイトバッファーです。

| フィールド | 値 |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | `[N_bytes]`。ここで、`N_bytes`は、モデルアーカイブから得られる、パックされたバッファーの容量である。 |

テンソルの形状は、検出回数ではなく、バイト数で表されます。ペイロードはリトルエンディアン形式を使用します。

```text
offset  size  content
------  ----  -------
  0      4    uint32  N = valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   trailing bytes are padding and must be ignored
```

各`RawBox`レコードは24バイトです。

| オフセット | サイズ | タイプ | フィールド | 意味 |
| --- | --- | --- | --- | --- |
| 0 | 4 | `int32` | `x` | ソース画像の左上のx座標（ピクセル単位）。 |
| 4 | 4 | `int32` | `y` | ソース画像の左上の y 座標。 |
| 8 | 4 | `int32` | `w` | 元画像のピクセル単位での幅。|
| 12 | 4 | `int32` | `h` | 元の画像のピクセル単位の高さ。|
| 16 | 4 | `float32` | `score` | NMS後の`[0.0, 1.0]`における信頼度。|
| 20 | 4 | `int32` | `class_id` | モデルで定義されたクラスID。|

1つのレコードに対応するPythonの`struct`形式は`"<iiiifi"`です。

座標は、上流のプリプロセス処理メタデータが存在する場合、元の画像のピクセル単位で表されます。座標は`[0, 1]`に正規化されることも、モデルの内部レターボックス形式の入力空間で表現されることもありません。

## `model.run` が生のヘッドを返す場合

一部のモデルの処理経路では、デコードされた`BBOX`テンソルではなく、`model.run(...)`から生の特徴マップヘッドが返されます。これは、処理が失敗したことを意味するのではなく、モデルは正常に実行されたものの、出力の読み取り時にBoxDecodeが処理経路に含まれていなかったことを意味します。

次のルールを使用してください。

- `detections=...`または`BBOX`テンソル：パックされたBBOXペイロードを解析するか、または使用します。
  デコードヘルパー。
- `raw_output_heads=...`: BoxDecodeステージを追加するか、モデルのルーティングを検査するか。
  モデル固有の後処理を用いて、生のテンソルを処理します。

生のヘッダーをボックスとして解析しないでください。生のテンソルのレイアウトは、エクスポートされたモデルのファミリーとモデルアーカイブの仕様によって異なります。

## 契約を上書きする

モデルアーカイブは、デコードタイプ、閾値、`top_k`、およびソースジオメトリのデフォルト値を設定できます。ランタイム引数は、空でない、または正の値が渡された場合にのみ、これらのデフォルト値を上書きします。

| ランタイム引数 | 渡された値 | 動作 |
| --- | --- | --- |
| `decode_type` | 空 / `Unspecified` | サポートされている場合は、モデルアーカイブまたはルートプランナー推論を保持します。|
| `decode_type` | 具体的な型 | この実行において、デコード処理を上書きします。|
| `original_width` / `original_height` | `0` | パッケージ化されたジオメトリまたは上流のプリプロセスメタデータを保持します。|
| `original_width` / `original_height` | 正の整数 | 座標マッピングのために、元の画像のサイズを上書きします。|
| `detection_threshold` / `score_threshold` | `0.0` | パッケージ化されたしきい値を保持します。|
| `detection_threshold` / `score_threshold` | `> 0.0` | スコアの閾値を上書きします。|
| `nms_iou_threshold` | `0.0` | パッケージ化されたNMS IoUを保持します。|
| `nms_iou_threshold` | `> 0.0` | NMS IoU の値を上書きします。|
| `top_k` | `0` | パッケージ化された上位K個の要素を保持します。|
| `top_k` | `> 0` | 保持する検出結果の最大数を上書きします。|
| `num_classes` | `0` | MPKから推測されるクラスヘッダーの深さを利用してください。|
| `num_classes` | MPKに一致する正の整数。 | 明示的なクラス数を指定してください。MPKが単一クラスのヘッドを確実に分割できるかどうかを推測できない場合に必要となります。 |
| `num_classes` | は、YOLO26 MPKと矛盾する正の整数です。| パイプラインの構築前にエラーが発生し、両方の値が報告されます。YOLO26は、クラスの深さからグループ化された生のヘッドレイアウトを導き出すため、この不一致はモデルの契約違反です。|
| `num_classes` | SSD用の正の整数、またはYOLO26より前の非姿勢推定YOLOファミリー。 | 既存の明示的なオーバーライドの動作を維持します。ポーズデコーダーとSuperPointは、それぞれのファミリーに固有のルールを保持します。 |

`detection_threshold` は、BoxDecodeノード/ステージのコンストラクタで使用される名前です。`ModelOptions.score_threshold` は、同じ制御に渡されるモデルルートオプションです。

## デコードタイプのマッピング

| API列挙型 | バックエンドトークン | 一般的なモデルファミリー |
| --- | --- | --- |
| `BoxDecodeType::Yolo` | `yolo` | 一般的なYOLOスタイルのヘッド |
| `BoxDecodeType::YoloV5` | `yolov5` | YOLOv5による検出 |
| `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLOv5セグメンテーション |
| `BoxDecodeType::YoloV7` | `yolov7` | YOLOv7による検出 |
| `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLOv7セグメンテーション |
| `BoxDecodeType::YoloV8` | `yolov8` | YOLOv8による検出 |
| `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLOv8セグメンテーション |
| `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLOv8 ポーズ |
| `BoxDecodeType::YoloV9` | `yolov9` | YOLOv9による検出 |
| `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLOv9セグメンテーション |
| `BoxDecodeType::YoloV10` | `yolov10` | YOLOv10による検出 |
| `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLOv10セグメンテーション |
| `BoxDecodeType::YoloV26` | `yolo26` | YOLO26による検出 |
| `BoxDecodeType::YoloV26Pose` | `yolo26-pose` | YOLO26 ポーズ |
| `BoxDecodeType::YoloV26Seg` | `yolo26-seg` | YOLO26セグメンテーション |
| `BoxDecodeType::YoloV6` | `yolov6` | YOLOv6による検出 |
| `BoxDecodeType::YoloX` | `yolox` | YOLOXによる検出 |
| `BoxDecodeType::Ssd` | `ssd` | 注文されたヘッドジオメトリから選択された、正確に準備されたSSD300、SSD-Mobile-300、SSD-Mobile-320、またはSSDlite-Mobile-320の契約。 |
| `BoxDecodeType::SuperPoint` | `superpoint` | SuperPoint検出器および特徴記述子の後処理 |
| `BoxDecodeType::Detr` | `detr` | DETRスタイルのトランスフォーマーによる検出 |
| `BoxDecodeType::EffDet` | `effdet` | EfficientDetによる検出 |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | R-CNNのプロポーザル段階 |
| `BoxDecodeType::Centernet` | `centernet` | CenterNetによる物体検出 |

`BoxDecodeType::Unspecified` は未設定のプレースホルダーであり、ランタイム前にエラーが発生します。SSDレシピの識別子は、別の公開デコードタイプやバックエンドトークンではなく、内部のCore契約（`ssd300-v1`、`ssd-mobile-300-v1`、`ssd-mobile-320-v1`、または`ssdlite-mobile-320-v1`）です。Coreは、最適化処理の前にこれを解決し、インストールされたオブジェクトデコーダーは引き続き、サポートする`ssd` ファミリーのトークンを受け取り、すでに検証済みのヘッドジオメトリから対応する固定実装を選択します。

## 適切なタイプを選ぶ

- SiMaが提供またはコンパイルしたモデルパックを使用している場合は、モデルファミリーに一致する`BoxDecodeType`を選択し、`decode_type_option`を`Auto`のままにしてください。
- 検出結果が不足している場合、またはすべてのスコアが予想外に低い場合は、まず、デコードファミリーがエクスポートされたモデルのヘッドと一致していることを確認してください。YOLOX、YOLOv6、およびYOLO26は、生の出力またはロジット形式のヘッドを使用しており、確率のみを出力するYOLOヘッドとは異なる扱いをする必要があります。
- ボックスが正しく配置または拡大・縮小されていない場合は、画像のサイズ変更ポリシーを確認してください。`resize_mode_override` は、グラフに、サイズ変更メタデータを書き込む上流の `Preproc` ステージがない場合にのみ使用してください。
- カスタムモデルパックを作成する場合は、アーカイブに、検出ヘッドに関する正確な情報（テンソルの順序、論理的な形状、物理的な格納方法、dtype/量子化、スコアの範囲、クラス数、およびスライスされた出力など）が記載されていることを確認してください。アプリケーションコードは、これらの詳細を補正する必要はありません。

## 形状とレイアウトに関するガイドライン

異なる検出モデルは、それぞれ異なるヘッドレイアウトを使用します。あるモデルは、各特徴マップレベルに対して1つのテンソルを使用し、別のモデルは、バウンディングボックス、オブジェクトネス、クラス、キーポイント、またはマスクを個別のテンソルに分割します。一部のモデルの出力は、密なHWCテンソルであり、他のモデルの出力は、コンパイラ/ランタイムによってパックまたはスライスされます。

モデルパックフローの場合、これはパッケージ化されたコントラクトによって処理されます。手動で接続されたテンソルの場合、重要なルールは次のとおりです。エクスポートされたヘッド形式と完全に一致させます。ランクまたはチャネル数のみに基づいてデコードタイプを選択しないでください。

高度なテンソルコントラクトルール：

- YOLOファミリーのデコードタイプ：`Yolo`、`YoloV5`、`YoloV7`、`YoloV8`、`YoloV9`。
  `YoloV10`およびセグメンテーション/ポーズ検出モデルは、分離されたヘッド構造または、モデルファミリーに適合するパックされたヘッド構造を必要とします。
- パックされたYOLOヘッドは、クラス数とヘッドの深さを一貫して保つ必要があります。
  機能レベル。
- `YoloV26`は、グループ化された生の左/上/右/下のバウンディングボックスのヘッドと、クラススコアのヘッドを使用します。
- `Ssd`は、汎用的なSSDデコーダーではありません。あらかじめ用意された**4つのプロファイル**のみを処理します。
  コンパイル時に、完全な順序で指定されたロケーション/構成のヘッダー/ハードウェア/コンポーネントの署名を使用します。それ以外のヘッダーセットや順序が指定された場合、エラーが発生し、実際に使用された署名とサポートされている署名が表示されます。
  - **SSD300** (`dboxes300_coco`): 300×300の入力サイズ、特徴マップ
    `{38,19,10,5,3,1}`、セルごとの事前分布 `{4,6,6,6,4,4}`、信頼度チャンネルの順序
    `class*A + anchor`、クラスのスコアは、クラス次元に沿って**ソフトマックス**関数を適用することで算出されます（背景はインデックス0として含みます）。
  - **SSD-Mobile-300-v1** (`ssd_anchor_generator`): 300×300の入力、特徴
    マップ `{19,10,5,3,2,1}`、セルごとの事前分布 `{3,6,6,6,6,6}`、信頼度チャンネル
    順序 `anchor*C + class`、クラスごとの**シグモイド関数**を使用してクラスのスコアを算出します（背景は無視されます）。
  - **SSD-Mobile-320-v1** (`ssd_anchor_generator`): 320×320の入力、特徴
    マップ `{20,10,5,3,2,1}`、セルごとの事前分布 `{3,6,6,6,6,6}`、信頼度チャンネル
    順序 `anchor*C + class`、クラスごとの**シグモイド関数**を使用してクラスのスコアを算出します（背景は無視されます）。
  - **SSDlite-Mobile-320-v1**（TorchVision `DefaultBoxGenerator`）：320×320
    入力、特徴マップ `{20,10,5,3,2,1}`、各レベルでセルあたり6つの事前ボックス、
    位置合わせの順序 `anchor*4 + {dx,dy,dw,dh}`、信頼度の順序
    `anchor*C + class`、および背景を含むすべての91クラスに対する**ソフトマックス**によるクラススコア。

  すべてのレシピでは、グループ化されたレベルごとの位置合わせヘッド（深さ =
  `4 * priors-per-cell`）とクラス-信頼度ヘッド（深さ =
  `num_classes * priors-per-cell`）がペアになっており、FasterRcnnBoxCoderの分散スケーリング
  (`scale_xy 0.1`、`scale_wh 0.2`）、および**ストレッチ**（異方性）による前処理リサイズが使用されます。スコア活性化はレシピによって固定されます（デバイス上のデコーダーと一致します）。役割ごとにグループ化されたレイアウトは自動的に選択されます。`decode_type_option`を`Auto`のままにします。グループ化されていないレイアウトトークンは拒否されます。

  **モデルフレームは、ヘッドのジオメトリだけでなく、プロファイルの一部です。** SSD300-v1とSSD-Mobile-300-v1では、300×300が必要です。両方の320-v1プロファイルでは、320×320が必要です。解決された前処理リサイズターゲット、またはそれ以外のサイズのモデル次元のオーバーライドは、ビルド時に拒否されます。これは、事前ボックステーブルとストレッチによる逆投影が、そのフレームでのみ有効であるためです。

  生の/スタンドアロンの`SimaBoxDecode`の構築では、リサイズモードは決して生成されません。上流の`Preproc`メタデータ要件を維持するか、明示的な生のオーバーロードを使用して、外部で実行される`ResizeMode::Stretch`をアサートします。LetterboxとCropは拒否されます。

  **`num_classes`契約。** エンコードされたクラス数は、常に信頼度ヘッドの深さ（`conf_depth / priors-per-cell`、インデックス0に背景が含まれる）から導き出されます。SSD300-v1では、準備された81から8への連続したプレフィックス選択が許可されます。他の3つのプロファイルでは、正確なエンコードされた数が必須です。無効な選択は、ビルド時に拒否されます。設定しない場合は、プロファイルのデフォルトが使用されます。
- `Detr` は、最大のヘッド深度からクラスチャネルを推論し、有効なものを必要とします。
  クラス：次元。
- `EffDet`、`RcnnStage1`、および`Centernet`は、それぞれのモデルファミリーの規約を使用します。
  YOLOデコードタイプのルーティングを経由させないでください。
- `*-seg` デコードタイプは、ボックスの先頭部分の出力と、タスク固有のマスクデータを生成します。

カスタムモデルパックが、いずれかの完全な順序付きシグネチャと一致しない場合は、マッチャーを弱体化させるのではなく、新たに明示的にサポートするプロファイルを作成してください。

## Pythonに関するメモ

Pythonからモデルのオプションを設定する場合、文字列ではなく、型付きの列挙型を使用できる場合はそちらを使用してください。

```python
opt = pyneat.ModelOptions()
opt.decode_type = pyneat.BoxDecodeType.YoloV8
```

モデルのタスクに合ったヘルパーを使用して、出力を解析します。

```python
outputs = model.run([image])

boxes = pyneat.decode_bbox(outputs)[0].to_numpy()

pose = pyneat.decode_pose(outputs)[0]
pose_boxes = pose.boxes.to_numpy()
keypoints = pose.keypoints.to_numpy()

seg = pyneat.decode_segmentation(outputs)[0]
seg_boxes = seg.boxes.to_numpy()
masks = seg.masks.to_numpy()
```
