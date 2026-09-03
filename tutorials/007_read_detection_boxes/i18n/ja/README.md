# 007 モデルの出力から検出ボックスを読み取る

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | yolo_v8s |
| Labels | postprocessing, boxdecode, detection |

## Concept

`SimaBoxDecode` を使用して、生のモデル出力を、使用可能なバウンディングボックスにデコードします。これには、閾値処理、NMS（Non-Maximum Suppression）、および座標マッピングが含まれており、これらはすべて1つの後処理段階に組み込まれています。その後、結果を、解析済みのボックスまたは生のパックされたバイトバッファとして読み取ります。

## Walkthrough

検出器は、直接バウンディングボックスを返しません。その生の出力は、意味のあるものになる前に、しきい値処理、非最大抑制、および座標マッピングが必要な特徴マップのスタックです。`SimaBoxDecode`は、これらすべてを1つの最適化されたステップで実行する後処理ステージであり、推論テンソルをソース画像のピクセル単位の最終的な検出に変換します。

この章では、そのデコードを構成します。つまり、`decode_type`を使用してモデルファミリーを選択し、スコアしきい値で信頼度を制御し、NMS IoUしきい値で重複を抑制し、`top_k`で出力を制限します。その後、モデルを実行し、検出された数を読み取ります。最終的には、構成された検出器パイプラインと、その出力から読み取った検出数のカウントが得られ、さらに（以下に示す「実践」の参照）完全なワイヤ形式も得られるため、任意のランタイムでバウンディングボックスを自分で解析できます。

### デコードを構成する {#step-configure-decode}

これらのオプションは、入力の契約と後処理の動作の両方を設定します。`decode_type`（ここでは`YoloV8`）は、モデルファミリーのデコードパスを選択します。信頼度しきい値は、NMSの前に弱い候補を削除します。NMS IoUしきい値は、重複するバウンディングボックスの結合をどの程度積極的に行うかを制御します。`top_k`は、決定的な下流のコストのために最終的なカウントを制限します。そして、`boxdecode_original_width`/`boxdecode_original_height`は、デコードされた座標をソース画像のピクセルにマッピングします。これらの各項目の調整に関するガイダンスは、以下に示す「実践」にあります。

**C++:** `decode_type`は、`BoxDecodeType::YoloV8` enumを受け取ります。しきい値/NMS/`top_k`の値は、`Model::Options`ではなく、後で`stages::BoxDecodeOptions`を通じて渡されます。

**Python:** `decode_type`は、`pyneat.BoxDecodeType.YoloV8` enumを受け取り、`score_threshold`、`nms_iou_threshold`、`top_k`、および`boxdecode_original_width`/`boxdecode_original_height`は、`ModelOptions`に直接設定されます。（`score_threshold`とC++の`detection_threshold`は、同じ制御を行います。命名に関する注意については、「実践」を参照してください。）

### モデルを構築する {#step-load-model}

アーカイブとオプションから`Model`を構築すると、デコード構成がモデルにバインドされ、そこから派生した推論および後処理ステージで上記の設定が使用されます。

### 前処理、推論、およびデコードを実行する {#step-run-decode}

ここでは、フレームが前処理、MLA推論、およびボックスデコーダーを通過し、検出出力が生成されます。

**C++:** 処理フローは段階的に明確に定義されています。`stages::Preproc` は入力テンソルを生成し、`stages::Infer` はモデルを実行し、`stages::BoxDecodeOptions`（`detection_threshold = 0.55`、`nms_iou_threshold = 0.5`、`top_k = 100` を含む）が、次に実行されるデコードを構成します。

**Python:** `model.run([tensor])` は、構成された処理フロー全体を1回の呼び出しで実行し、`TensorList` を返します。BoxDecode がモデルの処理フローに組み込まれている場合、最初のテンソルはパックされた `BBOX` 出力になります。

### ボックスの読み込み {#step-read-boxes}

最後に、デコードの出力を、実際に使用できる形式に変換します。

**C++:** `stages::BoxDecodeResults(...)` は `BoxDecodeResultList` を返し、最初の結果の `boxes` ベクトルは、すでにソースピクセルにクランプされた `{x1, y1, x2, y2, score, class_id}` にパースされているため、`decoded.boxes.size()` が検出の数になります。

**Python:** 結果は、`outputs[0]` 内の単一の `BBOX` `uint8` テンソルです。最初の4バイト（リトルエンディアン）は、検出の数です（`struct.unpack_from("<I", buf, 0)`）。完全なレコードレイアウトは、「実践」に記載されています。ランタイムが BoxDecode を `model.run` に組み込まない場合、返される `TensorList` には、生の特徴マップのヘッドが含まれます。

## In Practice

`SimaBoxDecode` は、`BBOX` というタグが付けられた単一の出力テンソルを出力します。このテンソルには、ランタイムパーサーが浮動小数点数の検出に解釈する、パックされたバイトバッファが含まれています。この2層の契約（ワイヤバッファとパースされた `Box` レコード）を理解することが、PythonまたはC++のいずれかから出力を読み取るための鍵となります。

### BBOX テンソル

デコードステージは、入力フレームごとに1つの `BBOX` テンソルを生成します。

| フィールド | 値 |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | ランク1: `[N_bytes]`。ここで、`N_bytes` は、モデルアーカイブにパックされたバッファの容量です（たとえば、標準の YOLOv8 パックでは `[20160]`）。 |

テンソルの形状は、**バイト数**であり、検出の数ではありません。パックされたバイトには、小さなヘッダーと、固定サイズのボックスレコードの連続した配列が含まれています。`N_bytes` は、モデルアーカイブの `buffers.input[0].size` フィールド（ボックスデコードステージの構成JSON内）によって決定され、デコーダーが1つのフレームで出力できる最大検出数を制限します（ランタイムの次元がパッケージ化された値とどのように相互作用するかについては、「契約のオーバーライド」を参照）。

### パックされたワイヤ形式

`uint8` バッファは、リトルエンディアン形式でレイアウトされています。

```
offset  size  content
------  ----  -------
  0      4    uint32  N = number of valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   (trailing bytes up to buffer capacity are padding, ignored)
```

各`RawBox`レコードは24バイトです。

| レコード内のオフセット | サイズ | 型 | フィールド | 意味 |
|---|---|---|---|---|
| 0 | 4 | int32 | `x` | ソースピクセルにおける左上のx座標 |
| 4 | 4 | int32 | `y` | ソースピクセルにおける左上のy座標 |
| 8 | 4 | int32 | `w` | ソースピクセルにおける幅 |
| 12 | 4 | int32 | `h` | ソースピクセルにおける高さ |
| 16 | 4 | float32 | `score` | NMS後の検出信頼度（`[0.0, 1.0]`における値で、`detection_threshold`の値でフィルタリングされる） |
| 20 | 4 | int32 | `class_id` | 予測されたクラスID（モデル定義、0から始まるインデックス、クラス名マップはモデルアーカイブのメタデータに格納） |

1つのレコードに一致する標準的なPython `struct`形式は`"<iiiifi"`です（リトルエンディアン、4つの符号付き整数、1つの浮動小数点数、1つの符号付き整数）。

ランタイムの解析ヘルパー（`parse_bbox_bytes` / `decode_bbox_tensor`（`include/pipeline/DetectionTypes.h`内）、`tests/unit_testing/unit_detection_types_bbox_test.cpp`はワイヤ契約を固定します）は、各`RawBox`を、後続のコードで使用するための`Box`構造体に拡張します。

```cpp
struct Box {
  float x1, y1, x2, y2;  // x2 = x + w, y2 = y + h; clamped to [0, img_w|h]
  float score;
  int   class_id;
};
```

### 座標空間

`BBOX`からデコードされた座標は、**元の画像ピクセル**にあり、これは、`original_width` / `original_height`として渡された（またはモデルアーカイブにパッケージ化された）のと同じ座標系です。これらは`[0, 1]`に正規化されておらず、モデルの内部のレターボックス形式の入力空間で表現されていません。パーサーは`(x1, y1, x2, y2)`を`[0, original_width]` / `[0, original_height]`にクリップするため、呼び出しコードはこれらをソースフレームに直接描画できます。

### 動作例

チュートリアルのランタイム構成（`original_width = 640`、`original_height = 640`、`top_k = 100`）と、標準のYOLOv8パック（boxdecode構成内の`buffers.input[0].size = 20160`）を使用すると、デコードされた単一のフレームは次のようになります。

- `out.kind == SampleKind.Tensor`
- `out.payload_tag == "BBOX"`
- `out.tensor.dtype == UInt8`、`out.tensor.shape == [20160]`
- バイト`[0:4]`はリトルエンディアンで`N`を表します。`0 <= N <= 100`は、`top_k = 100`のためです。`N`が`0`の場合、「このフレームで閾値を超える検出がない」という意味であり、0回反復して何も出力しません。
- バイト`[4 : 4 + 24 * N]`には有効な検出が含まれており、それ以降のすべてのバイトはゼロ/パディングであり、無視する必要があります。

Pythonでボックスを読み取るには、`struct.unpack_from`を使用します。

```python
import struct
payload = out.tensor.copy_payload_bytes()
count = struct.unpack_from("<I", payload, 0)[0]
for i in range(count):
    x, y, w, h, score, cls = struct.unpack_from("<iiiifi", payload, 4 + 24 * i)
    # (x, y, w, h) in source pixels; x2 = x + w, y2 = y + h
```

C++では、`stages::BoxDecode`ヘルパー関数は、この処理を済ませた`BoxDecodeResult`を返します。`result.boxes[i]`は、`(x, y, x+w, y+h)`から`(x1, y1, x2, y2)`がすでに設定され、画像に合わせてクリップされた`Box`です。

### オーバーライド契約：ランタイムの次元とパッケージ化されたモデルアーカイブのデフォルト値

`SimaBoxDecode`は、`decode_type`、`detection_threshold`、`nms_iou_threshold`、`top_k`、`original_width`、および`original_height`のパッケージ化されたデフォルト値を含む、トレーニング済みのモデルアーカイブから構築されます。パブリックコンストラクタは```cpp
SimaBoxDecode(const Model& model,
              const std::string& decode_type = "",
              int original_width = 0, int original_height = 0,
              double detection_threshold = 0.0,
              double nms_iou_threshold = 0.0,
              int top_k = 0);
```です。

そして、その Python 版である `pyneat.nodes.sima_box_decode(model, ...)` は、フィールドごとに単純な「肯定的な値は優先され、ゼロ/空の値は保持される」というルールを使用します。

> **命名に関する注意:** `detection_threshold` は、`SimaBoxDecode` のコンストラクタで使用される名前です。`ModelOptions.score_threshold` (Python のチュートリアルで使用) は、同じ引数に渡されます。これら 2 つの名前は、同じ基盤となる制御を指します。

| ランタイム引数 | 渡される値 | 動作 |
|---|---|---|
| `decode_type` | `""` (空) | モデルアーカイブ / モデルパス推論を保持 |
| `decode_type` | 空でない文字列 | この実行のためにモデルアーカイブの値を上書き |
| `original_width` / `original_height` | `0` | モデルアーカイブにパッケージ化された次元を保持 |
| `original_width` / `original_height` | 正の整数 | 有効な構成における `original_width` / `original_height` を書き換える |
| `detection_threshold` | `0.0` | モデルアーカイブにパッケージ化された閾値を保持 |
| `detection_threshold` | `> 0.0` | 上書き (YOLOv8 のクリフ警告もトリガー) |
| `nms_iou_threshold` | `0.0` | モデルアーカイブにパッケージ化された NMS IoU を保持 |
| `nms_iou_threshold` | `> 0.0` | 上書き |
| `top_k` | `0` | モデルアーカイブにパッケージ化されたトップ K を保持 |
| `top_k` | `> 0` | 上書き |

このルールは、フィールドごとに厳密に適用されます。

- **Python パス** — チュートリアルではすべてのフィールドが上書きされます。なぜなら、`ModelOptions` が正の値に設定されるからです。
- **C++ パス** — `read_detection_boxes.cpp` は `0.55f, 0.5f, 100` を渡します (したがって、`detection_threshold`、`nms_iou_threshold`、および `top_k` が上書きされます) さらに `bgr.cols, bgr.rows` を正の値で渡します (したがって、`original_width` / `original_height` も上書きされます)。

実用的な影響:

- モデルアーカイブが、ソースフレームとは異なる解像度でパックされている場合、`original_width` と `original_height` を明示的に渡して、座標がソースピクセルに一致するようにします。
- `detection_threshold` と `nms_iou_threshold` を `0.0` のままにしておくことは、モデルアーカイブの検証済みのデフォルトを取得する最も安全な方法です。意図的に再調整する場合にのみ上書きしてください。
- `detection_threshold` を低い値に設定する場合は、注意してください。値が低いほど、しきい値処理を通過する候補ボックスが多くなり、NMS のコストは、生き残ったボックスの数の 2 乗に比例して増加します。したがって、非常に低いしきい値は、後処理の計算量とレイテンシーを大幅に増加させる可能性があります。弱い検出を捉えるために必要な範囲までのみ値を下げ、`top_k` と組み合わせて、最悪の場合の数を制限します。

### デコードタイプとテンソルの契約

`BoxDecodeType` は型付き API (`simaai::neat::BoxDecodeType` / `neat.BoxDecodeType`) であり、デコードステージでは常に明示的に設定する必要があります。 以下のランタイムコントラクトは、`internals/gst_plugins/genericboxdecode_v2/gstneatboxdecode.cpp` (`infer_num_classes`、`infer_yolo_decoupled_classes`、`infer_yolo_packed_classes`、`compute_required_output_size`) から派生します。

主要なテンソルコントラクトルール：
- `yolov5` 検出以外の YOLO ファミリーのデコードタイプ (`yolo`、`yolov5-seg`、`yolov7*`、`yolov8*`、`yolov9*`、`yolov10*`):
  - 分離されたヘッド：クラスヘッドの深さは繰り返し可能で、`> 4` である必要があります。
  - パックされたヘッド：各ヘッドの深さは、`depth = 3 * (num_classes + 5)` を満たし、ヘッド間で一貫している必要があります。
- `yolov5` 検出：ストライド 8/16/32 の形状と `3 * (num_classes + 5)` の深さを持つ、未デコードの P3/P4/P5 パック済みヘッドが正確に 3 つ必要です。
- `yolo26`：4チャンネルの生の l/t/r/b バウンディングボックステンソルと、繰り返し可能なクラスヘッドの深さ `> 4` を持つ、分離されたグループ化されたヘッド。
- `detr`：クラスチャネルは、ヘッド全体の最大深度から推測され、`> 4` である必要があります。
- その他の非 YOLO デコードタイプ (`effdet`、`rcnn-stage1`、`centernet`): フォールバッククラス推論では、最大深度を使用し、`> 4` が必要です。
- セグメンテーションデコードトークン (`*-seg`) は、v2 でセグメンテーションのような出力サイズを有効にします（検出ごとにマスクペイロードを追加します）。

| API 列挙型 | バックエンド・トークン | 期待される契約 |
|---|---|---|
| `BoxDecodeType::Yolo` | `yolo` | YOLO 分離またはパックされた深度契約 |
| `BoxDecodeType::YoloV5` | `yolov5` | 未デコードの P3/P4/P5 パック済みヘッド 3 つ |
| `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLO 深度契約 + セグメンテーションパス |
| `BoxDecodeType::YoloV7` | `yolov7` | YOLO 分離またはパックされた深度契約 |
| `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLO 深度契約 + セグメンテーションパス |
| `BoxDecodeType::YoloV8` | `yolov8` | YOLO 分離またはパックされた深度契約 |
| `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLO 深度契約 + セグメンテーションパス |
| `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLO 分離またはパックされた深度契約 |
| `BoxDecodeType::YoloV9` | `yolov9` | YOLO 分離またはパックされた深度契約 |
| `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLO 深度契約 + セグメンテーションパス |
| `BoxDecodeType::YoloV10` | `yolov10` | YOLO 分離またはパックされた深度契約 |
| `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLO 深度契約 + セグメンテーションパス |
| `BoxDecodeType::YoloV26` | `yolo26` | YOLO26 グループ化された生の l/t/r/b バウンディングボックスヘッド + クラススコアヘッド |
| `BoxDecodeType::Detr` | `detr` | `num_classes = max(depth)` (必ず `> 4` であること) |
| `BoxDecodeType::EffDet` | `effdet` | フォールバック最大深度推論 (`> 4`) |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | フォールバック最大深度推論 (`> 4`) |
| `BoxDecodeType::Centernet` | `centernet` | フォールバック最大深度推論 (`> 4`) |

早期失敗動作:
- `stages::BoxDecodeOptions` は、デコードタイプを使用して明示的に構築する必要があります。
- `stages::BoxDecode(...)` および `nodes::SimaBoxDecode(...)` は、`BoxDecodeType::Unspecified` の場合に早期に失敗します。

デコードタイプを明示的に設定する:

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.5;
opt.top_k = 100;
```

```python
opt = neat.ModelOptions()
opt.decode_type = neat.BoxDecodeType.YoloV8
```

## Run

**Python** および **C++ (事前にビルドされたもの)** コマンドを、**Neat インストールルート** ( `share/` と `lib/` を含むディレクトリ) から実行します。**ソースからビルド** コマンドは、**リポジトリルート** から実行します。

**Python:**
```bash
python3 share/sima-neat/tutorials/007_read_detection_boxes/read_detection_boxes.py \
  --model /tmp/yolo_v8s.tar.gz --width 640 --height 640
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_007_read_detection_boxes \
  --model /tmp/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_007_read_detection_boxes
./build/tutorials-standalone/tutorial_007_read_detection_boxes \
  --model /tmp/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

期待される出力 (ボックスの数はフレームによって異なります。合成フレームではゼロになります):

```text
boxes=0
[OK] 007_read_detection_boxes
```

（Pythonビルドでは、`detections=...`が出力されます。ランタイムでBoxDecodeを`model.run`に接続していない場合は、`raw_output_heads=...`が出力されます。）この章のC++ソースを、カスタムの`CMakeLists.txt`を使用して独自のプロジェクトに統合する方法（追加のフォルダーは不要）については、ランディングページにある[チュートリアルの実行方法](/tutorials#compile-a-copy-yourself)を参照してください。

## ソースファイル
- C++: `tutorials/007_read_detection_boxes/read_detection_boxes.cpp`
- Python: `tutorials/007_read_detection_boxes/read_detection_boxes.py`
