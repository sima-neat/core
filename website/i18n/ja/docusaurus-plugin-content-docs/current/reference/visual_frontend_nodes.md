---
title: "EV74 ビジュアルフロントエンドノード"
description: "FeatureHistogram、GriderFast、TrackDescriptor、TrackKLT のパブリック Graph 契約"
sidebar_position: 8
---

# EV74 ビジュアルフロントエンドノード

EV74 の演算には、パブリックな `Graph` ノードファクトリを使用します。Core、Scheduler、Runner、Kernels、EV74 ファームウェアは整合するパッケージセットをインストールしてください。旧グラフ ID はネイティブファームウェアの互換性チェックには使えません。アプリケーションにディスパッチャー ID は不要です。

品質オプションは既存のソースコードでの使用方法とデフォルトの出力位置を維持しますが、`TrackKLTOptions` の C++ レイアウトを変更するため、既存の C++ バイナリとの ABI 互換性はありません。C++ アプリケーションと Python 拡張を、対応する Core バージョンに対して再ビルドしてください。v2 トラッカーリクエストには、対応する Kernels、Runner、Scheduler、EV74 ファームウェアの組み合わせを使用します。既存のネイティブ v1 リクエストは引き続きサポートされますが、v2 リクエストが古いファームウェアと互換になるわけではありません。

| ノードファクトリ | 用途 |
| --- | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | グレースケールヒストグラム |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | グリッド分布型 FAST 特徴点 |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | FAST 特徴点と BRIEF 記述子 |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | ピラミッド KLT 追跡と任意の現フレーム特徴検出 |

## テンソル契約

画像テンソルの論理形状は `[B,H,W]` で、`B` は `batch_size` です。バッチを画像の高さにまとめないでください。デバイス入力は独立した割り当て、または有効な共有ストレージビューを使用できます。CPU コピーによる回避策は不要です。

| ノード | 入力 | 既定のパブリック出力 |
| --- | --- | --- |
| `FeatureHistogram` | UInt8 `[B,H,W]` | Int32 `[B,256]` |
| `GriderFast` | UInt8 `[B,H,W]` | Int32 `[B,1+3*max_features]` |
| `TrackDescriptor` | UInt8 `[B,H,W]` | Int32 `[B,1+3*max_features]`; Int32 `[B,max_features,8]` |
| `TrackKLT` | UInt8 `[B,H,W]`; UInt8 `[B,H,W]`; Int32 `[B,num_points,2]` | Float32 `[B,num_points,2]`; Int32 `[B,num_points,1]` |

特徴リストはバッチ項目ごとに `[count,x0,y0,score0,...]` です。先頭の `count` レコードのみを読み取ってください。未使用容量は特徴リストではありません。TrackDescriptor では `descriptor_words == 8` が必須です。記述子の各行は、選択された特徴の同じ行に対応します。

## グリッド選択と境界

`grid_x` と `grid_y` は画像全体をセルに分割します。容量 `K`、セル数 `M` の場合、各セルの枠は `K / M`、行優先で先頭の `K % M` セルにはさらに 1 枠です。各セルで強いスコアを選び、全体をスコア降順、同点は `(y,x)` 昇順に並べます。疎なセルの余りは貸し出しません。`min_px_dist` はセル境界を越えて近い点を抑制し、距離が等しい場合は許可します。抑制後の補充はないため、結果は `max_features` 未満になることがあります。

通常の FAST は `x=3`、`y=3`、`x=W-4`、`y=H-4` を含みます。TrackDescriptor には 31×31 の BRIEF パッチ全体が必要なので、中心は各端から 15 ピクセル以上離れている必要があります。行 3 に記述子がないのは意図した動作です。

## KLT の入力と出力

`TensorList` は **前の画像、現在の画像、入力点** の順に渡します。この直接のパブリック複数入力経路では、空でない論理 `route.name` は設定された位置と一致する必要があり、リストを並べ替えません。正しい順の無名テンソルも有効です。入力不足、名前の矛盾、型・形状の誤り、アクセス不能な範囲は、後続フレームも含めて送信前に拒否されます。

互換性のある出力テンソルを再利用する際は、浅いコピーを作り、新しいパブリック位置に合わせて `route.name` だけを変更または消去します。ストレージ、オフセット、物理経路メタデータは保持してください。この規則は内部グラフ接続を改名せず、明示的なモデル入力経路プロセッサーを置き換えません。

TrackKLT の既定出力は点とステータスです。`detect_new_features != 0` は **現在** の画像の検出特徴を追加します。検出が無効なら隠れた特徴出力は割り当てません。`publish_error = true` は既存出力の後に Float32 `[B,num_points,1]` を追加します。検出なしでは位置 2、検出ありでは位置 3 です。既存出力の位置は変わりません。

### C++ の例

`640×480` の UInt8 画像ベクトルを 2 つと、`64×2` の Int32 初期座標を用意します。以下の入力は独立した EV74 ストレージを持ち、戻り値は点、ステータス、誤差です。

```cpp
#include <neat.h>
#include <cstdint>
#include <vector>

using namespace simaai::neat;

TensorList track_pair(const std::vector<std::uint8_t>& previous,
                     const std::vector<std::uint8_t>& current,
                     const std::vector<std::int32_t>& xy) {
  constexpr int width = 640, height = 480, count = 64;
  auto image = [&](const auto& pixels, const char* name) {
    auto tensor = Tensor::from_vector(pixels, {1, height, width}, TensorMemory::EV74);
    tensor.layout = TensorLayout::HW;
    tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H,
                            TensorAxisSemantic::W};
    tensor.route.name = name;
    return tensor;
  };
  auto points = Tensor::from_vector(xy, {1, count, 2}, TensorMemory::EV74);
  points.route.name = "input_points";
  TensorList inputs{image(previous, "prev_image"), image(current, "cur_image"), points};

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = input.max_width = width;
  input.height = input.max_height = height;
  input.depth = input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "prev_image";
  input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                        "storage=(string)tensorbuffer";

  TrackKLTOptions klt;
  klt.width = width;
  klt.height = height;
  klt.num_points = count;
  klt.win_half = 7;
  klt.max_level = 2;
  klt.max_iters = 20;
  klt.publish_error = true;

  Graph graph;
  graph.add(nodes::Input(input));
  graph.add(nodes::TrackKLT(klt));
  graph.add(nodes::Output());
  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  Run run = graph.build(inputs, options);
  TensorList outputs;
  try {
    outputs = run.run(inputs, 30000);
  } catch (...) {
    run.close();
    throw;
  }
  run.close();
  return outputs;
}
```

## 品質制御

| オプション | 既定値 | 意味 |
| --- | --- | --- |
| `publish_error` | `false` | 最終座標の平均絶対輝度残差を公開します。`output_error_name` の既定値は `output_error` です。 |
| `error_threshold` | `-1.0` | `-1` は無効、それ以外は `[0,255]` の有限なしきい値です。残差がしきい値を超えると拒否し、等しい場合は許可します。誤差の公開なしでもフィルターは動作します。 |
| `min_eig_threshold` | `0.0` | `0` は無効、それ以外は有限で正の正規化最小固有値しきい値です。テクスチャの条件判定に使用します。 |
| `require_convergence` | `false` | 最終の原解像度更新に収束条件を要求します。false の場合、反復上限に達しただけでは失敗としません。 |

残差は輝度差であり、ピクセル変位や信頼度ではありません。返された座標で評価し、`+inf` は有効な残差を評価できなかったことを示します。テクスチャ指標は中央差分勾配行列の最小固有値を窓面積と `255²` で割ります。OpenCV のしきい値と同じ尺度ではありません。代表的な校正データでしきい値を選び、誤追跡の拒否と有効な追跡の保持率を両方検証してください。

座標を使う前にステータスを確認します。解法開始前の無効な初期点は元の座標とステータス 0 を保持します。有限の計算結果が画像外にある場合や品質しきい値で拒否された場合は、計算座標とステータス 0 が保持されることがあります。既定ステータスは解法・幾何の妥当性であり、実世界の対応の正しさの証明ではありません。

## 動き、境界、固定点数

改良中の位置が現在の画像タイルを出ると、タイルを再中心化して再読み込みします。±4 ピクセルの探索領域に黙って制限しません。再読み込みと反復は有限です。境界タイルは画像内 DMA のみを使い、端の画素を複製します。旧来のピラミッド依存の除外帯はなくなりますが、任意の動き、遮蔽、曖昧な模様での追跡を保証するものではありません。

Core の初期点は Int32、結果は Float32 です。`num_points` はグラフ構築時に固定されます。未使用の位置には `(-1,-1)` などの無効座標を与え、ステータスを確認します。`(0,0)` は有効な端の座標であり、無効マーカーではありません。両画像のピラミッドは呼び出しごとに再構築し、前フレームの再利用は行いません。

## Python オプション

Python も同じオプションを公開します。`Tensor.from_numpy(..., memory="ev74")` で入力を作り、入力順を保持して C++ と同様にパブリックノードを構築します。

```python
import pyneat

options = pyneat.TrackKLTOptions()
options.width, options.height = 640, 480
options.num_points = 64
options.win_half, options.max_level, options.max_iters = 7, 2, 20
options.publish_error = True
options.error_threshold = -1.0
options.min_eig_threshold = 0.0
options.require_convergence = False
node = pyneat.nodes.track_klt(options)
```

## 検証

設定したポート名は空でなく一意である必要があります。無効な寸法、点数、窓、ピラミッド段数、記述子サイズ、非有限または範囲外の品質パラメーターは拒否されます。検証を迂回したり、値の変換のためにテンソル型だけを変更したりしないでください。

## 正確性テストの実行

DevKit 上で `CORE_TEST_DIR` をインストール済み Core テストディレクトリまたはネイティブビルドのテストディレクトリに設定してから実行します。

```bash
ctest --test-dir "$CORE_TEST_DIR" \
  -R '^(grider_fast|track_descriptor|track_klt)_ev_test$' --output-on-failure
```

各登録テストはカーネルごとの全ケースを実行します。GriderFast はグリッド・距離と FAST 境界、TrackDescriptor は正確な記述子対応も、TrackKLT は独立入力、フレーム再利用、検証、品質、動き、境界、検出器を確認します。別の高速プリセットはヒストグラム、検出なし KLT、バッチ拒否のスモークテストであり、数値検証の代わりにはなりません。

## Metoak 深度（C++）

`nodes::MetoakDepth` は既存の `MetoakDepthOptions` API を復元します。
6 個の名前付きテンソルを受け取り、RGB、深度、XYZ を同時に公開します。
カメラの取得ではなく EV74 計算を行うため、カメラなしで合成テンソルを使用できます。

上記のバッチ対応ノードとは異なり、Metoak は常にバッチ 1 で、先頭にバッチ次元を付けません。
`W` は `[8,2048]`、`H` は `[8,1536]` の偶数に設定します。
標準のルート名を維持し、以下の順序で入力を渡してください。

| テンソル | データ型 | 形状 |
| --- | --- | --- |
| y_src | UInt8 | [H,W] |
| u_src, v_src | UInt8 | [H/2,W/2] |
| disp_src | UInt16 | [H,W] |
| bf_mm_src | Float32 | [1] |
| proj_src | Float32 | [3] |
| rgb_dst | UInt8 | [H,W,3] |
| depth_dst | UInt16 | [H,W] |
| points_dst | Float32 | [H,W,3] |

`y_src`、`u_src`、`v_src` は独立した I420 プレーンです。視差の倍率は 32 です。
各フレームに正の有限値 `bf_mm_src`（基線長 × 焦点距離、mm）と
`proj_src = {fx_fy, cx, cy}` を渡します。`fx_fy` は正、3 値すべては有限である必要があります。
キャリブレーションはフレーム間で変更できます。

深度は mm 単位に丸められ、UInt16 の上限で飽和します。XYZ は丸め前の深度を m 単位で使用します。
視差がゼロの場合、深度はゼロ、XYZ は NaN です。深度が主出力の説明であっても、3 出力すべてを利用できます。

```cpp
MetoakDepthOptions depth;
depth.width = 640;
depth.height = 360;
graph.add(nodes::MetoakDepth(depth));
```

既存の `debug`、`num_buffers`、`element_name` フィールドは引き続き使用できます。
`num_buffers = 0` はグラフの既定容量を使用し、正の値はこのステージの実行容量を選択します。
復元される公開 API は C++ のみであり、Python の Metoak ファクトリは追加しません。
