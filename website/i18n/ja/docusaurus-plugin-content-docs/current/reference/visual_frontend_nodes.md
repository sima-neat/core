---
title: "EV74 ビジュアルフロントエンドノード"
description: "FeatureHistogram、GriderFast、TrackDescriptor、TrackKLT、MetoakDepth の Neat Graph 使用方法"
sidebar_position: 8
---

# EV74のビジュアルフロントエンドノード

Neat は、EV74のビジュアル・フロントエンドのグラフを通常の`Graph`ノードとして公開します。パブリックなノードファクトリとオプション構造体を使用し、アプリケーションコードから`processcvu`、ConfigManager、またはディスパッチャーAPIを直接呼び出さないでください。

| ノードファクトリ | グラフ名 | グラフID | 目的 |
| --- | --- | ---: | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | `feature_histogram` | 235 | グレースケール画像のヒストグラム |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | `grider_fast` | 236 | グリッド分布型 FAST 特徴 |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | `track_descriptor` | 237 | FAST特徴量と記述子 |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | `track_klt` | 238 | ピラミッド型KLTトラッキング。検出された代替特徴点を使用する場合あり |
| `nodes::MetoakDepth` | `simor_depth_map` | 20 | I420 と視差から RGB、距離深度、XYZ を生成 |

グラフIDは、診断やファームウェア/パッケージの整合性チェックに役立ちます。アプリケーションコードでは必須ではありません。

## テンソル縮約

特徴抽出と追跡のテンソルは、**論理的なバッチ形状**を使用します。`batch_size == B`の場合、グレースケール画像は`[B,H,W]`であり、`[B*H,W]`ではありません。ランタイムは、EV74トランスポートパッキングを内部的に処理します。

| ノード | 入力 | 公開出力 |
| --- | --- | --- |
| `FeatureHistogram` | `input_image`: UInt8 `[B,H,W]` | `output_hist`: Int32 `[B,256]` |
| `GriderFast` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]` |
| `TrackDescriptor` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]`; `output_descriptors`: Int32 `[B,max_features,8]` |
| `TrackKLT` | `prev_image`: UInt8 `[B,H,W]`; `cur_image`: UInt8 `[B,H,W]`; `input_points`: Int32 `[B,num_points,2]` | `output_points`: Float32 `[B,num_points,2]`; `output_status`: Int32 `[B,num_points,1]`; さらに、`output_features`: Int32 `[B,1 + max_features*3]` は、`detect_new_features != 0` の場合にのみ出力されます |

特徴量リストのテンソルは、このバッチごとのレイアウトを使用します。

```text
[count, x0, y0, score0, x1, y1, score1, ...]
```

現在のディスクリプタグラフには、`descriptor_words == 8` が必要です。それを変更すると、EV74 ABI の変更となり、処理前に拒否されます。

## C++のクイックスタート

```cpp
#include <neat.h>

#include <cstdint>
#include <vector>

using namespace simaai::neat;

Tensor make_gray_batch(int width, int height, int batch) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * batch);
  // Fill pixels in batch-major order: b*height*width + y*width + x.
  auto tensor = Tensor::from_vector(pixels, {batch, height, width}, TensorMemory::EV74);
  tensor.layout = TensorLayout::HW;
  tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H, TensorAxisSemantic::W};
  tensor.route.name = "input_image";
  tensor.route.segment_name = "input_image";
  return tensor;
}

int main() {
  constexpr int width = 320;
  constexpr int height = 240;
  constexpr int batch = 2;

  Graph graph;

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = width;
  input.height = height;
  input.depth = 1;
  input.max_width = width;
  input.max_height = height * batch; // transport capacity; public tensor remains [B,H,W]
  input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "input_image";

  graph.add(nodes::Input(input));

  GriderFastOptions fast;
  fast.width = width;
  fast.height = height;
  fast.batch_size = batch;
  fast.max_features = 64;
  fast.threshold = 30;
  graph.add(nodes::GriderFast(fast));

  graph.add(nodes::Output());

  RunOptions run_opt;
  run_opt.output_memory = OutputMemory::Owned;

  Tensor image = make_gray_batch(width, height, batch);
  Run run = graph.build({image}, run_opt);
  TensorList outputs = run.run({image}, /*timeout_ms=*/30000);
  run.close();
}
```

## 3つの入力を持つKLT

`TrackKLT` は、テンソルセット（前の画像、現在の画像、および入力点）を受け取ります。オプションフィールドに一致するように、ルートに名前を付けてください。

```cpp
TrackKLTOptions klt;
klt.width = 320;
klt.height = 240;
klt.batch_size = 2;
klt.num_points = 32;
klt.max_features = 64;
klt.detect_new_features = 1; // publish output_features as the third output

graph.add(nodes::TrackKLT(klt));
```

`detect_new_features == 1` が実行された場合の、想定される公開出力：

```text
output_points   Float32 [2,32,2]
output_status   Int32   [2,32,1]
output_features Int32   [2,193]
```

`detect_new_features == 0` が実行されると、Neat は `output_points` と `output_status` のみを公開し、EV で確認可能な機能バッファーは内部ランタイム割り当てのままになります。

## 6 入力の Metoak 深度

`MetoakDepth` は `simor_depth_map`（グラフ 20）を使用する C++ 専用ノードです。生の SIMOR カメラフレームではなく、デコード済み I420 プレーン、生の視差、フレームごとのキャリブレーションを受け取ります。このノードの前で、アプリケーションまたは ROS アダプターが SIMOR をアンパックし、キャリブレーションを選択してください。Neat はそのアダプターを置き換えません。

偶数の `width` を `[8,2048]`、偶数の `height` を `[8,1536]` に設定します。S315 のネイティブ深度解像度は `640x360` です。バッチは 1 に固定され、先頭のバッチ次元はありません。次の標準ルート名と入力順序を維持してください。別名は契約コンパイル時に拒否されます。

| 入力ルート | 型 | 形状 | 意味 |
| --- | --- | --- | --- |
| `y_src` | UInt8 | `[H,W]` | I420 Y |
| `u_src` | UInt8 | `[H/2,W/2]` | I420 U |
| `v_src` | UInt8 | `[H/2,W/2]` | I420 V |
| `disp_src` | UInt16 | `[H,W]` | 生の視差；固定サブピクセルスケール 32 |
| `bf_mm_src` | Float32 | `[1]` | キャリブレーション済み基線長 × 焦点距離（mm） |
| `proj_src` | Float32 | `[3]` | 投影 `{fx_fy,cx,cy}` |

| 出力ルート | 型 | 形状 | 意味 |
| --- | --- | --- | --- |
| `rgb_dst` | UInt8 | `[H,W,3]` | インターリーブ RGB |
| `depth_dst` | UInt16 | `[H,W]` | 深度（mm）；0 は無効 |
| `points_dst` | Float32 | `[H,W,3]` | インターリーブ XYZ（メートル）；NaN は無効 |

3 つの出力は常に一緒に公開されます。`depth_dst` は主境界の記述であり、出力セレクターではありません。キャリブレーションの BF と焦点距離は正の有限値、主点座標は有限値である必要があります。

入力表に一致する名前付きテンソル 6 個を EV74 メモリに用意して Graph を構築します。この例はノードを設定します。実際のデコード済みフレームとキャリブレーションのテンソルはアダプターから提供してください。

```cpp
#include <neat.h>

using namespace simaai::neat;

// inputs contains the six named, decoded EV74 tensors from the table above.
Run build_metoak_depth(const TensorList& inputs) {
  Graph graph;
  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.caps_override =
      "application/vnd.simaai.tensor, representation=(string)tensor-set, storage=(string)tensorbuffer";
  graph.add(nodes::Input(input));

  MetoakDepthOptions depth;
  depth.width = 640;
  depth.height = 360;
  graph.add(nodes::MetoakDepth(depth));
  graph.add(nodes::Output());

  RunOptions options;
  options.output_memory = OutputMemory::Owned;
  return graph.build(inputs, options);
}
```

グラフ 20 を含む対応する Internals と EV74 ファームウェアでのみこの Graph を実行してください。下記の特徴抽出/追跡の検証コマンドは他の 4 グラフを対象とし、`MetoakDepth` は対象外です。

## Pythonの表面

4 つの特徴抽出/追跡ノードには、C++ のオプション/ファクトリ形式に対応する Python バインディングがあります。`MetoakDepth` に Python バインディングはまだありません。オプションオブジェクトを作成し、公開設定を行い、ノードを `Graph` に追加してください。

```python
import numpy as np
import pyneat

width, height, batch = 320, 240, 2

opt = pyneat.GriderFastOptions()
opt.width = width
opt.height = height
opt.batch_size = batch
opt.max_features = 64
print(opt.summary())

graph = pyneat.Graph()
input_opt = pyneat.InputOptions()
input_opt.payload_type = pyneat.PayloadType.Tensor
input_opt.format = pyneat.Format.UINT8
input_opt.width = width
input_opt.height = height
input_opt.max_width = width
input_opt.max_height = height * batch
input_opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
input_opt.buffer_name = "input_image"

graph.add(pyneat.nodes.input(input_opt))
graph.add(pyneat.nodes.grider_fast(opt))
graph.add(pyneat.nodes.output())

image_np = np.zeros((batch, height, width), dtype=np.uint8)
image = pyneat.Tensor.from_numpy(image_np, memory="ev74")
image.layout = pyneat.TensorLayout.HW
# If setting route metadata from Python in a custom app, keep it aligned with
# the option names used above.
```

## 安全確認

4 つの特徴抽出/追跡ノードは、EV へのディスパッチ前にグラフの範囲を検証します。以下の場合は拒否します。

- 0以下の次元またはカウント。
- サポートされていないバッチサイズです。
- `[0,255]`の範囲外の閾値。
- 重複または空のテンソル名。
- `TrackDescriptorOptions.descriptor_words != 8`;
- 無効な KLT ウィンドウ、レベル、および検出モードの値です。
- プリディスパッチネゴシエーション中に、ランタイムで使用するテンソルが不足している。

これは重要なことです。不正なバッファが EV74 に悪影響を及ぼす可能性があるからです。検証に失敗した場合は、ホスト側のエラーとして扱い、Node の契約パスを迂回しないでください。

## 高速な検証コマンド

迅速な顧客対応を可能にする DevKit は次のとおりです。

```bash
ctest --test-dir /workspace/core_graph_changes/build/tests \
  -R visual_frontend_ --output-on-failure
```

実行結果：

- 4つの視覚的なグラフを、それぞれ `320x240`、`batch_size=2`、`detect_new_features=1` の設定で表示します。
- 特定の KLT 検出回避 ABI チェック。
- 不正なバッチ入力がないことを確認する、出荷前のネガティブチェック。
  EV74より前に却下されました。
