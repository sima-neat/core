---
title: "EV74 ビジュアルフロントエンドノード"
description: "機能ヒストグラム、グリダーファスト、トラック記述子、およびトラックKLTの利用状況を、顧客向け形式のNeatグラフで表示します。"
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

グラフIDは、診断やファームウェア/パッケージの整合性チェックに役立ちます。アプリケーションコードでは必須ではありません。

## テンソル縮約

すべてのテンソルは、**論理的なバッチ形状**を使用します。`batch_size == B`の場合、グレースケール画像は`[B,H,W]`であり、`[B*H,W]`ではありません。ランタイムは、EV74トランスポートパッキングを内部的に処理します。

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

## Pythonの表面

Python APIは、C++のオプション/ファクトリ形式を模倣しており、意図的に階層構造になっています。オプションオブジェクトを作成し、公開設定を行い、ノードを`Graph`に追加します。

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

これらのノードは、EV（電気自動車）の配車前に、グラフの範囲を検証します。以下の場合は拒否します。

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
