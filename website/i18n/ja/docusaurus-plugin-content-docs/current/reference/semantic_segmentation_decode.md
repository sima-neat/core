---
title: "セマンティックセグメンテーションのデコード"
description: "SimaResizeArgmax または Model のセマンティックセグメンテーションオプションで、クラスごとのスコアを EV74 上でリサイズしたクラスマップに変換する"
sidebar_position: 9
---

# セマンティックセグメンテーションのデコード

`SimaResizeArgmax` は、セグメンテーションモデルのクラスごとのスコアを、指定した
サイズのクラスマップに変換します。出力ピクセルごとに各クラスのスコアをバイリニア
補間し、スコアが最大のクラスを返します。これは SegFormer などのモデルが MLA の後に
必要とするデコードで、`interpolate(mode="bilinear")` で目的のサイズへ補間してから、
クラス方向に `argmax` を取ります。`target_sizes` を指定した Hugging Face の
`SegformerImageProcessor.post_process_semantic_segmentation` と同じクラスマップが得られ、
リサイズ後のスコアを保存することはありません。

デコードは EV74 上で実行されます。A65 の実装やフォールバックはありません。

## モデルの出力をデコードする

`Model::Options::semantic_segmentation` を設定すると、モデルの dequantize または cast の
後処理がこのデコードに置き換わります。ノードは MLA のスコアをその場で読み取り、
保存チャネルがパディングされた MLA 出力にも対応します。モデルはスコアの代わりに
クラスマップを 1 つ返します。

```cpp
#include <neat.h>

using namespace simaai::neat;

Model::Options options;
options.semantic_segmentation.enable = true;
options.semantic_segmentation.output_width = 1280; // camera frame size
options.semantic_segmentation.output_height = 720;
Model model("segformer_b0_ade_512_mpk.tar.gz", options);
```

```python
import pyneat

options = pyneat.ModelOptions()
options.semantic_segmentation.enable = True
options.semantic_segmentation.output_width = 1280  # camera frame size
options.semantic_segmentation.output_height = 720
model = pyneat.Model("segformer_b0_ade_512_mpk.tar.gz", options)
```

これらのオプションは、モデルに必要な前処理オプションと組み合わせて使います。
モデルの出力は 1 つのクラスマップテンソル `[N,H,W]` で、
`model.info().selection.selected_post_kind` は `"resize_argmax"` になります。

- `output_width` と `output_height` を両方 0 のままにするとモデルの入力サイズを使います。
  元画像のすべてのピクセルにクラスを割り当てるには、両方を元のフレームサイズに設定します。
- 既定の `align_corners = false` は、PyTorch や Hugging Face と同じくピクセル中心を
  半ピクセルずらした座標を使います。角を揃えた補間で学習したモデルの場合にだけ
  `true` に設定してください。
- モデルの後処理は dequantize または cast のステージだけで構成され、
  `decode_type` は未設定でなければなりません。そうでない場合、`Model` のコンストラクターは
  `std::invalid_argument`（Python では `ValueError`）を送出します。

## グラフでノードを使う

C++ の `nodes::SimaResizeArgmax(...)` と Python の `pyneat.nodes.sima_resize_argmax(options)`
は、以下の契約を満たす任意のテンソルに同じデコードを実行します。次のフラグメントは、
1 つのスコアテンソル `scores` を 512 x 512 のクラスマップにデコードします。

```cpp
SimaResizeArgmaxOptions decode;
decode.output_width = 512;
decode.output_height = 512;

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaResizeArgmax(decode));
graph.add(nodes::Output());
auto run = graph.build(TensorList{scores});
TensorList class_map = run.run(TensorList{scores}, 1000);
```

## 入力と出力の契約

| ポート | テンソル |
| --- | --- |
| `scores` | `[h,w,C]` または `[N,h,w,C]`、クラスは最後の軸。INT8、UINT8、BF16、FP32。密、またはピクセルピッチが `C` より大きいストライド付き |
| `class_map` | `[H,W]` または `[N,H,W]`、密。`C` が 256 以下なら UINT8、それ以外は INT32 |

- 整数スコアは逆量子化せずに比較します。すべてのクラスに 1 つの正のスケールが
  適用される場合、逆量子化したスコアと同じクラスになります。軸ごとの量子化を持つ
  スコアや、スケールが正でないスコアは、グラフのビルド時に拒否されます。
- スコアが等しい場合は最も小さいクラスインデックスを選びます。BF16 と FP32 の
  スコアでは、補間結果の NaN が優先され、最初の NaN のクラスが選ばれます。
- パディングされた MLA 出力の余分なチャネルなど、保存上のパディングチャネルを
  クラスとして読むことはありません。
- MLA タイル形式のスコアは拒否されます。先にデテッセレーションしてください。論理的な
  CHW または NCHW の軸を持つスコアも拒否されます。クラスは最後の軸に置く必要があります。
- DevKit は、この演算を含む EV74 ファームウェアで動作している必要があります。古い
  ファームウェアでは、最初のフレームを送信する前に失敗します。
