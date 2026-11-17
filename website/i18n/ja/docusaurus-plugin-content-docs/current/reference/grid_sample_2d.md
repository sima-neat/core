---
title: 動的グリッドで特徴をサンプリングする
description: Neat グラフで A65 NEON GridSample2D を実行する
sidebar_position: 9
---

# 動的グリッドで特徴をサンプリングする

`SimaGridSample2D` は A65 でバイリニアサンプリングを行います。各入力サンプルには、特徴 `[N,H,W,C]`（FP32 または BF16）、次いで正規化された `(x,y)` 座標のグリッド `[N,Ho,Wo,2]`（FP32）を格納します。出力は特徴と同じ型でチャネルラストの `sampled` `[N,Ho,Wo,C]` です。全次元は正である必要があります。

## ビルドと実行

`grid_sample_2d` を含む、対応する Kernels、Runner、Scheduler、Core パッケージ一式を使用してください。この C++ 断片は `SimaNeat::sima_neat` にリンクしたアプリケーション内で実行します。グラフを構築する前に、連続 NHWC 特徴と NHW2 グリッドの値を 2 個のベクトルに格納してください。

```cpp
#include <neat.h>
using namespace simaai::neat;

std::vector<float> features(1 * 16 * 16 * 16, 1.0f);
std::vector<float> grid(1 * 32 * 8 * 2, 0.0f);
auto feature = Tensor::from_vector(features, {1,16,16,16}, TensorMemory::CPU);
feature.layout = TensorLayout::HWC;
auto coordinates = Tensor::from_vector(grid, {1,32,8,2}, TensorMemory::CPU);

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::SystemMemory;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";
SimaGridSample2DOptions sampling;
sampling.padding = SimaGridSample2DPadding::Zeros;
sampling.align_corners = false;
Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaGridSample2D(sampling));
graph.add(nodes::Output());
auto run = graph.build(TensorList{feature, coordinates});
auto output = run.run(TensorList{feature, coordinates}, 30000);
run.close();
```

形状と型を固定したまま、投入ごとにグリッド値を変更できます。複数の処理中サンプルには通常の非同期 push/pull API を使います。保持中のゼロコピー出力は生成側バッファの貸し出しを保持するため、プールを進めるには解放してください。所有出力は実行オブジェクトから独立して有効です。

## サンプリングのオプションと精度

`SimaGridSample2DPadding::Zeros` は無効な特徴タップをゼロに置き換えます。`SimaGridSample2DPadding::Border` は座標を端のピクセル中心に制限します。`align_corners=false` は半ピクセル中心を使い、true は正規化座標 -1/+1 を角のピクセル中心に写像します。NaN のグリッド成分は正規化座標 -1 を意味します。無限大成分はゼロパディングではサンプル全体をゼロにし、境界パディングでは端を選びます。

座標、重み、累積は FP32 です。各合計は最初の乗算に続く 3 回の明示的な融合積和で計算します。BF16 特徴は正確に拡張し、最終出力で一度だけ最近接偶数に丸めます。この精度は、グリッド値や中間座標を BF16 に丸める MLA 実装と異なります。

NCHW、タイル格納、非密ビュー、非恒等量子化には、このノードの前で明示的な変換が必要です。最近傍、三次、反射モードは未対応です。このノードを追加しても、コンパイル済み MLA モデル内の GridSample は置き換わりません。性能比較では同じ保存済み入力と計測範囲を使用してください。グリッド初期化が不明な ELF プロファイルのグリッド由来は未確認です。