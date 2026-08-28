---
title: "ROIリストの前処理"
description: "1つまたは複数の画像に対して、複数のランタイムROIウィンドウにわたって前処理を実行します。"
sidebar_position: 7
---

# ROIリストの前処理

アプリケーションにすでに 1 つ以上の画像ウィンドウがあり、各ウィンドウをモデルの入力として正確に前処理（リサイズまたはレターボックス処理、カラー変換、正規化、量子化、およびオプションでテッセレーション）したい場合は、前処理済み ROI リストを使用します。

これは、2 次分類器、検出器カスケード、トラッカー、またはソース画像のバッチから動的なクロップのリストが生成されるようなフローに最適なツールです。

## 最小限の例

C++：

```cpp
#include <neat.h>
#include <opencv2/imgcodecs.hpp>

using namespace simaai::neat;

Model model("/path/to/model.tar.gz");

std::vector<cv::Mat> images = {
    cv::imread("/data/camera0.jpg", cv::IMREAD_COLOR),
    cv::imread("/data/camera1.jpg", cv::IMREAD_COLOR),
};

std::vector<PreprocessRoi> rois = {
    {0, 0, 0, 320, 240},
    {0, 120, 80, 160, 160},
    {1, -20, 30, 224, 224},
};

TensorList out = stages::Preproc(images, model, rois);

for (std::size_t i = 0; i < out.size(); ++i) {
  const auto& meta = out[i].semantic.preprocess;
  // out[i] is the preprocessed tensor for rois[i].
  // meta carries the ROI and inverse-coordinate breadcrumbs.
}
```

Python：

```python
import cv2
import pyneat

model = pyneat.Model("/path/to/model.tar.gz")

images = [
    cv2.imread("/data/camera0.jpg", cv2.IMREAD_COLOR),
    cv2.imread("/data/camera1.jpg", cv2.IMREAD_COLOR),
]

rois = [
    pyneat.PreprocessRoi(0, 0, 0, 320, 240),
    pyneat.PreprocessRoi(0, 120, 80, 160, 160),
    pyneat.PreprocessRoi(1, -20, 30, 224, 224),
]

out = pyneat.stages.preproc(
    images,
    model,
    rois=rois,
    image_format=pyneat.PixelFormat.BGR,
)

for i, tensor in enumerate(out):
    meta = tensor.semantic.preprocess
    # tensor is the preprocessed output for rois[i].
    # meta carries the ROI and inverse-coordinate breadcrumbs.
```

## リサイズ、レターボックス、および正規化の設定を行います。

ROIリストの前処理は、フルフレームの前処理と同じモデル前処理オプションを使用します。

C++:

```cpp
Model::Options opt;
opt.preprocess.resize.enable = AutoFlag::On;
opt.preprocess.resize.width = 640;
opt.preprocess.resize.height = 640;
opt.preprocess.resize.mode = ResizeMode::Letterbox;
opt.preprocess.resize.pad_value = 114;
opt.preprocess.resize.scaling_type = "BILINEAR";
opt.preprocess.normalize.enable = AutoFlag::On;
opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
opt.preprocess.normalize.stddev = {1.0f, 1.0f, 1.0f};
opt.preprocess.tessellate.enable = AutoFlag::Auto;

Model model("/path/to/model.tar.gz", opt);
TensorList out = stages::Preproc(images, model, rois);
```

Python：

```python
opt = pyneat.ModelOptions()
opt.preprocess.resize.enable = pyneat.AutoFlag.On
opt.preprocess.resize.width = 640
opt.preprocess.resize.height = 640
opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
opt.preprocess.resize.pad_value = 114
opt.preprocess.resize.scaling_type = "BILINEAR"
opt.preprocess.normalize.enable = pyneat.AutoFlag.On
opt.preprocess.normalize.mean = [0.0, 0.0, 0.0]
opt.preprocess.normalize.stddev = [1.0, 1.0, 1.0]
opt.preprocess.tessellate.enable = pyneat.AutoFlag.Auto

model = pyneat.Model("/path/to/model.tar.gz", opt)
out = pyneat.stages.preproc(
    images,
    model,
    rois=rois,
    image_format=pyneat.PixelFormat.BGR,
)
```

サポートされている`scaling_type`トークンは、`BILINEAR`、`NEAREST_NEIGHBOUR`、`BICUBIC`、`INTERAREA`、および`NO_SCALING`です。`NEAREST_NEIGHBOR`と`INTER_AREA`は、有効な別名として受け入れられます。

## バッチ処理のセマンティクス

| 入力 | 意味 |
| --- | --- |
| `images` | 入力画像のバッチ。ROIが指定されている場合、ベクトル/リストは空であってはなりません。Pythonでは、uint8形式のHW/HWC形式のNumPy/Torch/`pyneat.Tensor`形式の画像を入力として受け付けます。|
| `rois` | ランタイムにおけるROIリスト。出力されるテンソルの順序は、このベクトルの順序と一致します。|
| `PreprocessRoi::batch_index` | は、`images` 内のどのソース画像からROIが読み取るかを指定します。|
| `PreprocessRoi::x`, `y` | 署名されたソースフレームの左上座標。負の値も許可されます。 |
| `PreprocessRoi::width`, `height` | ROIの寸法。どちらも正の値である必要があります。 |

ROIリスト呼び出しで使用するすべてのソース画像は、幅、高さ、OpenCVのタイプ、およびチャンネル数が一致している必要があります。ステージAPIは、ROIリスト用の、パックされた8ビットRGB/BGR（`CV_8UC3`）およびGRAY/GRAY8（`CV_8UC1`）のソース画像をサポートします。

Pythonの場合、`cv2.imread`で読み込んだ画像には`image_format=pyneat.PixelFormat.BGR`を、RGB配列には`RGB`を、HW形式のグレースケール配列には`GRAY8`を渡します。CHW形式のテンソルは`pyneat.stages.preproc`によって拒否されるため、ステージを呼び出す前にHWC形式に変換してください。

## 出力の意味

`stages::Preproc(images, model, rois)` は、以下の値を返します。

- 有効な空でないリクエストの場合の、`out.size() == rois.size()`。
- ROI `rois[i]` から生成された出力テンソル `out[i]`。
- モデルのルーティングによって選択されたデータ型とレイアウト（BF16やINT8、および密な出力またはテッセレーションされた出力など）
- 出力ごとの `tensor.semantic.preprocess` メタデータ。

各出力の事前処理メタデータには、選択された関心領域（ROI）、レターボックスのジオメトリ、正規化/量子化/テッセレーションのフラグ、およびモデル/事前処理された座標を元のソースフレーム座標系にマッピングするアフィン変換が含まれます。

## フレーム外にある関心領域

関心領域（ROI）は、元の画像の外側にまで及ぶことがあります。

```cpp
std::vector<PreprocessRoi> rois = {
    {0, -16, -16, 128, 128},
    {0, image.cols - 64, image.rows - 64, 128, 128},
};
```

フレーム内の領域はコピーされ、フレーム外の領域は、Preproc パッドの値を使用してパディングされます。これにより、出力形状が安定し、画像の端付近での特別な処理を回避できます。

## レターボックスとアスペクト比

`ResizeMode::Letterbox` の場合、Preproc は ROI ごとにレターボックスのスケールとパディングを計算します。したがって、縦長の ROI と横長の ROI では、同じターゲットサイズであっても、異なる `scaled_width`、`scaled_height`、`pad_left`、および `pad_top` メタデータが生成される可能性があります。

後続のコードは、仮定に基づいてパディングを再計算するのではなく、メタデータを参照する必要があります。

## 検証チェックリスト

推論結果を批判する前に、まず以下の点を確認してください。

- 画像ベクトルは空ではありません。
- すべての画像は、同じサイズ、OpenCVの型、およびチャンネル数を持っています。
- Pythonへの入力は、CHWテンソルではなく、uint8形式のHW/HWC画像です。
- 各関心領域には、有効な`batch_index`が割り当てられています。
- 各関心領域（ROI）は、正の`width`と`height`を持ちます。
- ROIリストモードの場合、ソース画像の形式はRGB、BGR、GRAY、またはGRAY8です。
- リサイズモードと正規化は、モデルの学習時に使用された前処理と一致します。
- 下流の座標処理モジュールは、`tensor.semantic.preprocess`メタデータを使用します。

## この動作を検証するテスト

このリポジトリには、ユーザーインターフェースの主要な部分と機能的なROI（投資対効果）の動作を迅速にテストするための機能が含まれています。

- `preproc_roi_batch_functional_test`
- `preproc_roi_user_smoke_test`

これらは、複数の画像、複数の関心領域（ROI）、フレーム外のパディング、リサイズ/レターボックス処理、正規化された出力、および高密度/テッセレーションされたBF16/INT8形式の処理を網羅しています。

## 関連項目

- [前処理ノード](/reference/nodes/preproc)
- [推論前に画像を前処理する](/tutorials/preprocess-images)
- [データ形式](/develop-apps/advanced-concepts/data_formats)
- [BoxDecode デコードの種類](/reference/boxdecode_decode_types)
