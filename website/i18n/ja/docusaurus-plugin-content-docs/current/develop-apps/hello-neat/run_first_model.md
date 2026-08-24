---
title: "モデルを実行する"
description: "サンプル画像で Neat を使用してモデルを実行します。"
sidebar_position: 2
---

# モデルを実行する

[こんにちは、Neat！](/develop-apps/hello-neat/minimal)で使用したのと同じワーキングディレクトリを使用して、オブジェクト検出のための実際のモデルを実行します。
このアプリケーションは、YOLOv8モデルをロードし、サンプル画像を読み込み、推論を実行し、バウンディングボックスをデコードし、検出されたオブジェクトの数を表示します。

このページでは、次の2つのNeatの概念を紹介します。

* [`Model`](/develop-apps/development-workflow/model)は、コンパイルされたモデルパッケージをロードし、`run(...)`のエントリポイントを提供します。
* [`ModelOptions`](/tutorials/configure-model-options)は、Neatに、画像をどのように準備し、検出器の出力をどのようにデコードするかを指示します。

まだ完全なAPIを習得する必要はありません。現時点では、`Model`と`ModelOptions`がどのように連携して、コンパイルされたモデルを実行するかという点に焦点を当ててください。

![こんにちは、Neat YOLOv8フロー。](@site/../docs/images/hello-neat-yolov8-flow.svg)

## モデルとサンプル画像を取得する

1. **モデルと入力画像を保存するアセットディレクトリを作成します。**
    ```bash
    mkdir -p assets
    cd assets
    ```
2. **モデルをダウンロードします:**
    ```bash
    sima-cli modelzoo -v 2.0.0 get yolo_v8s
    ```
    :::note sima-cli モデルのダウンロード
    `sima-cli` がモデルを `assets` ディレクトリ以外の場所に書き込む場合は、そのファイルを `assets/yolo_v8s_mpk.tar.gz` にコピーしてください。
    :::
3. **サンプル画像をダウンロード**し、`assets` ディレクトリに `tutorial_sample_image.png` として保存します。

    [サンプル画像を閲覧またはダウンロードしてください。](../../images/tutorial_sample_image.png)。
4. **プロジェクトディレクトリに戻ります:**
    ```bash
    cd ..
    ```

## 概要

このプログラムをさらに発展させます。 [こんにちは、Neat！](/develop-apps/hello-neat/minimal)：変更しない `CMakeLists.txt` （すでにリンクされています） Neat および OpenCV を使用し、プログラムの本体を以下の 4 つの手順に置き換えます。各手順は、最終的なプログラムの一部です。手順を順番に読み、次に次の手順に進んでください。 [完全版プログラム](#full-program) 貼り付けて実行します。任意のブロックにある言語タブを選択してください。選択内容はサイト全体の設定に反映されます。

### 1. 画像を読み込み、サイズを変更します。 {#step-load-image}

YOLOv8s は固定サイズの `640×640` BGR 画像を想定しているため、OpenCV でサンプル画像を読み込み、サイズを変更します。これは通常の画像入出力処理であり、この時点ではまだ Neat API を使用しません。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
cv::Mat load_sample_image() {
  cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png", cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to load sample image");
  cv::resize(bgr, bgr, cv::Size(640, 640));   // YOLOv8s input size
  return bgr;
}
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
def load_image(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.resize(bgr, (640, 640))   # YOLOv8s input size
```

</CodeTab>
</CodeTabs>

### 2. 入力とデコードについて説明します。 `ModelOptions` {#step-model-options}

`ModelOptions` これは、お客様のイメージとモデル間のランタイム契約です。ここでは、以下の2つのことを規定します。 Neat 推論の前に、デコードされたピクセルをどのように前処理すべきか、また、検出器の生の出力データをどのようにボックスに変換すべきか。 `decode_type` YOLOv8デコーダーを選択し、閾値を使用して、検出精度の低い、または重複するバウンディングボックスを排除します。 `top_k` 上限を設定します。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::Model::Options opt;
opt.preprocess.kind = simaai::neat::InputKind::Image;
opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
opt.score_threshold = 0.55f;
opt.nms_iou_threshold = 0.5f;
opt.top_k = 100;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
opt = pyneat.ModelOptions()
opt.preprocess.kind = pyneat.InputKind.Image
opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
opt.decode_type = pyneat.BoxDecodeType.YoloV8
opt.score_threshold = 0.55
opt.nms_iou_threshold = 0.5
opt.top_k = 100
```

</CodeTab>
</CodeTabs>

### 3. モデルをロードし、推論を実行する {#step-run}

コンパイルされた `.tar.gz` パッケージとオプションから `Model` を構築し、次にタイムアウトを指定して `run(...)` を呼び出します。これは同期的に実行され、出力テンソルを返します。`timeout_ms` を設定することで、処理が停止した場合に、ハングアップするのではなく、エラーを明確に表示するようにします。

**C++** では、モデル入力ごとに 1 つの `cv::Mat` を渡します。**Python** では、まず NumPy イメージを `BGR` タグが付いた `Tensor` としてラップし、Neat にバイトレイアウトを認識させた後、Python モデル入力はシーケンスであるため、`[tensor]` を渡します。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::Model yolo("assets/yolo_v8s_mpk.tar.gz", opt);
simaai::neat::TensorList outputs = yolo.run(std::vector<cv::Mat>{bgr}, /*timeout_ms=*/2000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
model = pyneat.Model(str(mpk), opt)
tensor = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)
outputs = model.run([tensor], timeout_ms=2000)
```

</CodeTab>
</CodeTabs>

### 4. 検出数の読み込み {#step-read}

`decode_type` が設定されているため、最初の出力テンソルにはデコードされたバウンディングボックスが含まれます。BBOXテンソルは、`uint32` 形式の検出数で始まり、最初の4バイトを読み取ります。完全なワイヤー形式（ボックスごとの座標、スコア、クラス）については、[モデルの出力から検出ボックスを読み取る](/tutorials/read-detection-boxes) を参照してください。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
std::uint32_t detections = 0;
if (!outputs.empty()) {
  simaai::neat::Mapping view = outputs.front().map_read();
  if (view.size_bytes >= sizeof(detections))
    std::memcpy(&detections, view.data, sizeof(detections));
}
std::cout << "detections=" << detections << "\n";
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
detections = 0
if len(outputs) > 0:
    payload = outputs[0].to_numpy(copy=False).tobytes()
    detections = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else 0
print(f"detections={detections}")
```

</CodeTab>
</CodeTabs>

## プログラム全体 {#full-program}

[こんにちは、Neat！](/develop-apps/hello-neat/minimal) の `CMakeLists.txt` をそのまま使用します（すでにアプリケーションと Neat および OpenCV をリンクしています）。プログラムの本体を、以下に示す完全なファイルに置き換えます。強調表示されている行が、プログラムの主要な3つの部分です。`Model` を作成し、入力を構築し、`run()` を呼び出します。

<details>
<summary>プログラム全体を表示する</summary>

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp title="main.cpp" {43-45}
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

cv::Mat load_sample_image() {
  cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png", cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to load sample image");

  // YOLOv8s expects a 640 x 640 input in this tutorial.
  cv::resize(bgr, bgr, cv::Size(640, 640));
  return bgr;
}

int main() {
  // 1. Load the sample image and resize it for the model.
  cv::Mat bgr = load_sample_image();

  // 2. Tell Neat how to preprocess pixels and decode YOLO boxes.
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  opt.score_threshold = 0.55f;
  opt.nms_iou_threshold = 0.5f;
  opt.top_k = 100;

  // 3. Load the compiled model package and run inference.
  simaai::neat::Model yolo("assets/yolo_v8s_mpk.tar.gz", opt);
  simaai::neat::TensorList outputs = yolo.run(std::vector<cv::Mat>{bgr}, /*timeout_ms=*/2000);

  // 4. The BBOX output starts with a uint32 detection count.
  std::uint32_t detections = 0;
  if (!outputs.empty()) {
    simaai::neat::Mapping view = outputs.front().map_read();
    if (view.size_bytes >= sizeof(detections))
      std::memcpy(&detections, view.data, sizeof(detections));
  }
  std::cout << "detections=" << detections << "\n";
  std::cout << "[OK] YOLOv8 completed\n";
  return 0;
}
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python title="hello_neat.py" {50-53}
#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
from pathlib import Path

try:
    import pyneat
except ImportError:
    sys.exit(
        "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
        "Run: source ~/pyneat/bin/activate"
    )

import cv2


def load_image(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    # YOLOv8s expects a 640 x 640 input in this tutorial.
    return cv2.resize(bgr, (640, 640))


def main() -> int:
    mpk = Path("assets/yolo_v8s_mpk.tar.gz")
    image = Path("assets/tutorial_sample_image.png")

    # 1. Load the sample image and resize it for the model.
    bgr = load_image(image)

    # 2. Tell Neat how to preprocess pixels and decode YOLO boxes.
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV8
    opt.score_threshold = 0.55
    opt.nms_iou_threshold = 0.5
    opt.top_k = 100

    # 3. Load the compiled model package and run inference.
    model = pyneat.Model(str(mpk), opt)
    tensor = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)
    outputs = model.run([tensor], timeout_ms=2000)

    # 4. The BBOX output starts with a uint32 detection count.
    detections = 0
    if len(outputs) > 0:
        payload = outputs[0].to_numpy(copy=False).tobytes()
        detections = struct.unpack_from("<I", payload, 0)[0] if len(payload) >= 4 else 0
    print(f"detections={detections}")

    print("[OK] YOLOv8 completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

</CodeTab>
</CodeTabs>

</details>

## ビルドと実行

これらをプロジェクトディレクトリ（`assets/` が含まれるディレクトリ）から実行してください。

<CodeTabs>
<CodeTab label="C++" lang="cpp">

「Hello Neat!」と同じコマンドで再構築し、その後、バイナリを実行します。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/sima_neat_hello      # on the DevKit
dk build/sima_neat_hello     # from the Palette SDK host
```

</CodeTab>
<CodeTab label="Python" lang="python">

スクリプトを実行してください。

```bash
source ~/pyneat/bin/activate
python3 hello_neat.py        # on the DevKit
dk hello_neat.py             # from the Palette SDK host
```

</CodeTab>
</CodeTabs>

以下のような検出結果の概要が表示されるはずです。

```text
detections=3
[OK] YOLOv8 completed
```

正確な数は、モデルパックとランタイムのバージョンによって異なります。重要なのは、アプリがビルドされ、実行され、`[OK] YOLOv8 completed` に到達することです。

## 作成した内容

この例では、より大規模な Neat アプリケーションで使用されるのと同じ基本的な手順に従います。

- コンパイルされたモデルパッケージ（`.tar.gz`）を`Model` としてロードします。
- 入力画像を、モデルが期待する形式に変換します。
- Neat ランタイムステージを通じて推論を実行します。
- 生の検出器出力を、バウンディングボックスにデコードします。

バウンディングボックスのデコード、閾値、NMS、および検出器出力構造の詳細な説明については、[モデルの出力から検出ボックスを読み取る](/tutorials/read-detection-boxes) を参照してください。

## 次のステップ

YOLOv8の実行後、より広範な SiMa.ai Neat の学習リソースに進みます。

- [アプリを実行する](/develop-apps/hello-neat/run_an_app) を使用して、この同じモデルを`Graph` アプリケーションに組み込みます。これは、名前付きの入力 → モデル → 出力パイプラインであり、一度構築して、`Model.run(...)` を直接呼び出す代わりに、プッシュ/プルで実行します。
- [基本的なプログラミングモデル](/develop-apps/development-workflow/overview) を学習します。これには、モデル、グラフ、およびランタイム実行など、主要な Neat の概念が説明されています。
- [チュートリアル](/tutorials/) を参照して、特定の概念とワークフローを段階的に確認します。
- [アプリポータル](https://apps.neat.sima.ai/portal) で厳選されたアプリケーションを探索し、[GitHub上のアプリケーションリポジトリ](https://github.com/sima-neat/apps) でソースコードを確認します。
