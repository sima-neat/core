---
title: "アプリを実行する"
description: "PythonまたはC++で、グラフアプリケーション内でYOLOv8を実行し、デコードされたバウンディングボックスを読み取ります。"
sidebar_position: 3
mdx:
  format: mdx
---

# アプリケーションの実行

![グラフの作成](@site/../docs/images/hello-neat-graph-add-animation.svg)

![元の画像における Neat による YOLOv8 の検出結果](@site/../docs/images/first_inference_hook.png)

*以下のプログラムで生成された検出結果を、元の画像に重ねて表示します。*

これは、小さな**アプリケーション**として実行するのと同じYOLOv8推論です。`Model.run(...)`を直接呼び出す代わりに（[モデルを実行する](/develop-apps/hello-neat/run_first_model)の場合と同様）、モデルを[`Graph`](/develop-apps/development-workflow/graph)（入力、モデル、および出力を持つ名前付きグラフフロー）に組み込み、それを構築してプッシュ/プルします。PythonとC++で同じプログラムを使用できます。各コードブロックで言語タブを選択してください。

この最初のアプリケーションでは、意図的に単純な構造にしています。

- 名前付きの_入力_ (`nodes.input("image")`)は、データがアプリケーションにどこから入力されるかを示します。
- _モデル_ (`graph.add(model)`)は、グラフ内の1つのステップとしてモデルを実行します。
- 名前付きの_出力_ (`nodes.output("detections")`)は、アプリケーションが結果を読み取る場所を示します。

同じAPIは、後により複雑なアプリケーションにも拡張できます。ここでは、基本的な構成パターンを理解することが目的です。

:::tip 言語を選択してください。
任意のコードブロックにある「**Python / C++**」タブを使用してください。選択はサイト全体の言語セレクターに従うため、すべてのコードスニペットとプログラム全体がまとめて切り替わります。
:::

## プロジェクトのセットアップ

:::tip すでに[モデルを実行する](/develop-apps/hello-neat/run_first_model)を実行しましたか？
このセクションはスキップできます。ここでは、同じ`assets/`ディレクトリ、モデルパッケージ、およびサンプル画像を使用します。直接、[コードをステップごとに確認する](#walk-through-the-code)に進んでください。
:::

1. **モデルと入力画像用のアセットディレクトリを作成します。**
    ```bash
    mkdir -p assets
    ```
2. **モデルをダウンロードします:**
    ```bash
    sima-cli modelzoo -v 2.0.0 get yolo_v8s
    ```
    :::note sima-cli モデルのダウンロード
    `sima-cli` がモデルを `assets` ディレクトリ以外の場所に書き込む場合は、そのファイルを `assets/yolo_v8s_mpk.tar.gz` にコピーしてください。
:::
3. **サンプル画像をダウンロード**し、`assets/tutorial_sample_image.png` として保存します。

[サンプル画像を閲覧またはダウンロードしてください。](../../images/tutorial_sample_image.png)。

## コードの解説

このプログラムは、短い8つの部分で構成されています。各ブロックで言語タブを切り替えてください。

### 1. 画像を読み込む

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
import cv2

bgr = cv2.imread("assets/tutorial_sample_image.png")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
#include <opencv2/opencv.hpp>

cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png");
cv::Mat rgb;
cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
```

</CodeTab>
</CodeTabs>

OpenCVはBGR形式で画像を読み込みますが、YOLOv8はRGB形式を想定しています。この処理は必須ではありません。 Neat —アプリケーションは、ファイル、カメラ、またはデコーダーからピクセルデータを取得します。 Neat 次のステップで入力します。

### 2. モデルのオプションについて説明します。

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
import pyneat as neat

opt = neat.ModelOptions()
opt.preprocess.kind   = neat.InputKind.Image
opt.preprocess.preset = neat.NormalizePreset.COCO_YOLO
opt.decode_type       = neat.BoxDecodeType.YoloV8
opt.score_threshold   = 0.25
opt.nms_iou_threshold = 0.45
opt.top_k             = 100
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>
namespace neat = simaai::neat;

neat::Model::Options opt;
opt.preprocess.kind   = neat::InputKind::Image;
opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
opt.decode_type       = neat::BoxDecodeType::YoloV8;
opt.score_threshold   = 0.25f;
opt.nms_iou_threshold = 0.45f;
opt.top_k             = 100;
```

</CodeTab>
</CodeTabs>

`ModelOptions` は、1つのオブジェクトでモデルの処理方法を宣言します。具体的には、Neat が入力ピクセルをどのように前処理し、検出器の出力をどのようにデコードするかを定義します。

| フィールド | 設定内容 |
|---|---|
| `preprocess.kind = Image` | 入力は、あらかじめ整形されたテンソルではなく、生のピクセルです。 |
| `preprocess.preset = COCO_YOLO` | リサイズとレターボックス処理を行い、モデルの入力に合うように調整します。RGB形式で、`1/255` でスケールし、平均値の引き算は行いません。 |
| `decode_type = YoloV8` | 検出ヘッドデコーダーのファミリー。 |
| `score_threshold` / `nms_iou_threshold` / `top_k` | 信頼度の閾値、NMSのオーバーラップ、および保持する最大ボックス数。 |

### 3. モデルをロードします

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
model = neat.Model("assets/yolo_v8s_mpk.tar.gz", opt)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Model model("assets/yolo_v8s_mpk.tar.gz", opt);
```

</CodeTab>
</CodeTabs>

`Model` は、`.tar.gz` を読み込み、渡された `ModelOptions` に基づいて、その **MPK コントラクト** の有効性を検証し、モデルの断片をインスタンス化します。まだ何も実行されていません。

### 4. 画像を `Tensor` としてラップします。

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
tensor = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Tensor input = neat::Tensor::from_cv_mat(
    rgb,
    neat::ImageSpec::PixelFormat::RGB);
```

</CodeTab>
</CodeTabs>

`Tensor` は、Neat の型付きデータコンテナです。これには、形状、データ型、レイアウト、およびフレームワークがバイトを解釈するために必要なピクセル形式が含まれます。Neat がバイトだけでなくレイアウトも認識できるように、`PixelFormat` を渡す必要があります。

### 5. グラフを構成する

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
graph = neat.Graph("hello_neat_app")
graph.add(neat.nodes.input("image"))
graph.add(model)
graph.add(neat.nodes.output("detections"))
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Graph graph("hello_neat_app");
graph.add(neat::nodes::Input("image"));
graph.add(model);
graph.add(neat::nodes::Output("detections"));
```

</CodeTab>
</CodeTabs>

`Graph` はアプリケーションのフローです。各 `add(...)` は次のステップを追加するため、これにより線形のフロー `image → model → detections` が構築されます。ステップ 3 のモデルの一部が、その中の 1 つのステップになります。

### 6. グラフを構築して実行する

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
run = graph.build()
try:
    run.push("image", [tensor])
    run.close_input()
    outputs = run.pull_tensors("detections", timeout_ms=2000)
finally:
    run.close()
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Run run = graph.build();
run.push("image", neat::TensorList{input});
run.close_input();
neat::TensorList outputs = run.pull_tensors("detections", /*timeout_ms=*/2000);
run.close();
```

</CodeTab>
</CodeTabs>

`build()` は、公開されているグラフを、ノード名を保持したまま、実行可能なランタイムグラフに変換します。次に、入力データを名前付きの入力に`push`し、それ以上の入力が不要になったら`close_input()`を呼び、タイムアウトを設定して、名前付きの出力から結果を`pull`します。`pull_tensors`は、`TensorList`を返します。これは、`Model.run`が生成するのと同じ形状のテンソルであり、ここでは、パックされたYOLOv8の`BBOX`出力です。

### 7. バウンディングボックスをデコード

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
decoded = neat.decode_bbox(outputs)
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
neat::TensorList decoded = neat::decode_bbox(outputs);
```

</CodeTab>
</CodeTabs>

`decode_bbox` は、1:1 の位置関係を持つ `TensorList → TensorList` 変換です。各デコードされた出力は、`[num_detections, 6]` の形状を持つ `float32` テンソルであり、列は `(x1, y1, x2, y2, score, class_id)` です。

### 8. バウンディングボックスを読み取る

<CodeTabs>
<CodeTab label="Python" lang="python">

```python
labels = {0: "person", 27: "tie"}
for x1, y1, x2, y2, score, cls in decoded[0].to_numpy():
    name = labels.get(int(cls), f"id{int(cls)}")
    print(f"{name:<8} {score:.2f}  [{x1:4.0f} {y1:4.0f} {x2:4.0f} {y2:4.0f}]")
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

```cpp
const neat::Tensor& boxes = decoded.front();      // [num_detections, 6] float32
auto m = boxes.map_read();
const float* d = static_cast<const float*>(m.data);
for (int64_t i = 0; i < boxes.shape[0]; ++i) {
  const float* r = d + i * 6;                     // x1 y1 x2 y2 score class_id
  const int cls = static_cast<int>(r[5]);
  const char* name = (cls == 0) ? "person" : (cls == 27) ? "tie" : "?";
  std::printf("%-8s %.2f  [%4.0f %4.0f %4.0f %4.0f]\n", name, r[4], r[0], r[1], r[2], r[3]);
}
```

</CodeTab>
</CodeTabs>

Pythonでは、デコードされたテンソルは次のように読み込まれます。 `[N, 6]` NumPy配列を介して `to_numpy()`C++では、テンソルをマッピングし、浮動小数点数を読み取ります。モデルはCOCOクラスIDを出力します。それらを表示名にマッピングするのはアプリケーション側で行います。

## 完全なプログラム

プロジェクトディレクトリにファイルを作成し、ビルドして実行します。

<CodeTabs>
<CodeTab label="Python" lang="python">

`app.py`:

```python {18-24,34-35,38-41,44-46,48}
#!/usr/bin/env python3
import sys

try:
    import pyneat as neat
except ImportError:
    sys.exit(
        "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
        "Run: source ~/pyneat/bin/activate"
    )

import cv2

LABELS = {0: "person", 27: "tie"}


def yolo_model_options():
    opt = neat.ModelOptions()
    opt.preprocess.kind   = neat.InputKind.Image
    opt.preprocess.preset = neat.NormalizePreset.COCO_YOLO
    opt.decode_type       = neat.BoxDecodeType.YoloV8
    opt.score_threshold   = 0.25
    opt.nms_iou_threshold = 0.45
    opt.top_k             = 100
    return opt


def main() -> int:
    bgr = cv2.imread("assets/tutorial_sample_image.png")
    if bgr is None:
        raise RuntimeError("failed to read assets/tutorial_sample_image.png")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    model = neat.Model("assets/yolo_v8s_mpk.tar.gz", yolo_model_options())
    tensor = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)

    # Compose the model into a Graph application: image -> model -> detections.
    graph = neat.Graph("hello_neat_app")
    graph.add(neat.nodes.input("image"))
    graph.add(model)
    graph.add(neat.nodes.output("detections"))

    # Build the app, push the image into the named input, pull the named output.
    run = graph.build()
    try:
        run.push("image", [tensor])
        run.close_input()
        outputs = run.pull_tensors("detections", timeout_ms=2000)
    finally:
        run.close()

    decoded = neat.decode_bbox(outputs)
    for x1, y1, x2, y2, score, cls in decoded[0].to_numpy():
        name = LABELS.get(int(cls), f"id{int(cls)}")
        print(f"{name:<8} {score:.2f}  [{x1:4.0f} {y1:4.0f} {x2:4.0f} {y2:4.0f}]")
    print("[OK] Graph app completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**実行:**

* **DevKit 上で**
  ```bash
  source ~/pyneat/bin/activate
  python3 app.py
  ```
* **Neat SDK がインストールされているホスト**
  ```bash
  dk app.py
  ```

</CodeTab>
<CodeTab label="C++" lang="cpp">

`CMakeLists.txt` と `main.cpp` を作成します。

```cmake title="CMakeLists.txt" {18,23-27}
cmake_minimum_required(VERSION 3.16)
project(sima_neat_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Supports both DevKit/native installs (system paths) and
# cross builds with SYSROOT exported (SDK sysroot paths).
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH
    "$ENV{SYSROOT}/usr"
    "$ENV{SYSROOT}/usr/lib"
    "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu"
  )
endif()

find_package(SimaNeat REQUIRED CONFIG)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

add_executable(sima_neat_app main.cpp)
target_link_libraries(sima_neat_app
  PRIVATE
    SimaNeat::sima_neat
    PkgConfig::OPENCV
)
```

強調表示されている2つの行は、アプリケーションとNeatをリンクさせる部分です。`find_package(SimaNeat REQUIRED CONFIG)`は、インストールされたNeatパッケージを（`SimaNeatConfig.cmake`を介して）検索し、`target_link_libraries(sima_neat_app PRIVATE SimaNeat::sima_neat ...)`は、それとリンクします。インポートされた`SimaNeat::sima_neat`ターゲットは、Neatのインクルードディレクトリと推移的な依存関係を自動的に伝播するため、手動でのインクルード/ライブラリパスの指定は不要です。（`PkgConfig::OPENCV`は、このアプリケーションがOpenCVを使用して画像をロードするため、必要です。）

```cpp title="main.cpp" {13-19,30-31,34-37,40-42,44-46}
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace neat = simaai::neat;

neat::Model::Options yolo_model_options() {
  neat::Model::Options opt;
  opt.preprocess.kind   = neat::InputKind::Image;
  opt.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  opt.decode_type       = neat::BoxDecodeType::YoloV8;
  opt.score_threshold   = 0.25f;
  opt.nms_iou_threshold = 0.45f;
  opt.top_k             = 100;
  return opt;
}

int main() {
  cv::Mat bgr = cv::imread("assets/tutorial_sample_image.png");
  if (bgr.empty())
    throw std::runtime_error("failed to read assets/tutorial_sample_image.png");
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  neat::Model model("assets/yolo_v8s_mpk.tar.gz", yolo_model_options());
  neat::Tensor input = neat::Tensor::from_cv_mat(
      rgb,
      neat::ImageSpec::PixelFormat::RGB);

  // Compose the model into a Graph application: image -> model -> detections.
  neat::Graph graph("hello_neat_app");
  graph.add(neat::nodes::Input("image"));
  graph.add(model);
  graph.add(neat::nodes::Output("detections"));

  // Build the app, push the image into the named input, pull the named output.
  neat::Run run = graph.build();
  run.push("image", neat::TensorList{input});
  run.close_input();
  neat::TensorList outputs = run.pull_tensors("detections", /*timeout_ms=*/2000);
  run.close();

  neat::TensorList decoded = neat::decode_bbox(outputs);
  const neat::Tensor& boxes = decoded.front();      // [num_detections, 6] float32
  auto m = boxes.map_read();
  const float* d = static_cast<const float*>(m.data);
  for (int64_t i = 0; i < boxes.shape[0]; ++i) {
    const float* r = d + i * 6;                     // x1 y1 x2 y2 score class_id
    const int cls = static_cast<int>(r[5]);
    const char* name = (cls == 0) ? "person" : (cls == 27) ? "tie" : "?";
    std::printf("%-8s %.2f  [%4.0f %4.0f %4.0f %4.0f]\n", name, r[4], r[0], r[1], r[2], r[3]);
  }
  std::printf("[OK] Graph app completed\n");
  return 0;
}
```

**構築：**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**実行:**

* **DevKit 上で**
  ```bash
  ./build/sima_neat_app
  ```
* **Neat SDK がインストールされているホスト**
  ```bash
  dk build/sima_neat_app
  ```

</CodeTab>
</CodeTabs>

検出ごとに1行表示されます。その後は：

```text
[OK] Graph app completed
```

## Neat がどのように組み立てられたか

![こんにちは。Neat グラフアプリの操作の流れです。](@site/../docs/images/hello-neat-graph-app-flow.svg)

API は、その構造に直接対応しています。

- `Graph` はアプリケーションのフローを保持し、`graph.add(...)` は各ステップを順番に追加します。
- 名前付きの入力と出力は、ランタイムのエンドポイントになります: `run.push("image", ...)` と `run.pull_tensors("detections")`。
- `Model` は、`Model.run` を使用して直接呼び出すのと同じフラグメントであり、ここではアプリケーション内の 1 つのノードとして実行されます。

## 次のステップ

より高度なグラフの構成については、[グラフプログラミングモデル](/develop-apps/development-workflow/graph) を参照してください。

そこから、より広範な SiMa.ai Neat の学習リソースに進みます。

- モデル、グラフ、実行などの主要な Neat の概念を説明する [基本的なプログラミングモデル](/develop-apps/development-workflow/overview) を学習します。
- 特定の概念とワークフローを段階的に説明する [チュートリアル](/tutorials/) を参照します。
- [アプリポータル](https://apps.neat.sima.ai/portal) で厳選されたアプリケーションを探索し、[GitHub上のアプリケーションリポジトリ](https://github.com/sima-neat/apps) でソースコードを確認します。
