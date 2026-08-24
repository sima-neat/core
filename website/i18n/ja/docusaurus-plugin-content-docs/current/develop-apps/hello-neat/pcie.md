---
title: "PCIeによる共同処理"
description: "最初の PCIe コプロセッシング推論を実行してください。"
sidebar_position: 4
---

# PCIe による共同処理

![PCIeによるコプロセッシングの流れ](@site/../docs/images/hello-neat-pcie-coprocessing-flow.svg)

PCIe による共同処理では、アプリケーションの制御と入力の準備をホストマシン上で行い、接続された Modalix PCIe カードでモデルを実行します。この例では、ホストが画像を読み込み、サイズを変更し、正規化し、準備されたテンソルをカードに送信して ResNet-50 の推論を実行し、返された分類結果を出力します。

## 始める前に

次のものが必要です。

- ホストマシンに [Neat PCIe ホストパッケージ](/getting-started/neat-library/pcie-host/) をインストールすること。
- カードに互換性のある Neat Library リリースをインストールすること。
- ホストからアクセス可能なカード管理インターフェース。カード 0 はデフォルトで `10.0.0.2` を使用します。

## モデルと画像の取得

ホストマシンに作業ディレクトリを作成し、Model Zoo から ResNet-50 をダウンロードします。

<ShellCommand prompt="pcie-host">
mkdir -p pcie-host-quickstart/assets
cd pcie-host-quickstart
sima-cli modelzoo get resnet_50
</ShellCommand>

ダウンロードしたアーカイブをこのディレクトリに、`resnet_50.tar.gz`として配置します。

[ラブラドール犬のサンプル画像](../../images/hello-neat-pcie-labrador.jpg)をダウンロードし、`assets/sample.png`として保存します。

この画像は、Elfが撮影し、DjmirkoとFT2が修正した[黄色のラブラドール犬。新品のように見える。jpg](https://commons.wikimedia.org/wiki/File:YellowLabradorLooking_new.jpg)を基にしており、[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)のライセンスで提供されています。

作業ディレクトリには、以下のファイルが含まれているはずです。

```text
pcie-host-quickstart/
├── assets/
│   └── sample.png
└── resnet_50.tar.gz
```

## アプリケーションの作成

PythonまたはC++を選択し、作業ディレクトリにアプリケーションを作成します。

<CodeTabs>
<CodeTab label="Python" lang="python">

`pcie_host.py` を作成します。

```python title="pcie_host.py"
import cv2
import numpy as np

import pyneatpcie as pcie


def load_input(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    bgr = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    stddev = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (rgb - mean) / stddev

connection = pcie.ConnectionOptions(
    card_host="10.0.0.2",
    card_id=0,
    queue=0,
)

with pcie.Model("resnet_50.tar.gz", connection=connection) as model:
    input_spec = model.info().inputs[0]
    input_tensor = pcie.Tensor.from_numpy(
        load_input("assets/sample.png"),
        copy=True,
        route_name=input_spec.name,
    )
    model.build(readiness_timeout_ms=180000)
    outputs = model.run([input_tensor], timeout_ms=30000)

output = outputs[0].to_numpy().reshape(-1)
print("output:", outputs[0].route.name, outputs[0].shape)
print("top1:", int(np.argmax(output)))
```

</CodeTab>
<CodeTab label="C++" lang="cpp">

`pcie_host.cpp` を作成します。

```cpp title="pcie_host.cpp"
#include <simaai/neat/pcie/Model.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pcie = simaai::neat::pcie;

pcie::Tensor load_input(const std::string& path,
                        const std::string& route_name) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty())
    throw std::runtime_error("failed to read image: " + path);

  cv::resize(bgr, bgr, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (!rgb.isContinuous())
    rgb = rgb.clone();

  constexpr float mean[] = {0.485F, 0.456F, 0.406F};
  constexpr float stddev[] = {0.229F, 0.224F, 0.225F};
  std::vector<float> input(rgb.total() * rgb.channels());
  for (int row = 0; row < rgb.rows; ++row) {
    const auto* pixels = rgb.ptr<std::uint8_t>(row);
    for (int col = 0; col < rgb.cols; ++col) {
      for (int channel = 0; channel < 3; ++channel) {
        const std::size_t index =
            (static_cast<std::size_t>(row) * rgb.cols + col) * 3 + channel;
        const float value = pixels[col * 3 + channel] / 255.0F;
        input[index] = (value - mean[channel]) / stddev[channel];
      }
    }
  }
  return pcie::Tensor::from_vector(
      std::move(input), {rgb.rows, rgb.cols, rgb.channels()}, route_name);
}

int main() {
  pcie::ConnectionOptions connection;
  connection.card_host = "10.0.0.2";
  connection.card_id = 0;
  connection.queue = 0;

  pcie::Model model("resnet_50.tar.gz", {}, connection);
  const pcie::ModelInfo info = model.info();
  if (info.inputs.empty())
    throw std::runtime_error("model reports no inputs");

  model.build(/*readiness_timeout_ms=*/180000);
  pcie::TensorList outputs = model.run(
      load_input("assets/sample.png", info.inputs[0].name),
      /*timeout_ms=*/30000);

  if (outputs.empty() || outputs[0].dtype != pcie::TensorDType::Float32 ||
      outputs[0].data == nullptr || outputs[0].size_bytes == 0)
    throw std::runtime_error("expected one FP32 classification output");

  const auto* scores = static_cast<const float*>(outputs[0].data);
  const std::size_t count = outputs[0].size_bytes / sizeof(float);
  const auto best = std::max_element(scores, scores + count);

  std::cout << "output: " << outputs[0].route.name << " [";
  for (std::size_t index = 0; index < outputs[0].shape.size(); ++index) {
    if (index != 0)
      std::cout << ", ";
    std::cout << outputs[0].shape[index];
  }
  std::cout << "]\n";
  std::cout << "top1: " << std::distance(scores, best) << '\n';
  model.close();
}
```

`CMakeLists.txt` を作成します。

```cmake title="CMakeLists.txt"
cmake_minimum_required(VERSION 3.16)
project(pcie_host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaPCIeHost REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(pcie_host pcie_host.cpp)
target_include_directories(pcie_host PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
  pcie_host
  PRIVATE SimaPCIeHost::sima_neat_pcie_host ${OpenCV_LIBS}
)
```

</CodeTab>
</CodeTabs>

## アプリケーションを実行します

<CodeTabs>
<CodeTab label="Python" lang="python">

ホストマシンで Python 環境を有効にし、スクリプトを実行します。

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 pcie_host.py
</ShellCommand>

</CodeTab>
<CodeTab label="C++" lang="cpp">

C++アプリケーションをビルドして実行します。

<ShellCommand prompt="pcie-host">
cmake -S . -B build
cmake --build build
./build/pcie_host
</ShellCommand>

</CodeTab>
</CodeTabs>

正常に実行されると、返された出力テンソルと、最も高いスコアを獲得した ImageNet クラスのインデックスが出力されます。ドキュメントに記載されている ResNet-50 アーティファクトとラブラドール犬の画像を使用した場合、期待される上位クラスはインデックス 208、つまりラブラドール・レトリバーです。

```text
output: resnetv17_dense0_fwd [1, 1000]
top1: 208
```

Pythonのコンテキストマネージャーは、`close()` を自動的に呼び出します。C++の例では、明示的に呼び出します。どちらのバージョンも、終了する前にキュー0を解放します。

## 次のステップ

[PCIeによる共同処理](/develop-apps/development-workflow/pcie-model/) を使用して、完全なモデルAPI、`push()` と `pull()` を使用したパイプラインリクエスト、および画像の前処理オプションについて学習します。
