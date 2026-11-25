---
title: "Запуск моделі"
description: "Запустіть модель на зразку зображення за допомогою Neat"
sidebar_position: 2
---

# Запуск моделі

Використовуйте ту саму робочу теку з [Hello Neat!](/develop-apps/hello-neat/minimal), щоб запустити справжню модель для виявлення об’єктів.
Цей застосунок завантажує модель YOLOv8, зчитує зразок зображення, виконує інференс, декодує обмежувальні рамки та виводить кількість виявлених об’єктів.

На цій сторінці представлено дві концепції Neat:

* [`Model`](/develop-apps/development-workflow/model) завантажує скомпільований пакет моделі та надає вам точку входу `run(...)`.
* [`ModelOptions`](/tutorials/configure-model-options) повідомляє Neat, як підготувати зображення та декодувати вихідні дані детектора.

Вам не потрібно одразу освоювати всі можливості API; наразі зосередьтеся на тому, як `Model` і `ModelOptions` взаємодіють між собою для запуску скомпільованої моделі.

![Схема роботи YOLOv8 у Hello Neat](@site/../docs/images/hello-neat-yolov8-flow.svg)

## Отримання моделі та зразка зображення

1. **Створіть теку для ресурсів**, у якій ми зберігатимемо модель і вхідне зображення:

    <ShellCommand prompt="sdk|devkit">
    mkdir -p assets
    cd assets
    </ShellCommand>

2. **Завантажте модель** із Model Browser і видобудьте архів моделі із завантаженого ZIP-файлу:

    <ShellCommand prompt="sdk|devkit">
    sima-cli models download --id yolov8s --variant w8a8
    unzip -p models/yolov8s/w8a8/yolov8s_w8a8_default.zip yolov8s_w8a8_mpk.tar.gz > yolo_v8s_mpk.tar.gz
    </ShellCommand>

    :::note Завантаження моделі через sima-cli
    Програма відкриває `assets/yolo_v8s_mpk.tar.gz`. Підготовку моделей для всіх навчальних матеріалів описано в розділі [Налаштування навчального посібника](/tutorials/before-you-run).
    :::
3. **Завантажте зразок зображення** з документації та збережіть його під назвою `tutorial_sample_image.png` у теці `assets`.

    [Відкрийте або завантажте зразок зображення](../../images/tutorial_sample_image.png).
4. **Поверніться до теки вашого проєкту:**

    <ShellCommand prompt="sdk|devkit">
    cd ..
    </ShellCommand>


## Покроковий огляд

Ми беремо за основу програму з [Hello Neat!](/develop-apps/hello-neat/minimal): залишаємо той самий файл `CMakeLists.txt` (він уже підключає Neat і OpenCV) і замінюємо тіло програми чотирма кроками, наведеними нижче. Кожен крок — це невелика частина остаточної програми: прочитайте їх по черзі, а потім візьміть [повну програму](#full-program), щоб вставити та запустити її. Оберіть вкладку мови в будь-якому блоці; ваш вибір відповідає перемикачу мови для всього сайту.

### 1. Завантаження зображення та зміна його розміру {#step-load-image}

YOLOv8s очікує зображення BGR фіксованого розміру `640×640`, тому ми зчитуємо зразок за допомогою OpenCV і змінюємо його розмір. Це просте введення/виведення зображень — поки що жодних API Neat.

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

### 2. Опис входу та декодування за допомогою `ModelOptions` {#step-model-options}

`ModelOptions` — це контракт середовища виконання між вашим зображенням і моделлю. Тут він визначає дві речі: як Neat має попередньо обробляти декодовані пікселі перед інференсом і як декодувати необроблений вихід детектора в рамки. `decode_type` вибирає декодер YOLOv8, порогові значення відсікають слабкі рамки або рамки, що перекриваються, а `top_k` обмежує їхню кількість.

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

### 3. Завантаження моделі та запуск інференсу {#step-run}

Створіть `Model` зі скомпільованого пакета `.tar.gz` і параметрів, а потім викличте `run(...)` із тайм-аутом. Виклик виконується синхронно та повертає вихідні тензори. Завдяки `timeout_ms` запуск, що застряг, явно завершується помилкою, а не зависає.

У **C++** для кожного входу моделі передається один `cv::Mat`. У **Python** зображення NumPy спочатку обгортається в `Tensor` з позначкою `BGR`, щоб Neat знав розкладку байтів, а потім передається `[tensor]`, оскільки в Python вхід моделі — це послідовність.

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

### 4. Зчитування кількості виявлень {#step-read}

Оскільки параметр `decode_type` встановлено, перший вихідний тензор містить декодовані рамки. Тензор BBOX починається з лічильника виявлень типу `uint32`, тому ми зчитуємо його перші чотири байти. Повний формат даних — координати, оцінки та класи для кожної рамки — описано в посібнику [Зчитування рамок виявлення з виходу моделі](/tutorials/read-detection-boxes).

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

## Повна програма {#full-program}

Залиште файл `CMakeLists.txt` з [Hello Neat!](/develop-apps/hello-neat/minimal) (він уже підключає до застосунку Neat і OpenCV) і замініть тіло програми повним файлом, наведеним нижче. Виділено три ключові місця: створення `Model`, підготовка входу та виклик `run()`.

<details>
<summary>Показати повну програму</summary>

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

## Збирання та запуск

Виконуйте ці команди з теки вашого проєкту (тієї, що містить `assets/`).

<CodeTabs>
<CodeTab label="C++" lang="cpp">

Перезберіть проєкт тими самими командами, що й у Hello Neat!:

<ShellCommand prompt="sdk|devkit">
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
</ShellCommand>

Потім запустіть виконуваний файл:

<ShellCommand prompt="devkit">
./build/sima_neat_hello
</ShellCommand>

<ShellCommand prompt="sdk">
dk build/sima_neat_hello
</ShellCommand>

</CodeTab>
<CodeTab label="Python" lang="python">

Запустіть скрипт:

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 hello_neat.py
</ShellCommand>

<ShellCommand prompt="sdk">
dk hello_neat.py
</ShellCommand>

</CodeTab>
</CodeTabs>

Ви повинні побачити зведену інформацію про результати виявлення, схожу на таку:

```text
detections=3
[OK] YOLOv8 completed
```

Точна кількість може відрізнятися залежно від пакета моделі та версії середовища виконання. Головне, щоб застосунок збирався, запускався та доходив до `[OK] YOLOv8 completed`.

## Що ви створили

Цей приклад повторює ту саму загальну схему, що й більші застосунки Neat:

- Завантажте скомпільований пакет моделі (`.tar.gz`) як `Model`.
- Перетворіть вхідне зображення у формат, який очікує модель.
- Виконайте інференс через етапи середовища виконання Neat.
- Декодуйте необроблений вихід детектора в обмежувальні рамки.

Щоб детальніше дізнатися про декодування рамок, порогові значення, NMS і структуру виходу детектора, перейдіть до посібника [Зчитування рамок виявлення з виходу моделі](/tutorials/read-detection-boxes).

## Наступні кроки

Коли YOLOv8 запрацює, переходьте до ширшого кола навчальних матеріалів SiMa.ai Neat:

- Перейдіть до сторінки **[Запуск застосунку](/develop-apps/hello-neat/run_an_app)**, щоб вбудувати цю саму модель у застосунок `Graph` — іменований конвеєр «вхід → модель → вихід», який ви збираєте один раз і далі керуєте ним через push/pull, — замість безпосереднього виклику `Model.run(...)`.
- Вивчіть [базову модель програмування](/develop-apps/development-workflow/overview), яка пояснює основні концепції Neat, як-от моделі, графи та виконання запусків.
- Пройдіть [навчальні матеріали](/tutorials/), де покроково розглядаються конкретні концепції та робочі процеси.
- Перегляньте добірку застосунків на [порталі застосунків](https://apps.neat.sima.ai/portal), а також їхній початковий код у [репозиторії застосунків на GitHub](https://github.com/sima-neat/apps).
