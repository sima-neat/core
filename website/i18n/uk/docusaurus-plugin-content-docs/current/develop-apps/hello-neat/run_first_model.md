---
title: "Запустіть модель."
description: "Запустіть модель на зразку зображення за допомогою Neat."
sidebar_position: 2
---

# Запустіть модель.

Використовуйте ту саму робочу директорію з [Привіт, Neat!](/develop-apps/hello-neat/minimal), щоб запустити реальну модель для виявлення об’єктів.
Ця програма завантажує модель YOLOv8, зчитує зразок зображення, виконує обчислення, декодує обмежувальні рамки та виводить кількість виявлених об’єктів.

На цій сторінці представлено дві концепції Neat:

* [`Model`](/develop-apps/development-workflow/model) завантажує скомпільований пакет моделі та надає вам точку входу `run(...)`.
* [`ModelOptions`](/tutorials/configure-model-options) повідомляє Neat, як підготувати зображення та декодувати вихідні дані детектора.

Вам не потрібно одразу освоювати всі можливості API; наразі зосередьтеся на тому, як `Model` і `ModelOptions` взаємодіють між собою для запуску скомпільованої моделі.

![Привіт, Neat YOLOv8, як справи?](@site/../docs/images/hello-neat-yolov8-flow.svg)

## Отримайте модель і зразок зображення.

1. **Створіть теку для ресурсів**, у якій ми зберігатимемо модель і вхідне зображення:
    ```bash
    mkdir -p assets
    cd assets
    ```
2. **Завантажте модель:**
    ```bash
    sima-cli modelzoo -v 2.0.0 get yolo_v8s
    ```
    :::note sima-cli завантажити модель
    Якщо `sima-cli` зберігає модель в іншому місці, ніж директорія `assets`, скопіюйте цей файл до `assets/yolo_v8s_mpk.tar.gz`.
3. **Завантажте зразок зображення** з документації та збережіть його під назвою `tutorial_sample_image.png` у каталозі `assets`.

    [Open or download the sample image](../../images/tutorial_sample_image.png).
4. **Поверніться до каталогу вашого проєкту:**
    ```bash
    cd ..
    ```

## Покрокова інструкція

Ми беремо за основу програму з [Привіт, Neat!](/develop-apps/hello-neat/minimal): зберігаємо той самий файл `CMakeLists.txt` (він уже містить посилання на Neat та OpenCV) і замінюємо основну частину програми на чотири кроки, наведені нижче. Кожен крок — це невелика частина остаточної програми, тому прочитайте їх по черзі, а потім скопіюйте [повна програма](#full-program), щоб вставити та запустити її. Оберіть потрібну мову в будь-якому блоці; ваш вибір буде застосовано до всього сайту.

### 1. Завантажте та змініть розмір зображення {#step-load-image}

YOLOv8s очікує на отримання зображення BGR фіксованого розміру `640×640`, тому ми зчитуємо зразок за допомогою OpenCV і змінюємо його розмір. Це просте введення/виведення зображень — поки що жодних API Neat.

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

### 2. Опишіть процес введення даних і декодування, використовуючи `ModelOptions` {#step-model-options}.

`ModelOptions` — це контракт середовища виконання, який визначає взаємодію між вашим зображенням і моделлю. Він визначає дві речі: як Neat має попередньо обробляти декодовані пікселі перед виконанням висновків, і як декодувати необроблені дані детектора у вигляді обмежувальних рамок. `decode_type` вибирає декодер YOLOv8, порогові значення відсікають слабкі або перекривні рамки, а `top_k` обмежує їх кількість.

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

### 3. Завантажте модель і запустіть процес обчислення {#step-run}.

Створіть `Model` з скомпільованого пакета `.tar.gz` та параметрів, а потім викличте `run(...)` із заданим часом очікування. Він виконується синхронно та повертає тензори на виході. Параметр `timeout_ms` змушує процес, який завис, завершитися з повідомленням про помилку, замість того, щоб просто «зависнути».

У **C++** на вхід моделі передається один `cv::Mat`. У **Python** спочатку зображення NumPy обгортається у `Tensor` з тегом `BGR`, щоб Neat знав структуру байтів, а потім передається `[tensor]`, оскільки вхідні дані моделі Python є послідовністю.

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

### 4. Перегляньте кількість виявлених об’єктів {#step-read}.

Оскільки параметр `decode_type` було встановлено, перший вихідний тензор містить декодовані обмежувальні рамки. Тензор BBOX починається з лічильника виявлень типу `uint32`, тому ми зчитуємо його перші чотири байти. Повний формат даних — координати, оцінки та класи для кожної рамки — описано в [Зчитайте координати обмежувальних прямокутників, отримані з вихідних даних моделі.](/tutorials/read-detection-boxes).

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

Залиште файл `CMakeLists.txt` з проєкту [Привіт, Neat!](/develop-apps/hello-neat/minimal) (він уже встановлює зв’язок між програмою та Neat і OpenCV), і замініть основну частину програми на повний файл, наведений нижче. Виділені рядки містять три основні частини: створіть `Model`, підготуйте вхідні дані та викличте функцію `run()`.

<details>
<summary>Покажіть повну програму.</summary>

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

## Створіть і запустіть.

Запустіть ці файли з каталогу вашого проєкту (того, який містить `assets/`).

<CodeTabs>
<CodeTab label="C++" lang="cpp">

Відновіть проєкт, використовуючи ті самі команди, що й для Hello Neat!:

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

Точна кількість може відрізнятися залежно від набору моделей і версії середовища виконання. Найважливіше те, що застосунок успішно збирається, запускається та досягає кінцевого результату: `[OK] YOLOv8 completed`.

## Що ви створили.

Цей приклад використовує ту саму загальну схему, що й більші програми Neat:

- Завантажте скомпільований пакет моделі (`.tar.gz`) як `Model`.
- Перетворіть вхідне зображення у формат, який очікує модель.
- Виконуйте обчислення на основі вхідних даних, проходячи через етапи середовища виконання Neat.
- Декодуйте необроблені дані, отримані від детектора, та перетворіть їх на обмежувальні рамки.

Для більш детального пояснення декодування обмежувальних рамок, порогових значень, алгоритму NMS та структури вихідних даних детектора, перейдіть до [Зчитайте координати обмежувальних прямокутників, отримані з вихідних даних моделі.](/tutorials/read-detection-boxes).

## Наступні кроки

Після запуску YOLOv8 продовжуйте використовувати ширший спектр навчальних матеріалів SiMa.ai Neat:

- Продовжуйте, щоб **[Запустіть програму.](/develop-apps/hello-neat/run_an_app)**, і створіть на основі цієї моделі застосунок у вигляді `Graph` — іменований конвеєр «вхід → модель → вихід», який ви створюєте один раз і використовуєте за допомогою механізму «push/pull», замість того, щоб безпосередньо викликати `Model.run(...)`.
- Вивчіть [основна модель програмування](/develop-apps/development-workflow/overview), яка пояснює основні концепції Neat, такі як моделі, графи та виконання програми.
- Дотримуйтесь інструкцій із серії [навчальні матеріали](/tutorials/), де покроково пояснюються конкретні концепції та робочі процеси.
- Перегляньте підбірку застосунків на [портал застосунків](https://apps.neat.sima.ai/portal), а також ознайомтеся з їх вихідним кодом у [репозиторій застосунків на GitHub](https://github.com/sima-neat/apps).
