---
title: "Запуск застосунку"
description: "Запустіть YOLOv8 у графовому застосунку та зчитайте декодовані обмежувальні рамки мовою Python або C++"
sidebar_position: 3
mdx:
  format: mdx
---

# Запуск застосунку

![Складання вашого графа](@site/../docs/images/hello-neat-graph-add-animation.svg)

![Результати виявлення YOLOv8, отримані за допомогою Neat, на вихідному зображенні](@site/../docs/images/first_inference_hook.png)

*Результати виявлення, записані наведеною нижче програмою, нанесені на вихідне зображення.*

Це той самий інференс YOLOv8, але у вигляді невеликого **застосунку**: замість безпосереднього виклику `Model.run(...)` (як на сторінці [Запуск моделі](/develop-apps/hello-neat/run_first_model)) ви вбудовуєте модель у [`Graph`](/develop-apps/development-workflow/graph) — іменований граф обробки зі входом, моделлю та виходом, — а потім збираєте його та надсилаєте/отримуєте дані (push/pull). Та сама програма на Python і C++; виберіть вкладку мови в кожному блоці коду.

Для цього першого застосунку структура навмисно проста:

- Іменований _вхід_ (`nodes.input("image")`) позначає, де дані потрапляють у застосунок.
- _Модель_ (`graph.add(model)`) виконує модель як один крок у графі.
- Іменований _вихід_ (`nodes.output("detections")`) позначає, звідки ваш застосунок зчитує результат.

Згодом той самий API масштабується до значно складніших застосунків; тут мета — базовий шаблон компонування.

:::tip Оберіть мову
Використовуйте вкладки **Python / C++** у будь-якому блоці коду — ваш вибір відповідає перемикачу мови для всього сайту, тому всі фрагменти коду та повна програма перемикаються разом.
:::

## Налаштування проєкту

:::tip Уже пройшли [Запуск моделі](/develop-apps/hello-neat/run_first_model)?
Ви можете пропустити цей розділ: у ньому використовуються та сама тека `assets/`, пакет моделі та зразок зображення. Перейдіть одразу до розділу [Огляд коду](#walk-through-the-code).
:::

1. **Створіть теку для ресурсів** для моделі та вхідного зображення:

    <ShellCommand prompt="sdk|devkit">
    mkdir -p assets
    </ShellCommand>

2. **Завантажте модель** із Model Browser і видобудьте архів моделі із завантаженого ZIP-файлу:

    <ShellCommand prompt="sdk|devkit">
    sima-cli models download --id yolov8s --variant w8a8
    unzip -p models/yolov8s/w8a8/yolov8s_w8a8_default.zip yolov8s_w8a8_mpk.tar.gz > assets/yolo_v8s_mpk.tar.gz
    </ShellCommand>

    :::note Завантаження моделі через sima-cli
    Програма відкриває `assets/yolo_v8s_mpk.tar.gz`. Підготовку моделей для всіх навчальних матеріалів описано в розділі [Налаштування навчального посібника](/tutorials/before-you-run).
    :::
3. **Завантажте зразок зображення** з документації та збережіть його під назвою `assets/tutorial_sample_image.png`.

    [Відкрийте або завантажте зразок зображення](../../images/tutorial_sample_image.png).

## Огляд коду

Програма складається з восьми коротких фрагментів. Перемикайте вкладку мови в кожному блоці.

### 1. Зчитування зображення

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

OpenCV зчитує BGR; YOLOv8 очікує RGB. Цей крок не стосується Neat — ваш застосунок отримує пікселі з файлу, камери або декодера; Neat вступає в дію на наступному кроці.

### 2. Опис параметрів моделі

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

`ModelOptions` визначає маршрут моделі в одному об’єкті: як Neat попередньо обробляє вхідні пікселі та як він декодує вихідні дані детектора.

| Поле | Що задає |
|---|---|
| `preprocess.kind = Image` | На вхід подаються необроблені пікселі, а не попередньо сформований тензор. |
| `preprocess.preset = COCO_YOLO` | Зміна розміру + letterbox (доповнення полями) під вхід моделі, RGB, масштабування на `1/255`, без віднімання середнього. |
| `decode_type = YoloV8` | Сімейство декодерів для виходів детектора. |
| `score_threshold` / `nms_iou_threshold` / `top_k` | Мінімальний поріг впевненості, ступінь перекриття NMS і максимальна кількість збережених обмежувальних рамок. |

### 3. Завантаження моделі

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

`Model` зчитує `.tar.gz`, перевіряє його **MPK-контракт** на відповідність `ModelOptions`, які ви надали, і створює фрагмент моделі. Поки що нічого не виконується.

### 4. Обгортання зображення в `Tensor`

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

`Tensor` — це типізований контейнер даних у Neat: форма, тип даних, розкладка та формат пікселів, потрібний фреймворку для інтерпретації байтів. Передавати `PixelFormat` обов’язково, щоб Neat знав розкладку, а не лише самі байти.

### 5. Компонування графа

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

`Graph` — це послідовність обробки в застосунку. Кожен `add(...)` додає наступний крок, тож тут будується лінійна послідовність `image → model → detections`. Фрагмент моделі з кроку 3 стає одним із кроків у ній.

### 6. Збирання та запуск графа

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

`build()` перетворює публічний граф на один виконуваний граф середовища виконання, зберігаючи назви ваших вузлів. Потім ви надсилаєте (`push`) вхідні дані в іменовані входи, викликаєте `close_input()`, коли вхідних даних більше не буде, і отримуєте (`pull`) результати з іменованих виходів із тайм-аутом. `pull_tensors` повертає `TensorList` — ту саму структуру, яку повернув би `Model.run`, — тут це упакований вихід YOLOv8 `BBOX`.

### 7. Декодування рамок

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

`decode_bbox` — це перетворення `TensorList → TensorList`, позиційно 1:1. Кожен декодований вихід — це тензор `float32` форми `[num_detections, 6]` зі стовпцями `(x1, y1, x2, y2, score, class_id)`.

### 8. Зчитування рамок

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

У Python декодований тензор читається як масив NumPy `[N, 6]` через `to_numpy()`. У C++ ви відображаєте тензор у пам’ять і зчитуєте значення з рухомою комою. Модель видає ідентифікатори класів COCO; зіставлення їх із назвами для відображення — завдання застосунку.

## Повна програма

Створіть файли в теці вашого проєкту, а потім зберіть і запустіть.

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

**Запуск:**

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 app.py
</ShellCommand>

<ShellCommand prompt="sdk">
dk app.py
</ShellCommand>

</CodeTab>
<CodeTab label="C++" lang="cpp">

Створіть файли `CMakeLists.txt` та `main.cpp`:

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

Саме два виділені рядки підключають ваш застосунок до Neat: `find_package(SimaNeat REQUIRED CONFIG)` знаходить встановлений пакет Neat (через `SimaNeatConfig.cmake`), а `target_link_libraries(sima_neat_app PRIVATE SimaNeat::sima_neat ...)` підключає його до застосунку — імпортована ціль `SimaNeat::sima_neat` автоматично передає теки заголовків і транзитивні залежності Neat, тому не потрібно вказувати шляхи до заголовків/бібліотек вручну. (`PkgConfig::OPENCV` потрібен лише тому, що цей застосунок використовує OpenCV для завантаження зображень.)

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

**Збирання:**

<ShellCommand prompt="sdk|devkit">
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
</ShellCommand>

**Запуск:**

<ShellCommand prompt="devkit">
./build/sima_neat_app
</ShellCommand>

<ShellCommand prompt="sdk">
dk build/sima_neat_app
</ShellCommand>

</CodeTab>
</CodeTabs>

Ви повинні бачити один рядок для кожного виявленого об’єкта, а потім:

```text
[OK] Graph app completed
```

## Що зібрав Neat

![Схема графового застосунку Hello Neat](@site/../docs/images/hello-neat-graph-app-flow.svg)

API безпосередньо відповідають цій структурі:

- `Graph` містить послідовність обробки застосунку; `graph.add(...)` додає кожен крок по черзі.
- Іменовані вхід і вихід стають кінцевими точками середовища виконання: `run.push("image", ...)` і `run.pull_tensors("detections")`.
- `Model` — це той самий фрагмент, який ви б викликали безпосередньо через `Model.run`; тут він виконується як один вузол усередині застосунку.

## Наступні кроки

Щоб глибше опанувати компонування графів, перейдіть до [моделі програмування з графами](/develop-apps/development-workflow/graph).

Звідти перейдіть до ширшого кола навчальних матеріалів SiMa.ai Neat:

- Вивчіть [базову модель програмування](/develop-apps/development-workflow/overview), яка пояснює основні концепції Neat, як-от моделі, графи та виконання запусків.
- Пройдіть [навчальні матеріали](/tutorials/), де покроково розглядаються конкретні концепції та робочі процеси.
- Перегляньте добірку застосунків на [порталі застосунків](https://apps.neat.sima.ai/portal), а також їхній початковий код у [репозиторії застосунків на GitHub](https://github.com/sima-neat/apps).
