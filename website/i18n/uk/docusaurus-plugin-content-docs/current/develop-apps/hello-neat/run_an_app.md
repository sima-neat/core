---
title: "Запустіть програму."
description: "Запустіть YOLOv8 усередині графової програми та зчитайте декодовані координати обмежувальних прямокутників, використовуючи Python або C++."
sidebar_position: 3
mdx:
  format: mdx
---

# Запустіть програму.

![Збірка вашого графа.](@site/../docs/images/hello-neat-graph-add-animation.svg)

![Результати детектування YOLOv8, отримані за допомогою Neat, на вихідному зображенні.](@site/../docs/images/first_inference_hook.png)

*Результати, отримані програмою, відображені на вихідному зображенні.*

Це такий самий висновок YOLOv8, як і в невеликій **програмі**: замість безпосереднього виклику `Model.run(...)` (як у [Запустіть модель.](/develop-apps/hello-neat/run_first_model)), ви створюєте модель у вигляді [`Graph`](/develop-apps/development-workflow/graph) — іменованого графа, що містить вхідні дані, модель і вихідні дані, — а потім збираєте його та передаєте/отримуєте дані. Та сама програма, написана мовами Python і C++; виберіть потрібну мову у відповідному блоці коду.

Для цієї першої програми структура навмисно проста:

- Іменований параметр _вхідних даних_ (`nodes.input("image")`) позначає місце, звідки дані надходять у застосунок.
- _Модель_ (`graph.add(model)`) виконує модель як один крок у графі.
- Іменований параметр _output_ (`nodes.output("detections")`) вказує, звідки ваша програма отримує результат.

Цей самий API можна використовувати для розробки набагато складніших застосунків у майбутньому; тут основна мета полягає в реалізації базової моделі компонування.

:::tip Оберіть мову.
Використовуйте вкладки **Python / C++** у будь-якому блоці коду — ваш вибір відповідатиме загальному мовному селектору сайту, тому всі фрагменти коду та повна програма перемикатимуться одночасно.
:::

## Налаштуйте проєкт.

:::tip Вже запустили [Запустіть модель.](/develop-apps/hello-neat/run_first_model)?
Ви можете пропустити цей розділ, оскільки в ньому використовуються ті самі `assets/`, пакунок моделі та зразок зображення. Перейдіть одразу до [Перегляньте код.](#walk-through-the-code).
:::

1. **Створіть теку для ресурсів** для моделі та вхідного зображення:
    ```bash
    mkdir -p assets
    ```
2. **Завантажте модель:**
    ```bash
    sima-cli modelzoo -v 2.0.0 get yolo_v8s
    ```
    :::note sima-cli завантажити модель
    Якщо `sima-cli` зберігає модель в іншому місці, ніж директорія `assets`, скопіюйте цей файл до `assets/yolo_v8s_mpk.tar.gz`.
3. **Завантажте зразок зображення** з документації та збережіть його під назвою `assets/tutorial_sample_image.png`.

    [Open or download the sample image](../../images/tutorial_sample_image.png).

## Перегляньте код.

Програма складається з восьми коротких фрагментів. Перемикайте вкладку мови для кожного блоку.

### 1. Зчитайте зображення.

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

OpenCV зчитує BGR; YOLOv8 очікує RGB. Цей крок не є Neat — ваша програма отримує пікселі з файлу, камери або декодера; Neat переходить до наступного кроку.

### 2. Опишіть доступні варіанти моделі.

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

| Поле | Що в ньому міститься. |
|---|---|
| `preprocess.kind = Image` | На вхід подаються необроблені пікселі, а не попередньо сформований тензор. |
| `preprocess.preset = COCO_YOLO` | Змініть розмір і додайте рамки, щоб відповідало вхідним даним моделі, RGB, масштабуйте на `1/255`, не виконуйте віднімання середнього значення. |
| `decode_type = YoloV8` | Сімейство декодерів, що використовують метод виявлення об’єктів. |
| `score_threshold` / `nms_iou_threshold` / `top_k` | Мінімальний поріг впевненості, ступінь перекриття NMS і максимальна кількість збережених обмежувальних рамок. |

### 3. Завантажте модель.

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

`Model` зчитує `.tar.gz`, перевіряє його **MPK-контракт** на відповідність `ModelOptions`, які ви надали, і створює фрагмент моделі. Наразі жоден процес не було запущено.

### 4. Перетворіть ваше зображення на `Tensor`.

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

`Tensor` — це контейнер для даних із зазначенням типу, який використовується в Neat: він містить інформацію про розмір, тип даних, структуру та формат пікселів, необхідний для інтерпретації байтів. Передача `PixelFormat` є обов’язковою, щоб Neat знав структуру даних, а не лише самі байти.

### 5. Створіть граф.

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

`Graph` – це схема взаємодії між компонентами застосунку. Кожен `add(...)` додає наступний крок, таким чином формуючи лінійний потік `image → model → detections`. Фрагмент моделі з кроку 3 стає одним із кроків у цьому потоці.

### 6. Створіть і запустіть граф.

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

`build()` перетворює загальнодоступний граф в один виконуваний граф для середовища виконання, зберігаючи назви ваших вузлів. Потім ви `push` вхідні дані в іменовані вхідні дані, викликаєте `close_input()`, коли більше не надходить вхідних даних, і `pull` результати з іменованих вихідних даних із заданим часом очікування. `pull_tensors` повертає `TensorList` — тензор зі структурою, яку б створив `Model.run` — тут це об’єднаний вихід YOLOv8 `BBOX`.

### 7. Розшифруйте вміст коробок.

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

`decode_bbox` — це `TensorList → TensorList` трансформація, що застосовується в співвідношенні 1:1. Кожен декодований вихід є `float32` тензором розмірності `[num_detections, 6]`, що містить стовпці `(x1, y1, x2, y2, score, class_id)`.

### 8. Зчитайте обмежувальні рамки

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

У Python декодований тензор зчитується як масив NumPy за допомогою `[N, 6]` та функції `to_numpy()`. У C++ ви відображаєте тензор і зчитуєте значення з плаваючою комою. Модель генерує ідентифікатори класів COCO; відображення їх на відображувані назви здійснюється в застосунку.

## Повна програма

Створіть файли у каталозі вашого проєкту, а потім скомпілюйте та запустіть їх.

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

**Запустити:**

* **Щодо DevKit**
  ```bash
  source ~/pyneat/bin/activate
  python3 app.py
  ```
* **На хості Neat SDK**
  ```bash
  dk app.py
  ```

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

Дві виділені рядки показують, як ваш застосунок пов’язаний із Neat: `find_package(SimaNeat REQUIRED CONFIG)` знаходить встановлений пакет Neat (через `SimaNeatConfig.cmake`), а `target_link_libraries(sima_neat_app PRIVATE SimaNeat::sima_neat ...)` встановлює зв’язок із ним — імпортована ціль `SimaNeat::sima_neat` автоматично передає каталоги заголовків і транзитивні залежності Neat, тому не потрібно вказувати шляхи до заголовків/бібліотек вручну. (`PkgConfig::OPENCV` потрібен лише тому, що цей застосунок використовує OpenCV для завантаження зображень).

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

**Збірка:**

<ShellCommand prompt="sdk|devkit">
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
</ShellCommand>

**Запустити:**

* **Щодо DevKit**
  ```bash
  ./build/sima_neat_app
  ```
* **На хості Neat SDK**
  ```bash
  dk build/sima_neat_app
  ```

</CodeTab>
</CodeTabs>

Ви повинні бачити один рядок для кожного виявленого об’єкта, а потім:

```text
[OK] Graph app completed
```

## Що зібрав Neat?

![Привіт Neat Схема взаємодії в графічному застосунку.](@site/../docs/images/hello-neat-graph-app-flow.svg)

Інтерфейси прикладного програмування (API) безпосередньо відображаються на цю структуру:

- `Graph` містить схему взаємодії застосунку; `graph.add(...)` додає кожен крок у відповідному порядку.
- Вказані вхідні та вихідні дані стають кінцевими точками середовища виконання: `run.push("image", ...)` і `run.pull_tensors("detections")`.
- `Model` — це той самий фрагмент, який ви б викликали безпосередньо за допомогою `Model.run`; тут він виконується як один вузол усередині програми.

## Наступні кроки

Для більш глибокої композиції графа продовжуйте використовувати [Модель графового програмування.](/develop-apps/development-workflow/graph).

Звідти перейдіть до ширшого кола навчальних матеріалів SiMa.ai Neat:

- Вивчіть [основна модель програмування](/develop-apps/development-workflow/overview), яка пояснює основні концепції Neat, такі як моделі, графи та виконання програми.
- Дотримуйтесь інструкцій із серії [навчальні матеріали](/tutorials/), де покроково пояснюються конкретні концепції та робочі процеси.
- Перегляньте підбірку застосунків на [портал застосунків](https://apps.neat.sima.ai/portal), а також ознайомтеся з їх вихідним кодом у [репозиторій застосунків на GitHub](https://github.com/sima-neat/apps).
