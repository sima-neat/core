---
title: "Попередня обробка списків ROI."
description: "Запустіть попередню обробку для кількох вікон ROI у середовищі виконання на одному або кількох зображеннях."
sidebar_position: 7
---

# Попередня обробка списків ROI.

Використовуйте список попередньо оброблених областей інтересу (ROI), якщо ваш застосунок вже має одне або кілька вікон зображень і ви хочете, щоб кожне вікно було попередньо оброблено точно так, як це потрібно для вхідних даних моделі: зміна розміру або застосування методу letterbox, перетворення кольору, нормалізація, квантування та, за бажанням, розбиття на частини.

Це ідеальний інструмент для класифікаторів другого рівня, каскадів детекторів, трекерів або будь-якого процесу, де пакет вихідних зображень генерує динамічний список фрагментів.

## Мінімальний приклад.

C++:

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

Python:

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

## Налаштуйте зміну розміру, додавання чорних смуг і нормалізацію.

ROI-list Preproc використовує ті самі параметри попередньої обробки моделі, що й повноекранний Preproc.

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

Python:

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

Підтримувані значення для `scaling_type` включають `BILINEAR`, `NEAREST_NEIGHBOUR`, `BICUBIC`, `INTERAREA` та `NO_SCALING`. `NEAREST_NEIGHBOR` і `INTER_AREA` є допустимими альтернативними назвами.

## Пакетна семантика.

| Вхідні дані | Значення |
| --- | --- |
| `images` | Пакет вихідних зображень. Вектор/список не повинен бути порожнім, якщо надаються області інтересу (ROIs). Python приймає вхідні дані у вигляді зображень у форматі uint8 HW/HWC NumPy/Torch/`pyneat.Tensor`. |
| `rois` | Список областей інтересу (ROI) для середовища виконання. Порядок елементів у вихідному тензорі відповідає порядку елементів у цьому векторі. |
| `PreprocessRoi::batch_index` | визначає, з якого вихідного зображення в `images` зчитуються дані для області інтересу (ROI). |
| `PreprocessRoi::x`, `y` | Підписані координати верхнього лівого кута вихідного кадру. Допускаються від’ємні значення. |
| `PreprocessRoi::width`, `height` | Розміри області інтересу (ROI). Обидва значення мають бути додатними. |

Усі вихідні зображення у списку ROI повинні мати однакову ширину, висоту, тип OpenCV і кількість каналів. API етапу підтримує упаковані 8-бітові RGB/BGR (`CV_8UC3`) і GRAY/GRAY8 (`CV_8UC1`) вихідні зображення для списків ROI.

Для Python передавайте `image_format=pyneat.PixelFormat.BGR` для зображень, отриманих за допомогою `cv2.imread`, `RGB` для RGB-масивів або `GRAY8` для HW-масивів у градаціях сірого. `pyneat.stages.preproc` відхиляє тензори CHW; перед викликом етапу перетворіть їх на HWC.

## Семантика виводу

`stages::Preproc(images, model, rois)` повертає:

- `out.size() == rois.size()` для коректного запиту, який не є порожнім;
- вихідний тензор `out[i]`, отриманий з області інтересу `rois[i]`;
- тип даних і структура, обрані моделлю, зокрема BF16 або INT8, а також щільний або розділений на блоки вихідний формат;
- на вихід `tensor.semantic.preprocess` метадані.

Кожен результат попередньої обробки містить інформацію про вибрану область інтересу (ROI), геометрію кадрування, прапори нормалізації/квантування/триангуляції та афінне перетворення, яке відображає координати моделі/обробленого зображення назад у вихідну систему координат.

## Області інтересу, що виходять за межі кадру.

Області інтересу (ROI) можуть виходити за межі вихідного зображення:

```cpp
std::vector<PreprocessRoi> rois = {
    {0, -16, -16, 128, 128},
    {0, image.cols - 64, image.rows - 64, 128, 128},
};
```

Область, що знаходиться в межах кадру, копіюється, а область, що знаходиться за межами кадру, заповнюється значенням, вказаним у параметрі Preproc pad. Це забезпечує стабільність розміру вихідного зображення та усуває необхідність обробки окремих випадків поблизу країв зображення.

## Формат зображення та співвідношення сторін.

Для режиму `ResizeMode::Letterbox` модуль попередньої обробки обчислює масштаб і відступи для кожного окремого регіону інтересу (ROI). Таким чином, високий ROI і широкий ROI можуть мати різні значення `scaled_width`, `scaled_height`, `pad_left` і `pad_top` у метаданих, навіть якщо вони мають однаковий цільовий розмір.

Подальший код повинен зчитувати метадані, а не перераховувати відступи на основі припущень.

## Список для перевірки.

Перш ніж критикувати результати роботи моделі, перевірте:

- вектор зображення не є порожнім;
- кожне зображення має однаковий розмір, тип OpenCV і кількість каналів;
- Вхідні дані Python – це зображення у форматі uint8 HW/HWC, а не тензори у форматі CHW.
- для кожного регіону інтересів (ROI) наявний дійсний `batch_index`;
- для кожного ROI значення `width` і `height` є додатними;
- формат вихідного зображення може бути RGB, BGR, GRAY або GRAY8 для режиму «список областей інтересу»;
- режим зміни розміру та нормалізація відповідають попередній обробці даних, що використовувалася під час навчання моделі;
- споживачі координат на наступних етапах обробки використовують метадані `tensor.semantic.preprocess`.

## Тести, що охоплюють цю поведінку.

Цей репозиторій містить швидке тестування для основних сценаріїв використання та функціональної поведінки, що впливає на користувача:

- `preproc_roi_batch_functional_test`
- `preproc_roi_user_smoke_test`

Вони охоплюють обробку кількох зображень, кількох областей інтересу, заповнення країв зображення, поведінку при зміні розміру/додаванні чорних смуг, нормалізовані вихідні дані, а також щільні/розділені на частини обчислення у форматах BF16/INT8.

## Див. також.

- [Вузол попередньої обробки.](/reference/nodes/preproc)
- [Підготуйте зображення перед виконанням обчислень.](/tutorials/preprocess-images)
- [Формати даних](/develop-apps/advanced-concepts/data_formats)
- [Декодування типів даних у форматі Box.](/reference/boxdecode_decode_types)
