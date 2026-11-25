---
title: "Попередня обробка списків ROI"
description: "Запуск Preproc для кількох вікон ROI, заданих під час виконання, на одному або кількох зображеннях"
sidebar_position: 7
---

# Попередня обробка списків ROI

Використовуйте список ROI для Preproc, якщо ваш застосунок уже має одне або кілька вікон на зображеннях і кожне вікно треба попередньо обробити точно так, як вхід моделі: змінити розмір або застосувати letterbox, перетворити колір, нормалізувати, квантувати та, за потреби, виконати теселяцію.

Цей інструмент підходить для класифікаторів другого етапу, каскадів детекторів, трекерів і будь-якого процесу, у якому пакет зображень джерела дає динамічний список вирізаних фрагментів.

## Мінімальний приклад

C++:

```cpp
#include <neat.h>
#include <opencv2/imgcodecs.hpp>

using namespace simaai::neat;

Model::Options opt;
opt.preprocess.resize.enable = AutoFlag::On;
Model model("/path/to/model.tar.gz", opt);

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

opt = pyneat.ModelOptions()
opt.preprocess.resize.enable = pyneat.AutoFlag.On
model = pyneat.Model("/path/to/model.tar.gz", opt)

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

## Налаштування зміни розміру, letterbox і нормалізації

Preproc зі списком ROI використовує ті самі параметри попередньої обробки моделі, що й Preproc для цілого кадру.

C++:

```cpp
Model::Options opt;
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

Підтримувані значення `scaling_type`: `BILINEAR`, `NEAREST_NEIGHBOUR`, `BICUBIC`, `INTERAREA` і `NO_SCALING`. `NEAREST_NEIGHBOR` і `INTER_AREA` приймаються як псевдоніми.

## Семантика пакета

| Вхід | Значення |
| --- | --- |
| `images` | Пакет зображень джерела. Якщо передано ROI, вектор або список не може бути порожнім. Python приймає зображення uint8 HW/HWC у вигляді NumPy, Torch або `pyneat.Tensor`. |
| `rois` | Список ROI, заданий під час виконання. Порядок вихідних тензорів відповідає порядку в цьому векторі. |
| `PreprocessRoi::batch_index` | Вибирає зображення джерела в `images`, з якого ROI читає дані. |
| `PreprocessRoi::x`, `y` | Знакова координата верхнього лівого кута в кадрі джерела. Від’ємні значення допускаються. |
| `PreprocessRoi::width`, `height` | Розміри ROI. Обидва значення мають бути додатними. |

Усі зображення джерела в одному виклику зі списком ROI повинні мати однакові ширину, висоту, тип OpenCV і кількість каналів. API етапу підтримує для списків ROI упаковані 8-бітні зображення джерела RGB/BGR (`CV_8UC3`) і GRAY/GRAY8 (`CV_8UC1`).

У Python передавайте `image_format=pyneat.PixelFormat.BGR` для зображень із `cv2.imread`, `RGB` для масивів RGB або `GRAY8` для масивів HW у відтінках сірого. `pyneat.stages.preproc` відхиляє тензори CHW; перетворіть їх на HWC перед викликом етапу.

## Семантика виходу

`stages::Preproc(images, model, rois)` повертає:

- `out.size() == rois.size()` для коректного непорожнього запиту;
- вихідний тензор `out[i]`, отриманий з ROI `rois[i]`;
- тип даних і розкладку, які вибирає маршрут моделі, зокрема BF16 або INT8 і щільний або теселований вихід;
- метадані `tensor.semantic.preprocess` для кожного виходу.

Метадані попередньої обробки кожного виходу містять вибрану ROI, геометрію letterbox, прапорці нормалізації, квантування й теселяції та афінне перетворення, яке відображає координати моделі або попередньо обробленого зображення назад у систему координат початкового кадру джерела. Підтримувані профілі ROI описано на сторінці [BoxDecode](/reference/boxdecode_decode_types#crop-coordinate-mapping).

## ROI за межами кадру

ROI можуть виходити за межі зображення джерела:

```cpp
std::vector<PreprocessRoi> rois = {
    {0, -16, -16, 128, 128},
    {0, image.cols - 64, image.rows - 64, 128, 128},
};
```

Частина в межах кадру копіюється, а частина за межами кадру заповнюється значенням доповнення Preproc. Так форма виходу залишається сталою, і краї зображення не потребують окремої обробки.

## Letterbox і співвідношення сторін

Для `ResizeMode::Letterbox` Preproc обчислює масштаб і доповнення letterbox окремо для кожної ROI. Тому висока ROI і широка ROI можуть мати різні значення `scaled_width`, `scaled_height`, `pad_left` і `pad_top` у метаданих, навіть якщо цільовий розмір у них однаковий.

Код на наступних етапах має читати метадані, а не обчислювати доповнення заново на основі припущень.

## Контрольний список перевірки

Перш ніж звинувачувати вихід інференсу, перевірте, що:

- вектор зображень не порожній;
- усі зображення мають однаковий розмір, тип OpenCV і кількість каналів;
- входи Python — це зображення uint8 HW/HWC, а не тензори CHW;
- кожна ROI має коректний `batch_index`;
- кожна ROI має додатні `width` і `height`;
- формат зображення джерела для режиму зі списком ROI — RGB, BGR, GRAY або GRAY8;
- режим зміни розміру й нормалізація відповідають попередній обробці, з якою навчали модель;
- споживачі координат на наступних етапах використовують метадані `tensor.semantic.preprocess`.

## Тести цієї поведінки

У репозиторії є швидкі тести для користувацького шляху та функціональної поведінки ROI:

- `preproc_roi_batch_functional_test`
- `preproc_roi_user_smoke_test`

Вони охоплюють кілька зображень, кілька ROI, доповнення за межами кадру, зміну розміру й letterbox, нормалізований вихід, а також щільні й теселовані маршрути BF16/INT8.

## Див. також

- [Вузол Preproc](/reference/nodes/preproc)
- [Попередня обробка зображень перед інференсом](/tutorials/preprocess-images)
- [Формати даних](/develop-apps/advanced-concepts/data_formats)
- [BoxDecode: типи декодування](/reference/boxdecode_decode_types)
