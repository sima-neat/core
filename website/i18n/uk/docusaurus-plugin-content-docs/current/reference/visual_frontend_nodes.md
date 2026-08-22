---
title: "EV74: Візуальні вузли інтерфейсу."
description: "У стилі клієнта. Neat Використання графа для FeatureHistogram, GriderFast, TrackDescriptor і TrackKLT."
sidebar_position: 8
---

# Візуальний інтерфейс EV74: вузли.

Neat надає доступ до візуальних графів EV74 як до звичайних вузлів `Graph`. Використовуйте загальнодоступні фабрики вузлів і структури параметрів; не викликайте `processcvu`, ConfigManager або API диспетчера безпосередньо з коду застосунку.

| Фабрика вузлів | Назва графа | Ідентифікатор графа | Призначення |
| --- | --- | ---: | --- |
| `nodes::FeatureHistogram` / `pyneat.nodes.feature_histogram` | `feature_histogram` | 235 | Гістограма зображення у відтінках сірого |
| `nodes::GriderFast` / `pyneat.nodes.grider_fast` | `grider_fast` | 236 | Функції FAST, розподілені по сітці |
| `nodes::TrackDescriptor` / `pyneat.nodes.track_descriptor` | `track_descriptor` | 237 | Швидкісні характеристики плюс дескриптори |
| `nodes::TrackKLT` / `pyneat.nodes.track_klt` | `track_klt` | 238 | Пірамідальне відстеження KLT, опціонально з виявленими замінюючими ознаками |

Ідентифікатори графів корисні для діагностики та перевірки відповідності версій програмного забезпечення/пакетів. Вони не є обов’язковими в коді застосунку.

## Тензорне скорочення.

Усі тензори використовують **логічні розміри пакету**. Якщо `batch_size == B`, то зображення у відтінках сірого має розмір `[B,H,W]`, а не `[B*H,W]`. Середовище виконання обробляє будь-яке внутрішнє пакування даних EV74.

| Вузол | Вхідні дані | Відкриті вихідні дані |
| --- | --- | --- |
| `FeatureHistogram` | `input_image`: UInt8 `[B,H,W]` | `output_hist`: Int32 `[B,256]` |
| `GriderFast` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]` |
| `TrackDescriptor` | `input_image`: UInt8 `[B,H,W]` | `output_features`: Int32 `[B,1 + max_features*3]`; `output_descriptors`: Int32 `[B,max_features,8]` |
| `TrackKLT` | `prev_image`: UInt8 `[B,H,W]`; `cur_image`: UInt8 `[B,H,W]`; `input_points`: Int32 `[B,num_points,2]` | `output_points`: Float32 `[B,num_points,2]`; `output_status`: Int32 `[B,num_points,1]`; плюс `output_features`: Int32 `[B,1 + max_features*3]` лише коли `detect_new_features != 0` |

Тензори зі списком характеристик використовують такий формат для кожної партії даних:

```text
[count, x0, y0, score0, x1, y1, score1, ...]
```

Наразі для дескриптора графа потрібно `descriptor_words == 8`. Зміна цього є зміною ABI EV74 і відхиляється до моменту відправки.

## Швидкий старт з C++

```cpp
#include <neat.h>

#include <cstdint>
#include <vector>

using namespace simaai::neat;

Tensor make_gray_batch(int width, int height, int batch) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * batch);
  // Fill pixels in batch-major order: b*height*width + y*width + x.
  auto tensor = Tensor::from_vector(pixels, {batch, height, width}, TensorMemory::EV74);
  tensor.layout = TensorLayout::HW;
  tensor.axis_semantics = {TensorAxisSemantic::N, TensorAxisSemantic::H, TensorAxisSemantic::W};
  tensor.route.name = "input_image";
  tensor.route.segment_name = "input_image";
  return tensor;
}

int main() {
  constexpr int width = 320;
  constexpr int height = 240;
  constexpr int batch = 2;

  Graph graph;

  InputOptions input;
  input.payload_type = PayloadType::Tensor;
  input.format = FormatTag::UINT8;
  input.width = width;
  input.height = height;
  input.depth = 1;
  input.max_width = width;
  input.max_height = height * batch; // transport capacity; public tensor remains [B,H,W]
  input.max_depth = 1;
  input.memory_policy = InputMemoryPolicy::Ev74;
  input.buffer_name = "input_image";

  graph.add(nodes::Input(input));

  GriderFastOptions fast;
  fast.width = width;
  fast.height = height;
  fast.batch_size = batch;
  fast.max_features = 64;
  fast.threshold = 30;
  graph.add(nodes::GriderFast(fast));

  graph.add(nodes::Output());

  RunOptions run_opt;
  run_opt.output_memory = OutputMemory::Owned;

  Tensor image = make_gray_batch(width, height, batch);
  Run run = graph.build({image}, run_opt);
  TensorList outputs = run.run({image}, /*timeout_ms=*/30000);
  run.close();
}
```

## KLT з трьома входами.

`TrackKLT` використовує набір тензорів: попереднє зображення, поточне зображення та вхідні точки. Назвіть шляхи, щоб вони відповідали полям параметрів.

```cpp
TrackKLTOptions klt;
klt.width = 320;
klt.height = 240;
klt.batch_size = 2;
klt.num_points = 32;
klt.max_features = 64;
klt.detect_new_features = 1; // publish output_features as the third output

graph.add(nodes::TrackKLT(klt));
```

Очікувані результати, які будуть доступні для загального користування, коли `detect_new_features == 1`:

```text
output_points   Float32 [2,32,2]
output_status   Int32   [2,32,1]
output_features Int32   [2,193]
```

Коли `detect_new_features == 0`, Neat публікує лише `output_points` та `output_status`; буфер видимих для EV функцій залишається внутрішнім ресурсом середовища виконання.

## Поверхня, створена за допомогою Python.

API Python повторює структуру C++, що передбачає використання опцій/фабрик, і має шароподібну структуру: створіть об’єкт опцій, задайте загальнодоступні параметри конфігурації та додайте вузол до `Graph`.

```python
import numpy as np
import pyneat

width, height, batch = 320, 240, 2

opt = pyneat.GriderFastOptions()
opt.width = width
opt.height = height
opt.batch_size = batch
opt.max_features = 64
print(opt.summary())

graph = pyneat.Graph()
input_opt = pyneat.InputOptions()
input_opt.payload_type = pyneat.PayloadType.Tensor
input_opt.format = pyneat.Format.UINT8
input_opt.width = width
input_opt.height = height
input_opt.max_width = width
input_opt.max_height = height * batch
input_opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
input_opt.buffer_name = "input_image"

graph.add(pyneat.nodes.input(input_opt))
graph.add(pyneat.nodes.grider_fast(opt))
graph.add(pyneat.nodes.output())

image_np = np.zeros((batch, height, width), dtype=np.uint8)
image = pyneat.Tensor.from_numpy(image_np, memory="ev74")
image.layout = pyneat.TensorLayout.HW
# If setting route metadata from Python in a custom app, keep it aligned with
# the option names used above.
```

## Перевірки безпеки.

Ці вузли перевіряють межі графа перед відправкою даних про електромобіль (EV). Вони відхиляють:

- невід’ємні розміри або кількості;
- непідтримувані розміри пакетів;
- порогові значення поза межами `[0,255]`;
- дублікати/порожні назви тензорів;
- `TrackDescriptorOptions.descriptor_words != 8`;
- недійсні значення для параметрів KLT: вікно, рівень і режим виявлення;
- тензори середовища виконання недостатнього розміру під час попереднього узгодження.

Це важливо, оскільки нелегальні буфери можуть спричинити проблеми з EV74. Залишайте помилки під час перевірки як помилки на стороні хоста і не обходьте шлях, визначений контрактом Node.

## Швидка команда для перевірки.

Швидкий клієнтський DevKit інтерфейс:

```bash
ctest --test-dir /workspace/core_graph_changes/build/tests \
  -R visual_frontend_ --output-on-failure
```

Він працює:

- усі чотири візуальні графіки з `320x240`, `batch_size=2`, `detect_new_features=1`;
- цілеспрямована перевірка KLT на відсутність виявлення за допомогою ABI;
- негативна перевірка перед відправленням, яка підтверджує наявність несанкціонованих даних у пакеті.
  відхилено до версії EV74.
