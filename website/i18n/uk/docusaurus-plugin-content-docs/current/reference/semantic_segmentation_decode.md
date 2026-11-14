---
title: "Декодування семантичної сегментації"
description: "Перетворюйте оцінки класів на масштабовану карту класів на EV74 за допомогою SimaResizeArgmax або параметрів семантичної сегментації Model"
sidebar_position: 9
---

# Декодування семантичної сегментації

`SimaResizeArgmax` перетворює оцінки класів моделі сегментації на карту класів
вибраного розміру. Для кожного вихідного пікселя він білінійно інтерполює оцінку
кожного класу й повертає клас із найбільшою оцінкою. Саме таке декодування потрібне
SegFormer та подібним моделям після MLA: `interpolate(mode="bilinear")` до цільового
розміру, а потім `argmax` за класами. Результат збігається з картою класів Hugging Face
`SegformerImageProcessor.post_process_semantic_segmentation` із `target_sizes`, а
масштабовані оцінки ніколи не зберігаються.

Декодування виконується на EV74. Реалізації для A65 і резервного шляху немає.

## Декодування виходу моделі

Задайте `Model::Options::semantic_segmentation`, щоб замінити постобробку dequantize або cast
моделі цим декодуванням. Тоді вузол читає оцінки MLA на місці, зокрема виходи MLA з
доповненими збереженими каналами, і модель повертає одну карту класів замість оцінок.

```cpp
#include <neat.h>

using namespace simaai::neat;

Model::Options options;
options.semantic_segmentation.enable = true;
options.semantic_segmentation.output_width = 1280; // camera frame size
options.semantic_segmentation.output_height = 720;
Model model("segformer_b0_ade_512_mpk.tar.gz", options);
```

```python
import pyneat

options = pyneat.ModelOptions()
options.semantic_segmentation.enable = True
options.semantic_segmentation.output_width = 1280  # camera frame size
options.semantic_segmentation.output_height = 720
model = pyneat.Model("segformer_b0_ade_512_mpk.tar.gz", options)
```

Поєднуйте ці параметри з параметрами попередньої обробки, потрібними вашій моделі.
Вихід моделі — один тензор карти класів `[N,H,W]`, а
`model.info().selection.selected_post_kind` дорівнює `"resize_argmax"`.

- Залиште `output_width` і `output_height` обидва рівними 0, щоб використати вхідний
  розмір моделі, або задайте обом розмір вихідного кадру, щоб кожен піксель джерела
  отримав клас.
- Типове значення `align_corners = false` використовує центри з півпіксельним зсувом, як
  PyTorch і Hugging Face. Встановлюйте `true` лише для моделі, навченої з інтерполяцією,
  вирівняною за кутами.
- Постобробка моделі має містити лише етапи dequantize або cast, а `decode_type` має
  залишатися незаданим. Інакше конструктор `Model` викидає `std::invalid_argument`
  (`ValueError` у Python).

## Використання вузла в графі

`nodes::SimaResizeArgmax(...)` у C++ і `pyneat.nodes.sima_resize_argmax(options)` у Python
виконують те саме декодування для будь-якого тензора, що відповідає наведеному нижче
контракту. Цей фрагмент декодує один тензор оцінок, `scores`, у карту класів 512 на 512:

```cpp
SimaResizeArgmaxOptions decode;
decode.output_width = 512;
decode.output_height = 512;

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaResizeArgmax(decode));
graph.add(nodes::Output());
auto run = graph.build(TensorList{scores});
TensorList class_map = run.run(TensorList{scores}, 1000);
```

## Контракт входу й виходу

| Порт | Тензор |
| --- | --- |
| `scores` | `[h,w,C]` або `[N,h,w,C]`, класи на останній осі; INT8, UINT8, BF16 або FP32; щільний або з кроками, де крок пікселя більший за `C` |
| `class_map` | `[H,W]` або `[N,H,W]`, щільний; UINT8, коли `C` не більше 256, інакше INT32 |

- Цілочислові оцінки порівнюються без деквантування. Це дає ті самі класи, що й
  деквантовані оцінки, коли один додатний масштаб охоплює всі класи. Оцінки з
  поосьовим квантуванням або з недодатним масштабом відхиляються під час побудови графа.
- За рівних оцінок вибирається найменший індекс класу. Для оцінок BF16 і FP32
  перемагає інтерпольоване значення NaN, і вибирається перший клас із NaN.
- Збережені канали доповнення, як-от додаткові канали доповненого виходу MLA, ніколи
  не читаються як класи.
- Оцінки у форматі плиток MLA відхиляються; спершу виконайте детеселяцію. Оцінки з
  логічними осями CHW або NCHW також відхиляються; класи мають бути на останній осі.
- DevKit має працювати з прошивкою EV74, що містить цю операцію. Зі старішою прошивкою
  збій стається до надсилання першого кадру.
