---
title: Вибірка ознак за динамічною сіткою
description: Виконання A65 NEON GridSample2D через граф Neat
sidebar_position: 9
---

# Вибірка ознак за динамічною сіткою

`SimaGridSample2D` виконує білінійну вибірку на A65. Кожен вхідний зразок містить ознаки `[N,H,W,C]` (FP32 або BF16), потім сітку `[N,Ho,Wo,2]` (FP32) з нормалізованими координатами `(x,y)`. Вихід — `sampled` `[N,Ho,Wo,C]` із типом ознак і каналами на останній осі. Усі виміри мають бути додатними.

## Збирання й виконання

Використовуйте узгоджений комплект пакетів Kernels, Runner, Scheduler і Core, що містить `grid_sample_2d`. Цей фрагмент C++ виконується всередині застосунку, скомпонованого з `SimaNeat::sima_neat`; перед побудовою графа заповніть два вектори неперервними ознаками NHWC і значеннями сітки NHW2.

```cpp
#include <neat.h>
using namespace simaai::neat;

std::vector<float> features(1 * 16 * 16 * 16, 1.0f);
std::vector<float> grid(1 * 32 * 8 * 2, 0.0f);
auto feature = Tensor::from_vector(features, {1,16,16,16}, TensorMemory::CPU);
feature.layout = TensorLayout::HWC;
auto coordinates = Tensor::from_vector(grid, {1,32,8,2}, TensorMemory::CPU);

InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::SystemMemory;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";
SimaGridSample2DOptions sampling;
sampling.padding = SimaGridSample2DPadding::Zeros;
sampling.align_corners = false;
Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaGridSample2D(sampling));
graph.add(nodes::Output());
auto run = graph.build(TensorList{feature, coordinates});
auto output = run.run(TensorList{feature, coordinates}, 30000);
run.close();
```

Змінюйте значення сітки в кожному поданні, зберігаючи форми й типи. Для кількох одночасно оброблюваних зразків використовуйте звичайний асинхронний API push/pull. Утримувані виходи без копіювання зберігають позики буферів виробника; звільняйте їх, щоб пул міг просуватися. Виходи з власними даними залишаються чинними незалежно від виконання.

## Параметри вибірки й точність

`SimaGridSample2DPadding::Zeros` підставляє нуль замість недійсних відліків ознак; `SimaGridSample2DPadding::Border` обмежує координати центрами крайніх пікселів. `align_corners=false` використовує напівпіксельні центри; true відображає нормалізовані -1/+1 у центри кутових пікселів. Компоненти NaN означають нормалізовану координату -1. Нескінченні компоненти дають нульову вибірку за нульового доповнення або вибирають край за граничного.

Координати, ваги й накопичення використовують FP32. Кожна сума починається множенням, після якого виконуються три явні злиті множення-додавання. Ознаки BF16 розширюються точно, а кінцевий вихід один раз округлюється до найближчого парного. Ця точність відрізняється від реалізації MLA, яка округлює сітку чи проміжні координати до BF16.

NCHW, плиткове зберігання, нещільні подання та неідентичне квантування потребують явного перетворення перед цим вузлом. Режими найближчого сусіда, кубічний і відбиття не підтримуються. Додавання вузла не замінює GridSample усередині скомпільованих моделей MLA. Порівнюйте продуктивність на однакових збережених входах з однаковими межами вимірювання; походження сітки в профілюванні ELF з невідомою ініціалізацією лишається непідтвердженим.