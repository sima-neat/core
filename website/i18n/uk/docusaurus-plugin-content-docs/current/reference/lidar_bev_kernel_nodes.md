---
title: "Вузли ядер LiDAR і BEV"
description: "Виконуйте операції бібліотеки ядер SimaVoxelize, SimaPillarScatter, SimaBEVPoolPlan і SimaBEVPool як вузли графа"
sidebar_position: 8
---

# Вузли ядер LiDAR і BEV

`SimaVoxelize`, `SimaPillarScatter`, `SimaBEVPoolPlan` і `SimaBEVPool` виконують етапи PointPillars і
BEVFusion, що не працюють на MLA. Кожен вузол виконує одну операцію бібліотеки ядер SiMa
на вибраній вами цілі. Вузол ніколи не переходить на іншу ціль і не копіює тензор, щоб
змінити його розміщення. Якщо бібліотека ядер відхиляє запит, побудова графа завершується
помилкою з поясненням від бібліотеки ядер.

| Вузол | Операція | Цілі |
| --- | --- | --- |
| `SimaVoxelize` | Точки LiDAR у декорований квантований тензор стовпців | A65, EV74 |
| `SimaPillarScatter` | Ознаки стовпців у щільне полотно BEV | A65, EV74 |
| `SimaBEVPoolPlan` | Калібрування камер у план пулінгу BEV | A65 |
| `SimaBEVPool` | Ознаки глибини й контексту камер у сітку BEV | A65, EV74 |

```cpp
#include <neat.h>

using namespace simaai::neat;

SimaVoxelizeOptions voxelize;
voxelize.voxel_size = {0.25f, 0.25f, 8.0f};
voxelize.point_cloud_range = {-50.0f, -50.0f, -5.0f, 50.0f, 50.0f, 3.0f};
voxelize.quantization.multiplier = 2.720007f; // 1 / the consuming MLA input's scale
voxelize.quantization.zero_point = 8;
voxelize.max_pillars = 40000;
voxelize.max_points_per_pillar = 64;
voxelize.target = SimaKernelTarget::EV74;

// One sample holds every input tensor of the node, in order.
InputOptions input;
input.payload_type = PayloadType::Tensor;
input.memory_policy = InputMemoryPolicy::Ev74;
input.caps_override = "application/vnd.simaai.tensor, representation=(string)tensor-set, "
                      "storage=(string)tensorbuffer";

std::vector<float> xyzt = read_sweep();  // x, y, z, time offset per point
const auto n = static_cast<std::int64_t>(xyzt.size() / 4);
auto points = Tensor::from_vector(xyzt, {n, 4}, TensorMemory::EV74);
auto point_count = Tensor::from_vector(std::vector<std::int32_t>{static_cast<std::int32_t>(n)},
                                       {1}, TensorMemory::EV74);

Graph graph;
graph.add(nodes::Input(input));
graph.add(nodes::SimaVoxelize(voxelize));
graph.add(nodes::Output());
auto run = graph.build(TensorList{points, point_count});
```

Python надає ті самі типи параметрів і фабрики `pyneat.nodes.sima_voxelize`,
`pyneat.nodes.sima_pillar_scatter`, `pyneat.nodes.sima_bev_pool_plan` і `pyneat.nodes.sima_bev_pool`.

## Контракти входів і виходів

Кожен вузол читає тензори свого вхідного зразка в наведеному нижче порядку й публікує свої
виходи в наведеному нижче порядку. Кількість тензорів, типи даних і форми перевіряються під
час побудови графа.

| Вузол | Входи | Виходи |
| --- | --- | --- |
| `SimaVoxelize` | FP32 `points` `[N,4]` (x, y, z, часовий зсув); INT32 `point_count` `[1]` | INT8 або BF16 `decorated` `[1,M,T,10]`; INT32 `coordinates` `[M,2]` як (row, column); INT32 `count` `[1]` |
| `SimaPillarScatter` | INT8 або BF16 `features` `[M,C]`; INT32 `coordinates` `[M,2]`; INT32 `count` `[1]` | `canvas` `[1,H,W,C]` того самого типу, що й `features` |
| `SimaBEVPoolPlan` | FP32 `camera2lidar`, `camera_intrinsics` і `img_aug_matrix` `[N,4,4]`; FP32 `lidar_aug_matrix` `[4,4]` | UINT8 `plan` `[P]` |
| `SimaBEVPool` | INT8 або BF16 `depth`; `context` того самого типу; UINT8 `plan` `[P]` | INT8 або BF16 `bev` `[1,nZ*C,nX,nY]` |

- **SimaVoxelize** читає лише перші `point_count` точок. `max_pillars` (`M`) і
  `max_points_per_pillar` (`T`) — фіксовані місткості; точки й стовпці понад них
  відкидаються. Невикористані рядки містять нульову точку квантування та нульові координати.
- **SimaPillarScatter** читає лише перші `count` стовпців. Він також приймає ознаки з
  додатковими осями розміру 1, наприклад `[1,M,1,C]` від моделі ознак стовпців. `C` має бути
  кратним 16. Порожні комірки містять `zero_point`; це має бути код INT8, який вхідне
  квантування моделі, що читає полотно, призначає значенню 0.0. Байти полотна розміщені в
  щільному порядку NHWC (для полотна INT8).
- **SimaBEVPoolPlan** бере свою зрізану піраміду огляду (`xs`, `ys`, `ds`) і сітку (`bx`, `dx`,
  `nx`) з перетворення вигляду контрольної точки. Типовий `SimaBEVPoolPlanFormat::V1` групує
  точки за комірками BEV; для `SimaBEVPool` на EV74 використовуйте `SimaBEVPoolPlanFormat::V2`
  разом із `context_channels`. Типово `P` — розмір плану в найгіршому випадку; байти після
  використаної довжини плану не записуються.
- **SimaBEVPool** підсумовує `depth x context` за точками плану кожної комірки й квантує суму за
  допомогою `quantization`. Задайте `cells` рівним `nx` плану, `plan_format` — `format`
  плану, а для входів INT8 `quantization.multiplier` — `depth_scale * context_scale / output_scale`. З
  типовим `SimaBEVPoolStorage::ChannelsLast` входи надходять як `[N,fH,fW,D]` і `[N,fH,fW,C]`,
  як їх створюють моделі камер. Вихід публікується як логічний `[1,nZ*C,nX,nY]` із кроками
  «канали останніми», тож його байти мають розміщення NHWC `[1,nX,nY,nZ*C]`.
  `SimaBEVPoolStorage::Contiguous` використовує входи `[N,D,fH,fW]` і `[N,C,fH,fW]` та щільний
  вихід.

Виходи INT8 містять своє квантування в `Tensor.semantic.quant`, коли вузол його знає:
`scale = 1 / quantization.multiplier` для `SimaVoxelize` і для `SimaBEVPool` із входами BF16;
`depth_scale * context_scale / quantization.multiplier` для `SimaBEVPool`, обидва входи INT8
якого мають масштаб; квантування ознак для `SimaPillarScatter`. В інших випадках квантування
не публікується.

## Тензори BF16

- **SimaVoxelize** з `output_dtype = TensorDType::BFloat16` округлює кожне значення
  декорації binary32 до найближчого парного BF16, зберігає кожне NaN як `0x7FC0` і заповнює
  порожні слоти значенням +0.0. Залиште `quantization` типовим; будь-яке інше значення
  відхиляється.
- **SimaPillarScatter** зберігає тип своїх ознак. Ознаки BF16 потребують `zero_point = 0`,
  тож порожні комірки містять +0.0. Полотно BF16 зберігається в плитковому розміщенні BF16
  для MLA: кожна комірка містить молодші байти своїх `C` каналів, а потім їхні старші
  байти. Це не щільний NHWC, і опублікований тензор цієї відмінності не позначає: його dtype —
  BF16, а форма — `[1,H,W,C]`, тож `copy_dense_bytes_tight()` і подання DLPack, як-от
  `numpy.from_dlpack`, повертають збережені байти, а не значення BF16 у порядку NHWC.
  Передайте полотно моделі MLA, яка читає це розміщення, або прочитайте його як байти й
  переставте байти кожної комірки.
- **SimaBEVPoolPlan** приймає `input_dtype` — тип входів `SimaBEVPool`. План `V2`
  розподіляє кеші EV74 у байтах, тож план BF16 може бути більшим. План `V1` однаковий
  для обох типів.
- **SimaBEVPool** приймає `depth` і `context` у BF16 з нульовими точками 0. Сума кожної
  комірки обчислюється в binary32 у порядку точок плану, тож плани `V1` і `V2`
  можуть дати різні байти BF16. `bev` у BF16 зберігає кожну суму, округлену до найближчого
  парного, з NaN як `0x7FC0`, і потребує типового `quantization`. `bev` у INT8 із входів
  BF16 квантує суму: задайте `quantization.multiplier` рівним `1 / output_scale`. Входи INT8
  не можуть дати `bev` у BF16.

## Під’єднайте вузли до конвеєра

Ці вузли — звичайні етапи графа, але їхні входи надходять від кількох попередніх етапів:

- У PointPillars `SimaPillarScatter` потребує ознак стовпців від моделі ознак стовпців разом із
  `coordinates` і `count` від `SimaVoxelize`.
- У BEVFusion `SimaBEVPool` потребує `depth` і `context` від моделі камер разом із планом від
  `SimaBEVPoolPlan`.

Об’єднайте ці тензори в один вхідний зразок у наведеному вище порядку перед вузлом.
Наприклад, отримайте обидва результати у своєму застосунку й надішліть їх разом до графа
вузла.

`SimaPillarScatter` копіює байти ознак без повторного квантування, тож ознаки вже мають
використовувати вхідне квантування моделі, що читає полотно. `SimaBEVPool` отримує всі камери в
одному тензорі `[N,...]`.

## Плани та стан

План залежить лише від калібрування та `SimaBEVPoolPlanOptions`. Із `cache_plan = true`
кадр, байти калібрування якого збігаються з попереднім кадром, копіює план попереднього кадру
замість того, щоб обчислювати його знову. Кеш зберігається в контексті бібліотеки ядер, яким
середовище виконання володіє для кожного кадру в обробці, тому влучання можливе, але не
гарантоване, коли одночасно обробляються кілька кадрів. Байти плану однакові в обох
випадках.

`SimaBEVPool` не зберігає стану між кадрами. Під час кожного виконання він перевіряє заголовок і
вміст плану та відхиляє план, що не відповідає його формам.

## Обмеження цілей

EV74 додає обмеження до контракту A65:

- `SimaBEVPool` потребує `context` із розміщенням «канали останніми» та кількістю каналів,
  кратною 16, і `bev` із розміщенням «канали останніми». План `V2` також потребує `depth`
  із розміщенням «канали останніми».
- `SimaBEVPoolPlan` виконується лише на A65.

Запит, який ціль не підтримує, завершується помилкою на цій цілі. Вузол не вибирає замість
вас іншу ціль.
