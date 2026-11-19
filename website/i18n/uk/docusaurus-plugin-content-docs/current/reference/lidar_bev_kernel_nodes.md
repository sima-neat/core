---
title: "Вузли ядер LiDAR і BEV"
description: "Виконуйте операції бібліотеки ядер SimaVoxelize, SimaPillarScatter, SimaBEVPoolPlan, SimaBEVPool і SimaCenterHeadDecode як вузли графа"
sidebar_position: 8
---

# Вузли ядер LiDAR і BEV

`SimaVoxelize`, `SimaPillarScatter`, `SimaBEVPoolPlan` і `SimaBEVPool` виконують етапи PointPillars і
BEVFusion, що не працюють на MLA. `SimaCenterHeadDecode` перетворює виходи голів моделей типу CenterPoint на 3D-рамки. Кожен вузол виконує одну операцію бібліотеки ядер SiMa
на вибраній вами цілі. Вузол ніколи не переходить на іншу ціль і не копіює тензор, щоб
змінити його розміщення. Якщо бібліотека ядер відхиляє запит, побудова графа завершується
помилкою з поясненням від бібліотеки ядер.

| Вузол | Операція | Цілі |
| --- | --- | --- |
| `SimaVoxelize` | Точки LiDAR у декорований квантований тензор стовпців | A65, EV74 |
| `SimaPillarScatter` | Ознаки стовпців у щільне полотно BEV | A65, EV74 |
| `SimaBEVPoolPlan` | Калібрування камер у план пулінгу BEV | A65 |
| `SimaBEVPool` | Ознаки глибини й контексту камер у сітку BEV | A65, EV74 |
| `SimaCenterHeadDecode` | Виходи CenterHead у 3D-рамки | A65 |

Це фрагмент побудови графа для використання всередині застосунку. Реалізуйте
`read_sweep()` у застосунку: функція має повертати непорожній вектор із `4*N` значень
float у порядку x, y, z, часовий зсув. Значення `N` має вміщуватися в INT32. Ця
функція не належить до API Neat. Фрагмент лише будує граф: він не подає кадр і
не отримує вихідний результат.

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
`pyneat.nodes.sima_pillar_scatter`, `pyneat.nodes.sima_bev_pool_plan`, `pyneat.nodes.sima_bev_pool` і
`pyneat.nodes.sima_centerhead_decode`.

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
| `SimaCenterHeadDecode` | Карти голів FP32 або BF16 на двох–шести портах | FP32 `boxes` `[K,9]` або `[K,7]`; FP32 `scores` `[K]`; INT32 `labels` `[K]`; INT32 `count` `[1]` |

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
- **SimaCenterHeadDecode** розміщує голови за значеннями від `heatmap_input` до `vel_input`. Типово три порти
  містять теплову карту; reg, height, dim і vel; та rot. `K` дорівнює `tasks.size() * post_max_per_task`.
  `boxes` містять x, y, z центра основи, dx, dy, dz, yaw, а з `velocity` також vx і vy. `labels` — це канали
  теплової карти, пораховані через усі завдання голови. Рядки починаючи з `count` нульові.

### Логічні осі та фізичне розміщення

`SimaBEVPool` розрізняє порядок логічних осей і фізичний порядок байтів. Тензор із
позначкою `TensorLayout::HWC` має логічні осі `[N,H,W,K]`, а з позначкою
`TensorLayout::CHW` — `[N,K,H,W]`. Вузол надає перевагу цій позначці й перевіряє
відповідність кроків значенню `input_storage`. Для `ChannelsLast` входи HWC мають
бути щільними, а входи CHW — мати кроки з каналами останніми. Для `Contiguous`
потрібні щільні входи CHW; позначка HWC відхиляється.
Той самий канонічний порядок осей можна задати через `Tensor.axis_semantics`;
якщо вказано обидві анотації, вони мають узгоджуватися.

Для входів `ChannelsLast` без позначки зберігається правило сумісності: щільний
NHWC або нещільний логічний NCHW із кроками «канали останніми». Якщо виміри
розміру 1 роблять логічне подання NCHW також щільним, явно задайте
`Tensor.layout = TensorLayout::CHW`. У цьому випадку самі кроки не визначають
логічні осі. Виходи Pool мають логічні осі CHW і кроки відповідно до
`output_storage`; вузол не транспонує й не копіює дані для зміни розміщення.

### Метадані квантування

Виходи INT8 містять своє квантування в `Tensor.semantic.quant`, коли вузол його знає:
`scale = 1 / quantization.multiplier` для `SimaVoxelize` і для `SimaBEVPool` із входами BF16;
`depth_scale * context_scale / quantization.multiplier` для `SimaBEVPool`, обидва входи INT8
якого мають масштаб; квантування ознак для `SimaPillarScatter`. В інших випадках квантування
не публікується.

Для входів INT8 у Scatter поканальне квантування підтримується лише на останній
осі ознак `C`, зокрема для форм на зразок `[1,M,1,C]`. Задайте `C` скінченних
додатних масштабів і `C` нульових точок INT8, кожна з яких дорівнює параметру
Scatter `zero_point` для заповнення порожніх комірок. Полотно зберігає ці значення,
а вісь квантування змінюється на `3` у `[1,H,W,C]`. Інші осі квантування
відхиляються: параметри осі стовпців не можуть описувати фіксоване полотно після
розподілу стовпців за координатами.

`SimaBEVPool` не підтримує метадані поканального квантування й відхиляє їх,
а не мовчки видаляє. Поведінка наявних потензорних метаданих і явно заданих
нульових точок не змінюється; відсутні вхідні метадані залишаються невідомими.

## Тензори BF16

- **SimaVoxelize** з `output_dtype = TensorDType::BFloat16` округлює кожне значення
  декорації binary32 до найближчого парного BF16, зберігає кожне NaN як `0x7FC0` і заповнює
  порожні слоти значенням +0.0. Залиште `quantization` типовим; будь-яке інше значення
  відхиляється.
- **SimaPillarScatter** зберігає тип ознак. Для BF16 потрібен `zero_point = 0`, тому
  порожні комірки містять +0.0. Кожна комірка полотна BF16 зберігає молодші байти
  `C` каналів, а потім старші. `semantic.tess` явно описує це фізичне розміщення:
  `slice_shape = [1,1,1,C]`, `axes = [0,1,2,3]`,
  `format = "MLA_TILED_BF16_COMPACT"`; `storage_bytes` — перевірений розмір даних.
  Плитки D/H/W/C впорядковано за C/D/H/W, дані всередині — за D/H/W/C. BF16
  використовує окремі молодші й старші байти кожної комірки; плитки вирівняно до 16 байтів.
  `copy_dense_bytes_tight()`, типізований `data_ptr()` і DLPack відхиляють це нещільне
  розміщення. `copy_payload_bytes()` читає лише визначений діапазон збережених байтів.
  `clone()` і вихід Owned зберігають байти та метадані без перетворення плиток.
  Наступні вузли, які розуміють таке розміщення, наприклад `SimaArgMax`, безпосередньо
  отримують перевірений опис виробника в тому самому графі. Споживачі лише щільних
  даних і непідтримувані плиткові входи графа його відхиляють.
  Повноканальні плитки-комірки INT8 побайтово тотожні щільному NHWC, тому звичайне
  щільне копіювання та DLPack працюють як раніше. Давні позначки `TessSpec` лише з
  форматом не змінюють поведінки; ненульове `storage_bytes` означає явне плиткове сховище.
- **SimaBEVPoolPlan** приймає `input_dtype` — тип входів `SimaBEVPool`. План `V2`
  розподіляє кеші EV74 у байтах, тож план BF16 може бути більшим. План `V1` однаковий
  для обох типів.
- **SimaBEVPool** приймає `depth` і `context` у BF16 з нульовими точками 0. Сума кожної
  комірки обчислюється в binary32 у порядку точок плану, тож плани `V1` і `V2`
  можуть дати різні байти BF16. `bev` у BF16 зберігає кожну суму, округлену до найближчого
  парного, з NaN як `0x7FC0`, і потребує типового `quantization`. `bev` у INT8 із входів
  BF16 квантує суму: задайте `quantization.multiplier` рівним `1 / output_scale`. Входи INT8
  не можуть дати `bev` у BF16.

Додані поля `TessSpec::axes` і `TessSpec::storage_bytes` мають типові значення, тому наявні агрегатні ініціалізатори з двома полями зберігають сумісність вихідного коду. Бінарне компонування публічної структури змінюється: перебудуйте споживачів C++ і прив’язки Python з узгодженими заголовками та бібліотеками Core; не використовуйте повторно бінарні файли, зібрані з попереднім компонуванням.

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

## Декодуйте виходи CenterHead

`SimaCenterHeadDecode` виконує кодувальник рамок CenterPoint із mmdetection3d і NMS для кожного завдання голови
для таких моделей, як CenterPoint і BEVDet. Одразу після `model.inference()` вузол читає BF16-карти голів там, де їх
записав MLA, тому етап постобробки моделі не виконується.

Цей фрагмент побудови графа належить до вашого застосунку. Значення відповідають конфігурації CenterPoint для nuScenes.

```cpp
#include <neat.h>

using namespace simaai::neat;

Model model("centerpoint_mpk.tar.gz");

const auto circle = [](int classes, float radius) {
  SimaCenterHeadTask task;
  task.classes = classes;
  task.nms = SimaCenterHeadNms::Circle;
  task.nms_radius = radius;
  return task;
};
SimaCenterHeadDecodeOptions decode;
decode.tasks = {circle(1, 4.0f), circle(2, 12.0f), circle(2, 10.0f),
                circle(1, 1.0f), circle(2, 0.85f), circle(2, 0.175f)};
decode.out_size_factor = 4;
decode.voxel_size = {0.2f, 0.2f};
decode.range_min = {-51.2f, -51.2f};
decode.score_threshold = 0.1f;
decode.top_k = 500;
decode.post_max_per_task = 83;
decode.post_center_range = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};

Graph graph;
graph.add(nodes::Input());
graph.add(model.inference());
graph.add(nodes::SimaCenterHeadDecode(decode));
graph.add(nodes::Output());
```

Кожен отриманий Sample містить `boxes`, `scores`, `labels` і `count` у цьому порядку. Читайте перші `count` рядків.
`model.inference()` починається з MLA, тому вхід графа вже має бути у форматі, який читає MLA. Паралельна робота вузла
використовує A65-виконавців графа (`GraphOptions.kernel_runner_a65_workers`).

### Карти голів

Порт `i` — це `i`-й тензор вище за потоком. Кожен порт — це карта FP32 або BF16 із каналами останніми, `[H,W,C]` або
`[1,H,W,C]`, і всі порти мають однакові H та W.

- Теплова карта має власний порт, з одним каналом на клас, завдання голови за завданням голови.
- Кожен інший порт містить одну або кілька голів рамок, завдання голови за завданням голови. Голови кожного завдання
  йдуть у такому порядку: reg (2 канали: зсуви x та y), height (1), dim (3: логарифми dx, dy та dz), rot (2: синус і
  косинус) і, з `velocity`, vel (2).
- Типові значення відповідають моделі з трьома виходами: теплова карта; reg, height, dim і vel; та rot. Для
  CenterPoint на nuScenes, із шістьма завданнями голови та десятьма класами, вони мають 10, 48 і 12 каналів. Модель з
  одним виходом на голову задає значення від `heatmap_input` до `vel_input` за своїм порядком виходів.

Модель з одним виходом на кожну пару завдання голови та голови має понад шість виходів. Об’єднайте їх за головами під
час експорту моделі.

Карти щільні або BF16 із проміжками після клітинок і рядків. Після `model.inference()` MLA пакета SDK 3.0 публікує
голови саме так — BF16, доповнені до 16 каналів на клітинку, — і вузол читає їх на місці. Рамки не залежать від
способу зберігання голів.

Пакет, вихід MLA якого квантований або зберігається тайлами, публікує після `model.inference()` цілі числа, і побудова
графа завершується помилкою з причиною від вузла. У такому разі використовуйте `model.graph()` замість
`model.inference()`. Він додає власні перетворення моделі навколо MLA, тож граф приймає вхід моделі, а вузол отримує
карти голів FP32.

### Параметри

Параметри — це налаштування декодування з конфігурації mmdetection3d моделі.

| Параметр | Конфігурація mmdetection3d |
| --- | --- |
| `tasks[i].classes` | Кількість `class_names` завдання голови `i` |
| `tasks[i].nms`, `tasks[i].nms_radius` | `test_cfg.nms_type = 'circle'`, `test_cfg.min_radius[i]` |
| `tasks[i].nms`, `tasks[i].nms_iou_threshold` | `test_cfg.nms_type = 'rotate'`, `test_cfg.nms_thr` |
| `nms_size_scale` | `test_cfg.nms_rescale_factor`, одне значення на клас |
| `out_size_factor`, `voxel_size` | Однойменні ключі `bbox_coder` |
| `range_min` | Мінімуми x та y з `bbox_coder.pc_range` |
| `score_threshold`, `top_k`, `post_center_range` | `score_threshold`, `max_num` і `post_center_range` з `bbox_coder` |
| `post_max_per_task` | `test_cfg.post_max_size` |
| `velocity` | Голова має гілку `vel` |

Кожне завдання голови вказує свій NMS. Circle NMS порівнює квадрат відстані між центрами двох рамок із самим значенням
`nms_radius`, як це робить mmdetection3d. Rotate NMS порівнює IoU двох повернутих рамок у вигляді згори з
`nms_iou_threshold`. Якщо `nms_size_scale` задано, він має одне значення на канал теплової карти та 1 для класів
завдань голови з circle NMS. `labels` — це канали теплової карти, тож перший клас другого завдання голови йде одразу
після останнього класу першого.

У Python передавайте значення завдання голови його конструктору та присвоюйте списки цілком. Список, прочитаний з
об’єкта параметрів, є копією, тому додавання до `decode.tasks` нічого не змінює.

```python
import pyneat

circle = pyneat.SimaCenterHeadNms.Circle
decode = pyneat.SimaCenterHeadDecodeOptions()
decode.tasks = [
    pyneat.SimaCenterHeadTask(classes, circle, nms_radius=radius)
    for classes, radius in [(1, 4.0), (2, 12.0), (2, 10.0), (1, 1.0), (2, 0.85), (2, 0.175)]
]
decode.out_size_factor = 4
decode.voxel_size = [0.2, 0.2]
decode.range_min = [-51.2, -51.2]
decode.score_threshold = 0.1
decode.top_k = 500
decode.post_max_per_task = 83
decode.post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
node = pyneat.nodes.sima_centerhead_decode(decode)
```

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
- `SimaBEVPoolPlan` і `SimaCenterHeadDecode` виконуються лише на A65.

Запит, який ціль не підтримує, завершується помилкою на цій цілі. Вузол не вибирає замість
вас іншу ціль.
