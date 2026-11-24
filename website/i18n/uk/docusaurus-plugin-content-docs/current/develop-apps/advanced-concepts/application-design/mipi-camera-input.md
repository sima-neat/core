---
title: "Використання камери MIPI"
description: "Додайте до графа Neat камеру MIPI з підтримуваним оверлеєм за допомогою CameraInput, попередньої обробки моделі, інференсу на MLA, необов’язкового BoxDecode на EV74 і отримання виходу."
sidebar_position: 4
slug: /develop-apps/advanced-concepts/mipi-camera-input
---

# Використання камери MIPI

Використовуйте `CameraInput`, коли граф має зчитувати кадри безпосередньо з камери MIPI на Modalix DevKit. `CameraInput` — це межа Neat між перевіреним потоком libcamera і графом моделі, у якому основну роботу виконують прискорювачі:

```text
CameraInput -> model-managed CVU preproc -> MLA -> Output
CameraInput -> model-managed CVU preproc -> MLA -> EV74 BoxDecode -> Output
```

Жодного `appsrc`. Жодного `ostosima` у коді користувача. Жодних `videoconvert` чи `videoscale` на CPU у виробничому шляху, якщо ви не додасте їх самі.

## Два кроки: спершу запуск камери на платі, потім Neat

MIPI CSI-2 — це канал зв’язку з камерою. Він не робить кожен сенсор сумісним за принципом «підключи й працюй». Для робочого шляху камери також потрібні правильний оверлей плати, драйвер сенсора, конвеєр libcamera, поведінка ISP і caps.

Розглядайте роботу з камерою MIPI як два кроки:

1. **Запуск камери на платі:** Modalix DevKit бачить сенсор, і libcamera може передавати потік у потрібному режимі.
2. **Запуск графа Neat:** `CameraInput` подає ці кадри на етапи CVU/MLA, якими керує модель.

Neat починає роботу з кроку 2. Він не вибирає оверлеї `.dtbo`, не завантажує драйвери сенсорів і не налаштовує ISP. Щоб налаштувати оверлей, дізнатися назви підтримуваних оверлеїв і виконати перевірку за допомогою `cam`, скористайтеся [посібником з інтерфейсу камери MIPI для Modalix DevKit](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces).

## Що підтримує Neat

`CameraInput` підтримує камери, які вже запущено стеком камер платформи:

- камеру під’єднано до порту MIPI на Modalix DevKit, коли плату вимкнено;
- активний правильний оверлей плати;
- драйвер ядра й конвеєр libcamera надають доступ до камери;
- `libcamerasrc` може узгодити запитані caps `video/x-raw`, зазвичай `NV12`;
- caps відповідають попередній обробці моделі, яку ви налаштовуєте в Neat.

Якщо ці умови ще не виконано, спершу виправте стек камер. Граф Neat може бути точним, але не може перетворити сенсор без драйвера на потік.

## Перевірка потоку камери

Перевірте камеру на рівні libcamera/GStreamer, перш ніж будувати граф. Якщо цей рівень не працює, граф Neat цього не виправить.

На DevKit переконайтеся, що `libcamerasrc` наявний:

<ShellCommand prompt="devkit">
gst-inspect-1.0 libcamerasrc
</ShellCommand>

Якщо утиліта `cam` доступна, виведіть список камер і перегляньте режими:

<ShellCommand prompt="devkit">
cam -l
cam -c 1 -I
</ShellCommand>

Потім спробуйте саме ті caps, які плануєте запитувати з Neat:

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

Лише для візуального димового тесту закодуйте кілька кадрів у JPEG:

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! videoconvert ! jpegenc ! \
  multifilesink location=/tmp/mipi-frame-%03d.jpg
</ShellCommand>

`videoconvert` і `jpegenc` годяться для такої разової перевірки під час налагодження. Не додавайте їх у конвеєр моделі, коли важлива пропускна здатність.

## Побудова графа для димового тесту з необробленим виходом MLA

Почніть із маршруту з необробленим виходом MLA. Він доводить, що кадри камери доходять до попередньої обробки на CVU та інференсу на MLA, ще до того, як ви додасте постобробку, специфічну для моделі.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::CameraInputOptions camera;
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.format = "NV12";
camera.buffer_name = "camera0";
camera.allow_cpu_fallback = true;

neat::Model::Options model_options;
model_options.preprocess.kind = neat::InputKind::Image;
model_options.preprocess.input_max_width = static_cast<int>(camera.width);
model_options.preprocess.input_max_height = static_cast<int>(camera.height);
model_options.preprocess.input_max_depth = 3;
model_options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
model_options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
model_options.preprocess.resize.enable = neat::AutoFlag::On;
model_options.preprocess.resize.width = 640;
model_options.preprocess.resize.height = 640;
model_options.preprocess.resize.mode = neat::ResizeMode::Letterbox;
model_options.preprocess.resize.pad_value = 114;
model_options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
model_options.advanced_execution.preprocess_target = "EV74";
model_options.inference_terminal.mla_only = true;

neat::Model model("/models/yolo.tar.gz", model_options);

neat::Model::RouteOptions route;
route.include_input = false;
route.include_output = true;
route.upstream_name = camera.buffer_name;
route.buffer_name = camera.buffer_name;
route.name_suffix = "_camera0";
route.advanced_execution.preprocess_target = "EV74";

neat::Graph graph("camera_mla_smoke");
graph.add(neat::nodes::CameraInput(camera));
graph.add(model.graph(route));

neat::Run run = graph.build();
std::optional<neat::Sample> output = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.CameraInputOptions()
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.format = "NV12"
camera.buffer_name = "camera0"
camera.allow_cpu_fallback = True

model_options = pyneat.ModelOptions()
model_options.preprocess.kind = pyneat.InputKind.Image
model_options.preprocess.input_max_width = int(camera.width)
model_options.preprocess.input_max_height = int(camera.height)
model_options.preprocess.input_max_depth = 3
model_options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
model_options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
model_options.preprocess.resize.enable = pyneat.AutoFlag.On
model_options.preprocess.resize.width = 640
model_options.preprocess.resize.height = 640
model_options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
model_options.preprocess.resize.pad_value = 114
model_options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
model_options.advanced_execution.preprocess_target = "EV74"
model_options.inference_terminal.mla_only = True

model = pyneat.Model("/models/yolo.tar.gz", model_options)

route = pyneat.ModelRouteOptions()
route.include_input = False
route.include_output = True
route.upstream_name = camera.buffer_name
route.buffer_name = camera.buffer_name
route.name_suffix = "_camera0"
route.advanced_execution.preprocess_target = "EV74"

graph = pyneat.Graph("camera_mla_smoke")
graph.add(pyneat.nodes.camera_input(camera))
graph.add(model.graph(route))

run = graph.build()
output = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

`inference_terminal.mla_only = true` задано навмисно. Так димовий шлях залишається `CameraInput -> CVU preproc -> MLA -> Output`, і ви можете налагодити рух даних від камери та інференс, перш ніж налагоджувати декодування результатів виявлення.

## Додавання BoxDecode на EV74, коли це потрібно моделі

Для моделей виявлення в стилі YOLO явно ввімкніть BoxDecode і залиште постобробку на EV74. Токен декодування має відповідати вихідному контракту MPK.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
model_options.inference_terminal.mla_only = false;
model_options.decode_type = neat::BoxDecodeType::YoloV9Seg;
model_options.advanced_execution.postprocess_target = "EV74";
model_options.score_threshold = 0.25f;
model_options.nms_iou_threshold = 0.45f;
model_options.top_k = 100;

route.advanced_execution.postprocess_target = "EV74";
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
model_options.inference_terminal.mla_only = False
model_options.decode_type = pyneat.BoxDecodeType.YoloV9Seg
model_options.advanced_execution.postprocess_target = "EV74"
model_options.score_threshold = 0.25
model_options.nms_iou_threshold = 0.45
model_options.top_k = 100

route.advanced_execution.postprocess_target = "EV74"
```

</CodeTab>
</CodeTabs>

Не задавайте `decode_type` і залиште `mla_only = true`, коли потрібні необроблені тензори MLA. Задавайте `decode_type` лише тоді, коли маршрут моделі має видавати декодовані результати виявлення або дані сегментації.

## Вибір правильного режиму пам’яті

`CameraInput` має два режими:

| Режим | Коли використовувати | Поведінка |
| --- | --- | --- |
| Строгий режим без копіювання | Ваш `libcamerasrc` передає кожен кадр як один DMA-BUF (звичайний випадок). | Neat передає DMA-BUF камери етапам прискорювача й завершується помилкою, якщо кадр потребує копіювання. |
| Адаптивний резервний варіант (лише за явного ввімкнення) | Потрібна сумісність зі стеком камер, у якому кадр не є одним DMA-BUF. | Neat копіює такі кадри в DMA-BUF CMA для передавання до CVU/MLA і, як і раніше, напряму передає кадри, що є одним DMA-BUF. |

Строгий режим без копіювання — режим фреймворку за замовчуванням. Neat ніколи не володіє пам’яттю камери:
будь-який фізично неперервний DMA-BUF є допустимим входом EV74, і перевірка допуску EV74 у Neat приймає його, якщо він походить із купи CMA або з неперервного розподільника буферів захоплення камери (`videobuf2_dma_contig`), тому жодна властивість `libcamerasrc` не потрібна. Встановлюйте
`camera.allow_cpu_fallback = true` лише як явний запасний вихід для сумісності.
Резервне копіювання — це міст у конвеєр прискорювачів; воно не дає
дозволу додавати перетворення кольору чи масштабування на CPU в гарячий шлях.

В обох режимах Neat розміщує свій приватний міст пам’яті одразу після
caps камери й перед будь-якою чергою. Міст просить джерело додавати
`GstVideoMeta` через `GST_QUERY_ALLOCATION` і передає кожен кадр, площини якого
розташовані впритул одна до одної в одному DMA-BUF, як єдину пам’ять, утримуючи
буфер камери, доки всі споживачі його не звільнять. Це звичайний шлях,
а не шлях резервного копіювання.

Політику утримання виходу ISP визначає застосунок. Залиште
аргумент `capture_buffer_count` для `nodes::CameraInputWithCaptureBuffers()` або
`pyneat.nodes.camera_input()` рівним `0`, щоб використовувати значення камери за замовчуванням, або запросіть
більший мінімум, коли часовий кодувальник чи асинхронний ML-граф довше утримує кадри.
На Platform 3.0 `libcamerasrc` не має `buffer-count`, тому там залишайте значення `0`
(див. [довідник CameraInput](/reference/nodes/camera-input)).
Активний конвеєр камери перевіряє власне обмеження; провайдер Neat підтримує
до 128. Ця кількість задається окремо від приватного кільцевого буфера передавання RAW із CSI до ISP
і від `queue_depth`, який керує лише живою чергою GStreamer у Neat. Використовуйте чергу з відкиданням (leaky),
коли актуальні кадри важливіші за повноту; використовуйте зворотний тиск від
наступних етапів, коли потрібно зберегти кожен кадр. У режимі копіювання для сумісності
приватний пул мосту за потреби зростає до 128 буферів, тому він не
блокується, перш ніж черга з відкиданням позбудеться застарілих кадрів.

## Попередня обробка на CVU/EV74

Для конвеєрів моделей надавайте перевагу попередній обробці, якою керує модель:

- задайте `Model::Options::preprocess` або `pyneat.ModelOptions.preprocess` для зміни розміру, перетворення кольору, нормалізації, квантування й теселяції;
- залишайте цілі CVU для попередньої обробки та постобробки, якими керує модель, на `EV74`, коли маршрут моделі це підтримує;
- не вставляйте `VideoConvert`, `VideoScale` чи GStreamer-елементи `videoconvert`/`videoscale` перед моделлю, якщо тільки ви не будуєте граф лише для налагодження.

Камера дає вам кадри. Обчислення над кадрами має виконувати CVU. CPU не повинен ненавмисно стати вашим рушієм обробки зображень.

## Коротка карта усунення несправностей

| Симптом | Що перевірити спершу |
| --- | --- |
| Камера не з’являється | Перевірте вибраний `.dtbo`, орієнтацію кабелю, вимкнення й увімкнення живлення, а також журнали ядра та libcamera. |
| `libcamerasrc` відсутній | Установіть відповідний образ Neat/середовища виконання з підтримкою камери або пакети камери для збірки DevKit. |
| `misconfig.media_caps` або `not-negotiated` | Перевірте точні значення `format,width,height,framerate` за допомогою `gst-launch-1.0`. Спробуйте відомий підтримуваний режим, наприклад `NV12 1920x1080@30`. |
| Строгий режим без копіювання завершується помилкою | Встановіть `allow_cpu_fallback = true` або використовуйте стек камер, що передає кожен кадр як один DMA-BUF. |
| Кольори на виході неправильні | Переконайтеся, що кадр інтерпретується як `NV12`, а не RGB/BGR. Якщо JPEG, отриманий безпосередньо з `libcamerasrc`, теж неправильний, налагодьте ISP чи налаштування камери, перш ніж налагоджувати Neat. |
| Низька пропускна здатність | Приберіть перетворення й масштабування відео на CPU, отримуйте вихід безперервно й використовуйте для живого джерела політику черги, яка надає перевагу актуальності. |

Інші симптоми див. на сторінці [Усунення несправностей](/reference/troubleshooting).
