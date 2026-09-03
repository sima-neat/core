---
title: "Використовуйте камеру MIPI."
description: "Додайте MIPI-камеру з підтримкою накладання до Neat Graph, використовуючи CameraInput, попередню обробку моделі, вивід MLA, необов’язковий модуль EV74 BoxDecode та вивід даних."
sidebar_position: 4
slug: /develop-apps/advanced-concepts/mipi-camera-input
---

# Використовуйте камеру MIPI.

Використовуйте `CameraInput`, коли потрібно, щоб граф зчитував кадри безпосередньо з MIPI-камери на Modalix DevKit. `CameraInput` є межею Neat між валідованим потоком libcamera та графом моделі, орієнтованим на прискорення обчислень:

```text
CameraInput -> model-managed CVU preproc -> MLA -> Output
CameraInput -> model-managed CVU preproc -> MLA -> EV74 BoxDecode -> Output
```

Відсутній `appsrc`. Відсутній `ostosima` у коді користувача. Відсутні обчислення `videoconvert` або `videoscale` у робочому середовищі, якщо ви не додасте їх самостійно.

## Дві фази: спочатку підготовка, потім Neat.

MIPI CSI-2 – це інтерфейс підключення камери. Він не забезпечує підтримку всіх сенсорів у режимі «підключи та використовуй». Для забезпечення працездатності камери також потрібна відповідна конфігурація плати, драйвер сенсора, конвеєр libcamera, правильна робота ISP та налаштування.

Розглядайте роботу з камерами MIPI як два етапи:

1. **Налаштування плати:** за допомогою Modalix DevKit можна перевірити, чи датчик і бібліотека libcamera можуть передавати дані у потрібному режимі.
2. **Запуск графа Neat:** `CameraInput` передає ці кадри на етапи CVU/MLA, якими керує модель.

Neat починає роботу з вихідного каналу 2. Він не вибирає набори `.dtbo`, не завантажує драйвери датчиків і не налаштовує ISP. Для налаштування набору, перегляду списку підтримуваних назв наборів і виконання `cam` перевірки, використовуйте [Modalix DevKit: посібник із використання інтерфейсу камери MIPI.](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces).

## Що підтримує Neat?

`CameraInput` підтримує камери, які вже ініціалізовані платформою за допомогою стеку камер:

- камера під’єднана до порту MIPI Modalix DevKit, коли плата вимкнена.
- активовано правильне накладання шару на плату;
- ядро драйвера та конвеєр libcamera забезпечують доступ до камери.
- `libcamerasrc` може узгодити запитані параметри `video/x-raw`, зазвичай `NV12`;
- параметри, які ви налаштовуєте в Neat, відповідають попередній обробці моделі.

Якщо ці умови ще не виконані, спочатку налаштуйте камери. Граф Neat може бути точним, але він не може перетворити незакріплений сенсор на потік даних.

## Перевірте потік відео з камери.

Перевірте роботу камери на рівні libcamera/GStreamer, перш ніж створювати граф. Якщо цей рівень не працює, Neat не зможе виправити це в графі.

На DevKit переконайтеся, що `libcamerasrc` присутній:

<ShellCommand prompt="devkit">
gst-inspect-1.0 libcamerasrc
</ShellCommand>

Якщо `cam` доступний, перелічіть камери та режими перевірки:

<ShellCommand prompt="devkit">
cam -l
cam -c 1 -I
</ShellCommand>

Потім спробуйте використати саме ті великі літери, які ви плануєте використати у своєму запиті до Neat:

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! fakesink
</ShellCommand>

Для візуального тестування, просто щоб перевірити, чи працює система, закодуйте кілька кадрів у формат JPEG:

<ShellCommand prompt="devkit">
gst-launch-1.0 -e libcamerasrc ! \
  'video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1' ! \
  identity eos-after=30 ! videoconvert ! jpegenc ! \
  multifilesink location=/tmp/mipi-frame-%03d.jpg
</ShellCommand>

`videoconvert` і `jpegenc` підходять для цієї разової перевірки під час налагодження. Не включайте їх до конвеєра моделі, коли важлива пропускна здатність.

## Створіть базовий графік розповсюдження диму MLA.

Почніть із базового маршруту MLA. Це підтверджує, що кадри з камери надходять до етапу попередньої обробки CVU та обчислення MLA, перш ніж ви додасте специфічну для моделі постобробку.

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

`inference_terminal.mla_only = true` – це навмисна дія. Вона зберігає траєкторію руху диму в межах `CameraInput -> CVU preproc -> MLA -> Output`, щоб ви могли відлагоджувати рух камери та процес виведення даних перед відлагодженням декодування виявлених об’єктів.

## Додайте EV74 BoxDecode, коли це буде необхідно для конкретної моделі.

Для моделей детектування, що використовують стиль YOLO, явно увімкніть BoxDecode і залиште постобробку на EV74. Токен декодування має відповідати вихідному контракту MPK.

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

Залиште `decode_type` без змін і встановіть значення `mla_only = true`, коли вам потрібні необроблені тензори MLA. Встановлюйте `decode_type` лише тоді, коли модель повинна видавати декодовані дані про виявлені об’єкти або дані сегментації.

## Оберіть відповідний режим пам’яті.

`CameraInput` має два режими:

| Режим | Використовуйте, коли | Поведінка |
| --- | --- | --- |
| Суворе дотримання принципу відсутності копіювання. | Ваш `libcamerasrc` надає доступ до `external-buffer-mode`, а бібліотека пам’яті підтримує експорт DMA-BUF. | Neat вимагає безпосереднього копіювання даних у відповідний пул DMA-BUF, інакше операція завершиться невдало, якщо джерело не може надати ці дані. |
| Адаптивне резервне рішення (з явним підтвердженням згоди користувача) | Вам потрібна сумісність із набором інструментів для роботи з камерою, який не підтримує експорт буферів пристрою. | Neat приймає буфери OS/libcamera, копіює їх у спільну пам’ять SiMaAI для передачі даних між CVU/MLA, і передає буфери SiMaAI, коли джерело вже надає їх. |

За замовчуванням використовується суворий режим нульового копіювання. Він вимагає наявності `libcamerasrc`, який надає загальну властивість `external-buffer-mode`, і бібліотеки пам’яті, яка підтримує експорт DMA-BUF. Встановлюйте `camera.allow_cpu_fallback = true` лише як явний механізм забезпечення сумісності. Резервне копіювання є мостом до конвеєра прискорення; це не дозвіл на додавання перетворення кольору або масштабування за допомогою ЦП до основного потоку обробки.

В обох режимах Neat розміщує свій приватний міст пам’яті відразу після параметрів камери та перед будь-якою чергою. Міст пропонує стандартний пул буферів за допомогою `GST_QUERY_ALLOCATION`. Цей пул виділяє перевірене розташування площин з одного упакованого виділення SiMaAI та експортує один DMA-BUF на кожну площину. `libcamerasrc` імпортує ці DMA-BUF у чергу захоплення ISP; після захоплення міст розпаковує те саме упаковане виділення для подальшої обробки. Це нормальний шлях, а не шлях резервного копіювання.

Застосунок контролює політику утримання вихідних даних ISP. Залиште аргумент `capture_buffer_count` для `nodes::CameraInputWithCaptureBuffers()` або `pyneat.nodes.camera_input()` на рівні `0`, щоб використовувати значення за замовчуванням для камери, або запросіть більше мінімальне значення, коли тимчасовий кодувальник або асинхронний ML-граф утримують кадри довше. Активний конвеєр камери перевіряє власне обмеження; провайдер Neat підтримує до 128. Ця кількість відрізняється від приватного циклічного буфера передачі RAW CSI-ISP і від `queue_depth`, який контролює лише активну чергу Neat у GStreamer. Використовуйте чергу з можливістю скидання, коли поточні кадри важливіші за повноту; використовуйте зворотний тиск у подальших етапах обробки, коли кожен кадр має бути збережено. У режимі копіювання для сумісності приватний пул мосту збільшується за потреби, тому він не блокує роботу, поки черга з можливістю скидання не зможе видалити застарілі кадри.

## Продовжуйте попередню обробку даних для CVU/EV74.

Для конвеєрів моделей рекомендується використовувати попередню обробку, керовану моделлю:

- встановіть `Model::Options::preprocess` або `pyneat.ModelOptions.preprocess` для зміни розміру, перетворення кольорів, нормалізації, квантування та теселяції.
- залишайте цільові значення CVU, які визначаються моделлю, для етапів до та після обробки, коли маршрут моделі їх підтримує, на `EV74`;
- уникайте вставки `VideoConvert`, `VideoScale` або GStreamer `videoconvert`/`videoscale` перед моделлю, якщо ви не створюєте граф лише для налагодження.

Камера надає вам кадри. Модуль обробки зображень (CVU) має виконувати обчислення для кожного кадру. Центральний процесор (CPU) не повинен випадково перетворюватися на ваш засіб обробки зображень.

## Швидкий довідник з усунення несправностей.

| Симптом | Спочатку перевірте. |
| --- | --- |
| Камера не знайдена. | Підтвердьте вибрані налаштування `.dtbo`, орієнтацію кабелю, процедуру перезавантаження та журнали ядра/бібліотеки libcamera. |
| Відсутній `libcamerasrc`. | Встановіть відповідне зображення камери або пакети програмного забезпечення для камери Neat/runtime для збірки DevKit. |
| `misconfig.media_caps` або `not-negotiated` | Перевірте точні значення `format,width,height,framerate` за допомогою `gst-launch-1.0`. Спробуйте відомий підтримуваний режим, наприклад `NV12 1920x1080@30`. |
| Сувора перевірка на відсутність копій. | Встановіть `allow_cpu_fallback = true` або використовуйте набір програмних модулів для камери, який надає доступ до можливостей SiMaAI, що передбачають відсутність копіювання даних. |
| Виведені кольори відображаються неправильно. | Переконайтеся, що кадр інтерпретується як `NV12`, а не RGB/BGR. Якщо JPEG-зображення, отримане безпосередньо з `libcamerasrc`, також відображається некоректно, спочатку налагодьте роботу ISP/налаштування камери, а потім переходьте до налагодження Neat. |
| Пропускна здатність низька. | Видаліть функції перетворення/масштабування відео за допомогою ЦП, використовуйте безперервне завантаження та застосовуйте політику черги для потокового джерела, яка надає перевагу найновішим даним. |

Для перегляду повного переліку симптомів зверніться до розділу [Усунення несправностей](/reference/troubleshooting) (усунення несправностей).
