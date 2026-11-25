# 023 Запуск моделі з камерою MIPI

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
| Difficulty | Intermediate |
| Estimated Read Time | 10–15 хвилин |
| Model | User-provided camera-compatible model |
| Labels | mipi, camera, live-input, model, ev74 |

## Concept

Під’єднайте камеру MIPI на Modalix DevKit до `Graph` через `CameraInput`, передайте живі кадри `NV12` на попередню обробку, якою керує модель, і отримайте виходи моделі. Це прямий шлях від камери для розгорнутих застосунків комп’ютерного зору: джерелом володіє камера, CVU/EV74 виконує попередню обробку зображень, MLA виконує інференс, а застосунок споживає результат.

## Walkthrough

Цей розділ передбачає, що камера вже працює через оверлей плати та libcamera. Neat не вибирає файли `.dtbo` і не налаштовує ISP; він споживає кадри, щойно `libcamerasrc` може їх видавати. Перед запуском посібника перевірте камеру за [інструкцією з апаратного підключення MIPI](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) і перевіркою caps у GStreamer.

Сприймайте цей посібник як крок 2. Крок 1 — введення камери в роботу: оверлей, драйвер, libcamera, ISP і точні caps. Крок 2 — граф Neat: кадри з камери на попередню обробку CVU, інференс на MLA, необов’язковий EV74 BoxDecode і отримання виходів.

### Налаштування джерела камери {#step-configure-camera}

`CameraInputOptions` описує caps джерела, які Neat запитує в `libcamerasrc`: роздільну здатність, частоту кадрів, формат і необов’язкове ім’я камери в libcamera. Задайте `allow_cpu_fallback = true` для поточних стеків камер, які ще не надають буферів SiMaAI без копіювання. Строгий режим без копіювання все ще доступний через `--strict-zero-copy`, якщо ваш `libcamerasrc` його підтримує.

### Налаштування маршруту моделі {#step-configure-model}

Модель бачить кадри з камери як зображення `NV12`. Налаштуйте керовану моделлю попередню обробку: перетворення кольору, зміну розміру, нормалізацію, квантування та теселяцію. Приклад прив’язує керовану моделлю попередню обробку CVU до `EV74`, щоб виробничий граф непомітно не перетворився на конвеєр обробки зображень на ЦП. З `--decode none` маршрут закінчується на MLA і повертає необроблені тензори моделі. З токеном YOLO в `--decode` BoxDecode виконується як керований моделлю етап постобробки на EV74.

### Компонування графа з власним джерелом {#step-compose-graph}

Спочатку додайте `CameraInput`, а потім маршрут моделі з `include_input = false`. Публічного вузла `Input` немає, бо кадри виникають усередині запущеного конвеєра. `include_output = true` залишає кінцеву точку для отримання виявлень або тензорів.

### Отримання виходів {#step-pull-output}

Побудуйте граф і отримайте задану кількість виходів. Тайм-аут означає, що жоден вихід моделі не дійшов до застосунку до спливання `--pull-timeout-ms`: камера могла зупинитися, caps могли не узгодитися, або наступний етап, як-от BoxDecode, може бути під зворотним тиском. Виведіть кількість тензорів і форму першого тензора, щоб переконатися, що дані рухаються, перш ніж додавати логіку застосунку.

## Run

Запускайте цей посібник безпосередньо на Modalix DevKit з налаштованою камерою MIPI. Команди для попередньо зібраних програм запускайте з кореневої теки встановлення Neat, а команди збирання з початкового коду — з кореневої теки репозиторію. Архів моделі має відповідати попередній обробці та необов’язковому режиму `--decode`, який ви запитуєте.

Тайм-аут отримання за замовчуванням — 15 секунд. Збільште `--pull-timeout-ms`, якщо збираєте діагностику першого запуску на холодній платі.

**Python:**
<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/model.tar.gz --frames 5 --decode none
</ShellCommand>

**C++ (prebuilt):**
<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /path/to/model.tar.gz --frames 5 --decode none
</ShellCommand>

Для моделей типу YOLO з підтримуваним маршрутом BoxDecode виберіть токен декодування, наприклад `yolov8` або `yolov9seg`:

<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /path/to/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /path/to/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

**C++ (build from source):**
<ShellCommand prompt="devkit">
./build.sh --target tutorial_023_run_mipi_camera_model
</ShellCommand>

<ShellCommand prompt="devkit">
./build/tutorials-standalone/tutorial_023_run_mipi_camera_model \
  --model /path/to/model.tar.gz --frames 5 --decode none
</ShellCommand>

Очікувана форма виходу залежить від моделі та маршруту декодування. Необроблений вихід MLA зазвичай містить тензори, специфічні для моделі:

```text
frame=0 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=1 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=2 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=3 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=4 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
[OK] 023_run_mipi_camera_model
```

З підтримуваним маршрутом BoxDecode вихід змінюється на декодовані тензори виявлення або сегментації. Використовуйте кількість тензорів і першу форму як перевірку того, що дані рухаються, а не як універсальний контракт.

Якщо бачите `output_timeout`, перевірте камеру за допомогою `gst-launch-1.0`, а потім перегляньте згенерований бекенд через `--print-backend`. Для маршрутів BoxDecode переконайтеся, що архів моделі, токен `--decode` і порогові значення відповідають моделі.

## In Practice

Використовуйте `--print-backend`, коли потрібно перевірити згенерований шлях GStreamer. Виробничий шлях має містити `libcamerasrc`, `neatcamerabridge` (коли ввімкнено резервний режим), `neatprocesscvu`, `neatprocessmla`, необов’язкову постобробку EV74 і `appsink`. Він не повинен містити `appsrc`, `ostosima`, `videoconvert` чи `videoscale`, якщо ви навмисно не додали шлях лише для налагодження.

## Файли початкового коду
- C++: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.cpp`
- Python: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py`
