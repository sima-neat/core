# 023 Запустіть модель камери MIPI

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Model | User-provided camera-compatible model |
| Labels | mipi, camera, live-input, model, ev74 |

## Concept

Прикріпіть Modalix DevKit Камера MIPI до `Graph` з `CameraInput`, передавайте в прямому ефірі `NV12` кадри передаються для попередньої обробки, якою керує модель, і виводяться результати моделі. Це прямий шлях передачі даних з камери для розгорнутих застосунків комп’ютерного зору: камера є джерелом даних, CVU/EV74 здійснює попередню обробку зображень, MLA виконує обчислення, а застосунок використовує отриманий результат.

## Walkthrough

У цьому розділі передбачається, що камера вже працює через накладення на плату та libcamera. Neat не вибирає файли `.dtbo` і не налаштовує ISP; він обробляє кадри після того, як `libcamerasrc` їх генерує. Перед запуском навчального посібника перевірте роботу камери за допомогою [інструкції з апаратного забезпечення MIPI](https://developer.sima.ai/hardware/getting-started/standalone-mode/mipi-camera-interfaces) і перевірки параметрів GStreamer.

Уявіть собі, що цей навчальний посібник – це другий етап. Перший етап – це налаштування камери: накладення, драйвер, libcamera, ISP і точні параметри. Другий етап – це Neat graph: кадри з камери передаються на попередню обробку CVU, для виконання висновків MLA, необов’язковий EV74 BoxDecode і виведення результатів.

### Налаштуйте джерело камери {#step-configure-camera}

`CameraInputOptions` описує параметри джерела, які Neat запитує від `libcamerasrc`: роздільна здатність, частота кадрів, формат і необов’язкове ім’я камери libcamera. Встановіть `allow_cpu_fallback = true` для поточних конфігурацій камери, які ще не підтримують буфери SiMaAI із нульовим копіюванням. Суворе нульове копіювання все ще доступне через `--strict-zero-copy`, якщо ваш `libcamerasrc` це підтримує.

### Налаштуйте маршрут моделі {#step-configure-model}

Модель розглядає кадри з камери як зображення у форматі `NV12`. Налаштуйте керовану моделлю попередню обробку для перетворення кольору, зміни розміру, нормалізації, квантування та теселяції. У прикладі керована моделлю попередня обробка CVU прив’язується до `EV74`, щоб у робочому графі не відбулося непомітного перетворення на конвеєр обробки зображень на ЦП. За допомогою `--decode none` маршрут завершується в MLA і повертає необроблені тензори моделі. За допомогою токена YOLO `--decode`, BoxDecode виконується як керована моделлю стадія постобробки EV74.

### Складіть граф, що належить джерелу {#step-compose-graph}

Спочатку додайте `CameraInput`, а потім додайте маршрут моделі з параметром `include_input = false`. Немає загальнодоступного вузла `Input`, оскільки кадри генеруються всередині працюючого конвеєра. `include_output = true` зберігає точку витягування для виявлень або тензорів.

### Отримайте вихідні дані {#step-pull-output}

Створіть граф і отримайте фіксовану кількість вихідних даних. Тайм-аут означає, що вихід моделі не досяг застосунку до `--pull-timeout-ms`; можливо, камера зупинилася, можливо, не вдалося узгодити параметри, або можливо, що наступна стадія, така як BoxDecode, зазнає обмежень. Виведіть кількість тензорів і форму першого тензора, щоб ви могли підтвердити, що дані передаються, перш ніж додавати логіку застосунку.

## Run

Запустіть цей посібник безпосередньо на Modalix DevKit з налаштованою камерою MIPI. Запустіть попередньо створені команди з кореневої директорії встановлення Neat; запускайте команди для збірки з вихідного коду з кореневої директорії репозиторію. Архів моделі має відповідати попередній обробці та обраному режиму `--decode`.

За замовчуванням час очікування завантаження становить 15 секунд. Збільште значення `--pull-timeout-ms`, коли ви збираєте дані для первинної діагностики на холодній платі.

**Python:**
<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /шлях/до/моделі.tar.gz --frames 5 --decode none
</ShellCommand>

**C++ (prebuilt):**
<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /шлях/до/моделі.tar.gz --frames 5 --decode none
</ShellCommand>

Для моделей у стилі YOLO, які підтримують маршрут BoxDecode, виберіть токен декодування, наприклад, `yolov8` або `yolov9seg`:

<ShellCommand prompt="devkit">
python3 share/sima-neat/tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py \
  --model /шлях/до/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

<ShellCommand prompt="devkit">
./lib/sima-neat/tutorials/tutorial_023_run_mipi_camera_model \
  --model /шлях/до/yolo.tar.gz --frames 5 --decode yolov8
</ShellCommand>

**C++ (build from source):**
<ShellCommand prompt="devkit">
./build.sh --target tutorial_023_run_mipi_camera_model
</ShellCommand>

<ShellCommand prompt="devkit">
./build/tutorials-standalone/tutorial_023_run_mipi_camera_model \
  --model /шлях/до/моделі.tar.gz --frames 5 --decode none
</ShellCommand>

Очікувана форма вихідних даних залежить від моделі та шляху декодування. Необроблені вихідні дані MLA зазвичай містять тензори, специфічні для конкретної моделі:

```text
frame=0 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=1 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=2 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=3 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
frame=4 tensors=<raw_tensor_count> first_shape=[<model_specific_shape>]
[OK] 023_run_mipi_camera_model
```

За наявності підтримуваного маршруту BoxDecode, вихідні дані змінюються на декодовані тензори для виявлення або сегментації. Використовуйте кількість тензорів і першу форму як перевірку руху, а не як універсальну угоду.

Якщо ви бачите `output_timeout`, перевірте камеру за допомогою `gst-launch-1.0`, а потім перевірте згенерований бекенд за допомогою `--print-backend`. Для маршрутів BoxDecode підтвердьте, що архів моделі, токен `--decode` і порогові значення відповідають моделі.

## In Practice

Використовуйте `--print-backend`, коли потрібно перевірити згенерований шлях GStreamer. Шлях виробництва повинен містити `libcamerasrc`, `neatcamerabridge` (коли ввімкнено резервний режим), `neatprocesscvu`, `neatprocessmla`, необов’язкову постобробку EV74 і `appsink`. Він не повинен містити `appsrc`, `ostosima`, `videoconvert` або `videoscale`, якщо ви не додали спеціально шлях лише для налагодження.

## Файли джерела
- C++: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.cpp`
- Python: `tutorials/023_run_mipi_camera_model/run_mipi_camera_model.py`
