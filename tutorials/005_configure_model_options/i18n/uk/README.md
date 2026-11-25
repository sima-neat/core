# 005 Налаштування параметрів моделі

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 хвилин |
| Model | yolo_v8s |
| Labels | model-options, configuration, contracts |

## Concept

`ModelOptions` — це єдина структура, що оголошує контракт між вашими вхідними даними, етапами конвеєра моделі та декодуванням її виходу. Саме до неї ви звертаєтеся насамперед, коли поведінки за замовчуванням уже недостатньо.

## Walkthrough

Розділ 001 завантажував модель із розумними параметрами за замовчуванням. Реальні моделі — особливо моделі виявлення, як-от YOLOv8, — потребують, щоб ви *оголосили* їхній контракт: у якому піксельному форматі й розмірі надходять вхідні дані, як їх нормалізувати і як необроблені виходи мережі перетворюються на відфільтровані рамки. `ModelOptions` збирає все це в одну структуру, яку ви заповнюєте до створення моделі.

У цьому розділі ми повністю налаштовуємо модель YOLOv8, а потім перевіряємо контракт, який середовище виконання визначило з цих параметрів. Наприкінці ви задасте параметри входу, попередньої обробки й постобробки, прочитаєте визначені `input_specs()`/`output_specs()`/`metadata` і пропустите один детермінований кадр через налаштовану модель.

### Оголошення входу й попередньої обробки {#step-set-input-preproc}

Перший блок описує, який вигляд має кадр і як підготувати його для мережі. `format` (тут `BGR`) і межі `input_max_width`/`height`/`depth` задають вхідний контракт, на відповідність якому середовище виконання перевіряє дані й під який виділяє буфери. Поля нормалізації задають середнє значення й стандартне відхилення для кожного каналу, з якими навчали модель, щоб необроблені пікселі масштабувалися до діапазону, якого очікує мережа.

**C++:** Поля містяться в `opt.preprocess.*`: `kind = InputKind::Image`, `color_convert.input_format = PreprocessColorFormat::BGR` і `normalize.enable = AutoFlag::On`, де `mean`/`stddev` мають тип `std::array<float, 3>`.

**Python:** Поля містяться в `opt.preprocess.*`: `kind = pyneat.InputKind.Image`, `color_convert.input_format = pyneat.PreprocessColorFormat.BGR` і `normalize.enable = pyneat.AutoFlag.On`, де `mean`/`stddev` — списки.

### Оголошення постобробки {#step-set-postproc}

Другий блок визначає вихід детектора. `decode_type` вибирає шлях декодування рамок YOLOv8, а `score_threshold`, `nms_iou_threshold` і `top_k` фільтрують необроблені виявлення: відкидають рамки з низькою впевненістю, об’єднують ті, що перекриваються, і обмежують кількість тих, що залишаються. `boxdecode_original_width`/`boxdecode_original_height` передають декодеру геометрію початкового кадру, потрібну, щоб перевести нормалізовані координати назад у пікселі, а `name_suffix` робить згенеровані назви етапів стабільними, щоб граф конвеєра залишався читабельним у композиції з іншими.

**C++:** `decode_type = BoxDecodeType::YoloV8`; поля геометрії — `boxdecode_original_width`/`boxdecode_original_height`.

**Python:** `decode_type = pyneat.BoxDecodeType.YoloV8`; поля геометрії — `boxdecode_original_width`/`boxdecode_original_height`.

### Завантаження й перевірка визначеного контракту {#step-load-and-inspect}

Під час створення `Model` із цими параметрами контракт визначається з урахуванням архіву. Потім ми зчитуємо його: `input_specs()` і `output_specs()` повідомляють узгоджені обмеження тензорів, а `metadata()` показує контракт «ключ–значення», вбудований в архів. Перевірка цих даних після завантаження підтверджує, що середовище виконання прийняло ваші параметри, і показує конкретні форми, з якими ви працюватимете.

**C++:** Специфікації — це значення `TensorConstraint`; ми виводимо конкретну форму.

**Python:** Ми виводимо форми з `input_specs()[0]` і `output_specs()[0]`, а також `len(model.metadata())`.

### Пропускання одного кадру {#step-run-inference}

Наостанок ми синтезуємо один кадр BGR розміром `640×640` і пропускаємо його через налаштовану модель, щоб підтвердити, що весь контракт виконується від початку до кінця, і вивести, скільки виходів повернулося.

**C++:** Кадр — це `cv::Mat`; `run()` повертає `TensorList`, `size()` якого ми виводимо як `outputs=`.

**Python:** Кадр обгортається в `Tensor` через `Tensor.from_numpy(...)`; `run()` повертає `TensorList`, тож ми виводимо його довжину.

## Run

Запустіть програму — ви маєте побачити визначені форми специфікацій, кількість ключів метаданих і кількість виходів. Команди **Python** і **C++ (prebuilt)** запускайте з **кореневої теки встановлення Neat** (теки, що містить `share/` і `lib/`), а команди **build from source** — з **кореневої теки репозиторію**.

**Python:**
```bash
python3 share/sima-neat/tutorials/005_configure_model_options/configure_model_options.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_005_configure_model_options
./build/tutorials-standalone/tutorial_005_configure_model_options \
  --model /tmp/yolo_v8s.tar.gz
```

Очікуваний результат (форми й кількість ключів залежать від архіву моделі; версія для C++ виводить докладні рядки специфікацій і `outputs=`, а версія для Python — форми й `output_count=`):

```text
input_specs[0]: shape=[640,640,3]
output_specs[0]: shape=[]
metadata_keys=8
outputs=1
[OK] 005_configure_model_options
```

Щоб інтегрувати початковий код C++ цього розділу у свій проєкт із власним файлом `CMakeLists.txt` (тека extras не потрібна), див. розділ [Як запускати навчальні посібники](/tutorials#compile-a-copy-yourself) на головній сторінці.

## In Practice

### Пресети деталізації

Повідомленнями фреймворку під час побудови й виконання керує `VerboseOptions` у `GraphOptions`, `Model::Options` і `Model::RouteOptions`.

Поточне значення за замовчуванням для розробки: `VerboseOptions::debug_all()`. Якщо потрібно менше виводу, явно викличте `production()` або `quiet()`.

| Пресет | Призначення |
| --- | --- |
| `VerboseOptions::quiet()` | Приглушує вивід фреймворку про прогрес і подробиці. |
| `VerboseOptions::production()` | Показує лише лаконічний прогрес за фазами. |
| `VerboseOptions::debug_plugins()` | Зберігає вивід виробничого режиму, але також показує теми плагінів і GStreamer. |
| `VerboseOptions::debug_all()` | Примусово вмикає повний докладний вивід для всіх тем. |

Про налаштування черги й пропускної здатності середовища виконання див. [Налаштування пропускної здатності та глибини черги](/tutorials/tune-throughput-and-queues).

## Файли початкового коду
- C++: `tutorials/005_configure_model_options/configure_model_options.cpp`
- Python: `tutorials/005_configure_model_options/configure_model_options.py`
