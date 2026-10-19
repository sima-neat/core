# 029 Запуск моделей GenAI через PCIe

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | Qwen3-0.6B-Autoround-a16w4, LFM2-VL-450M-Autoround-a16w4, whisper-small-a16w8 |
| Labels | PCIe, GenAI, LLM, VLM, Whisper |

## Concept

Виконуйте генерацію тексту, запити про зображення та транскрибування мовлення з хостового застосунку на карті Modalix PCIe. Файли моделі, запити й результати передаються через PCIe; застосунок використовує публічний GenAI API, а не конвеєр GStreamer.

## Walkthrough

Кожен приклад завантажує підготовлену теку моделі LLiMa на хості, виконує запит з обмеженнями та закриває модель. Спочатку запускайте приклади окремо. Вони використовують однакові параметри з’єднання, але різні вхідні дані.

### Завантаження LLM {#step-load-llm}

Створіть `pcie::genai::GenAIModel` у C++ або `pyneatpcie.genai.GenAIModel` у Python. Конструктор завантажує модель на карту; `build()` і вибір черги не потрібні. Укажіть `connection.card_id`, щоб вибрати карту. Можна передавати абсолютний шлях моделі або шлях відносно поточної робочої теки.

### Надсилання текстового запиту {#step-text-request}

Задайте `prompt` і `max_new_tokens`, а потім викличте `run()`. Результат містить текст відповіді та метрики генерації. За замовчуванням приклади обмежують генерацію 128 токенами; змініть цей ліміт за допомогою `--max-tokens`.

### Потокове отримання відповіді {#step-stream-answer}

Передайте `--stream` прикладу LLM, щоб натомість перебирати значення `TokenSample`. Виводьте кожен текстовий фрагмент одразу після отримання; фінальний зразок містить підсумкові метрики генерації. Для кожної моделі одночасно може бути активним лише один запит.

### Закриття моделі {#step-close-model}

C++ явно викликає `close()` після інференсу, а якщо виняток перериває приклад, очищення виконує деструктор. Python використовує менеджер контексту, який закриває модель при виході з блоку, зокрема при помилках. Обидва тайм-аути з’єднання скінченні: 15 хвилин для запуску та 5 хвилин для запиту.

### Підготовка RGB-зображення {#step-prepare-image}

Приклад VLM читає зображення на хості через OpenCV, перетворює BGR на RGB і створює неперервні пікселі UInt8 із формою `[H, W, 3]`. C++ обгортає вектор, що володіє пікселями, у PCIe-тензор; Python передає масив NumPy. Не нормалізуйте пікселі та не підставляйте локальний шлях зображення на карті.

### Запит про зображення {#step-image-request}

Завантажте VLM і додайте RGB-вхід до `request.images` разом із промптом. Модель сама виконує попередню обробку зображення. Повернений текст відповідає на запитання; PCIe-запит не має поля для файлу зображення.

### Транскрибування мовлення за допомогою Whisper {#step-audio-request}

Завантажте теку Whisper та задайте в `audio_file` шлях до аудіо на хості. Використовуйте `language="auto"` для визначення мови або задайте вихідну мову через `--language`. За замовчуванням запит виконує транскрибування; `--translate` вибирає переклад англійською. Прочитайте транскрипт, мову та необов’язкову ймовірність відсутності мовлення. Не додавайте чат-промпт до ASR-запиту.

## Run

Виконайте [встановлення PCIe на хості](/getting-started/neat-library/pcie-host/) та [підготовку до навчальних прикладів](/tutorials/before-you-run). Використовуйте сумісні пакети хоста й карти, що надають PCIe GenAI API. Усі наведені нижче команди запускайте на хості з кореневої теки розпакованого PCIe extras.

### Завантаження підготовлених моделей

Ці репозиторії SiMa.ai Model Zoo вже містять скомпільовані артефакти LLiMa: [Qwen3](https://huggingface.co/simaai/Qwen3-0.6B-Autoround-a16w4), [LFM2-VL](https://huggingface.co/simaai/LFM2-VL-450M-Autoround-a16w4) та [Whisper](https://huggingface.co/simaai/whisper-small-a16w8). Завантажуйте лише ті моделі, які плануєте запускати. Перегляньте картки моделей і використовуйте артефакти, сумісні зі встановленою платформою.

```bash
source ~/pyneatpcie/bin/activate
python -m pip install huggingface_hub opencv-python
mkdir -p models
hf download simaai/Qwen3-0.6B-Autoround-a16w4 \
  --local-dir models/Qwen3-0.6B-Autoround-a16w4
hf download simaai/LFM2-VL-450M-Autoround-a16w4 \
  --local-dir models/LFM2-VL-450M-Autoround-a16w4
hf download simaai/whisper-small-a16w8 \
  --local-dir models/whisper-small-a16w8
```

Зберігайте кожну теку повністю, включно з токенізатором, конфігурацією та скомпільованими артефактами. Розпакування або ручне копіювання на карту не потрібне. Початковий контрольний знімок Hugging Face не є підготовленою текою LLiMa; для власних моделей спершу дотримуйтесь [GenAI з LLiMa](/genai-llima/), а потім запускайте цей приклад.

### Підготовка медіа

Команди VLM використовують зображення вуличної сцени з комплекту. Для Whisper запишіть або підготуйте короткий фрагмент мовлення на хості. Якщо встановлено FFmpeg, перетворіть наявний запис на моно WAV із частотою 16 кГц:

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le speech.wav
```

Не оцінюйте якість транскрибування за тишею. Приклади приймають інший шлях зображення або аудіо через `--image` чи `--audio`.

**Python:**

```bash
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_llm.py \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_vlm.py \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
python share/sima-pcie-host/tutorials/029_run_genai_over_pcie/run_whisper.py \
  --model models/whisper-small-a16w8 --audio speech.wav
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_029_run_llm \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
./lib/sima-pcie-host/tutorials/tutorial_029_run_vlm \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
./lib/sima-pcie-host/tutorials/tutorial_029_run_whisper \
  --model models/whisper-small-a16w8 --audio speech.wav
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_029_run_llm
./build.sh --target tutorial_029_run_vlm
./build.sh --target tutorial_029_run_whisper
./build/tutorials-standalone/tutorial_029_run_llm \
  --model models/Qwen3-0.6B-Autoround-a16w4 --stream
./build/tutorials-standalone/tutorial_029_run_vlm \
  --model models/LFM2-VL-450M-Autoround-a16w4 \
  --image share/sima-pcie-host/tutorials/assets/street-scene.png
./build/tutorials-standalone/tutorial_029_run_whisper \
  --model models/whisper-small-a16w8 --audio speech.wav
```

Для збирання C++ з вихідного коду потрібні пакет розробки PCIe та файли розробки OpenCV. Усі приклади за замовчуванням використовують карту 0 і стандартного SSH-користувача з’єднання sima. Якщо ваше встановлення відрізняється, додайте `--card N`, `--user USER`, `--ssh-key PATH` або `--card-host ADDRESS`. Використовуйте налаштований SSH-акаунт; приклади не налаштовують доступ.

Очікуйте відповідь LLM, опис сцени або транскрипт, що відповідає записаному мовленню. Формулювання залежить від моделі та вхідних даних; сам успішний запуск не є перевіркою якості.

## In Practice

Для незалежних моделей використовуйте окремі дескриптори з урахуванням пам’яті й обчислювальних ресурсів карти. Для чату передавайте історію через `messages`, а не `prompt`; незалежні виклики не пам’ятають попередніх реплік. Для скасування потоку викличте `cancel()` і дочитайте до фінального зразка перед повторним використанням моделі.

Огляд API наведено в [PCIe GenAI API](/develop-apps/development-workflow/genai-model/pcie-api). Навчальні приклади автономного GenAI використовують інший простір імен і локальне виконання моделей; не підставляйте їхні дескриптори моделей тут.

## Вихідні файли

- `run_llm.cpp` / `run_llm.py`
- `run_vlm.cpp` / `run_vlm.py`
- `run_whisper.cpp` / `run_whisper.py`
- `tutorial_args.h` (спільний розбір командного рядка C++)
- `../assets/street-scene.png`
