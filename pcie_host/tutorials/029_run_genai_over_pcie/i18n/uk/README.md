# 029 Запуск моделей GenAI через PCIe

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 хвилин |
| Model | Qwen3-0.6B-Autoround-a16w4, LFM2-VL-450M-Autoround-a16w4, whisper-small-a16w8 |
| Labels | PCIe, GenAI, LLM, VLM, Whisper |

## Concept

Запускайте з застосунку на хості генерацію тексту, запитання про зображення та транскрибування мовлення на карті Modalix PCIe. Файли моделей, запити й результати передаються через PCIe; застосунок використовує публічний GenAI API, а не конвеєр GStreamer.

## Walkthrough

Кожен приклад завантажує одну підготовлену теку моделі LLiMa, що лежить на хості, виконує запит з обмеженнями та закриває модель. Спершу запускайте приклади окремо. Вони використовують однакові параметри з’єднання, але різні вхідні дані запитів.

### Завантаження LLM {#step-load-llm}

Створіть `pcie::genai::GenAIModel` у C++ або `pyneatpcie.genai.GenAIModel` у Python. Під час створення модель завантажується на карту; `build()` і вибору черги тут немає. Щоб вибрати карту, задайте `connection.card_id`. Приймаються абсолютні шляхи до моделі та шляхи відносно поточної робочої теки.

### Надсилання текстового запиту {#step-text-request}

Задайте `prompt` і `max_new_tokens`, а потім викличте `run()`. Результат містить текст відповіді та метрики генерації. За замовчуванням приклади обмежують генерацію 128 токенами; змініть цей ліміт за допомогою `--max-tokens`.

### Потокове отримання відповіді {#step-stream-answer}

Передайте прикладу LLM `--stream`, щоб натомість перебирати значення `TokenSample`. Виводьте кожен текстовий фрагмент одразу після надходження; останній зразок містить підсумкові метрики генерації. Для однієї моделі одночасно може бути активним лише один запит.

### Закриття моделі {#step-close-model}

У C++ приклад явно викликає `close()` після інференсу, а якщо виконання перериває виняток, ресурси звільняє деструктор. У Python використано менеджер контексту, який закриває модель під час виходу з блоку, зокрема й у разі помилки. Обидва тайм-аути з’єднання скінченні: 15 хвилин на запуск і 5 хвилин на запит.

### Підготовка зображення RGB {#step-prepare-image}

Приклад VLM читає зображення на хості за допомогою OpenCV, перетворює BGR на RGB і формує неперервний масив пікселів UInt8 форми `[H, W, 3]`. У C++ вектор, що володіє пікселями, обгортається в тензор PCIe; у Python передається масив NumPy. Не нормалізуйте пікселі й не підставляйте замість них шлях до зображення, що лежить на карті.

### Запитання про зображення {#step-image-request}

Завантажте VLM і додайте вхід RGB до `request.images` разом із промптом. Попередню обробку зображення модель виконує сама. Повернений текст — це відповідь на запитання; поля для файлу зображення в запиті PCIe немає.

### Транскрибування мовлення за допомогою Whisper {#step-audio-request}

Завантажте теку Whisper і задайте в `audio_file` шлях до аудіофайлу на хості. Використовуйте `language="auto"`, щоб мову було визначено автоматично, або вкажіть мову оригіналу через `--language`. За замовчуванням запит виконує транскрибування; `--translate` вмикає переклад англійською. Прочитайте транскрипт, мову та, якщо вона є, ймовірність відсутності мовлення. Не додавайте до запиту ASR промпт для чату.

## Run

Виконайте інструкції зі сторінок [Встановлення хоста PCIe](/getting-started/neat-library/pcie-host/) і [Налаштування навчального посібника](/tutorials/before-you-run). Використовуйте відповідні один одному пакети хоста й карти, що надають PCIe GenAI API. Усі наведені нижче команди виконуйте на хості з кореневої теки розпакованих додаткових матеріалів PCIe.

### Завантаження підготовлених моделей

Ці репозиторії SiMa.ai Model Zoo вже містять скомпільовані артефакти LLiMa: [Qwen3](https://huggingface.co/simaai/Qwen3-0.6B-Autoround-a16w4), [LFM2-VL](https://huggingface.co/simaai/LFM2-VL-450M-Autoround-a16w4) та [Whisper](https://huggingface.co/simaai/whisper-small-a16w8). Завантажуйте лише ті моделі, які плануєте запускати. Ознайомтеся з їхніми картками моделей і використовуйте артефакти, сумісні з установленою у вас платформою.

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

Зберігайте кожну теку повністю, разом із токенізатором, конфігурацією та скомпільованими артефактами. Ні розпаковувати, ні вручну копіювати щось на карту не потрібно. Необроблена контрольна точка Hugging Face не є підготовленою текою LLiMa; для власних моделей перед запуском цього навчального посібника виконайте інструкції зі сторінки [GenAI з LLiMa](/genai-llima/).

### Підготовка медіафайлів

Команди VLM використовують зображення вуличної сцени з набору навчальних матеріалів. Для Whisper запишіть або підготуйте на хості короткий фрагмент мовлення. Якщо встановлено FFmpeg, перетворіть наявний запис на моно-WAV із частотою дискретизації 16 кГц:

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le speech.wav
```

Не використовуйте тишу для оцінювання якості транскрибування. Приклади приймають інший шлях до зображення чи аудіофайлу через `--image` або `--audio`.

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

Для збирання C++ з початкового коду потрібні пакет розробки PCIe та файли розробки OpenCV. Усі приклади за замовчуванням використовують карту 0 і SSH-користувача з’єднання за замовчуванням — sima. Якщо ваше встановлення відрізняється, додайте `--card N`, `--user USER`, `--ssh-key PATH` або `--card-host ADDRESS`. Використовуйте заздалегідь підготовлений обліковий запис SSH; приклади не налаштовують доступ.

Очікуйте відповідь LLM, опис сцени або транскрипт, що відповідає записаному мовленню. Формулювання залежить від моделі та вхідних даних; сам по собі успішний запуск не є перевіркою якості.

## In Practice

Для незалежних моделей використовуйте окремі дескриптори, якщо вистачає пам’яті й обчислювальних ресурсів карти. Для чату передавайте історію через `messages`, а не через `prompt`; незалежні виклики не пам’ятають попередніх реплік. Щоб скасувати потокову генерацію, викличте `cancel()` і отримайте завершальний зразок, перш ніж знову використовувати модель.

Огляд API наведено на сторінці [GenAI API через PCIe](/develop-apps/development-workflow/genai-model/pcie-api). Автономні навчальні посібники з GenAI використовують інший простір імен і локальне виконання моделей; не підставляйте тут їхні дескриптори моделей.

## Файли початкового коду

- `run_llm.cpp` / `run_llm.py`
- `run_vlm.cpp` / `run_vlm.py`
- `run_whisper.cpp` / `run_whisper.py`
- `tutorial_args.h` (спільний розбір аргументів командного рядка для C++)
- `../assets/street-scene.png`
