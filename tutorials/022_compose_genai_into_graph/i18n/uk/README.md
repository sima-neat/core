# 022 Інтеграція GenAI у граф

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Advanced |
| Estimated Read Time | 20–25 хвилин |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, graph, composition, streaming, advanced |

## Concept

Використовуйте фрагменти графа GenAI, коли робота LLM, VLM або ASR — лише один етап більшого графа Neat.

## Walkthrough

Більшості застосунків GenAI варто починати з прямих API моделей. Компонування графа стає корисним, коли GenAI має працювати поруч з іншими етапами Neat, іменованими входами, іменованими виходами, маршрутизацією чи оркестрацією на рівні застосунку.

### Створення фрагмента графа GenAI {#step-create-fragment}

Створіть дескриптор моделі для конкретного завдання, налаштуйте параметри фрагмента графа й побудуйте публічний фрагмент `Graph`.

Візуально-мовний фрагмент надає входи `prompt`, `image` і `use_cached_image`, а також виходи `tokens`, `done`, `encoded` і `error`. Фрагмент транскрибування мовлення надає входи `audio` і `audio_path`, а також виходи `tokens`, `done` і `error`.

`SpeechTranscriberOptions` за замовчуванням виконує автоматичне визначення мови й транскрибування. Щоб перекласти мовлення англійською, задайте для `task` значення `ASRTask::Translate` у C++ або `ASRTask.Translate` у Python. Пакет `done` цього фрагмента повідомляє визначену мову оригіналу, а також, якщо вони доступні, `no_speech_prob` і `avg_logprob`.

### Додавання фрагмента до графа застосунку {#step-compose-graph}

Додайте фрагмент до більшого графа застосунку. Фрагмент зберігає імена своїх публічних кінцевих точок, тож код застосунку може передавати й отримувати дані за іменем.

### Побудова графа й передавання входів {#step-push-prompt}

Побудуйте з графа `Run`, передайте зразок зображення на вхід `image`, потім передайте текстовий зразок на вхід `prompt` і дайте етапу GenAI згенерувати токени.

### Отримання токенів і метаданих завершення {#step-pull-results}

Отримуйте дані з `tokens`, доки не надійде зразок `done`. Зразок `done` — це пакет із такими полями, як кількість згенерованих токенів і причина завершення.

## Run

На DevKit без NVMe перед `llima pull` задайте в `LLIMA_MODELS_PATH` теку з правом запису й використовуйте її замість `/media/nvme/llima/models` у командах нижче.

На Modalix DevKit завантажте VLM LFM2-VL 1.6B з Hugging Face за допомогою LLiMa CLI:

```bash
llima pull LFM2-VL-1.6B-a16w4
```

Запустіть посібник на Modalix із локальною текою моделі на DevKit і локальним зображенням:

**Python:**
```bash
python3 share/sima-neat/tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_022_compose_genai_into_graph \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_022_compose_genai_into_graph
./build/tutorials-standalone/tutorial_022_compose_genai_into_graph \
  --model /media/nvme/llima/models/LFM2-VL-1.6B-a16w4 \
  --image share/sima-neat/tutorials/assets/fronalpstock_1330.jpg
```

Очікуваний результат: програма виводить опис графа та відповідь, отриману потоком з виходу `tokens`.

## In Practice

Використовуйте цей шаблон, коли GenAI є частиною більшого графа застосунку. Для простого коду застосунку в стилі «запит — відповідь» залишайте прямі виклики `GenAIModel`, `VisionLanguageModel` і `ASRModel`.

## Файли початкового коду
- C++: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.cpp`
- Python: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py`
