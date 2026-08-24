# 022 Інтегруйте GenAI у граф

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | LFM2-VL-1.6B-a16w4 |
| Labels | genai, graph, composition, streaming, advanced |

## Concept

Використовуйте фрагменти графа GenAI, коли робота LLM, VLM або ASR є одним із етапів у більшому графі Neat.

## Walkthrough

Більшість застосунків генеративного штучного інтелекту (GenAI) повинні починатися з використання прямих API моделей. Компонування графа стає корисним, коли GenAI потрібно інтегрувати з іншими етапами Neat, такими як іменовані вхідні дані, іменовані вихідні дані, маршрутизація або оркестрація на рівні застосунку.

### Створення фрагмента графа GenAI {#step-create-fragment}

Створіть обробник моделі, специфічний для завдання, налаштуйте параметри фрагмента графа та створіть загальнодоступний `Graph` фрагмент.

Фрагмент, що поєднує обробку зображень і тексту, надає вхідні дані `prompt`, `image` і `use_cached_image`, а також вихідні дані `tokens`, `done`, `encoded` і `error`. Фрагмент транскриптора мовлення надає вхідні дані `audio` і `audio_path`, а також вихідні дані `tokens`, `done` і `error`.

`SpeechTranscriberOptions` за замовчуванням використовує автоматичне визначення мови та транскрипцію. Встановіть `task` на `ASRTask::Translate` в C++ або `ASRTask.Translate` в Python, щоб перекласти мовлення на англійську мову. Його пакет `done` містить інформацію про виявлену вихідну мову, а також, за наявності, `no_speech_prob` і `avg_logprob`.

### Додавання фрагмента до графа застосунку {#step-compose-graph}

Додайте фрагмент до більшого графа застосунку. Фрагмент зберігає назви своїх загальнодоступних кінцевих точок, тому код застосунку може передавати та отримувати дані за назвою.

### Створення та передавання вхідних даних графа {#step-push-prompt}

Створіть граф у вигляді `Run`, передайте зразок зображення у вхід `image`, потім передайте текстовий зразок у вхід `prompt` і дозвольте етапу GenAI генерувати токени.

### Отримання токенів і метаданих завершення {#step-pull-results}

Отримуйте дані з `tokens`, поки не надійде зразок `done`. Зразок `done` містить інформацію, таку як кількість згенерованих токенів і причина завершення.

## Run

У Modalix DevKit завантажте VLM LFM2-VL 1.6B з Hugging Face, використовуючи CLI LLiMa:

```bash
llima pull LFM2-VL-1.6B-a16w4
```

Запустіть навчальний посібник у Modalix з локальним каталогом моделі DevKit і локальним зображенням:

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

Очікуваний вивід містить опис графа та поточну відповідь, отриману з вихідних даних `tokens`.

## In Practice

Використовуйте цей шаблон, коли GenAI є частиною більшого графа застосунку. Забезпечте прямий `GenAIModel`, `VisionLanguageModel`і `ASRModel` вимагає простого коду застосунку для обробки запитів і відповідей.

## Файли коду:
- C++: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.cpp`
- Python: `tutorials/022_compose_genai_into_graph/compose_genai_into_graph.py`
