# 021 Обслуговування моделей GenAI

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Beginner |
| Estimated Read Time | 15–20 хвилин |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4, Qwen3-VL-4B-Instruct-GPTQ-a16w4, whisper-small-a16w8 |
| Labels | genai, server, llm, vlm, asr, http |

## Concept

Розмістіть кілька моделей GenAI на сервері Neat GenAI, щоб інтерфейс користувача, сервіс або віддалений клієнт міг викликати кінцеві точки LLM, VLM і ASR одного процесу.

## Walkthrough

У більшості застосунків починайте з `GenAIServer` і його сумісної з OpenAI кінцевої точки `POST /v1/chat/completions`. Використовуйте прямі виклики `model.run(request)`, коли виклик моделі має виконувати вбудована логіка застосунку в тому самому процесі.

Повний контракт кінцевих точок і запитів описано на довідковій сторінці [Сервер GenAI](/develop-apps/development-workflow/genai-model/genai-server).

### Налаштування сервера {#step-configure-server}

Виберіть хост і порт. Хост за замовчуванням — `0.0.0.0`: він приймає з’єднання від інших машин, які мають доступ до пристрою Modalix.

### Реєстрація тек моделей {#step-register-models}

Додайте кожну розгорнуту теку моделі разом з ім’ям, під яким сервер її надаватиме. Цей посібник реєструє `llm`, `vlm` і `asr`; саме це ім’я клієнти надсилають у полі `model`.

### Запуск обслуговування {#step-start-serving}

Викличте `serve()`, щоб процес працював на передньому плані в блокувальному режимі, або `start()`, якщо рештою життєвого циклу процесу керує ваш застосунок.

Після запуску сервера перевірте зареєстровані імена моделей через `GET /v1/models`:

```bash
curl http://<modalix-ip>:9998/v1/models
```

Відповідь має містити імена, зареєстровані в цьому посібнику: `llm`, `vlm` і `asr`.

## Run


На DevKit без NVMe перед `llima pull` задайте в `LLIMA_MODELS_PATH` теку з правом запису й використовуйте її замість `/media/nvme/llima/models` у командах нижче.

На Modalix DevKit завантажте моделі LLM, VLM і ASR з Hugging Face за допомогою LLiMa CLI:

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
llima pull whisper-small-a16w8
```

Запустіть сервер на Modalix з усіма трьома локальними теками моделей на DevKit:

**Python:**
```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/serve_genai_models.py \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_021_serve_genai_models \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_021_serve_genai_models
./build/tutorials-standalone/tutorial_021_serve_genai_models \
  --llm /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4 \
  --vlm /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --asr /media/nvme/llima/models/whisper-small-a16w8
```

Приберіть `--vlm` або `--asr`, якщо під час розробки хочете обслуговувати лише частину моделей.

Коли сервер запущено, спершу перевірте, що всі імена моделей зареєстровано:

```bash
curl http://<modalix-ip>:9998/v1/models
```

Потім викликайте кінцеві точки з клієнта. Замініть `<modalix-ip>` на IP-адресу або ім’я хоста вашого пристрою Modalix.
Наведені нижче клієнти запитів використовують Python-бібліотеку `requests`, отримують відповідь потоком і виводять TTFT на боці сервера, а також середнє, мінімальне й максимальне значення TPS на токен, якщо сервер їх повідомляє.

### Текстовий запит до LLM

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_text.py \
  --server-ip <modalix-ip> \
  --model llm \
  "Give me three tips for designing a small REST API."
```

### Запит із викликом інструментів до LLM

Сумісна з OpenAI кінцева точка `POST /v1/chat/completions` і сумісна з Ollama кінцева точка `POST /api/chat` приймають визначення функцій у масиві `tools`. Кожен елемент має містити `type: "function"`, об’єкт `function` і непорожній рядок `function.name`. Опис функції та параметри у форматі JSON Schema можна додати всередину об’єкта `function`:

```bash
curl http://<modalix-ip>:9998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "stream": false
  }'
```

Задайте для `tool_choice` значення `"auto"`, щоб модель могла вибрати оголошений інструмент, або `"none"`, щоб вимкнути додавання інструментів до промпту та розбір їхніх викликів. Якщо `tool_choice` пропущено або задано як `null`, поведінка така сама, як з `"auto"`, коли `tools` не порожній. Неправильні визначення інструментів, `tools`, що не є масивом, а також непідтримувані значення чи типи `tool_choice` повертають HTTP 400 з `invalid_request_error`. Така сама перевірка визначень інструментів і вибору інструменту застосовується до прямих викликів `GenerationRequest` ще до початку інференсу.

### Запит із текстом і зображенням до VLM

Скрипт запиту кодує зображення в Base64 і надсилає його як сумісну з OpenAI частину вмісту `image_url`.

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_image.py \
  --server-ip <modalix-ip> \
  --model vlm \
  image.jpg \
  "What is the main subject of this image?"
```

### Аудіозапит до моделі ASR

Клієнт транскрибування за замовчуванням автоматично визначає мову оригіналу. Якщо мову оригіналу відомо, використовуйте `--language`:

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  speech.wav
```

Щоб перекласти мовлення англійською, додайте `--translate`. Клієнт надсилає той самий multipart-запит до `POST /v1/audio/translations`:

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  --translate \
  speech-in-another-language.wav
```

Транскрибування використовує `POST /v1/audio/transcriptions`. Обидва маршрути підтримують `stream=true`; наданий клієнт виводить текст потоком і друкує з фінальної події визначену мову оригіналу, `no_speech_prob` і `avg_logprob`. Вище значення `no_speech_prob` означає, що, на думку Whisper, вхідні дані з більшою ймовірністю не містять мовлення. `avg_logprob` — середня логарифмічна ймовірність згенерованих токенів; вище (менш від’ємне) значення свідчить про впевненіше декодування.

## In Practice

Використовуйте сервер, коли мережева межа доречна. Для коду застосунку в тому самому процесі використовуйте прямі виклики `GenAIModel`, `VisionLanguageModel` і `ASRModel` — вони мають менші накладні витрати.

Для звичайних застосунків запускайте один процес `GenAIServer` з кількома зареєстрованими іменами моделей. Кілька серверних процесів можуть слухати різні порти, якщо DevKit має достатньо пам’яті, але кожен із них завантажує власні екземпляри моделей, а всі разом вони однаково ділять той самий диспетчер доступу до апаратного MLA, тому не варто вважати це способом помножити апаратну пропускну здатність.

Кінцева точка `/v1/models` — найшвидша димова перевірка: якщо вона повертає зареєстровані імена моделей, сервер доступний і реєстр моделей заповнено.

## Файли початкового коду
- C++: `tutorials/021_serve_genai_models/serve_genai_models.cpp`
- Python: `tutorials/021_serve_genai_models/serve_genai_models.py`
- Клієнти запитів:
  - `tutorials/021_serve_genai_models/request_chat_completion_text.py`
  - `tutorials/021_serve_genai_models/request_chat_completion_image.py`
  - `tutorials/021_serve_genai_models/request_audio_transcription.py`
