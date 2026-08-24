# 021 Розгортання моделей GenAI

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Beginner |
| Estimated Read Time | 15-20 minutes |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4, Qwen3-VL-4B-Instruct-GPTQ-a16w4, whisper-small-a16w8 |
| Labels | genai, server, llm, vlm, asr, http |

## Concept

Розмістіть декілька моделей GenAI на сервері Neat GenAI, щоб користувацький інтерфейс, сервіс або віддалений клієнт могли викликати кінцеві точки LLM, VLM та ASR з одного процесу. Хост.

## Walkthrough

Для більшості застосунків почніть з `GenAIServer` та його кінцевої точки, сумісної з OpenAI, `POST /v1/chat/completions`. Використовуйте прямі виклики `model.run(request)`, коли вбудована логіка застосунку повинна контролювати виклик моделі в одному й тому ж процесі.

Див. [довідник GenAI Server](/develop-apps/development-workflow/genai-model/genai-server) для отримання повної інформації про кінцеву точку та контракт запиту.

### Налаштуйте сервер {#step-configure-server}

Виберіть хост і порт. Хост за замовчуванням — `0.0.0.0`, який приймає з’єднання від інших машин, які можуть отримати доступ до пристрою Modalix.

### Зареєструйте каталоги моделей {#step-register-models}

Додайте кожен розгорнутий каталог моделей із назвою, яка використовуватиметься для обслуговування. У цьому посібнику реєструються `llm`, `vlm` і `asr`; назва, що використовується для обслуговування, — це те, що клієнти надсилають у полі `model`.

### Почніть обслуговування {#step-start-serving}

Викличте `serve()` для блокуючого процесу на передньому плані або `start()`, коли ваш застосунок контролює решту часу виконання процесу.

Після запуску сервера перевірте зареєстровані назви моделей за допомогою `GET /v1/models`:

```bash
curl http://<modalix-ip>:9998/v1/models
```

У відповіді мають бути вказані назви, що використовуються для обслуговування, зареєстровані в цьому посібнику: `llm`, `vlm` і `asr`.

## Run


На Modalix DevKit завантажте моделі LLM, VLM та ASR з Hugging Face, використовуючи CLI LLiMa:

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
llima pull whisper-small-a16w8
```

Запустіть сервер на Modalix з усіма трьома локальними директоріями моделей DevKit:

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

Видаліть `--vlm` або `--asr`, якщо ви хочете використовувати лише частину моделей під час розробки.

Після запуску сервера спочатку перевірте, чи всі використані імена зареєстровані:

```bash
curl http://<modalix-ip>:9998/v1/models
```

Потім зробіть запити до кінцевих точок з клієнта. Замініть `<modalix-ip>` на IP-адресу або ім’я хоста вашого пристрою Modalix.
Наступні клієнтські програми для надсилання запитів використовують Python `requests`, передають відповідь у вигляді потоку та виводять значення TTFT на стороні сервера, а також середнє, мінімальне та максимальне значення TPS для кожного токена, коли це повідомляється.

### Текстовий запит до LLM

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_text.py \
  --server-ip <modalix-ip> \
  --model llm \
  "Give me three tips for designing a small REST API."
```

### Запит на виклик інструменту до великої мовної моделі (LLM)

Кінцева точка `POST /v1/chat/completions`, сумісна з OpenAI, і кінцева точка `POST /api/chat`, сумісна з Ollama, приймають визначення функцій у масиві `tools`. Кожен елемент повинен містити `type: "function"`, об’єкт `function` і непустий рядок `function.name`. Опис функції та параметри JSON Schema можуть бути включені всередину об’єкта `function`:

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

Встановіть `tool_choice` на `"auto"`, щоб модель могла вибрати оголошений інструмент, або на `"none"`, щоб вимкнути підказки та аналіз інструментів. Якщо `tool_choice` не вказано або встановлено на `null`, це еквівалентно встановленню `"auto"`, коли `tools` не є порожнім. Неправильні визначення інструментів, `tools`, що не є масивом, і непідтримувані значення або типи `tool_choice` повертають HTTP 400 з помилкою `invalid_request_error`. Та сама перевірка визначень інструментів і вибору інструментів застосовується до прямих викликів `GenerationRequest` перед початком виведення.

### Запит тексту та зображення до візуальної мовної моделі (VLM)

Скрипт запиту кодує зображення у форматі Base64 і надсилає його як частину вмісту `image_url`, сумісну з OpenAI.

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_chat_completion_image.py \
  --server-ip <modalix-ip> \
  --model vlm \
  image.jpg \
  "What is the main subject of this image?"
```

### Аудіозапит до моделі ASR

За замовчуванням клієнт транскрипції автоматично визначає мову джерела. Використовуйте
`--language`, коли відома мова джерела:

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  speech.wav
```

Щоб перекласти мовлення на англійську мову, додайте `--translate`. Клієнт надсилає той самий
багатокомпонентний запит до `POST /v1/audio/translations`:

```bash
python3 share/sima-neat/tutorials/021_serve_genai_models/request_audio_transcription.py \
  --server-ip <modalix-ip> \
  --model asr \
  --translate \
  speech-in-another-language.wav
```

Для транскрипції використовується `POST /v1/audio/transcriptions`. Обидва маршрути підтримують
`stream=true`; наданий клієнт передає текст у потоковому режимі та виводить визначену мову джерела, `no_speech_prob` та `avg_logprob` з останньої події. Вище значення
`no_speech_prob` вказує на те, що Whisper вважає, що вхідні дані, ймовірно, не містять мовлення. `avg_logprob` – це середнє значення логарифмічної ймовірності згенерованих
токенів, де вище (менш від’ємне) значення вказує на більш впевнене декодування.

## In Practice

Використовуйте сервер, коли наявність мережевої межі є доцільною. Використовуйте прямі виклики `GenAIModel`, `VisionLanguageModel` та `ASRModel` для зменшення навантаження на код застосунку всередині одного процесу.

Запустіть один процес `GenAIServer` з кількома іменами моделей, які обслуговуються, для звичайних застосунків. Кілька процесів сервера можуть використовувати різні порти, якщо DevKit має достатньо пам’яті, але вони завантажують власні екземпляри моделей і все ще використовують спільний апаратний контролер MLA, тому їх не слід розглядати як спосіб збільшити апаратну пропускну здатність.

Кінцева точка `/v1/models` є найшвидшим способом перевірки: якщо вона повертає імена моделей, які обслуговуються, сервер доступний, і реєстр моделей заповнено.

## Файли коду
- C++: `tutorials/021_serve_genai_models/serve_genai_models.cpp`
- Python: `tutorials/021_serve_genai_models/serve_genai_models.py`
- Клієнти для надсилання запитів:
  - `tutorials/021_serve_genai_models/request_chat_completion_text.py`
  - `tutorials/021_serve_genai_models/request_chat_completion_image.py`
  - `tutorials/021_serve_genai_models/request_audio_transcription.py`
