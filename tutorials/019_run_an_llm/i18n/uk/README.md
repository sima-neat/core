# 019 Запуск LLM через прямий API

## Metadata
| Field | Value |
| --- | --- |
| Category | GenAI |
| Difficulty | Intermediate |
| Estimated Read Time | 10 хвилин |
| Model | Qwen3-4B-Instruct-2507-GPTQ-a16w4 |
| Labels | genai, llm, chat, history, streaming |

## Concept

Завантажте теку моделі GenAI, надішліть один простий промпт, додайте системний промпт, а потім розширте той самий шаблон до історії чату та потокового виведення.

## Walkthrough

Класичні посібники з `Model` використовують архіви MPK `.tar.gz`. Моделі GenAI натомість використовують теки моделей LLiMa та API `neat::genai`. Почніть із найпростішого запиту: завантажте модель, задайте `request.prompt`, запустіть запит і виведіть відповідь. Коли це запрацює, переходьте на `request.messages`, якщо потрібен стан розмови.

### Завантаження теки моделі {#step-load-model}

Вкажіть `GenAIModel` шлях до розгорнутої теки моделі LLiMa. Цей посібник використовує `GenAIModel`, бо він автоматично визначає, що містить тека: модель LLM, VLM чи ASR.

**C++:** Створіть `simaai::neat::genai::GenAIModel` зі шляху до моделі.

**Python:** Створіть `pyneat.genai.GenAIModel` зі шляху до моделі.

### Надсилання одного промпту {#step-send-prompt}

Створіть `GenerationRequest` з `prompt` і лімітом токенів. Це найкоротший шлях для разових запитань, тестів і скриптів.

### Визначення системного промпту {#step-system-prompt}

Використовуйте коротку системну інструкцію, щоб скерувати поведінку моделі. До простого запиту з промптом її можна додати через `system_prompt`; коли перейдете до історії чату, перенесіть ту саму інструкцію в список повідомлень як повідомлення `system`.

### Перехід до повідомлень {#step-store-history}

Для запитів у стилі чату використовуйте `messages` замість `prompt`: почніть із системного повідомлення та повідомлення користувача, виконайте запит, а потім збережіть відповідь асистента. Модель сама не пам’ятає попередніх викликів `run()`; історією повідомлень володіє ваш застосунок.

### Уточнювальне запитання з історією {#step-follow-up}

Додайте ще одне повідомлення користувача, надішліть оновлений список повідомлень і прочитайте відповідь. Тепер модель бачить усю розмову, яку зберіг ваш застосунок.

### Потокове виведення відповіді {#step-stream-answer}

Для виведення в інтерфейсі користувача викличте `stream()` і пройдіть циклом по поверненому `GenerationStream`. Кожен зразок токена містить останній фрагмент тексту.

## Run

На DevKit без NVMe перед `llima pull` задайте в `LLIMA_MODELS_PATH` теку з правом запису й використовуйте її замість `/media/nvme/llima/models` у командах нижче.

На Modalix DevKit завантажте LLM, наприклад Qwen3 4B, з Hugging Face за допомогою LLiMa CLI:

```bash
llima pull Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

Запустіть посібник на Modalix із локальною текою моделі на DevKit:

**Python:**
```bash
python3 share/sima-neat/tutorials/019_run_an_llm/run_an_llm.py \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_019_run_an_llm \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_019_run_an_llm
./build/tutorials-standalone/tutorial_019_run_an_llm \
  --model /media/nvme/llima/models/Qwen3-4B-Instruct-2507-GPTQ-a16w4
```

Очікуваний результат — відповідь на простий промпт, відповідь із системним промптом, відповідь на уточнювальне запитання з урахуванням контексту та фінальна відповідь, виведена потоком.

## In Practice

Зберігайте лише стільки історії повідомлень, скільки потрібно вашому застосунку. Довга історія витрачає токени контексту та збільшує час до першого токена. У застосунках із тривалим чатом зберігайте розмову поза об’єктом моделі й заново формуйте `GenerationRequest.messages` для кожної репліки.

## Файли початкового коду
- C++: `tutorials/019_run_an_llm/run_an_llm.cpp`
- Python: `tutorials/019_run_an_llm/run_an_llm.py`
