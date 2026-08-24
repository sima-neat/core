# 024 Запустіть свою першу модель через PCIe

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, inference, tensor, image, detection |

## Concept

PCIe host API приймає або тензор, готовий до використання моделлю, або декодовані пікселі зображення.
У режимі тензора попередня обробка відбувається на хості. У режимі зображення оригінальні
пікселі надсилаються, а карта Modalix змінює їх розмір, конвертує колір і нормалізує.
Додавання декодування обмежувальних рамок зберігає вхідне зображення, але замінює шість вихідних тензорів YOLO на один компактний список виявлень.

## Walkthrough

Запустіть три незалежні програми з одним і тим же архівом YOLOv8s і зображенням вулиці розміром 640x480. Кожна програма демонструє один режим, використовує чергу 0 синхронно та
закриває одну модель. Це дозволяє кожному прикладу бути достатньо коротким, щоб його можна було скопіювати окремо.

### Запустіть тензор, готовий до використання моделлю {#step-tensor-mode}

Хост змінює розмір зображення відповідно до вхідних даних `[640, 640, 3]`, зазначених моделлю,
конвертує BGR в RGB і масштабує пікселі до `[0, 1]`. `Model.run()` надсилає цей тензор FP32 без попередньої обробки зображення на карті та виводить усі шість необроблених вихідних каналів YOLO.

### Перенесіть попередню обробку на карту {#step-image-mode}

Встановіть `preprocess.kind` на `Image`, ідентифікуйте вхідні пікселі як BGR і виберіть
попередньо встановлений профіль `COCO_YOLO`. Тепер хост надсилає декодовані пікселі, а карта виконує
зміну розміру, конвертацію BGR в RGB і нормалізацію. Програма виводить
імена та розміри шести необроблених вихідних каналів, щоб ви могли порівняти їх із режимом тензора.

### Декодуйте виявлення на карті {#step-decode-boxes}

Додайте `BoxDecodeType.YoloV8`, порогове значення для оцінки, порогове значення NMS і ліміт виводу.
Повернений тензор BBOX починається з кількості виявлень, за яким слідують записи фіксованого розміру, що містять `(x, y, width, height, score, class_id)`. Приклад аналізує
та виводить перші десять записів у координатах вихідного зображення.

### Проаналізуйте тензор BBOX {#step-parse-boxes}

Перевірте, чи декодування обмежувальних рамок повернуло один заповнений тензор, прочитайте його початкову кількість і відхиліть кількість, яка перевищує корисне навантаження. Кожен наступний 24-байтовий запис
потім перетворюється на одне виявлення для виведення.

## Run

Встановіть пакет PCIe host і завантажте пакет для навчального посібника, як описано в
[Налаштування навчального посібника.](/tutorials/before-you-run). Запустіть наступні
команди з кореневої папки розпакованих додаткових матеріалів PCIe:

```bash
sima-cli modelzoo get yolo_v8s
```

Програми вимагають наявності `yolo_v8s_mpk.tar.gz` у цьому каталозі. Вихідні
імена та розташування Model Zoo можуть відрізнятися. Якщо команда не створила саме цей шлях,
скопіюйте завантажений архів на місце та перевірте його:

```bash
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
test -f yolo_v8s_mpk.tar.gz
```

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_tensor_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_mode.py
python3 share/sima-pcie-host/tutorials/024_run_your_first_model_over_pcie/run_image_boxdecode.py
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_024_run_tensor_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_mode
./lib/sima-pcie-host/tutorials/tutorial_024_run_image_boxdecode
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_024_run_tensor_mode
./build.sh --target tutorial_024_run_image_mode
./build.sh --target tutorial_024_run_image_boxdecode

./build/tutorials-standalone/tutorial_024_run_tensor_mode
./build/tutorials-standalone/tutorial_024_run_image_mode
./build/tutorials-standalone/tutorial_024_run_image_boxdecode
```

Відповідні програми C++ і Python виводять однакові шість необроблених вихідних каналів
для режиму тензора та режиму зображення, за якими слідують декодовані люди, автомобілі або інші видимі
об’єкти:

```text
Tensor mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_tensor_mode
Image mode raw outputs:
  bbox_0 FP32 [80, 80, 64]
  ...
[OK] 024_run_image_mode
Image + boxdecode detections=...
  person score=... box=(...)
[OK] 024_run_image_boxdecode
```

За замовчуванням використовується карта 0 і черга 0. Передавайте `--card N` лише під час використання іншої карти;
її адреса керування визначається автоматично.

## In Practice

Використовуйте режим тензора, коли ваш застосунок вже створює тензори з точно таким типом даних,
формою, розміщенням, порядком кольорів і числовим діапазоном, як зазначено в `model.info()`. Використовуйте
режим зображення, коли застосунок природним чином володіє декодованими пікселями, і ви хочете, щоб
карта застосовувала повторювану попередню обробку моделі. Увімкніть декодування обмежувальних рамок, коли
застосунку потрібні виявлення, а не необроблені карти ознак.

У кожному режимі використовується один і той же життєвий цикл `pcie::Model`/`pyneatpcie.Model`. Змінюються лише
`ModelOptions` і передане корисне навантаження. Продовжуйте з
[Запустіть асинхронний процес виведення даних PCIe.](/tutorials/run-pcie-inference-async),
щоб перекрити надсилання та завершення за допомогою `push()` і `pull()`.

## Файли вихідного коду

- `run_tensor_mode.cpp`
- `run_tensor_mode.py`
- `run_image_mode.cpp`
- `run_image_mode.py`
- `run_image_boxdecode.cpp`
- `run_image_boxdecode.py`
- `../assets/street-scene.png`
