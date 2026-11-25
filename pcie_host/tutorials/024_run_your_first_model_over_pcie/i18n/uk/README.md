# 024 Запуск першої моделі через PCIe

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 хвилин |
| Model | yolo_v8s |
| Labels | PCIe, inference, tensor, image, detection |

## Concept

API хоста PCIe приймає або тензор, готовий для моделі, або декодовані пікселі зображення. У тензорному режимі попередня обробка залишається на хості. У режимі зображення хост надсилає початкові пікселі, а карта Modalix сама змінює їхній розмір, перетворює колір і нормалізує їх. Якщо додати декодування рамок, вхід залишається зображенням, а один компактний список виявлень замінює шість необроблених вихідних тензорів YOLO.

## Walkthrough

Запустіть три незалежні програми з тим самим архівом YOLOv8s і вуличною сценою розміром 640x480. Кожна програма показує один режим, синхронно використовує чергу 0 і закриває одну модель. Завдяки цьому кожен приклад достатньо короткий, щоб скопіювати його окремо.

### Запуск тензора, готового для моделі {#step-tensor-mode}

Хост застосовує letterbox до зображення під вхід `[640, 640, 3]`, про який повідомляє модель, перетворює BGR на RGB і масштабує пікселі до `[0, 1]`. `Model.run()` надсилає цей тензор FP32 без попередньої обробки зображення на карті й виводить усі шість необроблених вихідних маршрутів YOLO.

### Перенесення попередньої обробки на карту {#step-image-mode}

Задайте для `preprocess.kind` значення `Image`, позначте вхідні пікселі як BGR і виберіть набір параметрів `COCO_YOLO`. Тепер хост надсилає декодовані пікселі, а карта виконує зміну розміру з letterbox, перетворення BGR на RGB і нормалізацію. Програма виводить імена й форми шести необроблених вихідних маршрутів, щоб ви могли порівняти їх із тензорним режимом.

### Декодування виявлень на карті {#step-decode-boxes}

Додайте `BoxDecodeType.YoloV8`, поріг оцінки, поріг NMS і обмеження кількості результатів. Повернений тензор BBOX починається з кількості виявлень, за якою йдуть записи фіксованого розміру з полями `(x, y, width, height, score, class_id)`. Приклад розбирає й виводить перші десять записів у координатах вхідного зображення.

### Розбір тензора BBOX {#step-parse-boxes}

Перевірте, що декодування рамок повернуло один заповнений тензор, прочитайте кількість на початку тензора й відхиліть кількість, яка перевищує корисні дані. Кожен наступний 24-байтовий запис потім перетворюється на одне виявлення для виведення.

## Run

Установіть пакет хоста PCIe і завантажте набір навчальних матеріалів, як описано в розділі [Налаштування навчального посібника](/tutorials/before-you-run). Виконайте наведені нижче команди з кореневої теки розпакованих додаткових матеріалів PCIe:

```bash
sima-cli models download --id yolov8s --variant w8a8
unzip -p models/yolov8s/w8a8/yolov8s_w8a8_default.zip yolov8s_w8a8_mpk.tar.gz > yolo_v8s_mpk.tar.gz
```

Програмам потрібен файл `yolo_v8s_mpk.tar.gz` у цій теці. Друга команда видобуває його із завантаженого ZIP-файлу.

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

Відповідні програми на C++ і Python виводять ті самі шість контрактів необроблених виходів для тензорного режиму й режиму зображення, а потім декодованих людей, автомобілі чи інші видимі об’єкти:

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

За замовчуванням використовуються карта 0 і черга 0. Передавайте `--card N` лише тоді, коли використовуєте іншу карту; її адреса керування визначається автоматично.

## In Practice

Використовуйте тензорний режим, коли застосунок уже створює дані саме з тим dtype, формою, розкладкою, порядком кольорів і числовим діапазоном, про які повідомляє `model.info()`. Використовуйте режим зображення, коли застосунок і так працює з декодованими пікселями, і ви хочете, щоб повторювану попередню обробку моделі виконувала карта. Увімкніть декодування рамок, коли застосунку потрібні виявлення, а не необроблені карти ознак.

Кожен режим використовує той самий життєвий цикл `pcie::Model`/`pyneatpcie.Model`. Змінюються лише `ModelOptions` і надіслані корисні дані. Далі перейдіть до навчального посібника [Асинхронний запуск інференсу через PCIe](/tutorials/run-pcie-inference-async), щоб надсилати запити й отримувати результати з перекриттям у часі за допомогою `push()` і `pull()`.

## Файли початкового коду

- `run_tensor_mode.cpp`
- `run_tensor_mode.py`
- `run_image_mode.cpp`
- `run_image_mode.py`
- `run_image_boxdecode.cpp`
- `run_image_boxdecode.py`
- `../assets/street-scene.png`
