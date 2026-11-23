# 026 Запуск кількох моделей

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | resnet_50, yolo_v8s |
| Labels | PCIe, queues, concurrency, classification, detection |

## Concept

PCIe-карта Modalix надає доступ до черг співобробки з 0 по 5. Кожна активна `Model` має одну чергу, тому незалежні моделі можуть виконуватися одночасно шляхом призначення кожній інстанції окремої `ConnectionOptions.queue`. У цьому посібнику використовується ResNet-50 у черзі 0 і YOLOv8s у черзі 1 без додавання глобального координатора.

## Walkthrough

Обидві моделі навмисно використовують різні зображення: ResNet-50 класифікує чітке фото лабрадора, а YOLOv8s виявляє людей і автомобілі на жвавій міській сцені.

### Завантаження специфічних для моделі зображень {#step-load-assets}

Перевірте обидва архіви моделей і декодуйте обидва упаковані ресурси, перш ніж займати чергу. Зберігання зображень окремо робить кожен результат значущим і запобігає використанню класифікаційного портрета як навантаження для виявлення об’єктів.

### Призначте одну модель кожній черзі {#step-assign-queues}

Створіть два звичайні об’єкти `Model`. Налаштуйте ResNet-50 із попередньою обробкою зображень ImageNet у черзі 0 і YOLOv8s із попередньою обробкою зображень COCO плюс декодуванням обмежувальних рамок у черзі 1. Помилки під час збирання ідентифікують чергу та модель, які не вдалися; модель, яка вже була зібрана, закривається, якщо друга збірка не вдається.

### Запустіть обидві черги одночасно {#step-run-concurrently}

Запустіть один блокуючий процес виведення зображень для кожної моделі в окремих потоках хоста. Кожен виклик все ще використовує просту синхронну поведінку `run`, але виклики перекриваються, оскільки вони націлені на різні фізичні черги.

### Інтерпретуйте кожен результат незалежно {#step-read-results}

Черга 0 повертає один тензор класифікації FP32 і виводить його клас ImageNet з найвищим балом. Черга 1 повертає декодовані записи BBOX і виводить клас виявлення, впевненість і координати вихідного зображення. Закриття будь-якої моделі звільняє лише призначену їй чергу.

## Run

Установіть пакет хоста PCIe та завантажте набір навчальних матеріалів, як описано в розділі [Налаштування навчальних матеріалів](/tutorials/before-you-run). Виконайте наведені нижче команди в корені розпакованих додаткових матеріалів PCIe. У Windows налаштуйте навчальні матеріали та запускайте програми, як описано в розділі [Запуск застосунків у Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows).

Програмі потрібні точні шляхи `resnet_50_mpk.tar.gz` і `yolo_v8s_mpk.tar.gz` у цьому каталозі. Завантажте обидві моделі з [Model Browser](/tools/model-browser/), видобудьте архіви із завантажених ZIP-файлів і збережіть їх під цими назвами:

```bash
sima-cli models download --id resnet_50 --variant w8a8 --output models
sima-cli models download --id yolov8s --variant w8a8 --output models
unzip models/resnet_50/w8a8/*.zip resnet_50_w8a8_mpk.tar.gz
unzip models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
mv resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

У Windows завантажте моделі в PowerShell і видобудьте архіви за допомогою `Expand-Archive`:

```powershell
sima-cli models download --id resnet_50 --variant w8a8 --output models
sima-cli models download --id yolov8s --variant w8a8 --output models
Expand-Archive models\resnet_50\w8a8\*.zip -DestinationPath models\resnet_50\w8a8
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8
Copy-Item models\resnet_50\w8a8\resnet_50_w8a8_mpk.tar.gz resnet_50_mpk.tar.gz
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

Запустіть Python:

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/026_run_multiple_models/run_multiple_models.py
```

Запустіть попередньо зібраний посібник C++:

```bash
./lib/sima-pcie-host/tutorials/tutorial_026_run_multiple_models
```

Або перезберіть його:

```bash
./build.sh --target tutorial_026_run_multiple_models
./build/tutorials-standalone/tutorial_026_run_multiple_models
```

За наявності задокументованих моделей і ресурсів, обидві версії виводять результати, подібні до:

```text
queue=0 model=resnet_50 output_shape=[1, 1000] top1=208 (Labrador retriever)
queue=1 model=yolo_v8s detections=...
  person score=... box=(...)
[OK] 026_run_multiple_models
```

У посібнику ResNet-50 навмисно фіксується до черги 0, а YOLOv8s – до черги 1. Передавайте `--card N` лише під час використання іншої карти.

## In Practice

Призначення черги є рішенням щодо ресурсів застосунку: дві активні моделі не можуть використовувати одну й ту саму фізичну чергу. Збирайте моделі перед початком роботи, повідомляйте про конкретну чергу у разі збою та закривайте кожну успішно зібрану модель як у звичайних, так і в аварійних сценаріях. Окремі екземпляри `Model` зберігають результати та збої ізольованими, залишаючись при цьому простими для розуміння.

Для діагностики розгортання продовжуйте роботу з [Модель робочого процесу PCIe](/develop-apps/development-workflow/pcie-model/) і [посібник з усунення несправностей](/reference/troubleshooting/).

## Файли коду

- `run_multiple_models.cpp`
- `run_multiple_models.py`
- `../assets/labrador.jpg`
- `../assets/street-scene.png`
