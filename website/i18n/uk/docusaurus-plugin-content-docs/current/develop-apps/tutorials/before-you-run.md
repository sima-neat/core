---
title: "Налаштування навчального посібника"
description: "Оберіть середовище виконання, завантажте навчальні матеріали та підготуйте архіви моделей."
sidebar_position: 2
slug: /tutorials/before-you-run
---

# Налаштування навчального посібника

Виконайте налаштування один раз перед початком роботи з навчальним посібником. Оберіть середовище, яке відповідає категорії навчального посібника; набори Neat Library та PCIe не є взаємозамінними.

## 1. Оберіть потрібне середовище.

| Категорія навчальних матеріалів | Запустіть на | Середовище Python |
| --- | --- | --- |
| Моделі та виведення даних, графи та конвеєри, камери та потокове передавання, генеративний штучний інтелект. | Modalix DevKit або середовище, назва якого вказана в навчальному посібнику. | `~/pyneat` |
| Співобробка через PCIe | Хост, підключений до PCIe-карти Modalix. | `~/pyneatpcie` |

Навчальні матеріали з PCIe запускаються на хості, а не всередині контейнера SDK або безпосередньо на платі.

## 2. Налаштуйте навчальні матеріали для Neat Library.

Переконайтеся, що [Neat Library встановлено.](/getting-started/neat-library/install-or-update/),
потім виконайте ці команди з теки, в якій ви хочете розмістити набір навчальних матеріалів:

<ShellCommand prompt="sdk|devkit">
sima-cli neat install core@v%%release.core%% -t extras
cd sima-neat-*-Linux-extras
</ShellCommand>

Для навчальних матеріалів Python, які запускаються безпосередньо на DevKit, активуйте PyNeat і перевірте імпорт:

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 -c "import pyneat; print('pyneat ready')"
</ShellCommand>

## 3. Налаштуйте навчальні матеріали щодо PCIe.

Спочатку [встановіть і перевірте пакет хоста PCIe](/getting-started/neat-library/pcie-host/).
Потім завантажте набір навчальних матеріалів для версії Ubuntu, яка працює на хості.
Запустіть команду з теки, в якій ви хочете розмістити цей набір.

**Ubuntu 22.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64@v%%release.core%% -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

Перевірте PCIe PyNeat:

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. Підготуйте архіви моделей {#prepare-model-archives}

У навчальних матеріалах використовуються ResNet-50 і YOLOv8s із [Model Browser](/tools/model-browser/). Завантажити їх можна за допомогою `sima-cli`, який надає Palette SDK.

Для навчальних матеріалів Neat Library завантажте обидві моделі й видобудьте кожен архів моделі з його ZIP-файлу за шляхом, який команди навчальних матеріалів передають як `--model`:

<ShellCommand prompt="sdk|devkit">
sima-cli models download --id resnet_50 --variant w8a8
unzip -p models/resnet_50/w8a8/resnet_50_w8a8_default.zip resnet_50_w8a8_mpk.tar.gz > /tmp/resnet_50.tar.gz
sima-cli models download --id yolov8s --variant w8a8
unzip -p models/yolov8s/w8a8/yolov8s_w8a8_default.zip yolov8s_w8a8_mpk.tar.gz > /tmp/yolo_v8s.tar.gz
</ShellCommand>

У навчальних матеріалах щодо PCIe використовуються фіксовані імена файлів у кореневій теці додаткових матеріалів PCIe:

| Навчальний посібник з PCIe | Необхідний файл моделі. |
| --- | --- |
| Запустіть свою першу модель через інтерфейс PCIe. | `yolo_v8s_mpk.tar.gz` |
| Запустіть асинхронний процес виведення даних PCIe. | `yolo_v8s_mpk.tar.gz` |
| Запустіть кілька моделей. | `resnet_50_mpk.tar.gz` та `yolo_v8s_mpk.tar.gz` |
| Обгорніть зовнішню пам’ять тензора. | `yolo_v8s_mpk.tar.gz` |

Кожне завантаження — це ZIP-файл у `models/<model id>/<variant>/`. Завантажте моделі, потрібні для навчального матеріалу, з кореневої теки додаткових матеріалів PCIe і видобудьте кожен архів моделі з його ZIP-файлу під необхідною назвою:

<ShellCommand prompt="pcie-host">
sima-cli models download --id resnet_50 --variant w8a8
unzip -p models/resnet_50/w8a8/resnet_50_w8a8_default.zip resnet_50_w8a8_mpk.tar.gz > resnet_50_mpk.tar.gz
sima-cli models download --id yolov8s --variant w8a8
unzip -p models/yolov8s/w8a8/yolov8s_w8a8_default.zip yolov8s_w8a8_mpk.tar.gz > yolo_v8s_mpk.tar.gz
</ShellCommand>

## 5. Перевірте шляхи та очікувані результати.

Запустіть навчальні команди з кореневої теки, куди було вилучено додаткові файли. Переконайтеся, що вона містить допоміжні інструменти для збірки, попередньо скомпільовані програми на C++ та початковий код для навчальних матеріалів:

<ShellCommand prompt="sdk|pcie-host">
test -x build.sh
ls lib/*/tutorials/
ls share/*/tutorials/
</ShellCommand>

- Готові програми, написані мовою C++, розміщено в `lib/<package>/tutorials/`.
- Початковий код C++ і Python розміщено за адресою `share/<package>/tutorials/`.
- `./build.sh --list-targets` містить перелік програм, написаних мовою C++, які ви можете перекомпілювати.
- Успішні навчальні матеріали з C++ завершуються позначкою `[OK]`; у навчальних матеріалах з Python виводиться стислий
  результат, наприклад, `top1=...`, `completed=...` або `detections=...`.

Якщо в інструкції зазначено про відсутній файл, спочатку перевірте поточну теку та назву файлу моделі. Для отримання додаткової допомоги зверніться до розділу [Усунення несправностей](/reference/troubleshooting/).
