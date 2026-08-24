---
title: "Налаштування навчального посібника."
description: "Оберіть середовище виконання, завантажте навчальні матеріали та підготуйте архіви моделей."
sidebar_position: 2
slug: /tutorials/before-you-run
---

# Налаштування навчального посібника.

Виконайте налаштування один раз перед початком роботи з навчальним посібником. Оберіть середовище, яке відповідає категорії навчального посібника; пакети Neat Library та PCIe не є взаємозамінними.

## 1. Оберіть потрібне середовище.

| Категорія навчальних матеріалів | Запустіть на | Середовище Python |
| --- | --- | --- |
| Моделі та виведення даних, графи та конвеєри, камери та потокове передавання, генеративний штучний інтелект. | Modalix DevKit або середовище, назва якого вказана в навчальному посібнику. | `~/pyneat` |
| PCIe – спільна обробка даних | Хост, підключений до PCIe-карти Modalix. | `~/pyneatpcie` |

Навчальні матеріали з PCIe запускаються на хості, а не всередині контейнера SDK або безпосередньо на платі.

## 2. Налаштуйте навчальні матеріали для Neat Library.

Переконайтеся, що [Neat Library встановлено.](/getting-started/neat-library/install-or-update/),
потім виконайте ці команди з каталогу, в якому ви хочете розмістити пакет навчальних матеріалів:

<ShellCommand prompt="sdk-or-devkit">
sima-cli neat install core -t extras
cd sima-neat-*-Linux-extras
</ShellCommand>

Для навчальних матеріалів Python, які запускаються безпосередньо на DevKit, активуйте PyNeat і перевірте імпорт:

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 -c "import pyneat; print('pyneat ready')"
</ShellCommand>

## 3. Налаштуйте навчальні матеріали щодо PCIe.

Спочатку [встановіть і перевірте пакет хоста PCIe.](/getting-started/neat-library/pcie-host/).
Потім завантажте пакет навчальних матеріалів для версії Ubuntu, яка працює на хості.
Запустіть команду з каталогу, в якому ви хочете розмістити цей пакет.

**Ubuntu 22.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu22/amd64 -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

**Ubuntu 24.04:**

<ShellCommand prompt="pcie-host">
sima-cli neat install core/pciehost/ubuntu24/amd64 -t extras
cd sima-pcie-host-*-Linux-amd64-extras
</ShellCommand>

Перевірте PCIe PyNeat:

<ShellCommand prompt="pcie-host">
source ~/pyneatpcie/bin/activate
python3 -c "import pyneatpcie; print('pyneatpcie ready')"
</ShellCommand>

## 4. Підготуйте архіви моделей.

Використовуйте Model Zoo, щоб завантажити модель, назва якої вказана в інструкції. Наприклад:

<ShellCommand prompt="sdk-devkit-or-pcie-host">
sima-cli modelzoo get resnet_50
sima-cli modelzoo get yolo_v8s
</ShellCommand>

Навчальні матеріали Neat Library приймають `--model`, тому ви можете безпосередньо передавати завантажений архів. У навчальних матеріалах щодо PCIe використовуються фіксовані імена файлів у кореневій теці додаткових матеріалів PCIe:

| Навчальний посібник з PCIe | Необхідний файл моделі. |
| --- | --- |
| Запустіть свою першу модель через інтерфейс PCIe. | `yolo_v8s_mpk.tar.gz` |
| Запустіть асинхронний процес виведення даних PCIe. | `yolo_v8s_mpk.tar.gz` |
| Запустіть кілька моделей. | `resnet_50_mpk.tar.gz` та `yolo_v8s_mpk.tar.gz` |

Назви та розташування вихідних файлів Model Zoo можуть відрізнятися. За потреби скопіюйте архіви в основну теку PCIe, використовуючи необхідні назви:

<ShellCommand prompt="pcie-host">
cp /absolute/path/to/downloaded-resnet-archive.tar.gz resnet_50_mpk.tar.gz
cp /absolute/path/to/downloaded-yolov8s-archive.tar.gz yolo_v8s_mpk.tar.gz
</ShellCommand>

## 5. Перевірте шляхи та очікувані результати.

Запустіть навчальні команди з кореневої директорії, куди було вилучено додаткові файли. Переконайтеся, що вона містить допоміжні інструменти для збірки, попередньо скомпільовані програми на C++ та вихідний код для навчальних матеріалів:

<ShellCommand prompt="sdk-or-pcie-host">
test -x build.sh
ls lib/*/tutorials/
ls share/*/tutorials/
</ShellCommand>

- Готові програми, написані мовою C++, розміщено в `lib/<package>/tutorials/`.
- Вихідний код C++ і Python розміщено за адресою `share/<package>/tutorials/`.
- `./build.sh --list-targets` містить перелік програм, написаних мовою C++, які ви можете перекомпілювати.
- Успішні навчальні матеріали з C++ завершуються позначкою `[OK]`; у навчальних матеріалах з Python виводиться стислий
  результат, наприклад, `top1=...`, `completed=...` або `detections=...`.

Якщо в інструкції зазначено про відсутній файл, спочатку перевірте поточну директорію та назву файлу моделі. Для отримання додаткової допомоги зверніться до розділу [Усунення несправностей](/reference/troubleshooting/) (усунення несправностей).
