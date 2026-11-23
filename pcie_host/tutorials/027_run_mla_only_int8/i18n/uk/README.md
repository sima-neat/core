# 027 Запуск лише MLA з тензорами INT8

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 хвилин |
| Model | yolo26n w8a8 (Model Browser, скомпільовано для прямого входу та виходу MLA) |
| Labels | PCIe, MLA, INT8, quantization, tensor |

## Concept

Типовий маршрут PCIe надсилає тензори FP32 на карту, де вхід квантується, виконується MLA, а вихід деквантується й повертається на хост як FP32. Застосунок, який уже має дані INT8 або хоче самостійно керувати кроком квантування, може встановити `ModelOptions.mla_only`. Тоді карта виконує лише MLA: хост надсилає тензори INT8, що відповідають вхідному контракту MLA, і отримує сирі голови INT8. `model.info()` публікує параметри квантування кожного тензора, тож хост може квантувати й деквантувати за однією формулою:

```text
x = (q - zero_point) * scale
q = clamp(round(x / scale) + zero_point, -128, 127)
```

Цей посібник показує випадок INT8. `mla_only` також підтримує сумісні архіви BF16; тензори BF16 не мають параметрів квантування.

## Walkthrough

Одна програма квантує зображення на хості, виконує маршрут лише MLA, деквантує голови та звіряє їх із типовим маршрутом на тій самій черзі.

### Перегляньте контракт лише MLA {#step-inspect-contract}

Створіть `Model` з увімкненим `mla_only`. Тепер `info()` повідомляє входи та виходи INT8, кожен із `quant.scale` та `quant.zero_point`. Еталонна модель має один вхід, `images`, тензор INT8 `[640, 640, 3]` HWC.

### Квантуйте на хості {#step-quantize-on-host}

Еталонна модель очікує одне зображення RGB з пікселями в діапазоні `[0, 1]`. Змініть розмір зображення до `640x640`, перетворіть BGR на RGB, поділіть на 255 і застосуйте формулу квантування з параметрами входу. Ця попередня обробка належить моделі, а не контракту архіву: іншій моделі потрібен власний рецепт. Збережіть деквантовані значення тих самих кодів: це точний вхід FP32, який потрібен типовому маршруту для порівняння за однакових умов.

### Виконайте маршрут INT8 {#step-run-int8}

`build()` запускає конвеєр на карті, що містить лише MLA. `run()` приймає тензори INT8 і повертає по одному щільному тензору INT8 на кожен вихід у порядку та з іменами, які повідомив `info().outputs`. Маршрут відхиляє будь-який інший dtype; надсилання FP32 завершується помилкою замість квантування на карті.

### Деквантуйте та порівняйте {#step-dequantize-and-compare}

Створіть другу `Model` без `mla_only` і надішліть деквантовані значення FP32 через типовий маршрут. Деквантуйте голови INT8 за параметрами кожного виходу та виведіть найбільше відхилення для кожної голови в одиницях її scale. Обидва маршрути виконують ту саму програму MLA з тими самими кодами, тому похибка дорівнює нулю.

## Run

Установіть пакет хоста PCIe та завантажте набір навчальних матеріалів, як описано в розділі [Налаштування навчальних матеріалів](/tutorials/before-you-run).

Посібник виконує архів YOLO26n `w8a8` з [Model Browser](/tools/model-browser/), скомпільований для прямого входу та виходу MLA. У корені розпакованих додаткових матеріалів PCIe завантажте модель і видобудьте архів із завантаженого ZIP-файлу:

```bash
sima-cli models download --id yolo26n --variant w8a8 --output models
unzip models/yolo26n/w8a8/*.zip yolo26n_w8a8_mpk.tar.gz
```

У Windows завантажте модель у PowerShell і видобудьте архів за допомогою `Expand-Archive`:

```powershell
sima-cli models download --id yolo26n --variant w8a8 --output models
Expand-Archive models\yolo26n\w8a8\*.zip -DestinationPath models\yolo26n\w8a8
Copy-Item models\yolo26n\w8a8\yolo26n_w8a8_mpk.tar.gz .
```

Інші архіви підходять, якщо їх скомпільовано з `tessellate_parameters` у Model SDK з `enable_mla=True`, розкладкою DRAM `HWC` на кожному вході та `HWC16` на кожному виході. Архіви, що натомість виконують теселяцію на CVU, як-от збірка `yolo_v8s` з Model Zoo, відхиляються, коли ввімкнено `mla_only`:

```text
mla_only does not support stage 'tessellate_quantize_0_MLA_0/...' (tess)
```

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/027_run_mla_only_int8/run_mla_only_int8.py \
  --model yolo26n_w8a8_mpk.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_027_run_mla_only_int8 \
  --model yolo26n_w8a8_mpk.tar.gz
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_027_run_mla_only_int8
./build/tutorials-standalone/tutorial_027_run_mla_only_int8 \
  --model yolo26n_w8a8_mpk.tar.gz
```

У Windows зберіть програму за допомогою CMake з **x64 Native Tools Command Prompt** у своїй копії навчальних матеріалів. Замініть `C:\opencv\build` на каталог `build` пакета OpenCV, описаного в розділі [Запуск застосунків у Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows):

```bat
call "%ProgramFiles%\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_027_run_mla_only_int8
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_027_run_mla_only_int8.exe --model yolo26n_w8a8_mpk.tar.gz
```

Обидві версії виводять контракт і нульове відхилення для кожної голови:

```text
MLA-only contract:
  input images INT8 [640, 640, 3] scale=0.00391965 zero_point=-128
  output bbox_0 INT8 [80, 80, 4] scale=0.0331556 zero_point=-108
  ...
Dequantized MLA-only outputs vs the default route (error in scale units):
  bbox_0 [80, 80, 4] max_err=0.0000
  ...
[OK] 027_run_mla_only_int8
```

Типово використовуються карта 0 і черга 0. Передавайте `--card N` лише за використання іншої карти.

## In Practice

Для архіву INT8 умикайте `mla_only`, коли застосунок сам відповідає за квантування: він уже отримує INT8 із сенсора або попередньої моделі, йому потрібні сирі голови INT8 для власної постобробки, або він хоче прибрати етапи квантування й деквантування із затримки на боці карти. Читайте кожен scale і zero point з `model.info()`; ніколи не копіюйте їх з іншої збірки моделі.

Маршрут працює за принципом «усе або нічого». Кожен вхід має відповідати dtype, формі та розміру в байтах свого запису `info().inputs`, а попередню обробку зображень чи декодування рамок не можна поєднувати з `mla_only`. Залишайте типовий маршрут, коли хост має дані FP32 і не потребує виконувати перетворення самостійно.

Для діагностики розгортання перейдіть до [робочого процесу моделі PCIe](/develop-apps/development-workflow/pcie-model/).

## Файли коду

- `run_mla_only_int8.cpp`
- `run_mla_only_int8.py`
- `../assets/street-scene.png`
