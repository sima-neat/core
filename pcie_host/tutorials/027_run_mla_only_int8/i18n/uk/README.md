# 027 Запуск лише MLA з тензорами INT8

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 хвилин |
| Model | any archive compiled for direct MLA input and output |
| Labels | PCIe, MLA, INT8, quantization, tensor |

## Concept

Типовий маршрут PCIe надсилає тензори FP32 на карту, де EV74 квантує їх, виконується MLA, а EV74 деквантує результати назад у FP32. Застосунок, який уже має дані INT8 або хоче самостійно керувати кроком квантування, може встановити `ModelOptions.mla_only`. Тоді карта виконує лише MLA: хост надсилає тензори INT8, що відповідають вхідному контракту MLA, і отримує сирі голови INT8. `model.info()` публікує параметри квантування кожного тензора, тож хост може квантувати й деквантувати за однією формулою:

```text
x = (q - zero_point) * scale
q = clamp(round(x / scale) + zero_point, -128, 127)
```

## Walkthrough

Одна програма квантує зображення на хості, виконує маршрут лише MLA, деквантує голови та звіряє їх із типовим маршрутом на тій самій черзі.

### Перегляньте контракт лише MLA {#step-inspect-contract}

Створіть `Model` з увімкненим `mla_only`. Тепер `info()` повідомляє входи та виходи INT8, кожен із `quant.scales[0]` та `quant.zero_points[0]`. Входи також містять `input_range` — діапазон із рухомою комою, для якого модель була відкалібрована. Модель із кількома входами перелічує по одному тензору INT8 на кожен вхід у порядку подання.

### Квантуйте на хості {#step-quantize-on-host}

Змініть розмір зображення до геометрії входу, перетворіть BGR на RGB, відобразіть пікселі на `input_range` і застосуйте формулу квантування з параметрами входу. Збережіть декванотовані значення тих самих кодів: це точний вхід FP32, який потрібен типовому маршруту для порівняння за однакових умов.

### Виконайте маршрут INT8 {#step-run-int8}

`build()` запускає конвеєр на карті, що містить лише MLA. `run()` приймає тензори INT8 і повертає по одному щільному тензору INT8 на кожен вихід у порядку та з іменами, які повідомив `info().outputs`. Маршрут відхиляє будь-який інший dtype; надсилання FP32 завершується помилкою замість квантування на карті.

### Деквантуйте та порівняйте {#step-dequantize-and-compare}

Створіть другу `Model` без `mla_only` і надішліть деквантовані значення FP32 через типовий маршрут. Деквантуйте голови INT8 за параметрами кожного виходу та виведіть найбільше відхилення для кожної голови в одиницях її scale. Обидва маршрути виконують ту саму програму MLA з тими самими кодами, тому похибка дорівнює нулю.

## Run

Установіть пакет хоста PCIe та завантажте набір навчальних матеріалів, як описано в розділі [Налаштування навчальних матеріалів](/tutorials/before-you-run).

Для цього посібника потрібен архів, скомпільований для прямого входу та виходу MLA: `tessellate_parameters` у Model SDK з `enable_mla=True`, розкладкою DRAM `HWC` на кожному вході та `HWC16` на кожному виході. Архіви Model Zoo натомість виконують теселяцію на EV74 і відхиляються, коли ввімкнено `mla_only`:

```text
mla_only does not support stage 'tessellate_quantize_0_MLA_0/...' (tess)
```

Скопіюйте відповідний архів у корінь розпакованих додаткових матеріалів PCIe, наприклад як `model_mlatess_int8.tar.gz`, і передайте його шлях через `--model`.

**Python:**

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/027_run_mla_only_int8/run_mla_only_int8.py \
  --model model_mlatess_int8.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_027_run_mla_only_int8
./build/tutorials-standalone/tutorial_027_run_mla_only_int8 \
  --model model_mlatess_int8.tar.gz
```

З архівом YOLOv8n, скомпільованим для прямого входу та виходу MLA, обидві версії виводять контракт і нульове відхилення для кожної голови:

```text
MLA-only contract:
  input images INT8 [640, 640, 3] scale=0.00391965 zero_point=-128 range=[0, 1]
  output bbox_0 INT8 [80, 80, 64] scale=0.0828159 zero_point=-60
  ...
Dequantized MLA-only outputs vs the default route (error in scale units):
  bbox_0 [80, 80, 64] max_err=0.0000
  ...
[OK] 027_run_mla_only_int8
```

Типово використовуються карта 0 і черга 0. Передавайте `--card N` лише за використання іншої карти.

## In Practice

Умикайте `mla_only`, коли застосунок сам відповідає за квантування: він уже отримує INT8 із сенсора або попередньої моделі, йому потрібні сирі голови INT8 для власної постобробки, або він хоче прибрати етапи EV74 із затримки на боці карти. Читайте кожен scale, zero point і діапазон входу з `model.info()`; ніколи не копіюйте їх з іншої збірки моделі.

Маршрут працює за принципом «усе або нічого». Кожен вхід моделі з кількома входами має надходити як INT8, а попередню обробку зображень чи декодування рамок не можна поєднувати з `mla_only`. Залишайте типовий маршрут, коли хост має дані FP32 і не потребує керування квантуванням.

Для діагностики розгортання перейдіть до [робочого процесу моделі PCIe](/develop-apps/development-workflow/pcie-model/).

## Source Files

- `run_mla_only_int8.cpp`
- `run_mla_only_int8.py`
- `../assets/street-scene.png`
