# 028 Обгортання зовнішньої пам’яті тензора

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Intermediate |
| Estimated Read Time | 15 хвилин |
| Model | yolo_v8s |
| Labels | PCIe, C++, tensor, external memory, zero-copy wrapping |

## Concept

`Tensor::from_external()` дає змогу застосунку C++ обгорнути наявну неперервну ділянку пам’яті без копіювання в новий тензор на хості. Тензор зберігає власника `std::shared_ptr<void>`, тому пам’ять залишається чинною, поки її використовують PCIe і GStreamer. Застосунок також не повинен змінювати байти, доки не буде отримано відповідний результат інференсу.

Це усуває додаткове проміжне копіювання на хості для одного неперервного входу. Це не наскрізний режим без копіювання: транспорт PCIe все одно копіює корисне навантаження в пам’ять, якою володіє карта.

## Walkthrough

Програма створює три повторно використовувані вхідні слоти й надсилає вісім синтетичних кадрів FP32. Слот повертається до черги доступних лише після отримання впорядкованого результату для нього.

### Перевірка контракту моделі {#step-inspect-contract}

Створіть модель і прочитайте `info().inputs` до виділення пам’яті. У цьому посібнику використано архів YOLOv8s з одним входом. Програма перевіряє, що оголошений dtype — FP32, а форма займає рівно `size_bytes` байтів.

Зовнішнє представлення має відповідати dtype, формі, розміру в байтах та імені відповідного `TensorInfo`. Не виводьте ці значення з іншої збірки моделі.

### Обгортання пам’яті застосунку {#step-wrap-memory}

Кожен слот кільця володіє `std::shared_ptr<std::vector<float>>`. Виклик `Tensor::from_external()` отримує базовий вказівник, повну кількість елементів базової пам’яті, спільного власника, форму моделі та ім’я маршруту. Оскільки представлення неперервне, хост PCIe може обгорнути його без створення проміжної ділянки пам’яті.

Зберігати лише сирий вказівник недостатньо. Спільний власник обов’язковий, оскільки транспорт може утримувати тензор після повернення з `push()`.

### Побудова моделі {#step-build-model}

Будуйте модель після перевірки контракту та виділення кільця. У прикладі `max_inflight` дорівнює розміру кільця, тому застосунок і транспорт мають однакову явну межу.

### Надсилання й безпечне повторне використання кільця {#step-submit-ring}

Заповніть доступний слот, викличте `push()` і перенесіть слот до черги активних. Не змінюйте й не використовуйте його пам’ять повторно лише тому, що `push()` повернув керування. Коли доступних слотів немає, приклад викликає `pull()` і повертає найстаріший слот до черги доступних тільки після надходження відповідного впорядкованого результату.

Кожному прийнятому надсиланню відповідає одне отримання, включно з фінальним спорожненням. У разі тайм-ауту модель закривається замість повторного використання пам’яті запиту, який усе ще може бути активним.

## Run

Установіть пакет хоста PCIe та завантажте набір навчальних матеріалів, як описано в розділі [Налаштування навчальних матеріалів](/tutorials/before-you-run). Виконайте наведені нижче команди в корені розпакованих додаткових матеріалів PCIe. У Windows налаштуйте навчальні матеріали та запускайте програми, як описано в розділі [Запуск застосунків у Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows).

Програмі потрібен файл `yolo_v8s_mpk.tar.gz` у цьому каталозі. Завантажте YOLOv8s з [Model Browser](/tools/model-browser/), видобудьте архів із завантаженого ZIP-файлу та збережіть його під цією назвою:

```bash
sima-cli models download --id yolov8s --variant w8a8 --output models
unzip models/yolov8s/w8a8/*.zip yolov8s_w8a8_mpk.tar.gz
mv yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

У Windows завантажте модель у PowerShell і видобудьте архів за допомогою `Expand-Archive`:

```powershell
sima-cli models download --id yolov8s --variant w8a8 --output models
Expand-Archive models\yolov8s\w8a8\*.zip -DestinationPath models\yolov8s\w8a8
Copy-Item models\yolov8s\w8a8\yolov8s_w8a8_mpk.tar.gz yolo_v8s_mpk.tar.gz
```

**C++ (prebuilt):**

```bash
./lib/sima-pcie-host/tutorials/tutorial_028_wrap_external_tensor_memory
```

**C++ (build from source):**

```bash
./build.sh --target tutorial_028_wrap_external_tensor_memory
./build/tutorials-standalone/tutorial_028_wrap_external_tensor_memory
```

У Windows зберіть програму за допомогою CMake з **x64 Native Tools Command Prompt** у своїй копії навчальних матеріалів. Замініть `C:\opencv\build` на каталог `build` пакета OpenCV, описаного в розділі [Запуск застосунків у Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows):

```bat
call "%ProgramFiles%\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_028_wrap_external_tensor_memory
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_028_wrap_external_tensor_memory.exe
```

Типово використовується карта 0 і черга 0. Передавайте `--card N` лише для іншої карти. Успішний запуск виводить:

```text
input=images
ring_slots=3
completed=8
[OK] 028_wrap_external_tensor_memory
```

## In Practice

Для прямого обгортання на хості потрібна неперервна пам’ять. Тензор із непослідовними кроками також приймається, якщо його дескриптор коректний, але хост ущільнює його в проміжну ділянку. Кілька окремо виділених входів також пакуються в проміжну пам’ять.

Для моделі з кількома входами уникнути такого виділення можна лише тоді, коли всі тензори є послідовними представленнями однієї спільної упакованої ділянки. Надсилайте їх у порядку з `info().inputs` і використовуйте ім’я та форму кожного входу. Наприклад, коли обидва входи мають FP32:

```cpp
const auto& first = info.inputs.at(0);
const auto& second = info.inputs.at(1);
const std::size_t first_count = first.size_bytes / sizeof(float);
const std::size_t second_count = second.size_bytes / sizeof(float);

auto packed =
    std::make_shared<std::vector<float>>(first_count + second_count);
pcie::Tensor input0 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, first.shape, first.name);
pcie::Tensor input1 = pcie::Tensor::from_external(
    packed->data(), packed->size(), packed, second.shape, second.name,
    static_cast<std::int64_t>(first.size_bytes));

model.push({input0, input1});
```

Ця оптимізація усуває лише пакувальне копіювання на хості. PCIe однаково копіює упаковане корисне навантаження в транспортну пам’ять карти перед інференсом.

## Файли джерельного коду

- `run_external_tensor_memory.cpp`
