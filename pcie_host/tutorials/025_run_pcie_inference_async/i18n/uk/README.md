# 025 Асинхронний запуск PCIe-інференсу

## Metadata

| Field | Value |
| --- | --- |
| Category | PCIe Co-Processing |
| Difficulty | Beginner |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | PCIe, asynchronous, throughput, detection |

## Concept

Синхронний `run()` ідеально підходить для першого інференсу, але він чекає на кожен результат, перш ніж надсилати наступне зображення. Якщо виробник викликає `push()`, а споживач обробляє `pull()`, одна PCIe-модель залишається в роботі. Пропускну здатність необхідно розраховувати на основі завершених результатів після періоду «розігріву», а не лише на основі зображень, наданих моделі.

## Walkthrough

У цьому навчальному посібнику повторно використовується конфігурація YOLOv8s (зображення плюс декодування обмежувальних рамок) і зображення міської сцени розміром 640x480 з навчального посібника 024. Він надсилає одне й те саме зображення кілька разів, щоб зберігання та декодування зображень не спотворювали вимірювання PCIe.

### Налаштування однієї моделі виявлення {#step-configure-model}

Завантажте зображення один раз, налаштуйте попередню обробку COCO на стороні відеокарти та декодування обмежувальних рамок YOLOv8, а потім створіть одну `Model` у черзі 0. Відсутні файли та помилки запуску відеокарти зупиняють програму до початку вимірювання.

### Підготовка конвеєра {#step-warm-up}

Запустіть кілька повних виявлень, не вимірюючи час. Підготовка усуває ефекти запуску моделі та першого буфера з отриманих даних про навантаження.

### Одночасна відправка та отримання {#step-push-pull}

Один потік надсилає зображення за допомогою `push()`, а інший отримує вихідні дані BBOX за допомогою `pull()` з обмеженим часом очікування. Невеликий FIFO, що належить застосунку, зберігає час початку кожної відправки. Будь-яке відхилення, перевищення часу очікування або неправильний результат закриває модель і активує інший потік.

Приклад покладається лише на звичайну поведінку керування потоком `Model`; у застосунку немає налаштування глибини черги.

### Звіт про виконану роботу {#step-report-results}

Зупиніть вимірювання часу лише після того, як обидва потоки завершать роботу, і всі прийняті зображення будуть отримані. Кількість кадрів в секунду використовує кількість завершених вихідних даних. Середня затримка вимірюється від кожної спроби відправки до надходження відповідного впорядкованого результату.

## Run

Установіть пакет хоста PCIe та завантажте набір навчальних матеріалів, як описано в розділі [Налаштування навчальних матеріалів](/tutorials/before-you-run). Виконайте наведені нижче команди в корені розпакованих додаткових матеріалів PCIe. У Windows налаштуйте навчальні матеріали та запускайте програми, як описано в розділі [Запуск застосунків у Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows).

Програмі потрібен точний шлях `yolo_v8s_mpk.tar.gz` у цьому каталозі. Якщо файлу ще немає, завантажте YOLOv8s з [Model Browser](/tools/model-browser/), видобудьте архів із завантаженого ZIP-файлу та збережіть його під цією назвою:

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

Якщо завантажено файл `<id>_<variant>_mpk.tar.gz`, а не ZIP-файл (так буває з моделлю, востаннє зібраною до того, як Model Browser почав пакувати моделі у ZIP-файли), скопіюйте цей файл із `models/<id>/<variant>` замість розпакування.

Запустіть Python:

```bash
source ~/pyneatpcie/bin/activate
python3 share/sima-pcie-host/tutorials/025_run_pcie_inference_async/run_pcie_inference_async.py
```

Запустіть попередньо зібраний навчальний посібник C++:

```bash
./lib/sima-pcie-host/tutorials/tutorial_025_run_pcie_inference_async
```

Або перезберіть його:

```bash
./build.sh --target tutorial_025_run_pcie_inference_async
./build/tutorials-standalone/tutorial_025_run_pcie_inference_async
```

У Windows перезберіть її за допомогою CMake з **x64 Native Tools Command Prompt** у своїй копії навчальних матеріалів. Замініть `C:\opencv\build` на каталог `build` пакета OpenCV, описаного в розділі [Запуск застосунків у Windows](/getting-started/neat-library/pcie-host/#run-applications-on-windows):

```bat
call "%ProgramFiles%\SimaPCIeHost\bin\sima-pcie-host-env.cmd"
set "OpenCV_DIR=C:\opencv\build"
cmake -S share\sima-pcie-host\tutorials -B build
cmake --build build --config Release --target tutorial_025_run_pcie_inference_async
set "PATH=%OpenCV_DIR%\x64\vc16\bin;%PATH%"
build\Release\tutorial_025_run_pcie_inference_async.exe
```

Точний час залежить від хоста та відеокарти, але обидві програми використовують однакові межі вимірювання та виводять:

```text
completed=1000
elapsed_seconds=...
throughput_fps=...
average_latency_ms=...
total_detections=...
[OK] 025_run_pcie_inference_async
```

Навчальний посібник завжди починає з п’яти кадрів, а потім вимірює 1000 завершених кадрів. Передавайте `--card N` лише під час використання іншої відеокарти.

## In Practice

Забезпечте збалансованість відправки та отримання. Якщо застосунок нескінченно надсилає дані, не отримуючи їх, нормальний механізм зворотного зв’язку зрештою сповільнить відправку. Виділений споживач також спрощує виявлення збоїв: обмежений час очікування ідентифікує результат, який не надходить, і закриття моделі звільняє чергу 0, навіть якщо виробник чекає.

Для репрезентативного тестування замініть повторюваний кадр фіксованим набором зображень і виключіть читання з диска з вимірюваного часового проміжку. Продовжуйте з [Запустіть кілька моделей.](/tutorials/run-multiple-models), щоб одночасно запустити дві різні моделі.

## Файли коду

- `run_pcie_inference_async.cpp`
- `run_pcie_inference_async.py`
- `../assets/street-scene.png`
