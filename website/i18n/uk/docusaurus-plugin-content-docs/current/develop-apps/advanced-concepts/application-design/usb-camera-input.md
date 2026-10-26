---
title: "Використання USB-камери"
description: "Зчитуйте кадри MJPEG з USB-вебкамери (UVC) за допомогою UsbCameraInput: апаратний декодер декодує їх на місці в NV12, без номерів пристроїв і рядків конвеєра."
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# Використання USB-камери

`groups::UsbCameraInput` зчитує USB-вебкамери (UVC), що надсилають MJPEG. Передайте стабільний ідентифікатор камери, і граф отримуватиме кадри NV12 у пам’яті SiMaAI, як і з [камери MIPI](./mipi-camera-input.md). `UsbCameraInputOptions` використовує назви полів і типові значення `CameraInputOptions`, але `camera_name` обов’язкове, а `format` немає: вихід завжди NV12.

```text
USB camera -> UsbCameraInput -> model-managed CVU preproc -> MLA -> Output
```

Усередині камера записує кожен стиснений кадр безпосередньо в буфери CMA, які апаратний декодер MJPEG читає на місці, тож між захопленням і декодуванням процесор нічого не копіює. Для камери, що надсилає JPEG 4:2:2, як більшість вебкамер, після декодера додається одне перетворення EV74 з NV16 у NV12.

## Що підтримує Neat

- Камери UVC, під’єднані до USB-порту Modalix DevKit, на Platform 3.0.
- Захоплення MJPEG із субдискретизацією кольоровості 4:2:0 або 4:2:2.
- Точні ширина, висота й частота кадрів, які камера пропонує для MJPEG. Neat не масштабує зображення й не підбирає найближчий режим.

Камери лише з H.264, захоплення нестисненого YUYV/NV12, потоки глибини чи ІЧ (наприклад, глибина RealSense) і керування камерою, як-от експозиція, до цього шляху не входять.

## Пошук ідентифікатора камери

Кожна USB-камера має стабільний ідентифікатор, складений з її USB-порту та ідентифікаторів виробника й продукту, у тому самому форматі, що й у libcamera:

```text
/base/pci@7000000000-3.1:1.0-046d:08e5
```

Ідентифікатор не змінюється після перезавантаження чи зміни порядку під’єднання. Він змінюється, коли камеру переносять на інший USB-порт, тому дві однакові камери на різних портах мають різні ідентифікатори. Перелічіть під’єднані камери:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
for (const std::string& id : simaai::neat::usb_camera_ids()) {
  std::cout << id << "\n";
}
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

print(pyneat.usb_camera_ids())
```

</CodeTab>
</CodeTabs>

Щоб побачити, які режими MJPEG пропонує камера, перегляньте її формати на DevKit. Номер пристрою потрібен лише для цієї перевірки; Neat знаходить пристрій за ідентифікатором.

<ShellCommand prompt="devkit">
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video1 --list-formats-ext
</ShellCommand>

## Створення графа

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::UsbCameraInputOptions camera;
camera.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5";
camera.width = 1920;
camera.height = 1080;
camera.framerate_num = 30;
camera.framerate_den = 1;
camera.buffer_name = "camera0";

neat::Graph graph("usb_camera");
graph.add(neat::nodes::groups::UsbCameraInput(camera));
graph.add(neat::nodes::Output());

neat::Run run = graph.build();
std::optional<neat::Sample> frame = run.pull(/*timeout_ms=*/5000);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
import pyneat

camera = pyneat.UsbCameraInputOptions()
camera.camera_name = "/base/pci@7000000000-3.1:1.0-046d:08e5"
camera.width = 1920
camera.height = 1080
camera.framerate_num = 30
camera.framerate_den = 1
camera.buffer_name = "camera0"

graph = pyneat.Graph("usb_camera")
graph.add(pyneat.groups.usb_camera_input(camera))
graph.add(pyneat.nodes.output())

run = graph.build()
frame = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

Модель додають так само, як для камери MIPI; див. розділ про базовий граф MLA у статті [Використання камери MIPI](./mipi-camera-input.md).

`UsbCameraInput` один раз відкриває камеру під час створення графа, щоб перевірити режим і прочитати субдискретизацію кольоровості першого кадру. Спершу закрийте інші програми, що використовують камеру.

Після від’єднання камери або перенесення на інший порт знову викличте `UsbCameraInput`. Збирання раніше створеного графа завершиться помилкою, тож він ніколи не відкриє іншу камеру, що отримала старий номер пристрою.

## Пам’ять і відкинуті кадри

| Параметр | Дія |
| --- | --- |
| `allow_cpu_fallback` | Для USB-камер не діє: захоплення завжди потрапляє у власні буфери декодера, тому жоден кадр не копіюється. Параметр залишено, щоб параметри збігалися з `CameraInputOptions`. |
| `queue_depth`, `leaky_queue` | Невелика черга з відкиданням перед декодером відкидає застарілі кадри, коли граф не встигає. |
| `capture_buffer_count` (другий аргумент) | Буфери CMA, спільні для камери й декодера. Значення 0 означає глибину черги плюс два вхідні слоти декодера. |

Нульове копіювання тут означає, що стиснений кадр не копіюється між захопленням і декодуванням. Декодований кадр NV12 є окремим буфером, а перетворення EV74 для 4:2:2 записує ще один.

Вебкамери часто додають байти після маркера кінця JPEG, а деякі кадри надходять неповними або позначеними драйвером USB як пошкоджені. Декодер декодує кожен кадр до маркера кінця й відкидає неповні чи пошкоджені кадри, не зупиняючись. Елемент декодера рахує їх у властивості `dropped-count`.

## Усунення несправностей

| Повідомлення | Що робити |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | Використайте один із наведених ідентифікаторів. Ідентифікатор змінюється, коли камеру переносять на інший порт. |
| `no USB cameras are connected` | Перевірте кабель і `lsusb`. |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | Виберіть один із наведених режимів. Деякі камери пропонують високу роздільність лише з нижчою частотою кадрів. |
| `is in use by another process` | Закрийте іншу програму, яка відкрила камеру. |
| `sent no usable MJPEG frame` | Камера надсилає JPEG, які Neat не може декодувати, наприклад із субдискретизацією 4:4:4, або не надсилає кадрів зовсім. |
| `is no longer /dev/videoN` | Після створення графа камеру від’єднали, перенесли або її номер змінився. Знову викличте `UsbCameraInput`. |
| `requires a neatdecoder with the input-pool-buffers property` | Встановлений декодер застарий. Оновіть пакет `simaai-scheduler`. |
