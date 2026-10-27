---
title: "Використання USB-камери"
description: "Зчитуйте кадри MJPEG з USB-вебкамери (UVC) за допомогою UsbCameraInput і передавайте їх як JPEG або декодуйте для моделі, без номерів пристроїв і рядків конвеєра."
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# Використання USB-камери

`nodes::UsbCameraInput` зчитує USB-вебкамери (UVC), що надсилають MJPEG, і виводить кадри JPEG камери. Передайте стабільний ідентифікатор камери. `UsbCameraInputOptions` використовує назви полів і типові значення `CameraInputOptions` ([камери MIPI](./mipi-camera-input.md)), але `camera_name` обов’язкове, а `format` немає.

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model
```

Більшість вебкамер надсилає JPEG 4:2:2, які декодуються в NV16. `usb_camera_info()` повідомляє, у яке розміщення декодуються JPEG камери: використайте його як `out_format` для `SimaDecode`, а коли воно повертає `NV16`, задайте `preprocess.color_convert.input_format` моделі значення `PreprocessColorFormat::NV16`, і кадри надходитимуть до моделі без етапу перетворення.

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
graph.add(neat::nodes::UsbCameraInput(camera));
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
graph.add(pyneat.nodes.usb_camera_input(camera))
graph.add(pyneat.nodes.output())

run = graph.build()
frame = run.pull(timeout_ms=5000)
```

</CodeTab>
</CodeTabs>

Вузол виводить кадри JPEG камери. Щоб запустити на них модель, декодуйте їх за допомогою `SimaDecode`, як показано в наступному розділі.

`UsbCameraInput` один раз відкриває камеру під час створення вузла, щоб перевірити режим і прочитати субдискретизацію кольоровості першого кадру. Спершу закрийте інші програми, що використовують камеру. `usb_camera_info(camera)` виконує ту саму перевірку без створення вузла й повертає пристрій захоплення та `decoded_format`: `NV12` або `NV16`.

Після від’єднання камери або перенесення на інший порт знову викличте `UsbCameraInput`. Збирання раніше створеного графа завершиться помилкою, тож він ніколи не відкриє іншу камеру, що отримала старий номер пристрою.

## Передавання JPEG камери

Щоб надсилати відео камери до переглядача й водночас запускати на ньому модель, під’єднайте вузол до наскрізного `VideoSender`, який надсилає JPEG як RTP/JPEG без повторного кодування, і до `SimaDecode`, у якого `out_format` дорівнює `decoded_format` камери:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::SimaDecodeOptions decode;
decode.type = neat::SimaDecodeType::MJPEG;
decode.out_format = neat::usb_camera_info(camera).decoded_format;  // NV12 or NV16
decode.dec_width = 1920;
decode.dec_height = 1080;
decode.dec_fps = 30;

auto sender = neat::nodes::groups::VideoSenderOptions::Passthrough(
    neat::nodes::groups::RtspCodec::MJPEG);
sender.host = "127.0.0.1";

neat::GraphLinkOptions video_link;
video_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

neat::Graph source;
source.add(neat::nodes::UsbCameraInput(camera));
neat::Graph decoder;
decoder.add(neat::nodes::SimaDecode(decode));

neat::Graph graph("usb_camera_forward");
graph.connect(source, decoder);
graph.connect(decoder, detector);  // your model Graph
graph.connect(source, neat::nodes::groups::VideoSender(sender), video_link);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
decode = pyneat.SimaDecodeOptions()
decode.type = pyneat.SimaDecodeType.MJPEG
decode.out_format = pyneat.usb_camera_info(camera).decoded_format  # NV12 or NV16
decode.dec_width = 1920
decode.dec_height = 1080
decode.dec_fps = 30

sender = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
sender.host = "127.0.0.1"

video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

source = pyneat.Graph()
source.add(pyneat.nodes.usb_camera_input(camera))
decoder = pyneat.Graph()
decoder.add(pyneat.nodes.sima_decode(decode))

graph = pyneat.Graph("usb_camera_forward")
graph.connect(source, decoder)
graph.connect(decoder, detector)  # your model Graph
graph.connect(source, pyneat.groups.video_sender(sender), video_link)
```

</CodeTab>
</CodeTabs>

Декодер копіює кожен JPEG у власний вхідний буфер перед декодуванням. До відправника застосовуються обмеження RTP/JPEG з розділу [Передавання відео](./video_sender.md).

## Пам’ять і відкинуті кадри

| Параметр | Дія |
| --- | --- |
| `queue_depth`, `leaky_queue` | Невелика черга з відкиданням після камери відкидає застарілі кадри, коли граф не встигає. |

Вебкамери часто додають байти після маркера кінця JPEG, а деякі кадри надходять неповними або позначеними драйвером USB як пошкоджені. `UsbCameraInput` відкидає кадри, які драйвер позначив як пошкоджені, ще до свого виходу, тож їх не отримують ні `VideoSender` passthrough, ні декодер. Декодер декодує кожен кадр до маркера кінця й відкидає неповні чи пошкоджені кадри, не зупиняючись. Елемент декодера рахує їх у властивості `dropped-count`.

## Усунення несправностей

| Повідомлення | Що робити |
| --- | --- |
| `no USB camera with ID ...; connected USB camera IDs: ...` | Використайте один із наведених ідентифікаторів. Ідентифікатор змінюється, коли камеру переносять на інший порт. |
| `no USB cameras are connected` | Перевірте кабель і `lsusb`. |
| `has no MJPEG mode 1920x1080 at 30/1 fps; its MJPEG modes are: ...` | Виберіть один із наведених режимів. Деякі камери пропонують високу роздільність лише з нижчою частотою кадрів. |
| `is in use by another process` | Закрийте іншу програму, яка відкрила камеру. |
| `sent no usable MJPEG frame` | Камера надсилає JPEG, які Neat не може декодувати, наприклад із субдискретизацією 4:4:4, або не надсилає кадрів зовсім. |
| `is no longer /dev/videoN` | Після створення графа камеру від’єднали, перенесли або її номер змінився. Знову викличте `UsbCameraInput`. |
