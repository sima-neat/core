---
title: "Використання USB-камери"
description: "Зчитуйте кадри MJPEG або YUYV з USB-вебкамери (UVC) за допомогою UsbCameraInput, передавайте чи декодуйте JPEG або подавайте кадри YUYV до моделі й переглядача H.264, без номерів пристроїв і рядків конвеєра."
sidebar_position: 5
slug: /develop-apps/advanced-concepts/usb-camera-input
---

# Використання USB-камери

`nodes::UsbCameraInput` зчитує USB-вебкамери (UVC). Типово він захоплює MJPEG і виводить кадри JPEG камери; якщо задати `format` значення `UsbCameraFormat::YUYV`, він натомість захоплює нестиснені кадри YUYV, як описано в розділі [Захоплення YUYV](#capture-yuyv). Передайте стабільний ідентифікатор камери. `UsbCameraInputOptions` використовує назви полів і типові значення `CameraInputOptions` ([камери MIPI](./mipi-camera-input.md)), але `camera_name` обов’язкове, а `format` має тип `UsbCameraFormat` і типове значення `MJPEG`.

```text
USB camera -> UsbCameraInput (JPEG) -+-> VideoSender passthrough (MJPEG)
                                     `-> SimaDecode (NV12 or NV16) -> model

USB camera -> UsbCameraInput (YUYV) -+-> Preproc -> model
                                     `-> ColorConvert (NV12) -> VideoSender (H.264)
```

Більшість вебкамер надсилає JPEG 4:2:2, які декодуються в NV16. `usb_camera_info()` повідомляє, у яке розміщення декодуються JPEG камери: використайте його як `out_format` для `SimaDecode`, а коли воно повертає `NV16`, задайте `preprocess.color_convert.input_format` моделі значення `PreprocessColorFormat::NV16`, і кадри надходитимуть до моделі без етапу перетворення.

## Що підтримує Neat

- Камери UVC, під’єднані до USB-порту Modalix DevKit, на Platform 3.0.
- Захоплення MJPEG із субдискретизацією кольоровості 4:2:0 або 4:2:2.
- Захоплення YUYV з парною шириною, кольором BT.601 або BT.709 і повним або обмеженим діапазоном, на Platform 3.0.1.
- Точні ширина, висота й частота кадрів, які камера пропонує для вибраного формату. Neat не масштабує зображення й не підбирає найближчий режим.

Камери лише з H.264, інші нестиснені формати, як-от NV12 чи UYVY, потоки глибини чи ІЧ і керування камерою, як-от експозиція, до цього шляху не входять.

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

Щоб побачити, які режими MJPEG і YUYV пропонує камера, перегляньте її формати на DevKit. Номер пристрою потрібен лише для цієї перевірки; Neat знаходить пристрій за ідентифікатором.

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

Якщо вам також потрібен `decoded_format`, викличте `usb_camera_info` один раз і передайте його результат у `UsbCameraInput(camera, info)`, як показано в наступному розділі. Тоді вузол повторно використовує цю перевірку замість того, щоб знову відкривати камеру, і лише перевіряє через sysfs, що пристрій досі належить цій камері. Передавайте однакові параметри в обидва виклики.

Після від’єднання камери або перенесення на інший порт знову викличте `UsbCameraInput`. Збирання раніше створеного графа завершиться помилкою, тож він ніколи не відкриє іншу камеру, що отримала старий номер пристрою.

## Передавання JPEG камери

Щоб надсилати відео камери до переглядача й водночас запускати на ньому модель, під’єднайте вузол до наскрізного `VideoSender`, який надсилає JPEG як RTP/JPEG без повторного кодування, і до `SimaDecode`, у якого `out_format` дорівнює `decoded_format` камери:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
const neat::UsbCameraInfo info = neat::usb_camera_info(camera);  // opens the camera once

neat::SimaDecodeOptions decode;
decode.type = neat::SimaDecodeType::MJPEG;
decode.out_format = info.decoded_format;  // NV12 or NV16
decode.dec_width = 1920;
decode.dec_height = 1080;
decode.dec_fps = 30;

auto sender = neat::nodes::groups::VideoSenderOptions::Passthrough(
    neat::nodes::groups::RtspCodec::MJPEG);
sender.host = "127.0.0.1";

neat::GraphLinkOptions video_link;
video_link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

neat::Graph source;
source.add(neat::nodes::UsbCameraInput(camera, info));  // reuses info, no second open
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
info = pyneat.usb_camera_info(camera)  # opens the camera once

decode = pyneat.SimaDecodeOptions()
decode.type = pyneat.SimaDecodeType.MJPEG
decode.out_format = info.decoded_format  # NV12 or NV16
decode.dec_width = 1920
decode.dec_height = 1080
decode.dec_fps = 30

sender = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
sender.host = "127.0.0.1"

video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

source = pyneat.Graph()
source.add(pyneat.nodes.usb_camera_input(camera, info))  # reuses info, no second open
decoder = pyneat.Graph()
decoder.add(pyneat.nodes.sima_decode(decode))

graph = pyneat.Graph("usb_camera_forward")
graph.connect(source, decoder)
graph.connect(decoder, detector)  # your model Graph
graph.connect(source, pyneat.groups.video_sender(sender), video_link)
```

</CodeTab>
</CodeTabs>

Камера захоплює кадри в буфери CMA, які апаратний декодер читає на місці, тож між захопленням і декодуванням жоден кадр не копіюється. Драйвер UVC ядра, як і раніше, копіює USB-пакети в кожен буфер кадру. До відправника застосовуються обмеження RTP/JPEG з розділу [Передавання відео](./video_sender.md).

## Пам’ять і відкинуті кадри

| Параметр | Дія |
| --- | --- |
| `queue_depth`, `leaky_queue` | Невелика черга з відкиданням після камери відкидає застарілі кадри, коли граф не встигає. Пул захоплення камери містить `queue_depth` + 6 буферів, не більше 64, тож `queue_depth` може бути не більше 58. |

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
| `is not at '/dev/videoN' as its UsbCameraInfo says` | Після виклику `usb_camera_info` камеру від’єднали, перенесли або її номер змінився, або ця інформація стосується іншої камери. Знову викличте `usb_camera_info` і передайте нову інформацію. |
| `has no YUYV mode 640x480 at 30/1 fps; YUYV modes: ...` | Виберіть один із наведених режимів YUYV. |
| `has no YUYV capture node` | Камера не надсилає YUYV. Захоплюйте MJPEG. |
| `YUYV requires an even width` | Виберіть режим YUYV з парною шириною. |
| `adjusted the requested progressive YUYV mode` або `did not accept 30/1 fps` | Камера навела цей режим, але не прийняла його. Виберіть інший наведений режим. |
| `has unsupported YUYV matrix/range/transfer metadata` або `has unsupported YUYV color primaries` | Камера повідомляє колірний простір, відмінний від BT.601 чи BT.709. Захоплюйте MJPEG. |
| `capture info format does not match the selected mode` | Передайте повну `UsbCameraInfo`, яку повернула `usb_camera_info`. |
| `capture info format does not match opt.format` | Інформацію перевірено для іншого `format`. Викличте `usb_camera_info` з параметрами, які ви передаєте в `UsbCameraInput`. |
| `capture info describes another mode` | Параметри відрізняються від переданих у `usb_camera_info`. Передайте однакові параметри в обидва виклики. |
| Run для YUYV завершується помилкою, коли камера починає передавати потік | Виконайте `dmesg` на DevKit. `Failed to submit URB 0 (-28)` (`ENOSPC`) означає, що USB-контролер не може зарезервувати пропускну здатність, потрібну нестисненому режиму. Виберіть нижчу роздільність чи частоту кадрів, зупиніть інші USB-камери або аудіопристрої, що одночасно передають потік, або захоплюйте MJPEG. |
| `YUYV requires explicit input color` | Задайте `PreprocOptions::input_color` значення `info.color`. |
| `YUYV input does not support antialiased sampling`, `FP32 transforms require supported RGB/Gray tensor output`, `crop currently requires RGB, BGR or Gray input` або `ROI lists require RGB/BGR/Gray inputs/outputs` | Передобробка моделі потребує параметра, який вхід YUYV не підтримує. Див. [Запуск моделі на кадрах YUYV](#run-a-model-on-yuyv-frames). |
| `Strict encoder ingress requires EncoderNv12 output from ColorConvert` | Розмістіть `ColorConvert` з `ImageSurfaceLayout::EncoderNv12` і виходом NV12 перед строгим `VideoSender`. |
| `Strict encoder ingress requires an exact integer frame rate` | Захоплюйте режим із цілою частотою кадрів і задайте її як `fps` кодувальника. |
| `Graph JSON does not store PreprocOptions::input_color or ColorConvertOptions::input_colorimetry` | Створюйте граф, що захоплює YUYV, у коді; `Graph::save` не може зберегти колір камери. |

## Захоплення YUYV {#capture-yuyv}

Багато камер UVC також надсилають нестиснений YUYV: упаковані кадри 4:2:2, які GStreamer називає `YUY2`. Кадрам YUYV не потрібен декодер; передобробка моделі та `ColorConvert` читають їх на місці з буферів захоплення. Нестиснені кадри потребують значно більшої пропускної здатності USB, ніж MJPEG, тому камери зазвичай пропонують YUYV з нижчою роздільністю або частотою кадрів.

Задайте `format` значення `UsbCameraFormat::YUYV`, викличте `usb_camera_info` і передайте її результат у `UsbCameraInput`. Виберіть режим YUYV, який пропонує камера, з парною шириною. Повернена `UsbCameraInfo` також описує кадри: `row_stride` — крок рядка в байтах, `storage_bytes` — розмір кожного буфера захоплення, `color` — матриця камери (BT.601 або BT.709) і діапазон, а `colorimetry` — те саме тлумачення у вигляді рядка colorimetry GStreamer.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
camera.format = neat::UsbCameraFormat::YUYV;
camera.width = 640;
camera.height = 480;
const neat::UsbCameraInfo info = neat::usb_camera_info(camera);  // opens the camera once
auto source = neat::nodes::UsbCameraInput(camera, info);         // reuses info
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
camera.format = pyneat.UsbCameraFormat.YUYV
camera.width = 640
camera.height = 480
info = pyneat.usb_camera_info(camera)  # opens the camera once
source = pyneat.nodes.usb_camera_input(camera, info)  # reuses info
```

</CodeTab>
</CodeTabs>

Вузол виводить кадри `UInt8` у розміщенні HWC з двома байтами на піксель. Кадри зберігають частоту кадрів камери, зокрема дробові частоти, як-от 30000/1001, а непарна висота допустима. Із типовим `UsbCameraFormat::MJPEG` вузол захоплює MJPEG.

### Запуск моделі на кадрах YUYV {#run-a-model-on-yuyv-frames}

Задайте `preprocess.color_convert.input_format` моделі значення `PreprocessColorFormat::YUYV`, а її максимальний вхідний розмір — відповідно до режиму камери. Кадри YUYV не вказують, яку матрицю й діапазон вони використовують, тому побудуйте маршрут моделі з `Preproc`, у якого `input_color` дорівнює `info.color`, після якого йдуть висновування й постобробка моделі:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::Model::Options options;
options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::YUYV;
options.preprocess.input_max_width = 640;
options.preprocess.input_max_height = 480;
neat::Model model("model.tar.gz", options);

neat::PreprocOptions preproc(model);
preproc.input_color = info.color;

neat::Graph graph("usb_camera_yuyv");
graph.add(neat::nodes::UsbCameraInput(camera, info));
graph.add(neat::nodes::Preproc(preproc));
graph.add(model.inference());
graph.add(model.postprocess());
graph.add(neat::nodes::Output());
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
options = pyneat.ModelOptions()
options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.YUYV
options.preprocess.input_max_width = 640
options.preprocess.input_max_height = 480
model = pyneat.Model("model.tar.gz", options)

preproc = pyneat.PreprocOptions(model)
preproc.input_color = info.color

graph = pyneat.Graph("usb_camera_yuyv")
graph.add(pyneat.nodes.usb_camera_input(camera, info))
graph.add(pyneat.nodes.preproc(preproc))
graph.add(model.inference())
graph.add(model.postprocess())
graph.add(pyneat.nodes.output())
```

</CodeTab>
</CodeTabs>

Передобробка моделі для кадрів YUYV не підтримує:

- згладжену дискретизацію під час зміни розміру (`ResizeInterpolation::BilinearAntialiased`);
- фіксовану арифметику джерела: `rescale_factor` і фіксовану нормалізацію, яку він вибирає, `quantization_step`, `PreprocFloatingConversion::NearestEven` і `PreprocColorConversionPlacement::BeforeResize`;
- `ResizeMode::Crop`;
- списки ROI під час виконання.

Модель, передобробка якої потребує одного з цих параметрів, завершується помилкою під час збирання графа. Натомість захоплюйте MJPEG або спершу перетворіть кадри за допомогою `ColorConvert` на `ImageFormat::RGB` (задайте його `color` значення `info.color`, а `input_colorimetry` — значення `info.colorimetry`) і задайте вхідний формат моделі `PreprocessColorFormat::RGB`.

### Перегляд кадрів YUYV як H.264 {#view-yuyv-frames-as-h264}

Щоб також надсилати відео камери до переглядача, розгалузьте граф перед `Preproc` і під’єднайте гілку переглядача через `RealtimeLatestByStream`, щоб вона ніколи не затримувала модель. Перетворіть кадри на NV12 у розміщенні кодувальника за допомогою `ColorConvert`, у якого `input_colorimetry` дорівнює `info.colorimetry`, а потім закодуйте їх строгим H.264 `VideoSender`, у якого `SimaEncodeOptions::input_policy` дорівнює `EncoderInputPolicy::RequireDirect`:

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
neat::ColorConvertOptions convert;
convert.output_format = neat::ImageFormat::NV12;
convert.surface = neat::ImageSurfaceLayout::EncoderNv12;
convert.color = info.color;
convert.color.output_siting = {neat::ImageChromaSiting::Cosited,
                               neat::ImageChromaSiting::Centered};
convert.input_colorimetry = info.colorimetry;

neat::SimaEncodeOptions encode;
encode.fps = 30;
encode.input_policy = neat::EncoderInputPolicy::RequireDirect;
auto sender = neat::nodes::groups::VideoSenderOptions::FromRaw(encode);
sender.host = "127.0.0.1";

neat::Graph viewer("usb_camera_yuyv_viewer");
viewer.add(neat::nodes::ColorConvert(convert));
viewer.add(neat::nodes::groups::VideoSender(sender));
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
convert = pyneat.ColorConvertOptions()
convert.output_format = pyneat.ImageFormat.NV12
convert.surface = pyneat.ImageSurfaceLayout.EncoderNv12
convert.color = info.color
convert.color.output_siting = [pyneat.ImageChromaSiting.Cosited, pyneat.ImageChromaSiting.Centered]
convert.input_colorimetry = info.colorimetry

encode = pyneat.SimaEncodeOptions()
encode.fps = 30
encode.input_policy = pyneat.EncoderInputPolicy.RequireDirect
sender = pyneat.VideoSenderOptions.from_raw(encode)
sender.host = "127.0.0.1"

viewer = pyneat.Graph("usb_camera_yuyv_viewer")
viewer.add(pyneat.nodes.color_convert(convert))
viewer.add(pyneat.groups.video_sender(sender))
```

</CodeTab>
</CodeTabs>

Строгий відправник ніколи не перетворює й не копіює кадри на CPU і передає в потоці H.264 первинні кольори, передавальну характеристику, матрицю й діапазон камери. Він потребує парних ширини й висоти та цілої частоти кадрів, що дорівнює `fps` кодувальника; інший вхід відхиляється без округлення. Див. [Передавання відео](./video_sender.md#strict-encoder-input).

Між захопленням і моделлю чи кодувальником кадри не копіюються; драйвер UVC ядра все ще копіює пакети USB у кожен буфер кадру. `UsbCameraInput` відкидає кадри, які драйвер позначає як пошкоджені. Кількість байтів кожного кадру від драйвера не доходить до графа, тому неповний кадр, який драйвер не позначив, проходить далі з рештою буфера, що залишилася від попереднього кадру.
