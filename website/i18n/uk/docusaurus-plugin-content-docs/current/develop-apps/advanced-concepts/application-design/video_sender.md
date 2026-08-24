---
title: "Надіслати відео"
description: "Формати передавання відеосигналу VideoSender H.264 і H.265 через протоколи RTP/UDP."
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# Надіслати відео

Використовуйте `VideoSender` коли граф має надсилати відео на зовнішній приймач. `VideoSender` повертає об’єкт, який можна повторно використовувати `Graph` фрагмент, тому додайте його разом із `Graph::add(...)`.

`VideoSender` передає H.264 або H.265 через RTP/UDP. Необроблений вхід кодується як H.264;
закодований вхід H.264 і H.265 передається без повторного кодування. H.264 за замовчуванням використовує тип корисного навантаження RTP 96, тоді як H.265 використовує 98. Правило за замовчуванням для UDP-порту:
`video_port_base + channel`, разом із `video_port_base = 9000`Якщо приймач працює з перенаправленням портів контейнера, передайте відповідний хост і відповідну `video_port_base` з програми.

## Необроблені кадри

Використовуйте необроблений шлях, коли вхідні дані конвеєра для `VideoSender` є необробленими відеокадрами.
Neat автоматично вибирає безпечний кодек для вхідного потоку:

```text
NV12 with a proven compatible boundary:
H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput

Other or unknown raw formats:
VideoConvert -> H264EncodeSima -> H264Parse -> H264Packetize -> UdpOutput
```

Автоматичний вибір не додає опцію застосунку або не змінює
`H264RtpUdpFromRaw(...)` API. Перевірений формат NV12 у системній або SiMaAI пам’яті може безпосередньо передавати дані
до кодувальника H.264, коли встановлений кодувальник підтримує
`input-layout-aware=true`. RGB, BGR, відтінки сірого, I420, невідомі формати/розміщення пам’яті та вхідні дані без надійного контракту щодо формату потребують одного перетворення у формат NV12.

### Необроблена геометрія та розмітка кадру.

`width` та `height` – це видимі розміри зображення. Їм не обов’язково бути кратними 8, 16 або 32. Для форматів NV12 та I420 4:2:0 обидва розміри мають бути додатними та парними; активний кодек, профіль, рівень і апаратне забезпечення визначають решту мінімальних і максимальних меж. Наприклад, `680x382`, `672x384` та `642x480` є допустимими розмірами, якщо встановлений кодувальник їх підтримує.

Апаратне вирівнювання пам’яті відрізняється від видимих геометричних параметрів. Neat зберігає запрошені розміри в метаданих і виділяє або готує поверхні кодувальника з кроком і висотою, необхідними для апаратного забезпечення. Необхідно, щоб необроблений буфер із визначеним фізичним розташуванням містив `GstVideoMeta` з точними зміщеннями та кроками площин. Без цих метаданих використовується узгоджений макет GStreamer; вхідний файл, керований властивостями, повинен містити рівно один щільно запакований кадр на буфер. Недійсні, обрізані або непідтримувані макети призводять до синхронної помилки замість часткового копіювання.

```cpp
simaai::neat::Graph graph;
const int channel = 0;

auto opt = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
    width, height, fps);
opt.host = "127.0.0.1";
opt.channel = channel;
opt.video_port_base = 9000;
opt.encoder.bitrate_kbps = 2500;

graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

Python:

```python
channel = 0

opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
    width=1920,
    height=1080,
    fps=30,
)
opt.host = "127.0.0.1"
opt.channel = channel
opt.video_port_base = 9000
opt.encoder.bitrate_kbps = 2500

graph = pyneat.Graph()
graph.add(pyneat.groups.video_sender(opt))
```

## Відео, кодоване за допомогою H.264 або H.265.

Для закодованих вхідних даних передайте кодек потоку у фабрику прямого копіювання. Neat
аналізує, розбиває на пакети та передає потік без повторного кодування.

| Кодек | Фабрика C++ | Фабрика Python | Тип корисного навантаження RTP за замовчуванням. |
| --- | --- | --- | ---: |
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |

Передача MJPEG відхиляється: у відправника немає модуля формування пакетів RTP/JPEG.

Приклад H.265:

```cpp
auto opt = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(
    simaai::neat::nodes::groups::RtspCodec::H265);
opt.host = "127.0.0.1";
opt.channel = 0;
graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

```python
opt = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.H265)
opt.host = "127.0.0.1"
opt.channel = 0
graph.add(pyneat.groups.video_sender(opt))
```

### Розповсюджуйте закодований сигнал RTSP для обробки та попереднього перегляду.

Коли один кодований джерело RTSP передає дані як для декодування/аналізу, так і для `VideoSender`, під’єднайте джерело безпосередньо до відправника. Для перегляду в реальному часі, наприклад, у Insight, встановіть для кодованого відправника значення `RealtimeLatestByStream`:

```cpp
simaai::neat::GraphLinkOptions video_link;
video_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;

graph.connect(encoded_source, decoder);
graph.connect(decoder, detector, detector_link);
graph.connect(encoded_source, video_sender, video_link);
```

```python
video_link = pyneat.GraphLinkOptions()
video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

graph.connect(encoded_source, decoder)
graph.connect(decoder, detector, detector_link)
graph.connect(encoded_source, video_sender, video_link)
```

Відправна гілка залишається перед `SimaDecode`, тому вона не повторно кодує відео та не копіює декодовані кадри до ЦП. За допомогою `RealtimeLatestByStream`, об’єднана відправна гілка зберігає не більше одного незавершеного закодованого блоку доступу та замінює застарілі дані, якщо швидкість передачі UDP зменшується. За замовчуванням використовується політика без втрат, яка може здійснювати зворотний тиск на спільне джерело кодування, включно з його гілкою декодера. Використовуйте налаштування за замовчуванням лише тоді, коли збереження кожного блоку доступу є важливішим, ніж підтримка актуальності даних для обчислень у реальному часі.
