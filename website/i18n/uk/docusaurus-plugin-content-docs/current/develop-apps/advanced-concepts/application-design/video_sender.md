---
title: "Передавання відео"
description: "Кодування сирих кадрів VideoSender та вихід H.264, H.265 і MJPEG через RTP/UDP"
sidebar_position: 2
slug: /develop-apps/advanced-concepts/video_sender
---

# Передавання відео

Використовуйте `VideoSender`, коли Graph має передавати відео зовнішньому приймачу. `VideoSender` повертає повторно використовуваний фрагмент `Graph`, тож додавайте його через `Graph::add(...)`.

`VideoSender` передає H.264, H.265 або MJPEG через RTP/UDP. `FromRaw` кодує сирі кадри відповідно до `SimaEncodeOptions.type`, типовим значенням якого є H.264. `Passthrough` передає вже кодовані кадри без повторного кодування. Типовий UDP-порт — `video_port_base + channel`, де `video_port_base = 9000`.

| Кодек | Тип енкодера сирих кадрів | Тип прямого передавання кодованих кадрів | Типовий тип корисного навантаження RTP |
| --- | --- | --- | ---: |
| H.264 | `SimaEncodeType::H264` | `RtspCodec::H264` | 96 |
| H.265 | `SimaEncodeType::H265` | `RtspCodec::H265` | 98 |
| MJPEG | `SimaEncodeType::MJPEG` | `RtspCodec::MJPEG` | 26 |

Переглядач RTP/WebRTC в Insight підтримує H.264 і H.265. Вихід MJPEG потребує сумісного приймача RTP/JPEG; цей переглядач його не підтримує.

Якщо приймач працює за перенаправленням портів контейнера, передайте із застосунку відповідну адресу хоста та узгоджений `video_port_base`.

## Сирі кадри

Використовуйте шлях сирих кадрів, коли вхід конвеєра до `VideoSender` — сирі відеокадри.
Neat автоматично вибирає безпечний вхід енкодера:

```text
NV12 in a compatible DMA-BUF:
SimaEncode -> codec parser -> RTP payloader -> UdpOutput

CPU input or raw frames requiring conversion:
Convert/upload into encoder DMA-BUF -> SimaEncode -> codec parser -> RTP payloader -> UdpOutput
```

Кодек разом визначає енкодер, парсер і пакувальник RTP. Застаріла фабрика
`H264RtpUdpFromRaw(...)` залишається доступною та зберігає типові параметри H.264.
Сумісний NV12 DMA-BUF вхід утримує базову виділену пам’ять. NV12 у CPU-пам’яті потребує завантаження; RGB, BGR, відтінки сірого та I420 потребують перетворення.
Neat виконує цю роботу на межі входу енкодера, записуючи в кінцеву DMA-поверхню, а не створюючи другу проміжну копію всередині енкодера. Застосункам не потрібно вибирати бекенд пам’яті.

### Геометрія та розміщення сирих кадрів

`width` і `height` задають видимі розміри зображення. Вони не мають бути кратними 8, 16 або 32. Для форматів NV12 та I420 4:2:0 обидва розміри мають бути додатними й парними; активний кодек, профіль, рівень і апаратне забезпечення визначають решту мінімальних та максимальних обмежень.
Наприклад, `680x382`, `672x384` та `642x480` є допустимими формами, якщо встановлений енкодер їх приймає.

RTP/JPEG має суворіші обмеження: передавання сирих кадрів MJPEG вимагає розмірів, кратних восьми й у межах 8..2040 за кожною віссю. Пряме передавання кодованого JPEG потребує базового JPEG зі стандартними таблицями Гаффмана й розмірами, які можна подати в RTP/JPEG. Окреме кодування та транспорт RTP мають різні обмеження розмірів.

Вирівнювання апаратного сховища відокремлене від видимої геометрії. Neat зберігає запитані розміри в caps і створює поверхні енкодера з кроком рядка та висотою сховища, потрібними апаратному забезпеченню.
Сирий буфер із власним фізичним розміщенням має містити `GstVideoMeta` з достовірними зсувами площин і кроками. Без цих метаданих використовується узгоджене розміщення GStreamer; файловий вхід, заданий властивостями, має містити рівно один щільно упакований кадр у буфері. Недопустимі, обрізані або непідтримувані розміщення завершуються синхронною помилкою замість часткового копіювання.

```cpp
simaai::neat::Graph graph;
const int channel = 0;

simaai::neat::SimaEncodeOptions encode;
encode.type = simaai::neat::SimaEncodeType::H265;
encode.fps = 30;
encode.bitrate_kbps = 2500;
encode.gop_length = 30;
auto opt = simaai::neat::nodes::groups::VideoSenderOptions::FromRaw(encode);
opt.host = "127.0.0.1";
opt.channel = channel;
opt.video_port_base = 9000;

graph.add(simaai::neat::nodes::groups::VideoSender(opt));
```

Python:

```python
channel = 0

encode = pyneat.SimaEncodeOptions()
encode.type = pyneat.SimaEncodeType.H265
encode.fps = 30
encode.bitrate_kbps = 2500
encode.gop_length = 30
opt = pyneat.VideoSenderOptions.from_raw(encode)
opt.host = "127.0.0.1"
opt.channel = channel
opt.video_port_base = 9000

graph = pyneat.Graph()
graph.add(pyneat.groups.video_sender(opt))
```

Передавання сирих кадрів визначає роздільність із вхідних кадрів і зберігає її без масштабування.
Щоб передавати сирі кадри з іншою роздільністю, почніть новий запуск.
Застаріла фабрика `H264RtpUdpFromRaw(width, height, fps)` зберігає фіксовані розміри входу.

Для MJPEG виберіть `SimaEncodeType::MJPEG` у C++ або `pyneat.SimaEncodeType.MJPEG` у Python і задайте `quality` від 1 до 100. Не задавайте бітрейт, керування швидкістю, профіль, рівень, GOP та IDR. `FromRaw` копіює передані параметри, тому налаштуйте їх перед викликом фабрики.

Наявні перевизначення бітрейту/профілю/рівня через `opt.encoder` діють для сирих передавачів H.264/H.265. `sync=true` планує надсилання за часовими позначками; типове `false` надсилає без очікування годинника.
`async=true` дозволяє UDP-вихідному вузлу під час запуску чекати першого буфера; типове значення — `false`. У Python цей параметр доступний як `async_`.

## Кодовані кадри

Для кодованого входу передайте кодек потоку фабриці прямого передавання. Neat розбирає, пакує та надсилає потік без повторного кодування.

| Кодек | Фабрика C++ | Фабрика Python | Типовий тип корисного навантаження RTP |
| --- | --- | --- | ---: |
| H.264 | `Passthrough(RtspCodec::H264)` | `passthrough(pyneat.RtspCodec.H264)` | 96 |
| H.265 | `Passthrough(RtspCodec::H265)` | `passthrough(pyneat.RtspCodec.H265)` | 98 |
| MJPEG | `Passthrough(RtspCodec::MJPEG)` | `passthrough(pyneat.RtspCodec.MJPEG)` | 26 |

Пряме передавання не створює енкодер. Вибраний викликачем кодек має відповідати кодованому входу; параметри енкодера не застосовуються.

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
