---
title: "Каталог кодів помилок."
description: "Стабільні коди помилок фреймворку, коли вони виникають, і способи їх усунення."
sidebar_position: 7
---

# Каталог кодів помилок.

Neat повідомляє про виявлені помилки за допомогою `NeatError` та `PullError`. Кожне повідомлення про помилку містить стабільний код помилки, зрозуміле для користувача повідомлення та, за наявності, `GraphReport` зі структурованим контекстом.

Використовуйте код помилки для програмної обробки. Показуйте повідомлення розробнику. Повний набір публічних констант міститься у файлі [`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h).

## Зміна поведінки, що впливає на сумісність, та інструкції з міграції.

Діагностична таксономія тепер зберігає конкретні GStreamer основні причини. Публічні інтерфейси методів не змінилися, але код, який порівнює точні рядки помилок, можливо, потрібно буде оновити:

| Попередній результат | Тепер повертається більш точний код | Міграція |
| --- | --- | --- |
| `misconfig.caps` є причиною помилки під час узгодження в середовищі виконання GStreamer, | `misconfig.media_caps`, або `misconfig.media_format`, коли несумісним є лише формат |. Обробляйте кодек медіа. Залиште `misconfig.caps` лише для перевірки в рамках фреймворку, що стосується перевизначень параметрів і суміжних контрактів вузлів. |
| `build.parse_launch` для кожної помилки `gst_parse_launch` | `build.plugin_missing`, `build.property_invalid` або `build.pipeline_syntax` | Обробляйте конкретні коди помилок. Використовуйте `build.parse_launch` як резервний варіант для некласифікованої помилки парсера. |
| `runtime.pull` для випадку поширеної помилки шини | Код основної причини, такий як `misconfig.media_caps`, `io.rtsp_connection_failed` або `resource.output_pool_exhausted` | Обробляйте коди основних причин і зберігайте основну гілку. `runtime.pull` залишається резервним варіантом для локальної помилки під час отримання даних, коли немає конкретної причини. |

Замість повторюваних рядкових літералів використовуйте константи C++ або Python. Завжди зберігайте шлях за замовчуванням для коду, доданого в новішій версії Neat Library.

## Публічні константи

Обидва мовні API містять однакові значення:

| Код помилки | C++ | Python |
| --- | --- | --- |
| `misconfig.pipeline_shape` | `error_codes::kPipelineShape` | `pyneat.ERROR_PIPELINE_SHAPE` |
| `misconfig.caps` | `error_codes::kCaps` | `pyneat.ERROR_CAPS` |
| `misconfig.input_shape` | `error_codes::kInputShape` | `pyneat.ERROR_INPUT_SHAPE` |
| `misconfig.runtime_abi_mismatch` | `error_codes::kRuntimeAbiMismatch` | `pyneat.ERROR_RUNTIME_ABI_MISMATCH` |
| `misconfig.graph_element_name` | `error_codes::kGraphElementName` | `pyneat.ERROR_GRAPH_ELEMENT_NAME` |
| `misconfig.media_caps` | `error_codes::kMediaCaps` | `pyneat.ERROR_MEDIA_CAPS` |
| `misconfig.media_format` | `error_codes::kMediaFormat` | `pyneat.ERROR_MEDIA_FORMAT` |
| `misconfig.input_capacity` | `error_codes::kInputCapacity` | `pyneat.ERROR_INPUT_CAPACITY` |
| `misconfig.tensor_dtype_missing` | `error_codes::kTensorDtypeMissing` | `pyneat.ERROR_TENSOR_DTYPE_MISSING` |
| `misconfig.option_out_of_range` | `error_codes::kOptionOutOfRange` | `pyneat.ERROR_OPTION_OUT_OF_RANGE` |
| `build.parse_launch` | `error_codes::kParseLaunch` | `pyneat.ERROR_PARSE_LAUNCH` |
| `build.pipeline_syntax` | `error_codes::kPipelineSyntax` | `pyneat.ERROR_PIPELINE_SYNTAX` |
| `build.plugin_missing` | `error_codes::kPluginMissing` | `pyneat.ERROR_PLUGIN_MISSING` |
| `build.property_invalid` | `error_codes::kPropertyInvalid` | `pyneat.ERROR_PROPERTY_INVALID` |
| `runtime.pull` | `error_codes::kRuntimePull` | `pyneat.ERROR_RUNTIME_PULL` |
| `runtime.element_failed` | `error_codes::kRuntimeElementFailed` | `pyneat.ERROR_RUNTIME_ELEMENT_FAILED` |
| `runtime.output_timeout` | `error_codes::kOutputTimeout` | `pyneat.ERROR_OUTPUT_TIMEOUT` |
| `runtime.unexpected_eos` | `error_codes::kUnexpectedEos` | `pyneat.ERROR_UNEXPECTED_EOS` |
| `io.parse` | `error_codes::kIoParse` | `pyneat.ERROR_IO_PARSE` |
| `io.open` | `error_codes::kIoOpen` | `pyneat.ERROR_IO_OPEN` |
| `io.file_not_found` | `error_codes::kFileNotFound` | `pyneat.ERROR_FILE_NOT_FOUND` |
| `io.permission_denied` | `error_codes::kPermissionDenied` | `pyneat.ERROR_PERMISSION_DENIED` |
| `io.rtsp_connection_failed` | `error_codes::kRtspConnectionFailed` | `pyneat.ERROR_RTSP_CONNECTION_FAILED` |
| `io.camera_not_found` | `error_codes::kCameraNotFound` | `pyneat.ERROR_CAMERA_NOT_FOUND` |
| `io.model_not_found` | `error_codes::kModelNotFound` | `pyneat.ERROR_MODEL_NOT_FOUND` |
| `io.source_ended` | `error_codes::kSourceEnded` | `pyneat.ERROR_SOURCE_ENDED` |
| `codec.invalid_h264_stream` | `error_codes::kInvalidH264Stream` | `pyneat.ERROR_INVALID_H264_STREAM` |
| `codec.decode_failed` | `error_codes::kDecodeFailed` | `pyneat.ERROR_DECODE_FAILED` |
| `codec.encode_failed` | `error_codes::kEncodeFailed` | `pyneat.ERROR_ENCODE_FAILED` |
| `resource.memory_allocation_failed` | `error_codes::kMemoryAllocationFailed` | `pyneat.ERROR_MEMORY_ALLOCATION_FAILED` |
| `resource.device_memory_exhausted` | `error_codes::kDeviceMemoryExhausted` | `pyneat.ERROR_DEVICE_MEMORY_EXHAUSTED` |
| `resource.output_pool_exhausted` | `error_codes::kOutputPoolExhausted` | `pyneat.ERROR_OUTPUT_POOL_EXHAUSTED` |
| `resource.buffer_too_small` | `error_codes::kBufferTooSmall` | `pyneat.ERROR_BUFFER_TOO_SMALL` |
| `resource.disk_full` | `error_codes::kDiskFull` | `pyneat.ERROR_DISK_FULL` |
| `infra.dispatcher_unavailable` | `error_codes::kDispatcherUnavailable` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE` |
| `infra.accelerator_execution_failed` | `error_codes::kAcceleratorExecutionFailed` | `pyneat.ERROR_ACCELERATOR_EXECUTION_FAILED` |
| `DispatcherUnavailable` (застаріла версія) | `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## Неправильне налаштування.

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| `misconfig.pipeline_shape` | У графі є недійсний топологічний зв’язок або відсутні вхідні/вихідні граничні вузли. | Виправте з’єднання графа та додайте необхідні вузли `Input` або `Output`. |
| Під час перевірки фреймворку неможливо використовувати одночасну зміну параметрів або сусідній контракт вузла. `misconfig.caps` | Узгодьте оголошений формат, розміри, частоту та сусідній контракт вузла. | |
| `misconfig.input_shape` | Тензор вхідних даних не відповідає очікуваній формі або типу даних. | Надайте очікувані вхідні дані або налаштуйте попередню обробку моделі через параметри моделі. |
| `misconfig.runtime_abi_mismatch` | Neat і встановлений плагін середовища виконання використовує несумісні ABI. | Встановіть відповідні. Neat Library та збірки плагінів для середовища виконання. |
| `misconfig.graph_element_name` | У спеціальному фрагменті міститься елемент, якому неможливо присвоїти стабільне ім’я вузла. | Присвоюйте спеціальним елементам стабільні, унікальні імена. |
| `misconfig.media_caps` | Підключені GStreamer етапи вимагають несумісних параметрів медіа. | Узгодьте етапи або вставте необхідний вузол для конвертації, масштабування або зміни частоти кадрів. |
| `misconfig.media_format` | З’єднані етапи вимагають несумісних форматів медіафайлів. | Налаштуйте спільний формат або додайте явне перетворення форматів. |
| `misconfig.input_capacity` | Зображення-джерело перевищує встановлену максимальну ємність вхідних даних для попередньої обробки. | Збільште значення `input_max_width` та `input_max_height`, або зменште розмір зображення-джерела перед передачею в модель. |
| `misconfig.tensor_dtype_missing` | У визначенні тензорної операції відсутній тип даних або формат. | Вкажіть підтримуваний тип даних у визначенні тензорної операції. |
| `misconfig.option_out_of_range` | Обраний параметр є недійсним для поточного вхідного контракту. | Встановіть для параметра значення в діапазоні, який показує діагностичний інструмент. |

## Помилки під час збірки.

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| Не вдалося створити згенерований конвеєр `build.parse_launch`. | Перевірте налаштований фрагмент, властивості елементів і наявність плагінів GStreamer. | |
| `build.pipeline_syntax` | У фрагменті GStreamer виявлено синтаксичну помилку. | Виправте фрагмент і перевірте його за допомогою `gst-launch-1.0`. |
| `build.plugin_missing` | Необхідний елемент або кодек GStreamer недоступний. | Встановіть або замініть компонент, а потім перевірте його за допомогою `gst-inspect-1.0`. |
| `build.property_invalid` | Назва або значення властивості елемента є недійсними. | Перевірте властивість за допомогою `gst-inspect-1.0 <element>`. |

## Помилки під час виконання (у середовищі виконання).

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| Операція завантаження даних завершується невдало через відсутність більш конкретного коду помилки. `runtime.pull` Перевірте прикріплений звіт і першу помилку, що виникла у вихідному джерелі. | | |
| `runtime.element_failed` | Етап конвеєра зупиняється без більш конкретної класифікації. | Виправте конфігурацію вказаного етапу та його вхідні дані. |
| `runtime.output_timeout` | Відсутність виводу до закінчення встановленого часу очікування. | Перевірте потік даних і механізм регулювання швидкості передачі, або відрегулюйте час очікування, якщо очікується, що дані будуть доступні пізніше. |
| `runtime.unexpected_eos` | Конвеєр досягає кінця вхідних даних (EOS) до того, як буде створено необхідний вихід. | Перевірте вхідні дані на наявність передчасного завершення (EOS) і переконайтеся, що було надано достатню кількість вхідних даних. |

## Помилки вводу/виводу.

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| `io.parse` | Neat не може обробити JSON, шаблон контракту або конфігурацію етапу. | Перевірте синтаксис, схему та обов’язкові поля конфігурації. |
| `io.open` | Neat не вдалося відкрити файл, пристрій або віддалений ресурс. | Перевірте шлях або адресу, права доступу та наявність ресурсів. |
| `io.file_not_found` | Вхідний файл не існує. | Перевірте шлях до файлу та переконайтеся, що файл існує в DevKit. |
| `io.permission_denied` | Не вдалося відкрити файл або пристрій із необхідним рівнем доступу. | Виправте права власності або дозволи для зазначеного ресурсу. |
| Не вдалося встановити з’єднання з джерелом RTSP. `io.rtsp_connection_failed` | Перевірте URL-адресу, сервер, доступність мережі та облікові дані. | Neat не може підключитися до джерела RTSP. |
| `io.camera_not_found` | Запитана камера недоступна. | Виберіть доступну камеру або використовуйте камеру за замовчуванням. |
| `io.model_not_found` | Запитний архів моделі не знайдено. | Перевірте шлях до моделі та переконайтеся, що архів встановлено. |
| `io.source_ended` | Джерело вхідних даних досягло свого звичайного кінця. | Припиніть отримувати дані з цього джерела або надайте додаткові дані, якщо застосунок вимагає більше інформації. |

## Збої під час матеріалізації конвеєра.

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Топологія конвеєра недійсна, або назви кінцевих елементів дублюються, є неоднозначними або відсутні після створення GStreamer. | Присвоюйте кожному явному елементу унікальну коротку назву в межах його матеріалізованого сегмента. Підтримуйте синхронізацію між оголошеннями `name=` та посиланнями на іменовані вхідні/вихідні порти. |
| `build.parse_launch` | GStreamer не може проаналізувати або створити кінцевий рядок запуску, оскільки синтаксис, плагін або властивість є недійсними. | Перевірте `GraphReport::pipeline_string`; перевірте фрагмент за допомогою `gst-launch-1.0` і плагін за допомогою `gst-inspect-1.0`. |

Ці перевірки виконуються автоматично під час `Graph::build()`. Для сегментів, які залежать від вхідних даних і є з’єднаними, той самий код і `GraphReport` можуть з’явитися, коли перший набір вхідних даних генерує цей сегмент.

## Збої в роботі кодеків.

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| `codec.invalid_h264_stream` | Вхідні дані не містять жодного дійсного H.264-кадру. | Надайте повний H.264-потік і перевірте налаштований кодек. |
| `codec.decode_failed` | Декодер не може декодувати отриманий потік даних. | Перевірте кодек і переконайтеся, що вхідні дані, які потрібно закодувати, є повними та не пошкодженими. |
| `codec.encode_failed` | Кодек не може обробити надані кадри. | Перевірте формат вхідних даних, роздільну здатність і налаштування кодека. |

## Збої в роботі ресурсів.

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| `resource.memory_allocation_failed` | Виникла помилка під час виділення необхідного обсягу пам’яті, і причина не пов’язана з конкретним пристроєм. | Зменште кількість потоків, роздільну здатність або обсяг буфера, а також звільніть пам’ять, яку використовують інші програми. |
| `resource.device_memory_exhausted` | Недостатньо суміжної пам’яті пристрою для DMA/CMA. | Зменште кількість одночасних потоків, роздільну здатність вхідного сигналу або глибину буфера. |
| Усі буфери виводу залишаються в активному стані. `resource.output_pool_exhausted` | Негайно звільняйте ресурси, що використовуються для виводу даних без копіювання, або використовуйте копії, якими ви володієте. | |
| `resource.buffer_too_small` | Буфер менший за оголошений розмір кадру або тензора. | Відкоригуйте розміри та крок у вихідному коді або виділіть необхідну кількість байтів. |
| `resource.disk_full` | Операція запису не вдалася, оскільки в місці призначення недостатньо вільного місця. | Звільніть місце або оберіть інше місце призначення. |

## Збої в роботі інфраструктури.

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat не може отримати доступ до середовища виконання прискорювача. | Переконайтеся у сумісності з DevKit і зупиніть робочі навантаження, які ексклюзивно використовують прискорювач. |
| Виникла помилка під час виконання етапу моделі прискорювачем. `infra.accelerator_execution_failed` | Прискорювач не може виконати етап моделі. | Перезапустіть конвеєр і зменште кількість одночасних завдань, що виконуються прискорювачем. |

## Внутрішні збої.

| Код | Виникає, коли | Що робити |
| --- | --- | --- |
| `internal.plugin_failure` | Плагін Neat не працює, і при цьому відсутня класифікація, яка б дозволила користувачеві вирішити проблему. | Збережіть прикріплений `GraphReport` і повідомте про помилку в службу підтримки. |

`DispatcherUnavailable` – це застаріле написання, яке використовується для забезпечення сумісності. У нових застосунках слід використовувати `infra.dispatcher_unavailable` та константу `error_codes::kDispatcherUnavailable`.

## Обробляйте помилки програмним шляхом.

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

try {
  auto run = graph.build();
  // Push and pull application data.
} catch (const simaai::neat::NeatError& error) {
  if (error.report().error_code == simaai::neat::error_codes::kInputShape) {
    handle_input_contract_error(error.report());
  } else {
    throw;
  }
}
```

`PullError.code` використовує ті самі константи. Не потрібно аналізувати `what()` або зіставляти його з текстом, зрозумілим для людини.

## Для подальшого ознайомлення

- [Діагностика та налагодження.](/reference/diagnostics) — повідомлення, що генеруються під час роботи, деталі налагодження та інше.
  Колекція `GraphReport`.
- [Формат повідомлення про помилку плагіна.](/reference/error_format) — це структурована схема для плагіна GStreamer.
  помилки.
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — тип винятку.
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — структурований контекст помилки.
