---
title: "Каталог кодів помилок"
description: "Стабільні коди помилок фреймворку, коли вони виникають і як на них реагувати"
sidebar_position: 7
---

# Каталог кодів помилок

Neat повідомляє про типізовані збої через `NeatError` і `PullError`. Кожен збій надає стабільний
код помилки, зрозуміле для людини повідомлення та, за наявності, `GraphReport` зі структурованим контекстом.

Використовуйте код помилки для програмного тріажу. Показуйте повідомлення розробнику. Повний набір
публічних констант міститься в
[`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h).

## Зміна поведінки, що порушує сумісність, і міграція

Діагностична таксономія тепер зберігає конкретні першопричини GStreamer. Сигнатури публічних методів
не змінилися, але код, який порівнює точні рядки помилок, може потребувати міграції:

| Попередній збіг | Конкретніший код, що повертається тепер | Міграція |
| --- | --- | --- |
| `misconfig.caps` для помилки узгодження GStreamer під час виконання | `misconfig.media_caps` або `misconfig.media_format`, якщо несумісний лише формат | Обробляйте медіакод. Залиште `misconfig.caps` лише для перевірки фреймворком перевизначень caps і контрактів суміжних вузлів. |
| `build.parse_launch` для кожного збою `gst_parse_launch` | `build.plugin_missing`, `build.property_invalid` або `build.pipeline_syntax` | Обробляйте конкретні коди збірки. Залиште `build.parse_launch` як резервний варіант для некласифікованого збою парсера. |
| `runtime.pull` для поширеного збою шини | Код першопричини, наприклад `misconfig.media_caps`, `io.rtsp_connection_failed` або `resource.output_pool_exhausted` | Обробляйте коди першопричин і залиште гілку за замовчуванням. `runtime.pull` залишається резервним варіантом для локального збою отримання (pull) без конкретної причини. |

Використовуйте константи C++ або Python замість повторення рядкових літералів. Завжди залишайте шлях за замовчуванням
для кодів, доданих у новішій збірці Neat Library.

## Публічні константи

Ті самі значення доступні в API обох мов:

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
| `io.response_too_large` | `error_codes::kResponseTooLarge` | `pyneat.ERROR_RESPONSE_TOO_LARGE` |
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
| `infra.peripheral_daemon_unavailable` | `error_codes::kPeripheralDaemonUnavailable` | `pyneat.ERROR_PERIPHERAL_DAEMON_UNAVAILABLE` |
| `infra.peripheral_daemon_timeout` | `error_codes::kPeripheralDaemonTimeout` | `pyneat.ERROR_PERIPHERAL_DAEMON_TIMEOUT` |
| `DispatcherUnavailable` (застарілий) | `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## Неправильна конфігурація

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Граф має неприпустиму топологію або в ньому бракує межі вводу/виводу. | Виправте з’єднання графа та потрібні вузли `Input` або `Output`. |
| `misconfig.caps` | Перевизначення caps або контракт суміжного вузла несумісні під час перевірки фреймворком. | Узгодьте оголошений формат, розміри, частоту та контракт суміжного вузла. |
| `misconfig.input_shape` | Вхідний тензор не відповідає очікуваній формі або типу даних. | Надайте очікувані вхідні дані або налаштуйте попередню обробку моделі через параметри моделі. |
| `misconfig.runtime_abi_mismatch` | Neat і встановлений плагін середовища виконання використовують несумісні ABI. | Встановіть відповідні одна одній збірки Neat Library і плагіна середовища виконання. |
| `misconfig.graph_element_name` | Користувацький фрагмент містить елемент, якому неможливо призначити стабільне ім’я вузла. | Дайте користувацьким елементам стабільні унікальні імена. |
| `misconfig.media_caps` | З’єднані етапи GStreamer вимагають несумісних медіа-caps. | Узгодьте етапи або вставте потрібний вузол перетворення, масштабування чи перетворення частоти. |
| `misconfig.media_format` | З’єднані етапи вимагають несумісних медіаформатів. | Налаштуйте спільний формат або додайте явне перетворення формату. |
| `misconfig.input_capacity` | Вихідне зображення перевищує налаштовану вхідну місткість попередньої обробки. | Збільште `input_max_width` і `input_max_height` або масштабуйте джерело перед етапом моделі. |
| `misconfig.tensor_dtype_missing` | У контракті тензора не вказано тип даних або формат. | Оголосіть підтримуваний тип даних у контракті тензора вище за потоком. |
| `misconfig.option_out_of_range` | Параметр неприпустимий для поточного вхідного контракту. | Задайте для параметра значення з діапазону, показаного в діагностиці. |

## Збої збірки

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `build.parse_launch` | GStreamer не може зібрати згенерований конвеєр. | Перевірте користувацький фрагмент, властивості елементів і наявність плагінів. |
| `build.pipeline_syntax` | Користувацький фрагмент GStreamer має неприпустимий синтаксис. | Виправте фрагмент і перевірте його за допомогою `gst-launch-1.0`. |
| `build.plugin_missing` | Потрібний елемент GStreamer або плагін кодека недоступний. | Встановіть або замініть компонент, а потім перевірте його за допомогою `gst-inspect-1.0`. |
| `build.property_invalid` | Ім’я або значення властивості елемента неприпустиме. | Перевірте властивість за допомогою `gst-inspect-1.0 <element>`. |

## Збої під час виконання

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `runtime.pull` | Операція отримання (pull) завершується збоєм без конкретнішого коду. | Перегляньте долучений звіт і першу помилку вище за потоком. |
| `runtime.element_failed` | Етап конвеєра зупиняється без конкретнішої класифікації. | Виправте конфігурацію зазначеного етапу та його вхідні дані вище за потоком. |
| `runtime.output_timeout` | Вихідні дані не надходять до завершення налаштованого часу очікування. | Перевірте потік даних із джерела та зворотний тиск (back-pressure) або змініть тайм-аут, якщо очікування є передбаченим. |
| `runtime.unexpected_eos` | Конвеєр досягає EOS, перш ніж створити потрібні вихідні дані. | Перевірте вхідні дані на передчасний EOS і переконайтеся, що надано достатньо вхідних даних. |

## Збої вводу/виводу

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `io.parse` | Neat не може розібрати JSON, контракт моделі або конфігурацію етапу. | Перевірте синтаксис конфігурації, схему та обов’язкові поля. |
| `io.open` | Neat не може відкрити файл, пристрій або віддалений ресурс. | Перевірте шлях або адресу, права доступу та доступність ресурсу. |
| `io.file_not_found` | Вхідний файл не існує. | Виправте шлях і переконайтеся, що файл існує на DevKit. |
| `io.permission_denied` | Файл або пристрій неможливо відкрити з потрібним доступом. | Виправте власника або права доступу для зазначеного ресурсу. |
| `io.rtsp_connection_failed` | Neat не може під’єднатися до джерела RTSP. | Перевірте URL, сервер, мережеву досяжність і облікові дані. |
| `io.camera_not_found` | Запитана камера недоступна. | Виберіть доступну камеру або використайте типову камеру. |
| `io.model_not_found` | Запитаний архів моделі не існує. | Виправте шлях до моделі та переконайтеся, що архів встановлено. |
| `io.source_ended` | Джерело вхідних даних досягає свого штатного кінця. | Припиніть читати це джерело або надайте додаткові вхідні дані, якщо застосунок очікує більше даних. |
| `io.response_too_large` | Обмежена відповідь локального протоколу перевищує задокументоване обмеження розміру. | Зменште розмір каталогу або встановіть відповідні одна одній версії клієнта та служби. |

## Збої матеріалізації конвеєра

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Топологія конвеєра неприпустима або фінальні імена елементів дублюються, неоднозначні чи відсутні після побудови GStreamer. | Дайте кожному явному елементу унікальне коротке ім’я в межах його матеріалізованого сегмента. Підтримуйте синхронізацію оголошень `name=` і посилань на іменовані pad. |
| `build.parse_launch` | GStreamer не може розібрати або побудувати фінальний рядок запуску, бо синтаксис, плагін або властивість неприпустимі. | Перегляньте `GraphReport::pipeline_string`; перевірте фрагмент за допомогою `gst-launch-1.0`, а плагін — за допомогою `gst-inspect-1.0`. |

Ці перевірки виконуються автоматично під час `Graph::build()`. Для з’єднаних сегментів, що залежать від вхідних даних,
той самий код і `GraphReport` можуть з’явитися, коли перші вхідні дані матеріалізують сегмент.

## Збої кодеків

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `codec.invalid_h264_stream` | Вхідні дані не містять жодного дійсного кадру H.264. | Надайте повний потік H.264 і перевірте налаштований кодек. |
| `codec.decode_failed` | Декодер не може декодувати прийнятий потік. | Перевірте кодек і переконайтеся, що закодовані вхідні дані повні й не пошкоджені. |
| `codec.encode_failed` | Кодер не може закодувати надані кадри. | Перевірте вхідний формат, роздільну здатність і налаштування кодера. |

## Збої ресурсів

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `resource.memory_allocation_failed` | Потрібне виділення пам’яті завершується збоєм без причини, пов’язаної з пристроєм. | Зменште кількість потоків даних, роздільну здатність або буферизацію та звільніть пам’ять, яку використовують інші навантаження. |
| `resource.device_memory_exhausted` | Неперервну пам’ять пристрою DMA/CMA вичерпано. | Зменште кількість одночасних потоків даних, вхідну роздільну здатність або глибину буфера. |
| `resource.output_pool_exhausted` | Усі вихідні буфери залишаються зайнятими. | Своєчасно звільняйте вихідні дані без копіювання (zero-copy) або використовуйте власні копії. |
| `resource.buffer_too_small` | Буфер менший за оголошене корисне навантаження кадру або тензора. | Виправте розміри та крок (stride) вище за потоком або виділіть потрібну кількість байтів. |
| `resource.disk_full` | Запис завершується збоєм, бо в місці призначення недостатньо вільного місця. | Звільніть місце або виберіть інше місце призначення. |

## Збої інфраструктури

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat не може отримати середовище виконання прискорювача. | Перевірте сумісність DevKit і зупиніть навантаження, що монопольно володіють прискорювачем. |
| `infra.accelerator_execution_failed` | Прискорювач не може виконати етап моделі. | Перезапустіть конвеєр і зменште кількість одночасних навантажень на прискорювач. |
| `infra.peripheral_daemon_unavailable` | Сокет API локального SiMa Sentinel відсутній або відмовляє в з’єднанні, виявлення периферійних пристроїв у Sentinel вимкнено чи зупинено, встановлений Sentinel надто старий, щоб надавати каталог периферійних пристроїв, або Sentinel повернув інший неочікуваний статус HTTP. | Встановіть або оновіть Sentinel командою `sima-cli neat install sentinel` або запустіть `simaai-sentinel.service`, а потім перегляньте його журнал. |
| `infra.peripheral_daemon_timeout` | Обмежений запит каталогу периферійних пристроїв не завершився. | Перевірте стан `simaai-sentinel.service` і провайдерів, а потім повторіть спробу. |

## Внутрішні збої

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `internal.plugin_failure` | Плагін Neat завершується збоєм без класифікації, за якою користувач міг би діяти. | Збережіть долучений `GraphReport` і повідомте про збій службі підтримки. |

`DispatcherUnavailable` — застаріле написання, яке приймається для сумісності. Нові застосунки мають
використовувати `infra.dispatcher_unavailable` і константу `error_codes::kDispatcherUnavailable`.

## Програмна обробка помилок

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

`PullError.code` використовує ті самі константи. Не розбирайте `what()` і не зіставляйте зрозумілий для людини текст.

## Додаткові матеріали

- [Діагностика та налагодження](/reference/diagnostics) — повідомлення для робочого середовища, подробиці налагодження та
  збирання `GraphReport`.
- [Формат помилок плагінів](/reference/error_format) — структурований контракт, за яким плагіни GStreamer повідомляють
  про помилки.
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — типізований виняток.
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — структурований контекст помилки.
