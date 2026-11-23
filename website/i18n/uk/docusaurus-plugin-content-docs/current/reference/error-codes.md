---
title: "Каталог кодів помилок"
description: "Стабільні коди помилок фреймворку: коли вони виникають і як на них реагувати"
sidebar_position: 7
---

# Каталог кодів помилок

Neat повідомляє про типізовані збої через `NeatError` і `PullError`. Кожен збій містить стабільний код помилки, зрозуміле людині повідомлення і, якщо доступно, `GraphReport` зі структурованим контекстом.

Використовуйте код, щоб програмно визначати тип помилки. Повідомлення показуйте розробникові. Повний набір публічних констант наведено у файлі [`pipeline/ErrorCodes.h`](/reference/cppapi/files/include-pipeline-errorcodes-h).

## Несумісна зміна поведінки та міграція

Таксономія діагностики тепер зберігає конкретні першопричини, що надходять від GStreamer. Сигнатури публічних методів не змінилися, але код, який порівнює точні рядки помилок, може потребувати міграції:

| Раніше повертався код | Конкретніший код, що повертається тепер | Міграція |
| --- | --- | --- |
| `misconfig.caps` для помилки узгодження GStreamer під час виконання | `misconfig.media_caps` або `misconfig.media_format`, якщо несумісний лише формат | Обробляйте код помилки медіаданих. Залиште `misconfig.caps` лише для перевірки фреймворком перевизначень caps і контрактів суміжних вузлів. |
| `build.parse_launch` для будь-якої помилки `gst_parse_launch` | `build.plugin_missing`, `build.property_invalid` або `build.pipeline_syntax` | Обробляйте конкретні коди збирання. Залиште `build.parse_launch` як резервний варіант для некласифікованої помилки парсера. |
| `runtime.pull` для помилки шини, що поширилася далі | Код першопричини, наприклад `misconfig.media_caps`, `io.rtsp_connection_failed` або `resource.output_pool_exhausted` | Обробляйте коди першопричин і залиште гілку за замовчуванням. `runtime.pull` залишається резервним варіантом для локальної помилки отримання даних (pull) без конкретної причини. |

Використовуйте константи C++ або Python замість того, щоб повторювати рядкові літерали. Завжди залишайте шлях за замовчуванням для кодів, які з’являться в новішій збірці Neat Library.

## Публічні константи

Ті самі значення доступні в API обох мов:

| Код помилки | C++ | Python |
| --- | --- | --- |
| `misconfig.pipeline_shape` | `error_codes::kPipelineShape` | `pyneat.ERROR_PIPELINE_SHAPE` |
| `misconfig.caps` | `error_codes::kCaps` | `pyneat.ERROR_CAPS` |
| `misconfig.input_shape` | `error_codes::kInputShape` | `pyneat.ERROR_INPUT_SHAPE` |
| `misconfig.sample_attributes` | `error_codes::kSampleAttributes` | `pyneat.ERROR_SAMPLE_ATTRIBUTES` |
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
| `DispatcherUnavailable` (застаріле) | `error_codes::kDispatcherUnavailableLegacy` | `pyneat.ERROR_DISPATCHER_UNAVAILABLE_LEGACY` |
| `internal.plugin_failure` | `error_codes::kInternalPluginFailure` | `pyneat.ERROR_INTERNAL_PLUGIN_FAILURE` |

## Неправильна конфігурація

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Граф має недійсну топологію або в ньому бракує межі входу чи виходу. | Виправте з’єднання графа й додайте потрібні вузли `Input` або `Output`, якщо їх бракує. |
| `misconfig.caps` | Під час перевірки фреймворком перевизначення caps або контракт суміжного вузла виявляються несумісними. | Узгодьте оголошений формат, розміри, частоту й контракт суміжного вузла. |
| `misconfig.input_shape` | Вхідний тензор не відповідає очікуваній формі або типу даних. | Надайте очікувані вхідні дані або налаштуйте попередню обробку моделі через параметри моделі. |
| `misconfig.sample_attributes` | Ключ або значення атрибута зразка містить байт NUL або некоректну послідовність UTF-8. | Для ключів і значень використовуйте коректний текст UTF-8 без байтів NUL. |
| `misconfig.runtime_abi_mismatch` | Neat і встановлений плагін середовища виконання використовують несумісні ABI. | Встановіть сумісні між собою збірки Neat Library і плагіна середовища виконання. |
| `misconfig.graph_element_name` | Власний фрагмент містить елемент, якому неможливо призначити стабільне ім’я вузла. | Дайте власним елементам стабільні унікальні імена. |
| `misconfig.media_caps` | З’єднані етапи GStreamer потребують несумісних caps медіаданих. | Узгодьте етапи або вставте потрібний вузол перетворення, масштабування чи перетворення частоти. |
| `misconfig.media_format` | З’єднані етапи потребують несумісних медіаформатів. | Налаштуйте спільний формат або додайте явне перетворення формату. |
| `misconfig.input_capacity` | Розміри зображення з джерела перевищують обмеження, задані для попередньої обробки. | Збільште `input_max_width` і `input_max_height` або змініть розмір зображення перед передаванням до моделі. |
| `misconfig.tensor_dtype_missing` | У контракті тензора не вказано тип даних або формат. | Оголосіть підтримуваний тип даних у контракті тензора на попередньому етапі. |
| `misconfig.option_out_of_range` | Параметр недійсний для поточного вхідного контракту. | Задайте параметру значення з діапазону, який показує діагностика. |

## Помилки збирання

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `build.parse_launch` | GStreamer не може зібрати згенерований конвеєр. | Перевірте власний фрагмент, властивості елементів і наявність плагінів. |
| `build.pipeline_syntax` | Власний фрагмент GStreamer має недійсний синтаксис. | Виправте фрагмент і перевірте його за допомогою `gst-launch-1.0`. |
| `build.plugin_missing` | Потрібний елемент GStreamer або плагін кодека недоступний. | Встановіть або замініть компонент, а потім перевірте його за допомогою `gst-inspect-1.0`. |
| `build.property_invalid` | Ім’я або значення властивості елемента недійсне. | Перевірте властивість за допомогою `gst-inspect-1.0 <element>`. |

## Помилки під час виконання

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `runtime.pull` | Операція отримання даних (pull) завершується помилкою без конкретнішого коду. | Перегляньте доданий звіт і першу помилку на попередніх етапах. |
| `runtime.element_failed` | Етап конвеєра зупиняється без конкретнішої класифікації. | Виправте конфігурацію зазначеного етапу та його вхідні дані з попередніх етапів. |
| `runtime.output_timeout` | Обчислювальне ядро (kernel) не завершило виконання за відведений час очікування. | Звільняйте задіяні буфери тільки після завершення роботи ядра або підтвердженого відновлення пристрою. |
| `runtime.unexpected_eos` | Конвеєр досягає EOS, перш ніж видати потрібний вихід. | Перевірте, чи немає у вхідних даних передчасного EOS, і переконайтеся, що вхідних даних надано достатньо. |

## Помилки введення/виведення

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `io.parse` | Neat не може розібрати JSON, контракт моделі або конфігурацію етапу. | Перевірте синтаксис конфігурації, схему та обов’язкові поля. |
| `io.open` | Neat не може відкрити файл, пристрій або віддалений ресурс. | Перевірте шлях або адресу, права доступу та доступність ресурсу. |
| `io.file_not_found` | Вхідний файл не існує. | Виправте шлях і переконайтеся, що файл існує на DevKit. |
| `io.permission_denied` | Файл або пристрій неможливо відкрити з потрібним рівнем доступу. | Змініть власника зазначеного ресурсу або налаштуйте права доступу до нього. |
| `io.rtsp_connection_failed` | Neat не може підключитися до джерела RTSP. | Перевірте URL, сервер, мережеву доступність і облікові дані. |
| `io.camera_not_found` | Запитана камера недоступна. | Виберіть доступну камеру або використовуйте камеру за замовчуванням. |
| `io.model_not_found` | Запитаний архів моделі не існує. | Виправте шлях до моделі та переконайтеся, що архів встановлено. |
| `io.source_ended` | Джерело вхідних даних штатно досягає кінця. | Припиніть читати з цього джерела або надайте додаткові вхідні дані, якщо застосунок очікує більше даних. |

## Помилки матеріалізації конвеєра

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `misconfig.pipeline_shape` | Топологія конвеєра недійсна, або після побудови в GStreamer остаточні імена елементів дублюються, неоднозначні чи відсутні. | Дайте кожному явно заданому елементу унікальне коротке ім’я в межах його матеріалізованого сегмента. Підтримуйте узгодженість оголошень `name=` і посилань на іменовані пади. |
| `build.parse_launch` | GStreamer не може розібрати або побудувати остаточний рядок запуску, бо синтаксис, плагін або властивість недійсні. | Перегляньте `GraphReport::pipeline_string`; перевірте фрагмент за допомогою `gst-launch-1.0`, а плагін за допомогою `gst-inspect-1.0`. |

Ці перевірки виконуються автоматично під час `Graph::build()`. Для з’єднаних сегментів, що залежать від вхідних даних, той самий код і `GraphReport` можуть з’явитися тоді, коли перші вхідні дані матеріалізують сегмент.

## Помилки кодеків

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `codec.invalid_h264_stream` | Вхідні дані не містять жодного дійсного кадру H.264. | Надайте повний потік H.264 і перевірте налаштований кодек. |
| `codec.decode_failed` | Декодер не може декодувати прийнятий потік. | Перевірте кодек і переконайтеся, що закодовані вхідні дані повні й не пошкоджені. |
| `codec.encode_failed` | Кодувальник не може закодувати надані кадри. | Перевірте формат вхідних даних, роздільну здатність і налаштування кодувальника. |

## Помилки ресурсів

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `resource.memory_allocation_failed` | Не вдалося виділити потрібну пам’ять. Причину, специфічну для пристрою, не визначено. | Зменште кількість потоків даних, роздільну здатність або обсяг буферизації та звільніть пам’ять, яку займають інші робочі навантаження. |
| `resource.device_memory_exhausted` | Вичерпано неперервну пам’ять DMA/CMA пристрою. | Зменште кількість одночасних потоків даних, роздільну здатність вхідних даних або глибину буфера. |
| `resource.output_pool_exhausted` | Усі вихідні буфери залишаються зайнятими. | Своєчасно звільняйте вихідні дані zero-copy або використовуйте власні копії. |
| `resource.buffer_too_small` | Буфер менший за оголошений обсяг даних кадру чи тензора. | Виправте розміри та крок (stride) на попередніх етапах або виділіть потрібну кількість байтів. |
| `resource.disk_full` | Запис не вдається, бо в місці призначення недостатньо вільного місця. | Звільніть місце або виберіть інше місце призначення. |

## Помилки інфраструктури

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `infra.dispatcher_unavailable` | Neat не може отримати середовище виконання прискорювача. | Перевірте сумісність DevKit і зупиніть робочі навантаження, які монопольно займають прискорювач. |
| `infra.accelerator_execution_failed` | Прискорювач не може виконати етап моделі. | Перезапустіть конвеєр і зменште кількість одночасних робочих навантажень на прискорювачі. |

## Внутрішні помилки

| Код | Коли виникає | Що робити |
| --- | --- | --- |
| `internal.plugin_failure` | Плагін Neat завершується помилкою без класифікації, за якою користувач міг би вжити заходів. | Збережіть доданий `GraphReport` і повідомте про збій у службу підтримки. |

`DispatcherUnavailable` — застаріле написання, яке приймається для сумісності. У нових застосунках слід використовувати `infra.dispatcher_unavailable` і константу `error_codes::kDispatcherUnavailable`.

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

`PullError.code` використовує ті самі константи. Не аналізуйте рядок, який повертає `what()`, і не визначайте тип помилки за текстом повідомлення.

## Додаткові матеріали

- [Діагностика та налагодження](/reference/diagnostics) — повідомлення в робочому режимі, подробиці налагодження та збір `GraphReport`.
- [Формат помилок плагінів](/reference/error_format) — структурований контракт для помилок плагінів GStreamer.
- [`NeatError`](/reference/cppapi/classes/simaai-neat-neaterror) — типізований виняток.
- [`GraphReport`](/reference/cppapi/structs/simaai-neat-graphreport) — структурований контекст помилки.
