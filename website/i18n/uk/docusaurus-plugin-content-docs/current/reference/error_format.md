---
title: "Формат повідомлення про помилку плагіна."
description: "Структуровані поля помилок, отримані внаслідок збоїв у роботі плагінів."
sidebar_position: 8
---

# Формат повідомлення про помилку плагіна.

Коли плагін стикається з критичною помилкою, він надсилає повідомлення `GST_MESSAGE_ERROR` до шини GStreamer. Neat підвищує пріоритет помилки до `NeatError` і зберігає підтримувані структуровані деталі для класифікації та відображення.

## Домени та коди помилок.

Це рекомендовані домени/коди, які використовуються в різних плагінах:

- Розбір/перевірка конфігурації: `GST_RESOURCE_ERROR_SETTINGS`
- Файл відсутній: `GST_RESOURCE_ERROR_NOT_FOUND`
- Диспетчер недоступний: `GST_RESOURCE_ERROR_BUSY`; скористайтеся.
  `GST_RESOURCE_ERROR_NOT_FOUND` лише з ідентифікатором діагностики, специфічним для диспетчера, або структурованим полем диспетчера.
- Помилки розподілу пам’яті: `GST_RESOURCE_ERROR_NO_SPACE_LEFT`
- Помилки під час узгодження параметрів/налаштування: `GST_STREAM_ERROR_FORMAT`
- Помилки під час обробки в середовищі виконання: `GST_STREAM_ERROR_FAILED`

## Версіоновані структуровані дані.

Нові помилки плагіна Neat додають структуру `GstStructure` під назвою `simaai-neat-error`. Версія 1 містить цілочисельне поле без знака `neat-schema-version=1`. Основний модуль зчитує структуровані поля з версії 1 і, у разі невідомої або відсутньої версії, використовує звичайні поля GStreamer: домен, код, повідомлення та рядок налагодження. Це запобігає інтерпретації майбутньої схеми з використанням застарілих припущень.

Загальні поля:
- `neat-schema-version`
- `neat-diagnostic-id`
- `neat-reason`
- `plugin`
- `node`
- `stage`
- `graph-id`
- `frame-id`
- `stream-id`
- `input-caps`
- `output-caps`
- `allocator`
- `dispatcher-error`

Помилки, пов’язані з обсягом вхідних даних, також надають інформацію про `actual-width`, `actual-height`, `actual-stride`,
`maximum-width`, `maximum-height`, `maximum-stride`, `resize-width`, `resize-height`,
`required-bytes`, `allocated-bytes` та `input-format`.

Помилки, пов’язані з контрактом вхідних даних, також надають інформацію про `input-name`, `segment-name`, `required-bytes`, `actual-bytes`,
`expected-shape`, `expected-layout`, `expected-dtype`, `received-shape`, `received-layout` та
`received-dtype`. Поля розмітки допомагають розрізняти такі форми, як `[3, 224, 224]` (`CHW`) та
`[224, 224, 3]` (`HWC`).

Старіші плагіни можуть розміщувати розділений пробілами `key='value'` список у рядку налагодження. Основний модуль продовжує
використовувати ці поля як резервний варіант для забезпечення сумісності.

## Приклад

```text
simaai-neat-error, neat-schema-version=(uint)1,
neat-diagnostic-id=(string)neatprocesscvu.input_contract_mismatch,
plugin=(string)neatprocesscvu, node=(string)model_0,
expected-shape=(string)"[3, 224, 224]", expected-layout=(string)CHW,
expected-dtype=(string)Float32, received-shape=(string)"[224, 224, 3]",
received-layout=(string)HWC, received-dtype=(string)UInt8;
```

## Примітки

- За замовчуванням `NeatError::what()` містить нормалізований код помилки, який призначений для відображення користувачеві.
  контекст, коригувальні дії та ідентифікатор для діагностики. Він не містить вихідне повідомлення GStreamer і рядок налагодження.
- Встановіть `SIMA_NEAT_VERBOSE_LEVEL=2` і `SIMA_NEAT_VERBOSE_TOPICS=gstreamer` на короткий проміжок часу.
  діагностичний запуск. Під час цього додаються відредаговані технічні деталі до
`NeatError::what()` та `GraphReport.repro_note`. `NeatError::report()` залишається структурованим інтерфейсом для діагностики.
- `NEAT_LOG_LEVEL=debug` не є налаштуванням Neat Library.
- URI, що містить інформацію про користувача, та поля, які містять інформацію про підтверджені облікові дані, зокрема `auth`, `playback-token`, `hdnts`.
  `stream-key` та `tkn` – ці значення видаляються з рядків, що використовуються в конвеєрі для формування звітів, команд для відтворення, структурованих даних та JSON. Перед тим, як надати доступ до пакета підтримки, перевірте, чи не містить він специфічних для розгортання шляхів і адрес медіафайлів.
