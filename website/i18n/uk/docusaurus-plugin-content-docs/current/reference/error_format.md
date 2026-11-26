---
title: "Формат помилок плагінів"
description: "Структуровані поля помилок, які надходять від плагінів у разі збою"
sidebar_position: 8
---

# Формат помилок плагінів

Коли в плагіні виникає ситуація, що унеможливлює подальше виконання, він надсилає `GST_MESSAGE_ERROR` на шину GStreamer. Neat перетворює цю помилку на `NeatError` і зберігає підтримувані структуровані дані для класифікації та відображення.

## Домени та коди помилок

Ось рекомендовані домени й коди, які використовують плагіни:

- Розбір і перевірка конфігурації: `GST_RESOURCE_ERROR_SETTINGS`
- Відсутній файл: `GST_RESOURCE_ERROR_NOT_FOUND`
- Диспетчер недоступний: `GST_RESOURCE_ERROR_BUSY`; використовуйте
  `GST_RESOURCE_ERROR_NOT_FOUND` лише разом з ідентифікатором діагностики, що стосується диспетчера,
  або зі структурованим полем із даними про диспетчер
- Збої виділення пам’яті: `GST_RESOURCE_ERROR_NO_SPACE_LEFT`
- Помилки caps і їхнього узгодження: `GST_STREAM_ERROR_FORMAT`
- Збої обробки під час виконання: `GST_STREAM_ERROR_FAILED`

## Версії структурованих даних

До повідомлень про помилки нового формату плагіни Neat додають структуру `GstStructure` з назвою `simaai-neat-error`. Версія 1 містить беззнакове цілочисельне поле `neat-schema-version=1`. Core зчитує структуровані поля з версії 1, а якщо версія невідома або відсутня, повертається до звичайних даних GStreamer: домену, коду, повідомлення та рядка налагодження. Так майбутню схему не буде інтерпретовано на основі старих припущень.

Спільні поля:
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

Помилки через перевищення обмежень входу попередньої обробки також містять поля `actual-width`, `actual-height`, `actual-stride`,
`maximum-width`, `maximum-height`, `maximum-stride`, `resize-width`, `resize-height`,
`required-bytes`, `allocated-bytes` і `input-format`.

Помилки контракту вхідних даних також містять поля `input-name`, `segment-name`, `required-bytes`, `actual-bytes`,
`expected-shape`, `expected-layout`, `expected-dtype`, `received-shape`, `received-layout` і
`received-dtype`. Поля розкладки уточнюють порядок вимірів для таких форм, як `[3, 224, 224]` (`CHW`) і
`[224, 224, 3]` (`HWC`).

Старіші плагіни можуть записувати в рядок налагодження список пар `key='value'`, розділених пробілами. Core і далі
використовує ці поля як резервний варіант для сумісності.

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

- За замовчуванням `NeatError::what()` містить нормалізований код помилки, контекст для користувача,
  дії для усунення помилки та ідентифікатор діагностики. Необроблене повідомлення GStreamer
  і рядок налагодження до нього не входять.
- Для короткого діагностичного запуску встановіть `SIMA_NEAT_VERBOSE_LEVEL=2` і `SIMA_NEAT_VERBOSE_TOPICS=gstreamer`.
  Тоді до `NeatError::what()` і `GraphReport.repro_note` додаються технічні подробиці з прихованими
  чутливими даними. `NeatError::report()` залишається структурованим інтерфейсом для діагностики.
- `NEAT_LOG_LEVEL=debug` не є налаштуванням Neat Library.
- Вміст компонента userinfo в URI й значення розпізнаних полів облікових даних, зокрема `auth`, `playback-token`, `hdnts`,
  `stream-key` і `tkn`, приховуються в рядках конвеєра для звітів, командах для відтворення,
  структурованих даних і JSON. Перед надсиланням діагностичного набору до служби підтримки перевірте, чи немає в ньому шляхів
  і адрес медіа, специфічних для розгортання.
