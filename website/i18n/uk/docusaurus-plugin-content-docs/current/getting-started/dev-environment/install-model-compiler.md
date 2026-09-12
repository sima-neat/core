---
title: "Встановіть Model Compiler."
description: "Встановіть Model Compiler в Neat SDK або на підтримуваному окремому хості."
sidebar_position: 5
---

:::tip Почніть звідси, лише якщо ви встановлюєте Model Compiler окремо.
Встановлення Model Compiler відбувається за бажанням під час встановлення SDK. Використовуйте цю сторінку лише в тому випадку, якщо ви пропустили відповідне повідомлення, бажаєте встановити новішу сумісну версію Model Compiler або вам потрібно встановити Model Compiler поза межами SDK на підтримуваному хості.
:::

Model Compiler квантує та компілює моделі ONNX, щоб їх можна було запускати на MLA від SiMa.ai. Це **обов’язково**, якщо ви самостійно компілюєте або квантуєте моделі, зокрема моделі GenAI, і є **необов’язковим**, лише якщо ви використовуєте виключно попередньо скомпільовані пакети моделей.

Під час встановлення/налаштування SDK, `sima-cli` пропонує встановити відповідний Model Compiler як розширення всередині Neat SDK. Ви також можете встановити його пізніше, або всередині контейнера Neat SDK, або окремо на підтримуваному хості Ubuntu. Інформацію про підтримувані комбінації версій і вимоги до окремого хоста див. у розділі [Сумісність](/getting-started/compatibility/#model-compiler).

## Встановіть у межах SDK.

Якщо ви пропустите Model Compiler під час налаштування SDK, встановіть його пізніше зсередини
Neat SDK. Запустіть команду, яка відповідає архітектурі вашого контейнера Neat SDK. Щоб перевірити це, запустіть `uname -m` в оболонці SDK: `aarch64` означає, що потрібно використовувати команду `arm64`, а `x86_64` означає, що потрібно використовувати команду `amd64`.

Для контейнерів `amd64` Neat SDK:

<ShellCommand prompt="sdk">
sima-cli neat install model-compiler/amd64@v2.1.3
</ShellCommand>

Для контейнерів `arm64`, що використовують Neat SDK:

<ShellCommand prompt="sdk">
sima-cli neat install model-compiler/arm64@v2.1.3
</ShellCommand>

Після встановлення активуйте середовище компілятора в командному рядку Neat SDK:

<ShellCommand prompt="sdk">
activate-model-compiler
</ShellCommand>

Щоб повернутися до стандартної оболонки Neat SDK, виконайте команду:

<ShellCommand prompt="sdk">
deactivate-model-compiler
</ShellCommand>

## Встановіть на окремому хості.

Автономна інсталяція підтримується лише на хост-середовищах, перелічених у [Сумісність](/getting-started/compatibility/#model-compiler). Запустіть відповідну команду `sima-cli neat install` із підтримуваного хост-середовища. Щоб перевірити архітектуру хоста, запустіть `uname -m`: `x86_64` використовує команду `amd64`, а `aarch64` використовує команду `arm64`.

Для Model Compiler 2.1.3 на хостах `amd64`:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3
</ShellCommand>

Для Model Compiler версії 2.1.3 на хостах `arm64`:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3
</ShellCommand>

Для Model Compiler 2.0.0 на хостах `amd64`:

<ShellCommand prompt="host">
sima-cli install -v 2.0.0 tools/model-compiler/amd64
</ShellCommand>

## Наступний крок

Продовжуйте [Складіть модель.](/compile-a-model/).
