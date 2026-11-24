---
title: "Встановлення Model Compiler"
description: "Встановіть Model Compiler у Palette SDK або на підтримуваному окремому хості"
sidebar_position: 5
---

:::tip Починайте тут, лише якщо встановлюєте Model Compiler окремо
Під час встановлення SDK Model Compiler встановлюється лише за вашою згодою. Користуйтеся цією сторінкою, лише якщо ви пропустили цей запит, хочете встановити новішу сумісну версію Model Compiler або вам потрібно встановити Model Compiler поза SDK на підтримуваному хості.
:::

Model Compiler квантує та компілює моделі ONNX, щоб їх можна було запускати на MLA від SiMa.ai. Він **обов’язковий**, якщо ви самостійно компілюєте або квантуєте моделі, зокрема моделі GenAI, і **необов’язковий**, лише якщо ви використовуєте виключно попередньо скомпільовані пакети моделей.

Під час встановлення/налаштування SDK `sima-cli` пропонує встановити відповідний Model Compiler як розширення всередині Palette SDK. Ви також можете встановити його пізніше: або всередині контейнера Palette SDK, або окремо на підтримуваному хості Ubuntu. Підтримувані комбінації версій і вимоги до окремого хоста див. у розділі [Сумісність](/getting-started/compatibility/#model-compiler).

## Встановлення всередині SDK

Якщо ви пропустили Model Compiler під час налаштування SDK, встановіть його пізніше зсередини
Palette SDK. Виконайте команду, яка відповідає архітектурі вашого контейнера Palette SDK. Щоб її перевірити, виконайте `uname -m` в оболонці SDK: `aarch64` означає, що треба використати команду `arm64`, а `x86_64` — команду `amd64`.

Для контейнерів Palette SDK `amd64`:

<ShellCommand prompt="sdk">
sima-cli neat install model-compiler/amd64@v%%release.model_compiler%%
</ShellCommand>

Для контейнерів Palette SDK `arm64`:

<ShellCommand prompt="sdk">
sima-cli neat install model-compiler/arm64@v%%release.model_compiler%%
</ShellCommand>

Після встановлення активуйте середовище компілятора в оболонці Palette SDK:

<ShellCommand prompt="sdk">
activate-model-compiler
</ShellCommand>

Щоб повернутися до звичайної оболонки Palette SDK, виконайте:

<ShellCommand prompt="sdk">
deactivate-model-compiler
</ShellCommand>

## Встановлення на окремому хості

Окреме встановлення підтримується лише в середовищах хоста, перелічених у розділі [Сумісність](/getting-started/compatibility/#model-compiler). Виконайте відповідну команду `sima-cli neat install` у підтримуваному середовищі хоста. Щоб перевірити архітектуру хоста, виконайте `uname -m`: для `x86_64` використовуйте команду `amd64`, а для `aarch64` — команду `arm64`.

Для Model Compiler %%release.model_compiler%% на хостах `amd64`:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v%%release.model_compiler%%
</ShellCommand>

Для Model Compiler %%release.model_compiler%% на хостах `arm64`:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v%%release.model_compiler%%
</ShellCommand>

Для Model Compiler 2.0.0 на хостах `amd64`:

<ShellCommand prompt="host">
sima-cli install -v 2.0.0 tools/model-compiler/amd64
</ShellCommand>

## Наступний крок

Перейдіть до сторінки [Компіляція моделі](/compile-a-model/).
