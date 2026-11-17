---
title: Встановлення в автономному режимі
description: Встановлення пакетів SDK та Model Compiler без доступу до інтернету на цільовому хості
sidebar_position: 6
---

:::tip Коли потрібне автономне встановлення
Використовуйте автономне встановлення, якщо цільовий хост не може завантажити пакети зі служб SiMa.ai.
Це може бути потрібно, якщо ваша організація:

- Блокує завантаження великих файлів.
- Обмежує доступ до реєстрів пакетів.
- Вимагає схвалення перед використанням завантажених пакетів.
- Відокремлює хости розробки від комп’ютерів із доступом до інтернету.
:::

Завантажте пакет на комп’ютері з доступом до інтернету.
Потім перенесіть пакет на цільовий хост для встановлення.

Пакети середовища виконання описано в розділах [автономного встановлення хоста PCIe](/getting-started/neat-library/pcie-host/#offline-installation) та [автономного встановлення Neat Library](/getting-started/neat-library/install-or-update/#offline-installation).

## Виберіть пакет

| Пакет | Призначення | Доступність |
| --- | --- | --- |
| Palette SDK | Налаштування середовища розробки SDK. | SDK 2.1.3.0 або новішої версії |
| Model Compiler | Встановлення Model Compiler у SDK або на окремому хості. | Model Compiler 2.1.3 або новішої версії |

## Перед початком

- Встановіть `sima-cli` на комп’ютері з доступом до інтернету. Див. [Встановлення SDK](/getting-started/dev-environment/install-the-environment/).
- Виберіть версію пакета, сумісну з вашою системою. Див. [Сумісність](/getting-started/compatibility/).
- Визначте операційну систему та архітектуру процесора цільового хоста.
- Підготуйте USB-накопичувач або внутрішній спільний файловий ресурс для перенесення.

Комп’ютер для завантаження та цільовий хост можуть мати різні архітектури процесорів.
Виберіть пакет для **цільового хоста**.
Зберігайте всі файли пакета разом, зокрема сценарії, метадані, контрольні суми та залежності.

## Встановіть Palette SDK {#download-the-sdk-offline-package}

### 1. Завантажте пакет

Виконайте одну відповідну команду на комп’ютері з доступом до інтернету.
У цих прикладах використано SDK %%release.sdk%%.

Для цільового хоста `amd64`:

<ShellCommand prompt="host">
sima-cli neat install sdk@v%%release.sdk%% -t offline-amd64
</ShellCommand>

Для цільового хоста `arm64`:

<ShellCommand prompt="host">
sima-cli neat install sdk@v%%release.sdk%% -t offline-arm64
</ShellCommand>

### 2. Встановіть пакет

Встановлення Model Compiler необов’язкове.
Якщо вам не потрібно компілювати моделі, пропустіть завантаження Model Compiler і виконайте наведені нижче кроки встановлення SDK.
Коли програма встановлення SDK запитає, чи встановлювати Model Compiler, виберіть пропуск.

Щоб встановити Model Compiler разом із SDK, [завантажте офлайн-пакет Model Compiler](#download-model-compiler-package).
Покладіть файл офлайн-пакета Model Compiler у ту саму теку, що й офлайн-пакет SDK.
Програма встановлення SDK автоматично знайде файл.
Коли з’явиться запит, виберіть встановлення з локального пакета.

1. Скопіюйте всю завантажену теку на цільовий хост.
2. Відкрийте термінал у цій теці.
3. Запустіть програму встановлення:

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

## Встановіть Model Compiler {#download-the-model-compiler-offline-package}

:::tip Коли використовувати цю процедуру
Використовуйте цю процедуру, щоб окремо встановити Model Compiler в наявному SDK або безпосередньо на хості.
Якщо ви встановили SDK і Model Compiler на попередньому кроці, пропустіть цей розділ.
:::

### 1. Завантажте пакет {#download-model-compiler-package}

Перевірте [вимоги до сумісності Model Compiler](/getting-started/compatibility/#model-compiler).
Виконайте одну відповідну команду на комп’ютері з доступом до інтернету.
У цих прикладах використано Model Compiler %%release.model_compiler%%.

Для цільового хоста `amd64`:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v%%release.model_compiler%% -t offline
</ShellCommand>

Для цільового хоста `arm64`:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v%%release.model_compiler%% -t offline
</ShellCommand>

### 2. Встановіть пакет

Виберіть одне місце встановлення.

**У Palette SDK:**

1. Скопіюйте всю завантажену теку в робочий простір хоста, зіставлений із `/workspace` у контейнері SDK.
2. Відкрийте оболонку SDK.
3. Перейдіть до теки пакета. Для пакета `amd64` використовуйте:

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
</ShellCommand>

Для пакета `arm64` використовуйте `/workspace/model-compiler-offline-arm64`.

4. Запустіть програму встановлення:

<ShellCommand prompt="sdk">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

**На окремому хості:**

1. Скопіюйте всю завантажену теку на цільовий хост.
2. Відкрийте термінал у цій теці.
3. Запустіть програму встановлення:

<ShellCommand prompt="host">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

### 3. Активуйте Model Compiler

1. Перезапустіть оболонку SDK або оболонку хоста, яку ви використовували для встановлення.
2. Активуйте Model Compiler:

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

Щоб вийти із середовища Model Compiler, виконайте:

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## Зберігайте пакети у внутрішній мережі {#hosting-packages-internally}

Ви можете зберігати схвалені пакети на внутрішньому файловому сервері.
Зберігайте кожну завантажену теку повністю й окремо від інших версій пакета.
Перенесіть усю теку на цільовий хост перед запуском програми встановлення.
