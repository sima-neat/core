---
title: "Встановлення PyNeat"
description: "Встановлення wheel-пакета PyNeat у власне віртуальне середовище Python"
sidebar_position: 4
---

:::note Лише для DevKit — пропустіть, якщо встановлюєте SDK
Ці кроки налаштовують PyNeat на DevKit (або на «чистому» хості, що виконує роль середовища виконання). Якщо ви працюєте всередині Palette SDK, пропустіть цю сторінку.
:::

:::tip PyNeat уже встановлено разом із Neat Library
PyNeat входить до складу Neat Library і встановлюється автоматично, коли ви встановлюєте Neat Library.

Типово він встановлюється у віртуальне середовище в `~/pyneat`. Цю сторінку можна пропустити, якщо ви не хочете встановити PyNeat у власне віртуальне середовище, наприклад в окреме середовище venv або conda на DevKit.
:::

Виконайте наведені нижче кроки на DevKit. Ця інструкція не встановлює й не оновлює пакети `.deb` середовища виконання, тож виконуйте її там, де вже встановлено відповідне середовище виконання Neat Library.

## Завантаження wheel-пакета

Виберіть версію wheel-пакета, що відповідає середовищу виконання Neat Library, уже встановленому на DevKit. Для поточного випуску:

<ShellCommand prompt="devkit">
sima-cli neat install core@v%%release.core%% -t pyneat
</ShellCommand>

Якщо встановлено іншу версію середовища виконання, замініть тег версії відповідним опублікованим тегом Core. Завантаження wheel-пакета не оновлює пакети середовища виконання.

## Створення середовища Python

Створіть і активуйте віртуальне середовище за допомогою `python3` цього середовища:

<ShellCommand prompt="devkit">
python3 -m venv ~/my-neat-env
source ~/my-neat-env/bin/activate
</ShellCommand>

## Встановлення wheel-пакета

<ShellCommand prompt="devkit">
pip install ./pyneat-*.whl
</ShellCommand>

Підтримувані комбінації Neat Library, SDK і програмного забезпечення DevKit див. у
[Посібнику із сумісності](/getting-started/compatibility/).

## Наступний крок

Перейдіть до розділу [Neat CLI](/getting-started/neat-library/neat-cli/), щоб перевірити
встановлене середовище.
