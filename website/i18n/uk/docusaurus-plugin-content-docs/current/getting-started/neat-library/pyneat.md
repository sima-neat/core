---
title: "Встановіть PyNeat."
description: "Встановіть пакет PyNeat у спеціально створене віртуальне середовище Python."
sidebar_position: 4
---

:::note DevKit — лише для цього випадку; пропускайте під час встановлення SDK.
Ці кроки встановлюють PyNeat на DevKit (або на звичайному хості, який виконує роль середовища виконання). Якщо ви працюєте в Neat SDK, пропустіть цю сторінку.
:::

:::tip PyNeat вже встановлено разом із Neat Library.
PyNeat постачається разом із Neat Library і встановлюється автоматично під час встановлення Neat Library.

За замовчуванням він встановлюється у віртуальне середовище за адресою `~/pyneat`. Ви можете пропустити цю сторінку, якщо не бажаєте встановлювати PyNeat у спеціальне віртуальне середовище, наприклад, окреме середовище venv або conda на DevKit.
:::

Виконайте наведені нижче дії на DevKit. Ця інструкція не встановлює та не оновлює пакети `.deb` для середовища виконання, тому виконайте її там, де вже встановлено відповідне середовище виконання Neat Library.

## Завантажте Wheel.

<ShellCommand prompt="devkit">
sima-cli neat install core -t pyneat
</ShellCommand>

Щоб завантажити пакет для певної версії Neat Library, вкажіть номер версії.

Для Neat Library 0.2.2:

<ShellCommand prompt="devkit">
sima-cli neat install core@v0.4.0 -t pyneat
</ShellCommand>

## Створіть середовище Python.

Створіть і активуйте віртуальне середовище, використовуючи `python3` для цього середовища:

<ShellCommand prompt="devkit">
python3 -m venv ~/my-neat-env
source ~/my-neat-env/bin/activate
</ShellCommand>

## Встановіть колесо.

<ShellCommand prompt="devkit">
pip install ./pyneat-*.whl
</ShellCommand>

Інформацію про сумісні комбінації програмного забезпечення, зокрема Neat Library, SDK та DevKit, можна знайти в
[Посібник зі сумісності](/getting-started/compatibility/).

## Наступний крок

Продовжуйте роботу в [Neat CLI](/getting-started/neat-library/neat-cli/), щоб перевірити встановлене середовище.
