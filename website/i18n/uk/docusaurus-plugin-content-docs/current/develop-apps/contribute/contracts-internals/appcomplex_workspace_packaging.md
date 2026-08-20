---
title: "Упаковка робочого середовища AppComplex."
description: "Створіть і встановіть пакет служб для захищеного комплексного робочого простору застосунків."
sidebar_position: 3
slug: /develop-apps/contribute/appcomplex_workspace_packaging
---

# Упаковка робочого середовища AppComplex.

Цей посібник об’єднує `tmp/core/sima-ai-appcomplex` в окремий пакет системи, який не замінює поточну версію `simaai-appcomplex.service`, якщо ви не зробите цього явно.

## Що саме встановлюється?

- Двійкові файли та бібліотеки в `/opt/simaai/appcomplex-workspace/`.
- Одиниця Systemd: `simaai-appcomplex-workspace.service`
- Файл конфігурації: `/etc/default/simaai-appcomplex-workspace`

За замовчуванням робоча область використовує ізольовані кінцеві точки:

- Керуючий сокет: `/tmp/mlactrl_workspace`
- Об’єкт SHM: `/mlashmdata_workspace`
- Ініціалізація MLA: `APP_COMPLEX_RUN_INIT=0` (пропустити ініціалізацію для паралельного запуску).

## Створити пакет.

```bash
./scripts/release/build_appcomplex_workspace_deb.sh
```

Скрипт виводить згенерований шлях до `.deb` у `build/packages/`.

## Встановити (за замовчуванням, але з можливістю відключення)

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb>
```

Стандартна поведінка під час інсталяції:

- Не зупиняє/не вимикає `simaai-appcomplex.service`.
- Не вмикається та не запускається автоматично `simaai-appcomplex-workspace.service`.

## Увімкніть службу робочого простору вручну.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now simaai-appcomplex-workspace.service
```

Якщо вам потрібно повторно ініціалізувати MLA перед запуском робочого середовища (для переходу в інший режим), встановіть:

```bash
sudo sed -i 's/^APP_COMPLEX_RUN_INIT=.*/APP_COMPLEX_RUN_INIT=1/' /etc/default/simaai-appcomplex-workspace
```

## Необов’язковий перехід (лише у разі явного зазначення).

Щоб надіслати запит на припинення надання старої послуги та активацію послуги робочого простору:

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb> --activate --switch-system
```

Або оновіть `/etc/default/simaai-appcomplex-workspace`:

- `APP_COMPLEX_ACTIVATE_ON_INSTALL=1`
- `APP_COMPLEX_SWITCH_SYSTEM_SERVICE=1`

а потім запустіть:

```bash
sudo dpkg-reconfigure simaai-appcomplex-workspace
```
