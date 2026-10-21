---
title: "Neat CLI"
description: "Використовуйте команду neat для перевірки та оновлення встановленого середовища Neat."
sidebar_position: 5
---

Встановлена бібліотека надає команду середовища `neat`. Запустіть її або з SDK, або з DevKit, щоб переглянути версії встановлених компонентів, встановлені сценарії `sima-cli` та наявність новіших артефактів.

У SDK вивід статусу також містить відображення хост-портів Insight з файлу `$HOME/.insight-config/neat-port-map.json`.

Відсутність установлених PyNeat або Insight не заважає звітуванню про стан.
Якщо перевірка версії завершується помилкою, компонент показує `unknown`, а `neat`
записує діагностику постачальника у стандартний потік помилок. Вивід JSON
зберігає наявну схему й повідомляє недоступні версії як `null`.

Розділена інсталяція показує `simaai-runner` і `simaai-scheduler`. На DevKit версії беруться з бази встановлених пакетів, у SDK — з керувальних метаданих кешованих DEB. Рядки застарілих `neat-runtime` і `neat-gst-plugins` залишаються видимими, якщо ці пакети присутні або жодного розділеного компонента не виявлено. JSON додає `components.runner` і `components.scheduler`, зберігаючи старі ключі `runtime` та `gstPlugins`; відсутні версії залишаються `null`.

<ShellCommand prompt="sdk-or-devkit">
neat
</ShellCommand>

Приклад виводу застарілої інсталяції:

```text
Neat Environment
  Mode               Neat SDK
  Sysroot            /opt/toolchain/aarch64/modalix
  Update check       online

Components
  Neat core              0.4.0 channel=release latest=0.4.0
  PyNeat                 0.4.0
  neat-runtime           0.4.0
  neat-gst-plugins       0.4.0
  neat-insight           0.0.6 channel=release status=Running venv=/opt/neat-insight/venv
  Model Compiler         2.1.3 run activate-model-compiler to activate

Exposed Ports
  Insight Web UI     https://10.0.0.22:9900

  Name               Protocol Host Port (Start) Host Port (End)
  ------------------ -------- ----------------- ---------------
  mainUI             tcp      9900              -
  metadataUDP        udp      9100              9179
  rtsp.tcp           tcp      8554              -
  videoUDP           udp      9000              9079
  videoUI            tcp      8081              -
  webRTC             udp      40000             40199
  webSSH             tcp      8022              -
```

## Вивід у форматі JSON

Для автоматизації та інтеграції інструментів використовуйте вивід у форматі JSON:

<ShellCommand prompt="sdk|devkit">
neat --json
</ShellCommand>

## Оновіть встановлені компоненти.

Щоб оновити середовище виконання Neat Library, `neat-insight` та встановлені сценарії `sima-cli` з виявленого каналу, виконайте команду:

<ShellCommand prompt="sdk|devkit">
neat update
</ShellCommand>

Щоб оновити лише середовище виконання Neat Library, не перевіряючи й не
оновлюючи Insight або сценарії, виконайте:

<ShellCommand prompt="sdk-or-devkit">
neat update --core-only
</ShellCommand>

## Наступний крок

Продовжуйте [Складіть модель.](/compile-a-model/), щоб підготувати модель для Modalix —
це запускається в SDK і не потребує DevKit. Щоб запустити повну програму на апаратному забезпеченні,
перегляньте [Привіт, Neat!](/develop-apps/hello-neat/minimal/), для чого потрібен комплект
DevKit.
