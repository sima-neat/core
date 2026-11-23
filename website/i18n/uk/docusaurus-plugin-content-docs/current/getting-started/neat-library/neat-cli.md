---
title: "Neat CLI"
description: "Перевірка та оновлення встановленого середовища Neat за допомогою команди neat"
sidebar_position: 5
---

Встановлена бібліотека надає команду середовища `neat`. Запустіть її в SDK або на DevKit, щоб переглянути версії встановлених компонентів, встановлені сценарії (playbooks) `sima-cli` і те, чи доступні новіші артефакти.

У SDK вивід стану також містить зіставлення портів хоста для Insight із файлу `$HOME/.insight-config/neat-port-map.json`.

Відсутність установлених PyNeat або Insight не заважає виводу стану.
Якщо перевірка версії завершується помилкою, компонент показує `unknown`, а `neat`
записує діагностичне повідомлення провайдера в стандартний потік помилок. Вивід JSON
зберігає наявну схему й повідомляє недоступні версії як `null`.

Розділене встановлення показує `simaai-runner` і `simaai-scheduler`. На DevKit версії беруться з бази встановлених пакетів, у SDK — з кешованих керувальних метаданих DEB. Рядки застарілих `neat-runtime` і `neat-gst-plugins` залишаються видимими, якщо ці пакети встановлено або не виявлено жодного розділеного компонента. JSON додає `components.runner` і `components.scheduler`, зберігаючи старі ключі `runtime` та `gstPlugins`; відсутні версії залишаються `null`.

<ShellCommand prompt="sdk|devkit">
neat
</ShellCommand>

Приклад виводу для застарілого встановлення:

```text
Neat Environment
  Mode               Palette SDK
  Sysroot            /opt/toolchain/aarch64/modalix
  Update check       online

Components
  Neat core              %%release.core%% channel=release latest=%%release.core%%
  PyNeat                 %%release.core%%
  neat-runtime           %%release.core%%
  neat-gst-plugins       %%release.core%%
  neat-insight           %%release.insight%% channel=release status=Running venv=/opt/neat-insight/venv
  Model Compiler         %%release.model_compiler%% run activate-model-compiler to activate

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

Для автоматизації та інтеграції з інструментами використовуйте вивід у форматі JSON:

<ShellCommand prompt="sdk|devkit">
neat --json
</ShellCommand>

## Оновлення встановлених компонентів

Щоб оновити середовище виконання Neat Library, `neat-insight` та встановлені сценарії `sima-cli` з виявленого каналу, виконайте команду:

<ShellCommand prompt="sdk|devkit">
neat update
</ShellCommand>

Щоб оновити лише середовище виконання Neat Library, не перевіряючи й не
оновлюючи Insight або сценарії, виконайте:

<ShellCommand prompt="sdk|devkit">
neat update --core-only
</ShellCommand>

## Наступний крок

Перейдіть до розділу [Компіляція моделі](/compile-a-model/), щоб підготувати модель для Modalix, —
це виконується в SDK і не потребує DevKit. Щоб запустити повноцінний застосунок на апаратному забезпеченні,
див. розділ [Привіт, Neat!](/develop-apps/hello-neat/minimal/); для цього потрібен під’єднаний
DevKit.

## Стан хоста та карт PCIe

Пакет `sima-pcie-host` також встановлює `neat` в Ubuntu. Виконайте `neat` або `neat --json`, щоб переглянути встановлені пакети хоста, PyNeatPCIe, а також плату та встановлене програмне забезпечення кожної карти SiMa PCIe. Карти перелічено за адресою PCI та інтерфейсом. Адреса PCI описує поточне під’єднання, а не постійний серійний номер плати.

```bash
neat
neat --json
neat --offline
```

У Windows пакет хоста PCIe встановлює `neat.cmd`. Запускайте його з PowerShell; `--json` виводить описаний нижче об’єкт стану PCIe без обгортки `sima.neat.status.v1`:

```powershell
& "$env:ProgramFiles\SimaPCIeHost\bin\neat.cmd"
& "$env:ProgramFiles\SimaPCIeHost\bin\neat.cmd" --json
```

У Windows немає переліку адрес PCI, тому карти відображаються за інтерфейсом мережевого адаптера PCIe від Platform та їхньою адресою. Пакетами хоста є встановлені продукти MSI Scheduler і Core.

Опитування карт використовує налаштований інтерфейс хоста `10.0.N.1` та адресу карти `10.0.N.2`, як і `pcie-setup.sh`. Спочатку налаштуйте мережу та довірені ключі SSH. Команда не налаштовує інтерфейси, не реєструє ключі хостів, не запитує паролі, не використовує sudo на картах і не змінює пакети. Відсутні інтерфейси, нестандартні адреси, збої SSH і недоступне ПЗ відображаються для кожної карти окремо; доступні карти й далі показують свої версії.

`NEAT_PCIE_SSH_USER` вибирає користувача SSH карти (типово `sima`). `NEAT_PCIE_SSH_KEY` вибирає приватний ключ (типово `~/.ssh/sima_neat_pcie_ed25519`). В Ubuntu наявні налаштування SSH та автентифікація через агент і надалі доступні. У Windows, якщо цей файл ключа існує, OpenSSH пропонує лише його та ключі, зазначені в `IdentityFile` налаштувань SSH, але не ключі ssh-agent. `--offline` пропускає перевірку оновлень через інтернет, але все одно опитує під’єднані карти через локальний SSH. Стан PCIe-хоста не перевіряє оновлень артефактів; для оновлення пакета хоста використовуйте `sima-cli neat install core/pciehost/<distro>/<arch>@<ref>`. `neat update` не оновлює карти PCIe.

JSON зберігає `sima.neat.status.v1` та додає `components.pcieHost`, `components.pcieScheduler`, `components.pyneatpcie` і `pcie`. Об’єкт `pcie` містить ПЗ хоста, масив `cards` із полями адреси PCI, інтерфейсів, адреси, стану, ПЗ та помилки, а також помилки топології. Відсутнє віддалене ПЗ подається як `null`, а не як припущена встановлена версія.

Стан PyNeat і PyNeatPCIe містить версію та шлях віртуального середовища для кожного виявленого встановлення. Окремі середовища відображаються як кілька середовищ, а не як конфлікт версій. Для пакетів поза віртуальним середовищем натомість показується тека пакета. Перевірка стану читає метадані пакетів без імпорту нативних модулів.

До кожного запису Python у JSON додається масив `installations`. Кожен елемент містить `version`, `environment` та `path` (теку пакета). Поза віртуальним середовищем `environment` має значення `null`. Якщо встановлені версії відрізняються, підсумкове значення `version` дорівнює `null`; наявність кількох встановлень сама по собі не задає `error`. Шляхи символічних посилань на те саме встановлення перелічуються лише один раз.
