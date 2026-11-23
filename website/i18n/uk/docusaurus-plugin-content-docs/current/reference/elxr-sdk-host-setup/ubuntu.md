---
title: "Нотатки щодо хоста Ubuntu"
description: "Підготовка хоста Ubuntu до роботи з Palette SDK і DevKit-Sync"
sidebar_position: 1
---

Скористайтеся цим посібником, якщо ваша хост-машина працює під керуванням Ubuntu і ви хочете запускати Palette SDK із DevKit-Sync.

## Необхідні умови

- Хост з Ubuntu 22.04 або 24.04.
- Docker Engine %%dependency.docker.minimum%% або новішої версії, встановлений і робочий. Перевірте версію командою `docker --version`.
- `sima-cli`, встановлений на хості.
- Modalix DevKit, доступний у тій самій мережі.

:::info Топологія мережі
В Ubuntu DevKit можна під’єднати до хоста безпосередньо через USB/Ethernet або розмістити хост і DevKit окремо в наявній мережі. Якщо вони перебувають у наявній мережі, спеціально налаштовувати спільний доступ не потрібно, за умови що хост і DevKit доступні одне одному для трафіку SSH і NFS.
:::

## Пряме з’єднання Ubuntu з DevKit

Використовуйте цей порядок дій, коли DevKit під’єднано до комп’ютера з Ubuntu безпосередньо через USB/Ethernet і йому потрібно спільно використовувати мережеве з’єднання цього комп’ютера.

Вимкніть IPv6 на спільному мережевому інтерфейсі, до якого під’єднано DevKit. DevKit-Sync покладається на передбачувану IPv4-адресацію для SSH і NFS, а ввімкнений IPv6 на спільному підключенні може зробити виявлення та вибір маршруту ненадійними.

### Графічний інтерфейс NetworkManager

1. Під’єднайте комп’ютер з Ubuntu до інтернету через Wi-Fi або інший висхідний інтерфейс.
2. Під’єднайте DevKit до комп’ютера з Ubuntu через адаптер USB/Ethernet.
3. На DevKit залиште для підключеного мережевого інтерфейсу режим `DHCP`.
4. В Ubuntu відкрийте `Settings > Network`.
5. Відкрийте налаштування дротового інтерфейсу, до якого під’єднано DevKit.
6. На вкладці `IPv4` встановіть для `IPv4 Method` значення `Shared to other computers`.
7. На вкладці `IPv6` встановіть для `IPv6 Method` значення `Disabled`.
8. Застосуйте зміни, потім від’єднайте й знову під’єднайте дротовий інтерфейс.

Коли з’єднання встановиться, знайдіть IPv4-адресу DevKit з Ubuntu:

```bash
ip neigh
```

Перш ніж починати налаштування SDK, перевірте доступ через SSH:

```bash
ssh sima@<devkit-ip>
```

Потім перейдіть до під’єднання DevKit:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

### Інтерфейс командного рядка NetworkManager

Якщо ви віддаєте перевагу налаштуванню з командного рядка, визначте інтерфейс, до якого під’єднано DevKit:

```bash
nmcli device status
```

Створіть спільне IPv4-з’єднання з вимкненим IPv6:

```bash
sudo nmcli connection add type ethernet ifname <devkit-interface> con-name devkit-shared ipv4.method shared ipv6.method disabled
sudo nmcli connection up devkit-shared
```

Якщо для цього інтерфейсу вже є профіль з’єднання, змініть його замість цього:

```bash
sudo nmcli connection modify "<connection-name>" ipv4.method shared ipv6.method disabled
sudo nmcli connection down "<connection-name>"
sudo nmcli connection up "<connection-name>"
```

## Примітки щодо брандмауера

Якщо в Ubuntu ввімкнено правила брандмауера, перед налаштуванням DevKit-Sync дозвольте трафік SSH і NFS на інтерфейсі або в підмережі, до яких під’єднано DevKit. Щонайменше DevKit має мати доступ до SSH і до NFS-експорту хоста, створеного командою `sima-cli sdk setup --devkit`.

## Наступний крок

Поверніться до [Palette SDK](/getting-started/dev-environment/) і продовжте встановлення/налаштування.
