---
title: "Нотатки щодо хоста Ubuntu."
description: "Підготуйте хост Ubuntu для Neat SDK та DevKit-Sync."
sidebar_position: 1
---

Використовуйте цей посібник, якщо ваша хост-машина працює під управлінням Ubuntu, і ви хочете запустити Neat, середовище розробки (Neat SDK) з DevKit-Sync.

## Необхідні умови.

- Ubuntu 22.04 або 24.04, використовується як хост.
- Docker Engine встановлено та працює.
- `sima-cli` встановлено на хості.
- Modalix і DevKit повинні бути доступні в одній мережі.

:::info Топологія мережі.
В Ubuntu ви можете під’єднати DevKit безпосередньо до хоста через USB/Ethernet або розмістити хост і DevKit окремо в існуючій мережі. Якщо вони знаходяться в існуючій мережі, не потрібно налаштовувати спеціальне спільне використання, якщо хост і DevKit можуть взаємодіяти між собою для SSH і NFS-трафіку.
:::

## Пряме підключення Ubuntu до DevKit.

Використовуйте цю схему, коли DevKit підключено безпосередньо до комп’ютера з Ubuntu через USB/Ethernet і потрібно, щоб він використовував мережеве підключення комп’ютера Ubuntu.

Вимкніть IPv6 на мережевому інтерфейсі, до якого підключено DevKit. DevKit-Sync потребує передбачуваної адресації IPv4 для SSH і NFS, і якщо залишити IPv6 увімкненим на спільному з’єднанні, це може зробити виявлення та вибір маршруту ненадійними.

### Графічний інтерфейс NetworkManager.

1. Під’єднайте комп’ютер з Ubuntu до Інтернету через Wi-Fi або інший мережевий інтерфейс.
2. Під’єднайте DevKit до комп’ютера з Ubuntu за допомогою USB/Ethernet-адаптера.
3. На DevKit залиште налаштування підключеного мережевого інтерфейсу в режимі `DHCP`.
4. На Ubuntu відкрийте `Settings > Network`.
5. Відкрийте налаштування для дротового інтерфейсу, підключеного до DevKit.
6. На вкладці `IPv4` встановіть параметр `IPv4 Method` на значення `Shared to other computers`.
7. На вкладці `IPv6` встановіть для параметра `IPv6 Method` значення `Disabled`.
8. Застосуйте зміни, потім від’єднайте та знову під’єднайте дротовий інтерфейс.

Після встановлення з’єднання знайдіть IPv4-адресу DevKit в Ubuntu:

```bash
ip neigh
```

Переконайтеся, що у вас є доступ за SSH, перш ніж розпочинати налаштування SDK:

```bash
ssh sima@<devkit-ip>
```

Then continue with DevKit pairing:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

### Інтерфейс командного рядка NetworkManager.

Якщо ви віддаєте перевагу налаштуванню через командний рядок, визначте інтерфейс, який використовується для взаємодії з DevKit:

```bash
nmcli device status
```

Створіть спільне IPv4-з’єднання, вимкнувши IPv6:

```bash
sudo nmcli connection add type ethernet ifname <devkit-interface> con-name devkit-shared ipv4.method shared ipv6.method disabled
sudo nmcli connection up devkit-shared
```

Якщо для цього інтерфейсу вже існує профіль підключення, змініть його:

```bash
sudo nmcli connection modify "<connection-name>" ipv4.method shared ipv6.method disabled
sudo nmcli connection down "<connection-name>"
sudo nmcli connection up "<connection-name>"
```

## Примітки щодо брандмауера.

Якщо ввімкнено правила брандмауера Ubuntu, дозвольте SSH та NFS трафік на інтерфейсі або підмережі, до якої звертається DevKit, перш ніж запускати налаштування DevKit-Sync. Мінімум, DevKit повинен мати можливість підключатися до SSH та до хосту NFS, експортованого `sima-cli sdk setup --devkit`.

## Наступний крок.

Поверніться до [Neat SDK](/getting-started/dev-environment/) і продовжуйте встановлення/налаштування.
