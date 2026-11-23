---
title: "Нотатки щодо хоста Windows"
description: "Підготовка хоста Windows 11 до роботи з Palette SDK і DevKit-Sync"
sidebar_position: 2
---

Скористайтеся цим посібником, якщо ваша хост-машина працює під керуванням Windows 11 і ви хочете запускати Neat SDK із DevKit-Sync.

## Необхідні умови

- Хост із Windows 11.
- [WSL](https://learn.microsoft.com/en-us/windows/wsl/install), встановлений і робочий.
- Docker Engine %%dependency.docker.minimum%% або новішої версії, встановлений у WSL. Перевірте версію командою `docker --version`.
- `sima-cli`, встановлений у WSL.

## Починайте з WSL

Виконуйте команди Palette SDK у своєму дистрибутиві Linux у WSL, а не в PowerShell чи Command Prompt. Це стосується й команди `sima-cli neat install sdk@v%%release.sdk%%`.

## Мережевий режим WSL

Налаштуйте `%UserProfile%\\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Потім перезапустіть WSL:

```powershell
wsl --shutdown
```

Це дає WSL змогу спільно використовувати мережеву конфігурацію хоста, що полегшує взаємодію DevKit-Sync і NFS.

## Рекомендована топологія: пряме з’єднання Windows із DevKit

Для хостів із Windows рекомендована топологія — пряме з’єднання USB/Ethernet між комп’ютером з Windows і DevKit. Зазвичай його простіше налаштувати, ніж розміщувати DevKit у ширшій спільній мережі, а зміни брандмауера Windows можна обмежити локальним інтерфейсом, до якого під’єднано DevKit, а не всією мережею. На відміну від Ubuntu і macOS, для Windows надавайте перевагу саме прямому підключенню, якщо тільки у вашому середовищі ще немає перевірених правил брандмауера й мережевих правил WSL для спільної мережі.

Використовуйте Internet Connection Sharing (ICS), коли DevKit потрібно спільно використовувати мережеве з’єднання комп’ютера з Windows через пряме підключення.

1. Під’єднайте комп’ютер з Windows до інтернету через Wi-Fi або інший висхідний інтерфейс.
2. Під’єднайте DevKit до комп’ютера з Windows через адаптер USB/Ethernet.
3. На DevKit залиште для підключеного мережевого інтерфейсу режим `DHCP`.
4. У Windows відкрийте `Control Panel > Network and Internet > Network Connections`.
   Також можна натиснути `Win + R`, виконати `ncpa.cpl` і натиснути Enter.
5. Клацніть правою кнопкою миші адаптер, підключений до інтернету, і виберіть `Properties`.
6. Відкрийте вкладку `Sharing`.
7. Увімкніть `Allow other network users to connect through this computer's Internet connection`.
8. У `Home networking connection` виберіть адаптер USB/Ethernet, до якого під’єднано DevKit.
9. Застосуйте зміни, а потім, якщо DevKit не отримує IPv4-адресу, повторно під’єднайте адаптер, до якого під’єднано DevKit.

Після ввімкнення ICS Windows зазвичай призначає спільному адаптеру адресу з мережі `192.168.137.0/24`. Знайдіть IPv4-адресу DevKit з WSL або з консолі DevKit, а потім перевірте доступ через SSH з WSL:

```bash
ssh sima@<devkit-ip>
```

Потім із WSL перейдіть до під’єднання DevKit:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

:::note Доступ до Insight у Windows із прямим з’єднанням
Коли Windows спільно використовує мережу через пряме з’єднання, брандмауер Windows і особливості переадресації портів у WSL можуть не дати іншим комп’ютерам у мережі отримати доступ до вебінтерфейсу Neat Insight. У такій конфігурації відкривайте Insight безпосередньо на хості Palette SDK із Windows, наприклад за адресою `https://localhost:9900`.
:::

## Правила брандмауера для NFS (PowerShell)

Дозвольте трафік NFS у брандмауері Windows. Запустіть PowerShell від імені адміністратора й додайте правила для потрібних портів/протоколів NFS за допомогою `New-NetFirewallRule`.

Приклад шаблону:

```powershell
New-NetFirewallRule -DisplayName "Allow NFS TCP 2049" -Direction Inbound -Protocol TCP -LocalPort 2049 -Action Allow
New-NetFirewallRule -DisplayName "Allow NFS UDP 2049" -Direction Inbound -Protocol UDP -LocalPort 2049 -Action Allow
```

Додайте всі інші порти, яких потребує конфігурація вашого NFS-сервера/клієнта.

## Наступний крок

Поверніться до [Palette SDK](/getting-started/dev-environment/) і продовжте встановлення/налаштування.
