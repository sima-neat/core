---
title: "Примітки щодо хоста Windows."
description: "Підготуйте хост із Windows 11 для Neat SDK та DevKit-Sync."
sidebar_position: 2
---

Використовуйте цей посібник, якщо ваша хост-машина працює під керуванням Windows 11, і ви хочете запустити середовище розробки Neat (далі – Neat SDK) із синхронізацією DevKit.

## Необхідні умови.

- Windows 11, хост.
- [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) встановлено та працює.
- Docker Engine встановлено всередині WSL.
- `sima-cli` встановлено в середовищі WSL.

## Почніть з WSL.

Виконуйте команди Neat SDK у середовищі вашої дистрибуції Linux WSL, а не в PowerShell або командному рядку. Це стосується і команди `sima-cli neat install sdk@release-2.1`.

## Режим мережі WSL.

Налаштуйте `%UserProfile%\\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Потім перезапустіть WSL:

```powershell
wsl --shutdown
```

Це дозволяє WSL використовувати спільну мережеву конфігурацію хоста, що полегшує синхронізацію та зв’язок через NFS для DevKit.

## Рекомендована топологія: пряме з’єднання Windows із DevKit.

Для хостів Windows рекомендується використовувати пряме USB/Ethernet-з’єднання між комп’ютером Windows і DevKit. Зазвичай налаштування такого з’єднання простіше, ніж розміщення DevKit у ширшій загальній мережі, і зміни в брандмауері Windows можна обмежити локальним інтерфейсом, що використовується для зв’язку з DevKit, замість застосування до всієї мережі. На відміну від Ubuntu та macOS, для Windows рекомендується використовувати саме це пряме з’єднання, якщо ваша мережа вже не має перевірених правил брандмауера та мережевих налаштувань WSL для загальної мережі.

Використовуйте функцію «Спільний доступ до Інтернет-з’єднання» (ICS), коли DevKit потрібно надати доступ до мережевого з’єднання комп’ютера Windows через пряме з’єднання.

1. Під’єднайте комп’ютер з Windows до Інтернету через Wi-Fi або інший мережевий інтерфейс.
2. Під’єднайте DevKit до комп’ютера з операційною системою Windows за допомогою USB/Ethernet-адаптера.
3. На DevKit залиште налаштування підключеного мережевого інтерфейсу в режимі `DHCP`.
4. У Windows відкрийте `Control Panel > Network and Internet > Network Connections`.
   Ви також можете натиснути `Win + R`, ввести `ncpa.cpl` і натиснути Enter.
5. Клацніть правою кнопкою миші на адаптері, який підключений до Інтернету, а потім виберіть `Properties`.
6. Відкрийте вкладку «`Sharing`».
7. Увімкніть `Allow other network users to connect through this computer's Internet connection`.
8. У `Home networking connection`, виберіть USB/Ethernet-адаптер, до якого підключено пристрій. DevKit.
9. Застосуйте зміни, а потім повторно під’єднайте адаптер, що підключається до DevKit, якщо DevKit не отримує сигнал.
   IPv4-адреса.

Після ввімкнення ICS, Windows зазвичай призначає спільному адаптеру адресу в мережі `192.168.137.0/24`.
Знайдіть IPv4-адресу DevKit у WSL або в консолі DevKit, а потім перевірте доступ через SSH з WSL:

```bash
ssh sima@<devkit-ip>
```

Потім продовжіть налаштування з’єднання з DevKit через WSL:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

:::note Insight: доступ до прямих посилань у Windows.
Завдяки функції прямого мережевого обміну у Windows, брандмауер Windows і поведінка перенаправлення портів WSL можуть перешкоджати доступу до веб-інтерфейсу Neat Insight з інших машин у мережі. У цьому випадку відкрийте Insight безпосередньо на хості Windows Neat SDK, наприклад, за адресою `https://localhost:9900`.
:::

## Правила брандмауера NFS (PowerShell)

Дозвольте трафік, пов’язаний з NFS, у брандмауері Windows. Запустіть PowerShell з правами адміністратора та додайте правила для необхідних портів/протоколів NFS, використовуючи `New-NetFirewallRule`.

Приклад:

```powershell
New-NetFirewallRule -DisplayName "Allow NFS TCP 2049" -Direction Inbound -Protocol TCP -LocalPort 2049 -Action Allow
New-NetFirewallRule -DisplayName "Allow NFS UDP 2049" -Direction Inbound -Protocol UDP -LocalPort 2049 -Action Allow
```

Додайте будь-які додаткові порти, необхідні для вашої конфігурації сервера/клієнта NFS.

## Наступний крок.

Поверніться до [Neat SDK](/getting-started/dev-environment/) і продовжуйте встановлення/налаштування.
