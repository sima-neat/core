---
title: "Neat Library"
description: "Встановлення або оновлення SiMa.ai Neat Library на DevKit або всередині Palette SDK"
sidebar_position: 1
---

Neat Library — це API для C++ і Python, призначений для запуску моделей і створення застосунків для Modalix.

:::tip Використовуєте Palette SDK? Цей розділ можна пропустити
Контейнер Palette SDK постачається з уже встановленою Neat Library, а під’єднання DevKit під час налаштування також автоматично встановлює на DevKit відповідну Neat Library і вбудоване середовище PyNeat. Якщо ви встановили Palette SDK — незалежно від того, під’єднали ви DevKit чи ні, — у вас уже є Neat Library, і ви можете пропустити цей розділ та почати роботу.

Використовуйте цей розділ лише тоді, коли хочете оновити Neat Library окремо від SDK або налаштувати її безпосередньо на DevKit чи на «чистому» хості без Palette SDK.
:::

Ви можете оновити Neat Library окремо, коли потрібна новіша сумісна версія без перевстановлення контейнера SDK. Підтримувані комбінації див. у [Посібнику із сумісності](/getting-started/compatibility/).

<div class="overview-section-label">Наступні кроки</div>

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Налаштування бібліотеки</h2>
    <p>Встановіть Neat Library, налаштуйте PCIe-хост, використовуйте прив’язки Python або перевіряйте середовище з командного рядка.</p>
    <ul class="overview-link-list neat-library-setup-grid">
      <li><a class="overview-link-card" href="/getting-started/neat-library/install-or-update/"><strong>Встановлення або оновлення Neat Library</strong><span>Встановіть пакет sima-neat, компоненти GStreamer і середовище PyNeat за замовчуванням.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/pcie-host/"><strong>Встановлення хоста PCIe</strong><span>Встановіть пакет для PCIe-хоста, щоб запускати застосунки співобробки з під’єднаною картою Modalix PCIe.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/pyneat/"><strong>Встановлення PyNeat</strong><span>Використовуйте wheel-пакет PyNeat у власному віртуальному середовищі Python.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/neat-cli/"><strong>Neat CLI</strong><span>Перевіряйте встановлені версії, стан SDK та оновлюйте встановлені компоненти.</span></a></li>
    </ul>
  </section>
</div>

## Наступний крок

Почніть із розділу [Встановлення або оновлення Neat Library](/getting-started/neat-library/install-or-update/).
