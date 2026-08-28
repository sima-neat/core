---
title: "Neat Library"
description: "Встановіть або оновіть SiMa.ai Neat Library на DevKit або всередині Neat SDK."
sidebar_position: 1
---

Neat Library — це API для мов C++ і Python, який використовується для запуску моделей і створення програм для Modalix.

:::tip Ви використовуєте Neat SDK? Тоді ви можете пропустити цей розділ.
Контейнер Neat SDK постачається з попередньо встановленою Neat Library, а під час налаштування, якщо ви під’єднаєте DevKit, автоматично встановлюється відповідна Neat Library та пакет PyNeat на DevKit. Якщо ви встановили Neat SDK — незалежно від того, чи під’єднали ви DevKit чи ні, — у вас вже є Neat Library, і ви можете пропустити цей розділ, щоб почати роботу.

Використовуйте цей розділ лише тоді, коли ви хочете оновити Neat Library незалежно від SDK або налаштувати її безпосередньо на DevKit або на хості без Neat SDK.
:::

Ви можете оновлювати Neat Library окремо, коли вам потрібна новіша сумісна версія, не перевстановлюючи контейнер SDK. Щодо підтримуваних комбінацій, див. [Посібник зі сумісності](/getting-started/compatibility/).

<div class="overview-section-label">Наступні кроки</div>

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Налаштування бібліотеки</h2>
    <p>Встановіть Neat Library, налаштуйте PCIe-хост, використовуйте Python-бібліотеки або перевірте середовище через командний рядок.</p>
    <ul class="overview-link-list neat-library-setup-grid">
      <li><a class="overview-link-card" href="/getting-started/neat-library/install-or-update/"><strong>Встановіть або оновіть Neat Library.</strong><span>Встановіть пакет sima-neat, компоненти GStreamer і стандартне середовище PyNeat.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/pcie-host/"><strong>Встановіть хост PCIe.</strong><span>Встановіть пакет PCIe для хоста, щоб запускати програми спільної обробки даних із підключеною платою PCIe Modalix.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/pyneat/"><strong>Встановіть PyNeat.</strong><span>Використовуйте пакет PyNeat у спеціально налаштованому віртуальному середовищі Python.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/neat-cli/"><strong>Neat CLI</strong><span>Перевірте встановлені версії, проаналізуйте стан SDK та оновіть встановлені компоненти.</span></a></li>
    </ul>
  </section>
</div>

## Наступний крок

Почніть з [Встановіть або оновіть Neat Library.](/getting-started/neat-library/install-or-update/).
