---
title: "Участь у розробці"
description: "Посібник для учасників розробки: архітектура фреймворку Neat, збірка, тести та робота над випусками"
sidebar_position: 1
slug: /develop-apps/contribute/
---

# Участь у розробці

Цей розділ призначено для учасників, які змінюють саму Neat Library. Тут пояснено, як організовано репозиторій, як збирати й тестувати зміни і які
контракти мають залишатися стабільними для розробників застосунків.

<div class="overview-section-label">Вступ для учасників</div>

**Neat Library** — це бібліотека C++/Python і середовище виконання з цього репозиторію.
Вона завантажує архіви моделей, компонує та перевіряє конвеєри, працює на апаратному забезпеченні Modalix і надає публічний API для застосунків. Palette SDK і DevKit Sync
утворюють довкола неї робочий процес розробки.

Змінюючи цей репозиторій, орієнтуйтеся на ті властивості фреймворку, які допомагають і людям, і розробці за участю агентів: явні API, детермінована поведінка, структурована діагностика, сувора перевірка та стабільні публічні контракти.

<div class="overview-section-label">Принципи для учасників</div>

- **Детермінованість понад усе** — назви елементів, згенеровані конвеєри, звіти й тести мають бути відтворюваними.
- **Зручність налагодження — першочергова вимога** — збої мають давати структуровані дані, а не лише рядки.
- **Без тихого переходу на резервний варіант** — не приховуйте помилок у вхідних даних моделі та збоїв апаратного забезпечення чи середовища виконання.
- **Перевірка перед запуском** — виявляйте структурні помилки, помилки caps, форми та контрактів ще до виконання.
- **Публічні API залишаються стабільними** — зміни у встановлених заголовних файлах у `include/*` мають лише доповнювати API і зберігати сумісність.
- **Паралельність має бути обмеженою та спостережуваною** — завершення роботи не повинно зависати, а діагностика має бути потокобезпечною.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Почніть тут</h2>
    <p>Перш ніж змінювати код, ознайомтеся з репозиторієм, правилами іменування та вимогами.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>Архітектура</strong><span>Вивчіть структуру репозиторію, розподіл відповідальності за модулі, логіку роботи середовища виконання та точки розширення.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/naming/"><strong>Контракт іменування</strong><span>Послідовно використовуйте канонічні назви продукту, API, пакетів і типів.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/coding_standard/"><strong>Стандарт кодування</strong><span>Дотримуйтеся вимог щодо стилю C++, публічного API, сумісності та документації.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>Збірка й тестування</h2>
    <p>Зберіть фреймворк, перевірте зміни та попрацюйте з прив’язками для Python.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/build/"><strong>Збірка</strong><span>Зберіть Neat з початкового коду та виберіть профілі CMake або перемикачі збірки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/test_requirements/"><strong>Вимоги до тестування</strong><span>З’ясуйте, які тести й оновлення документації очікуються для кожного типу змін.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/python_bindings/"><strong>Прив’язки для Python</strong><span>Збирайте, тестуйте й пакуйте прив’язки PyNeat як учасник розробки.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Контракти та внутрішня реалізація</h2>
    <p>Звертайтеся до цих сторінок, коли змінюєте архіви моделей, контракти плагінів або внутрішнє пакування.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/mpk_contract/"><strong>Контракт MPK</strong><span>Ознайомтеся з правилами приймання, перевірки та безпеки архівів моделей.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/sima_plugin_json_truth_map/"><strong>Карта істини JSON для плагінів</strong><span>Перегляньте зафіксовану карту контракту JSON для плагінів SIMA.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Випуск і супровід</h2>
    <p>Звертайтеся до цих сторінок, коли потрібні умови допуску випуску, плани впорядкування коду та довгострокові рекомендації щодо проєктування.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/release-checklist/"><strong>Контрольний список випуску</strong><span>Дотримуйтеся умов, що блокують випуск, обов’язкових перевірок і відтворюваних кроків.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/error-taxonomy-rollout/"><strong>Впровадження таксономії помилок</strong><span>Відстежуйте перехід на структуровані коди помилок і стан його перевірки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/agentic-workflow/"><strong>Агентний робочий процес</strong><span>Дізнайтеся, чому API побудовано з розрахунком на розробку як людьми, так і за допомогою ШІ.</span></a></li>
    </ul>
  </section>
</div>
