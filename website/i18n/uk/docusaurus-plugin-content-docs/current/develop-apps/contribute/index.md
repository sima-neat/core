---
title: "Зробіть внесок."
description: "Посібник для розробників щодо архітектури фреймворку Neat, процесів збірки, тестування та підготовки до випуску."
sidebar_position: 1
slug: /develop-apps/contribute/
---

# Зробіть внесок.

У цьому розділі наведено інструкції для розробників, які вносять зміни безпосередньо до Neat Library. Тут пояснюється, як організовано репозиторій, як створювати та тестувати зміни, а також які
інтерфейси мають залишатися стабільними для розробників застосунків.

<div class="overview-section-label">Інструктаж для авторів</div>

**Neat Library** — це бібліотека C++/Python і середовище виконання, що міститься в цьому репозиторії.
Вона завантажує архіви моделей, створює та виконує перевірку конвеєрів, працює на апаратному забезпеченні Modalix і надає загальнодоступний API для програм. Neat SDK і DevKit Sync
є частиною навколишнього процесу розробки.

Під час внесення змін до цього репозиторію оптимізуйте його для властивостей фреймворку, які полегшують розробку як для людей, так і для систем, що використовують штучний інтелект: чіткі API, детермінована поведінка, структурована діагностика, сувора перевірка та стабільні загальнодоступні інтерфейси.

<div class="overview-section-label">Принципи для авторів</div>

- **Детермінованість перемагає** — забезпечте відтворюваність назв елементів, згенерованих конвеєрів, звітів і тестів.
- **Зручність налагодження є пріоритетною** — у разі виникнення помилок повинні генеруватися структуровані дані, а не лише текстові рядки.
- **Не приховуйте помилки** — не маскуйте помилки вхідних даних моделі або збої апаратного забезпечення/середовища виконання.
- **Перевіряйте перед запуском** — виявляйте структурні помилки, помилки у великих літерах, помилки у формі та помилки в умовах контракту до початку роботи в середовищі виконання.
- **Публічні API залишаються стабільними** — встановлені заголовні файли в каталозі `include/*` вимагають поступових змін, що забезпечують сумісність.
- **Паралельність має бути обмеженою та відстежуваною** — процес завершення не повинен призводити до зависання, а діагностичні інструменти повинні бути безпечними для використання в багатопотоковому середовищі.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Почніть тут.</h2>
    <p>Перш ніж змінювати код, ознайомтеся зі структурою репозиторію, правилами іменування та вимогами.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>Архітектура</strong><span>Вивчіть структуру репозиторію, розподіл відповідальності за модулі, логіку роботи в середовищі виконання та точки розширення.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/naming/"><strong>Найменування договору</strong><span>Послідовно використовуйте стандартні назви продуктів, API, пакетів і типів.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/coding_standard/"><strong>Стандарт кодування</strong><span>Дотримуйтеся вимог щодо стилю коду C++, публічного API, сумісності та документації.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>Створення та тестування</h2>
    <p>Створіть структуру, перевірте внесені зміни та попрацюйте над Python-зв’язками.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/build/"><strong>Створити</strong><span>Зберіть Neat з вихідного коду та виберіть профілі CMake або параметри збірки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/test_requirements/"><strong>Вимоги до тестування</strong><span>З’ясуйте, які саме тести та оновлення документації передбачаються для кожного типу змін.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/python_bindings/"><strong>Зв’язки з Python</strong><span>Створюйте, тестуйте та збирайте пакети PyNeat як учасник розробки.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Контракти та внутрішні процеси</h2>
    <p>Використовуйте їх під час зміни архівів моделей, контрактів плагінів або внутрішніх пакетів.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/mpk_contract/"><strong>Контракт MPK</strong><span>Зрозумійте правила щодо завантаження, перевірки та безпеки даних в архіві моделі.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/sima_plugin_json_truth_map/"><strong>Плагін «JSON Truth Map»</strong><span>Перегляньте заморожену JSON-схему контракту для плагіна SIMA.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/appcomplex_workspace_packaging/"><strong>Складна упаковка застосунків.</strong><span>Створіть і встановіть пакет служб для захищеного комплексного робочого середовища застосунку.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Випуск і технічне обслуговування</h2>
    <p>Використовуйте їх для визначення етапів випуску, розробки планів з усунення проблем і надання довгострокових рекомендацій щодо проєктування.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/release-checklist/"><strong>Перелік завдань перед випуском</strong><span>Дотримуйтеся умов, що блокують випуск, необхідних перевірок і відтворюваних кроків.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/error-taxonomy-rollout/"><strong>Впровадження класифікації помилок</strong><span>Відстежуйте процес структурованої міграції кодів помилок і перевіряйте статус їхньої перевірки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/agentic-workflow/"><strong>Організований робочий процес</strong><span>Дізнайтеся, чому структура API розроблена з урахуванням можливостей розробки як людьми, так і за допомогою штучного інтелекту.</span></a></li>
    </ul>
  </section>
</div>
