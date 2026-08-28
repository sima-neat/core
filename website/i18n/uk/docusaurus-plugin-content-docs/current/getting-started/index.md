---
title: "Palette Neat"
sidebar_position: 1
---

# Palette Neat

Palette Neat — це набір інструментів SiMa.ai для розробки програмного забезпечення, призначений для створення програм штучного інтелекту на платформі Modalix. Він містить середовище розробки, бібліотеку середовища виконання, інструменти для роботи з моделями та робочий процес DevKit для перевірки. Разом ці компоненти підтримують повний цикл розробки: підготовка моделі, створення програми та перевірка результату на апаратному забезпеченні Modalix.

Використовуйте цей огляд, щоб зрозуміти основні частини Palette Neat і вибрати відповідну конфігурацію, процес підготовки моделі або шлях розробки програми.

![Набір програмного забезпечення Palette Neat, що демонструє Neat SDK, Neat Library, Model Compiler, LLiMa та Modalix DevKit.](@site/../docs/images/neat-software-stack-animated.svg)

<div class="overview-section-label">Шлях розробника</div>

:::tip Якщо ви вперше користуєтеся Palette Neat.
Існує два способи розробки програмних застосунків із використанням Palette Neat:

- **[Використовуйте Neat SDK.](/getting-started/dev-environment/)**, якщо вам потрібне більш продуктивне середовище розробки.
  особливо, якщо ви плануєте компілювати моделі або виконувати крос-компіляцію великих обсягів коду C++.
- **[Розробляйте безпосередньо на DevKit.](/getting-started/neat-library/)**, коли вам потрібно менше рухомих частин у середовищі розробки, особливо якщо ви не займаєтеся роботою з компіляції моделей.

Якщо ви обираєте шлях SDK, переконайтеся, що ваш хост відповідає
[вимоги до хоста](/getting-started/dev-environment/#host-requirements), а потім встановіть SDK. Під час встановлення SDK застосовуються сумісні параметри за замовчуванням, тому вам знадобиться лише довідник зі сумісності, якщо ви хочете фіксувати точні версії, оновлювати компоненти незалежно або усувати проблеми, пов’язані з несумісністю версій.

**Швидкий шлях лише для SDK (без DevKit):** встановіть SDK, а потім одразу перейдіть до
[Складіть модель.](/compile-a-model/). Налаштування SDK, DevKit Sync, бібліотеки Neat та PyNeat є необов’язковими додатковими кроками, які ви можете пропустити, поки не під’єднаєте DevKit.
:::

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Почніть тут.</h2>
    <p>Підготуйте ваш хост, Neat SDK, та DevKit для локальної розробки та апаратної перевірки.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/"><strong>Neat SDK</strong><span>Використовуйте SDK для створення високопродуктивного робочого процесу на основі хоста, компіляції моделей, створення збірок C++ та перевірки за допомогою DevKit.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/"><strong>Neat Library</strong><span>Встановіть середовище виконання та PyNeat безпосередньо, коли вам потрібно менше компонентів для створення прототипів застосунків на DevKit.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/compatibility/"><strong>Посібник зі сумісності</strong><span>Довідник із сумісних версій.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Підготовка моделі</h2>
    <p>Перетворіть навчені моделі на розгорнуті артефакти, які працюють на апаратному забезпеченні Modalix.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/compile-a-model/"><strong>Складіть модель.</strong><span>Зберіть попередньо навчені моделі ONNX для обробки зображень або моделі GenAI для Modalix.</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-zoo/"><strong>Використовуйте попередньо скомпільовану модель.</strong><span>Швидко розпочніть роботу з готовим до використання артефактом моделі.</span></a></li>
      <li><a class="overview-link-card" href="/genai-llima/"><strong>Генеративний ШІ з використанням LLiMa.</strong><span>Збирайте, тестуйте та оцінюйте продуктивність моделей генеративного штучного інтелекту на платформі Modalix.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>Створіть застосунок.</h2>
    <p>Використовуйте Neat Library для запуску моделей і створення конвеєрів для розробки готових до використання застосунків.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Привіт, Neat!</strong><span>Запустіть свою першу програму Neat і перевірте процес розробки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/"><strong>Розробляйте застосунки.</strong><span>Створюйте програми зі штучним інтелектом за допомогою Neat Library, використовуючи C++ або PyNeat.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>Навчальні матеріали</strong><span>Дотримуйтесь наведених прикладів, щоб ознайомитися з практичними способами застосування Neat.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Інструменти та довідкові матеріали</h2>
    <p>Використовуйте додаткові інструменти та довідкові матеріали, коли вам потрібна більш детальна інформація.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/tools/"><strong>Інструменти</strong><span>sima-cli, Model Zoo та Insight.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>Посилання</strong><span>Перегляньте інформацію про API, інструкції з усунення несправностей, змінні середовища, формати даних і примітки до випуску.</span></a></li>
      <li><a class="overview-link-card" href="/reference/troubleshooting/"><strong>Усунення несправностей</strong><span>Знайдіть рішення для проблем, пов’язаних із налаштуванням, середовищем виконання та конвеєром.</span></a></li>
    </ul>
  </section>
</div>
