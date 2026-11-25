---
title: "Palette SDK"
sidebar_position: 1
---

# Palette SDK

Palette SDK — це набір інструментів SiMa.ai для розробки програмного забезпечення, призначений для створення застосунків штучного інтелекту на Modalix. Він містить середовище розробки, бібліотеку середовища виконання, інструменти для роботи з моделями та робочий процес перевірки на DevKit. Разом ці компоненти охоплюють увесь шлях розробки: підготувати модель, створити застосунок і перевірити результат на апаратному забезпеченні Modalix.

Скористайтеся цим оглядом, щоб зрозуміти основні частини Palette SDK і вибрати відповідний шлях налаштування, підготовки моделі або розробки застосунку.

![Програмний стек Palette SDK: середовище розробки, Neat Library, Model Compiler, LLiMa та Modalix DevKit](@site/../docs/images/neat-software-stack-animated.svg)

<div class="overview-section-label">Шлях розробника</div>

:::tip Якщо ви вперше працюєте з Palette SDK
Є два підтримувані способи розробки застосунків за допомогою Palette SDK:

- **[Використовуйте Palette SDK](/getting-started/dev-environment/)**, якщо вам потрібне продуктивніше середовище розробки,
  особливо якщо ви плануєте компілювати моделі або виконувати крос-компіляцію великих кодових баз C++.
- **[Розробляйте безпосередньо на DevKit](/getting-started/neat-library/)**, якщо вам потрібно менше рухомих частин у середовищі розробки, особливо коли ви не займаєтеся компіляцією моделей.

Якщо ви обираєте шлях SDK, переконайтеся, що ваш хост відповідає
[вимогам до хоста](/getting-started/dev-environment/#host-requirements), а потім встановіть SDK. Під час встановлення SDK застосовуються сумісні значення за замовчуванням, тож довідник із сумісності знадобиться вам лише тоді, коли потрібно зафіксувати точні версії, оновити компоненти окремо один від одного або усунути невідповідність версій.

**Швидкий шлях лише з SDK (без DevKit):** встановіть SDK, а потім одразу перейдіть до розділу
[Компіляція моделі](/compile-a-model/). Налаштування SDK, DevKit Sync, Neat Library та PyNeat — необов’язкові відгалуження, які можна пропустити, доки ви не під’єднаєте DevKit.
:::

<div class="overview-section-label">Як читати команди</div>

Блоки команд у посібниках із налаштування та розробки застосунків позначено й виділено кольором відповідно до середовища, у якому вони виконуються, тож вам ніколи не доведеться вгадувати, де їх вводити:

| Запрошення | Де виконувати |
| --- | --- |
| `host$` | На вашій власній машині, поза SDK. |
| `sdk$` | В оболонці контейнера Palette SDK. |
| `devkit$` | На Modalix DevKit. |
| `pcie-host$` | На хост-машині, у якій встановлено карту Modalix PCIe. |

Блок, позначений кількома середовищами, наприклад `sdk or devkit$`, виконується однаково в будь-якому з них.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Почніть тут</h2>
    <p>Підготуйте хост-машину, Palette SDK і DevKit до локальної розробки та перевірки на апаратному забезпеченні.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/"><strong>Palette SDK</strong><span>Використовуйте SDK для високопродуктивного робочого процесу на хості, компіляції моделей, збирання коду C++ і перевірки на DevKit.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/neat-library/"><strong>Neat Library</strong><span>Встановіть середовище виконання та PyNeat безпосередньо, якщо для прототипування застосунків на DevKit вам потрібно менше рухомих частин.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/compatibility/"><strong>Посібник із сумісності</strong><span>Довідник із підтримуваних комбінацій версій.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Підготовка моделі</h2>
    <p>Перетворюйте навчені моделі на артефакти для розгортання, які працюють на апаратному забезпеченні Modalix.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/compile-a-model/"><strong>Компіляція моделі</strong><span>Компілюйте попередньо навчені моделі комп’ютерного зору ONNX або моделі GenAI для Modalix.</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-zoo/"><strong>Попередньо скомпільована модель</strong><span>Швидко почніть роботу з готовим до запуску артефактом моделі.</span></a></li>
      <li><a class="overview-link-card" href="/genai-llima/"><strong>Генеративний ШІ з LLiMa</strong><span>Компілюйте, тестуйте та вимірюйте продуктивність моделей генеративного ШІ на Modalix.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>Створення застосунку</h2>
    <p>Використовуйте Neat Library, щоб запускати моделі й компонувати конвеєри виробничих застосунків.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Привіт, Neat!</strong><span>Запустіть свій перший застосунок Neat і перевірте робочий процес розробки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/"><strong>Розробка застосунків</strong><span>Створюйте застосунки ШІ за допомогою Neat Library на C++ або PyNeat.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>Навчальні посібники</strong><span>Проходьте покрокові приклади типових підходів до створення реальних застосунків Neat.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Інструменти та довідка</h2>
    <p>Звертайтеся до допоміжних інструментів і довідкових матеріалів, коли потрібно більше подробиць.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/tools/"><strong>Інструменти</strong><span>sima-cli, Model Zoo та Insight.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>Довідка</strong><span>Переглядайте API, матеріали з усунення несправностей, змінні середовища, формати даних і примітки до випуску.</span></a></li>
      <li><a class="overview-link-card" href="/reference/troubleshooting/"><strong>Усунення несправностей</strong><span>Знайдіть рішення проблем із налаштуванням, середовищем виконання та конвеєрами.</span></a></li>
    </ul>
  </section>
</div>
