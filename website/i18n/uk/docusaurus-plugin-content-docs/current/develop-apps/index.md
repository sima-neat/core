---
title: "Розробка застосунків"
description: "Створюйте та запускайте застосунки штучного інтелекту на Modalix за допомогою SiMa.ai Neat"
sidebar_position: 1
---

# Розробка застосунків із SiMa.ai Neat

<LanguageContent lang="cpp">

<div className="overview-workflow-image overview-workflow-image-light">

![Робочий процес складеного застосунку SiMa.ai Neat для C++](@site/../docs/develop-apps/images/neat-overview-workflow-cpp.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![Робочий процес складеного застосунку SiMa.ai Neat для C++](@site/../docs/develop-apps/images/neat-overview-workflow-cpp-dark.svg)

</div>

</LanguageContent>

<LanguageContent lang="py">

<div className="overview-workflow-image overview-workflow-image-light">

![Робочий процес складеного застосунку SiMa.ai Neat для Python](@site/../docs/develop-apps/images/neat-overview-workflow-python.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![Робочий процес складеного застосунку SiMa.ai Neat для Python](@site/../docs/develop-apps/images/neat-overview-workflow-python-dark.svg)

</div>

</LanguageContent>

## Що таке SiMa.ai Neat

SiMa.ai Neat — це фреймворк для розробки застосунків: з ним створюють і
запускають застосунки штучного інтелекту на платформі SiMa.ai. Він надає API для Python і C++, щоб
завантажувати й запускати скомпільовані архіви моделей (`.tar.gz`), складати
наскрізні застосунки, які використовують обчислювальні ресурси Modalix, і
керувати виконанням.

У ширшому програмному стеку SiMa.ai фреймворк SiMa.ai Neat займає рівень застосунків.
Він працює поверх стеку середовища виконання SiMa.ai і використовує GStreamer
на нижчому рівні. Завдяки цьому розробники можуть зосередитися на логіці застосунку,
а не на ручному з’єднанні компонентів середовища виконання нижчого рівня.

Найкоротший шлях до інференсу — завантажити скомпільований архів моделі як
`Model` і запустити його напряму. Коли застосунку потрібні кілька входів, етапів
обробки, моделей або виходів, складіть ці компоненти в `Graph` і побудуйте з
нього `Run`. Ті самі публічні API підтримують і традиційну, і агентну розробку,
тож команди можуть переглядати, розширювати й підтримувати застосунки в
будь-якому з цих процесів.

### Виберіть модель розгортання

- **Запуск на Modalix DevKit** — застосунок і граф Neat працюють безпосередньо
  на пристрої. Використовуйте `simaai::neat` або `pyneat` із `Model`, `Graph`, `Node` і
  `Run`. Почніть зі сторінки [Запуск / інференс](/develop-apps/development-workflow/overview/).
- **Співобробка на PCIe-карті Modalix** — застосунок працює на хост-машині й
  надсилає тензори або зображення на карту для виконання моделі. Використовуйте
  `simaai::neat::pcie` або `pyneatpcie`. Почніть зі сторінки
  [Співобробка через PCIe](/develop-apps/development-workflow/pcie-model/).

### C++ чи PyNeat

Для застосунків, які працюють безпосередньо на Modalix DevKit, SiMa.ai Neat
надає той самий основний робочий процес через два мовні інтерфейси, тож ви
можете вибрати той, що підходить вашому застосунку:

- **PyNeat** — прив’язки для Python (`pyneat`). Найкраще підходять для швидких ітерацій, блокнотів, процесів аналізу даних і запуску застосунків Python безпосередньо на DevKit.
- **C++** — нативний API `simaai::neat`. Найкраще підходить для більших застосунків, тісної інтеграції з наявним кодом на C++ і процесів, де код крос-компілюється на хості для DevKit.

Обидва варіанти використовують ті самі скомпільовані артефакти моделей і
середовище виконання Modalix; поняття й сторінки нижче стосуються обох.
Співобробка через PCIe має окремі інтерфейси C++ і Python: `simaai::neat::pcie`
і `pyneatpcie`.

## Розробіть застосунок. <span className="neat-heading-highlight">SiMa.ai Neat розподілить його за вас.</span>

Modalix об’єднує в одній системі на кристалі (SoC) ядра для застосунків,
обробку зображень, прискорення машинного навчання, відеомодулі, спільну пам’ять
і високошвидкісне введення-виведення. Через свої API для Python і C++ SiMa.ai Neat
надає єдину модель програмування для створення застосунків на всіх обчислювальних
ресурсах системи, важливих для застосунку.

Побудуйте наскрізний шлях від камери або мережевого потоку через обробку та
інференс до кінцевого результату. SiMa.ai Neat будує конвеєр середовища
виконання, вибирає прискорені реалізації там, де це можливо, і координує виконання
та переміщення даних у Modalix. Ви зосереджуєтеся на застосунку, а SiMa.ai Neat
бере на себе складність апаратного забезпечення й середовища виконання.

<div className="overview-workflow-image modalix-application-map-desktop">

![Застосунок SiMa.ai Neat, розподілений на плані кристала MLSoC Modalix](@site/../docs/images/neat-modalix-floorplan-animated.svg)

</div>

<div className="overview-workflow-image modalix-application-map-mobile">

![Мобільне подання застосунку SiMa.ai Neat, розподіленого на плані кристала MLSoC Modalix](@site/../docs/images/neat-modalix-floorplan-mobile-animated.svg)

</div>

<p className="overview-figure-caption"><strong>Умовна схема розподілу:</strong> вибраний маршрут залежить від застосунку, моделі та доступного апаратного прискорення. Технічний опис розподілу див. на сторінці <a href="/develop-apps/advanced-concepts/processor_backends/">Процесорні бекенди</a>.</p>

## Опишіть застосунок. <span className="agentic-heading-highlight">Агент із навичками Neat його розробить.</span>

SiMa.ai Neat підтримує агентну розробку застосунків одразу після встановлення
завдяки навичкам, що входять до Palette SDK. Ці навички дають агентам для
програмування контекст, щоб використовувати публічні API Python і C++,
дотримуватися усталених шаблонів застосунків і працювати з процесом розробки
та перевірки на Modalix.

Рекомендований агентний шлях дає змогу створити застосунок, запустити його на
під’єднаному Modalix DevKit, переглянути результати й діагностику та вдосконалити
реалізацію. Традиційна розробка лишається паралельним шляхом для прямого
керування через ті самі API. Обидва шляхи дають стандартні застосунки SiMa.ai Neat,
які можна перевірити, тож ви можете переглядати або змінювати код, написаний агентом,
і переходити між двома процесами в міру розвитку застосунку. Щоб увімкнути агентну
розробку, див. [Налаштування Palette SDK](/getting-started/dev-environment/).

<div className="overview-workflow-image agentic-visual-desktop">

![Агент для програмування створює, запускає, діагностує та вдосконалює застосунок SiMa.ai Neat](@site/../docs/images/agentic-development-loop-animated.svg)

</div>

<div className="overview-workflow-image agentic-visual-mobile">

![Мобільне подання агентного циклу розробки SiMa.ai Neat](@site/../docs/images/agentic-development-loop-mobile-animated.svg)

</div>

## Вимоги

Перш ніж створювати застосунки, завершіть налаштування з розділу «Початок роботи»:

- **Встановлення для моделі розгортання** — для Modalix DevKit встановіть Neat
  Library у Palette SDK або безпосередньо на пристрій. Для співобробки через PCIe
  встановіть `core/pciehost` на хост-машину, а сумісну Neat Library — на
  PCIe-карту Modalix.
- **Артефакт моделі** — візьміть попередньо скомпільовану модель із Model Zoo або скомпілюйте власну модель в архів, готовий для Modalix.
- **Ціль виконання** — запускайте нативні застосунки на DevKit або збирайте й
  запускайте застосунок співобробки безпосередньо на хост-машині. Під’єднайте й
  синхронізуйте DevKit, коли крос-компілюєте нативні застосунки C++ у Palette SDK.

Сторінки «Знайомство з Neat» допоможуть виконати перший інференс, сторінки «Процес розробки» докладніше пояснюють основні поняття, а навчальні матеріали показують, як застосувати їх до реальних шаблонів застосунків.

Повні застосунки, які можна вивчити, адаптувати й запустити, дивіться серед [прикладів застосунків](https://developer.sima.ai/examples).

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Почніть тут</h2>
    <p>Почніть із налаштованого середовища й поступово опануйте основний робочий процес застосунку SiMa.ai Neat.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Знайомство з Neat</strong><span>Запустіть мінімальний застосунок Neat і перевірте цикл розробки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/development-workflow/overview/"><strong>Процес розробки</strong><span>Докладніше вивчіть робочий процес `Model`, `Graph` і `Run`.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>Навчальні матеріали</strong><span>Пройдіть покрокові приклади реальних шаблонів застосунків SiMa.ai Neat.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-explore">
    <h2>Більше можливостей</h2>
    <p>Перейдіть до цих розділів, коли будете готові створювати складніші застосунки або вивчати API.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/"><strong>Розширені концепції</strong><span>Зрозумійте графи, формати, пам’ять, потоки й поведінку середовища виконання.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>Довідка</strong><span>Переглядайте матеріали з C++, Python, Model Compiler, усунення несправностей і допоміжні матеріали.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>Участь у розробці</strong><span>Зрозумійте архітектуру, збирання з початкового коду, вимоги до тестування й домовленості репозиторію.</span></a></li>
    </ul>
  </section>
</div>
