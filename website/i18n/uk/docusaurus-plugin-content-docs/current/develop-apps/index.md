---
title: "Розробляйте застосунки."
description: "Створюйте та запускайте програми зі штучним інтелектом на платформі Modalix за допомогою SiMa.ai та Neat."
sidebar_position: 1
---

# Розробляйте застосунки за допомогою SiMa.ai Neat.

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

## Що таке SiMa.ai і Neat?

SiMa.ai Neat — це фреймворк для розробки застосунків, призначений для створення та запуску
застосунків штучного інтелекту на платформі SiMa.ai. Він надає API для Python і C++, які використовуються для
завантаження та запуску скомпільованих архівів моделей (`.tar.gz`), створення комплексних
застосунків, що використовують обчислювальні ресурси Modalix, і керування середовищем виконання.

У рамках ширшого програмного комплексу SiMa.ai, SiMa.ai Neat розміщується на рівні застосунків. Він базується на SiMa.ai середовищі виконання та використовує GStreamer на нижчому рівні, тому розробники можуть зосередитися на логіці застосунку, а не на ручному підключенні
компонентів середовища виконання нижчого рівня.

Для найшвидшого шляху до отримання результатів завантажте скомпільований архів моделі як `Model`
і запустіть його безпосередньо. Коли застосунку потрібні кілька вхідних даних, етапів обробки, моделей або вихідних даних, створіть ці компоненти у вигляді `Graph` і об’єднайте їх в `Run`. Ті самі загальнодоступні API підтримують як традиційну, так і агентну розробку,
тож команди можуть переглядати, розширювати та підтримувати застосунки, використовуючи будь-який з цих підходів.

### Оберіть модель розгортання.

- **Запустіть на Modalix DevKit** — застосунок і Neat граф запускаються безпосередньо на.
  пристрій. Використовуйте `simaai::neat` або `pyneat` разом із `Model`, `Graph`, `Node` та
`Run`. Почніть із [Запуск / здійснення обчислень](/develop-apps/development-workflow/overview/).
- **Використовуйте PCIe-карту Modalix для спільної обробки даних** — програма працює на
  хост-машина, яка надсилає тензори або зображення на карту для виконання моделі. Використовуйте `simaai::neat::pcie` або `pyneatpcie`. Почніть з [Копроцесор PCIe](/develop-apps/development-workflow/pcie-model/).

### C++ чи PyNeat

Для програм, які працюють безпосередньо на Modalix DevKit, SiMa.ai Neat забезпечує
однаковий базовий робочий процес через два мовні інтерфейси, тож ви можете обрати той, який найкраще підходить для вашої програми:

- **PyNeat** — Python-інтерфейс (`pyneat`). Найкраще підходить для швидкої розробки, використання в блокнотах, реалізації процесів обробки даних і безпосереднього запуску програм Python на DevKit.
- **C++** — нативний `simaai::neat` API. Найкраще підходить для великих застосунків, тісної інтеграції з існуючими кодовими базами C++ та міжплатформних робочих процесів, де код компілюється для хоста та DevKit.

Обидва рішення використовують однакові скомпільовані артефакти моделі та середовище виконання Modalix; наведені нижче концепції та сторінки стосуються обох. Копроцесор PCIe забезпечує окремі інтерфейси C++ і Python через `simaai::neat::pcie` і `pyneatpcie`.

## Розробіть застосунок. <span className="neat-heading-highlight">SiMa.ai Neat створить для вас карту.</span>

Modalix об’єднує обчислювальні ядра, обробку зображень, прискорення машинного навчання, відеомодулі, спільну пам’ять і високошвидкісний ввід-вивід в одному кристалі системи на чіпі (SoC).
Завдяки своїм API для Python і C++, SiMa.ai Neat надає єдину модель програмування для створення програм, які використовують відповідні обчислювальні ресурси в системі.

Створіть повноцінний конвеєр, починаючи від камери або мережевого потоку, через обробку та виведення результатів, до отримання кінцевого результату. SiMa.ai Neat створює конвеєр середовища виконання, вибирає прискорені реалізації, де це можливо, і координує виконання та переміщення даних у Modalix. Ви зосереджуєтесь на програмі, а SiMa.ai Neat обробляє базове апаратне забезпечення та складність середовища виконання.

<div className="overview-workflow-image modalix-application-map-desktop">

![Застосунок SiMa.ai Neat, відображений на плані MLSoC Modalix](@site/../docs/images/neat-modalix-floorplan-animated.svg)

</div>

<div className="overview-workflow-image modalix-application-map-mobile">

![Мобільне подання застосунку SiMa.ai Neat, відображеного на плані MLSoC Modalix](@site/../docs/images/neat-modalix-floorplan-mobile-animated.svg)

</div>

<p className="overview-figure-caption"><strong>Приклад відображення:</strong> обраний маршрут залежить від застосування, моделі та наявної апаратної підтримки для прискорення обчислень. Див. <a href="/develop-apps/advanced-concepts/processor_backends/">Процесорні модулі</a> для технічного відображення.</p>

## Опишіть вашу програму. <span className="agentic-heading-highlight">Агент, який володіє навичками Neat, розвиває [цей продукт/систему/інструмент].</span>

SiMa.ai і Neat підтримують розробку застосунків на основі агентів одразу після встановлення завдяки навичкам, що входять до складу Neat Development Environment (далі – Neat SDK). Ці навички надають кодуючим агентам контекст для використання загальнодоступних API Python і C++, дотримання встановлених шаблонів застосунків і роботи з робочим процесом розробки та __перевірки__ Modalix.

Рекомендований шлях розробки на основі агентів дозволяє створити застосунок, запустити його на з’єднаному Modalix DevKit, переглянути результати та діагностику, а також вдосконалити реалізацію. Традиційна розробка залишається паралельним шляхом для безпосереднього керування через ті самі API. Обидва підходи створюють стандартні застосунки SiMa.ai Neat, які можна перевірити, тому ви можете переглядати або змінювати код, розроблений агентами, і перемикатися між двома робочими процесами в міру розвитку застосунку. Див. [Налаштуйте Neat SDK.](/getting-started/dev-environment/), щоб увімкнути розробку на основі агентів.

<div className="overview-workflow-image agentic-visual-desktop">

![Агент програмування створює, запускає, діагностує та вдосконалює застосунок SiMa.ai Neat](@site/../docs/images/agentic-development-loop-animated.svg)

</div>

<div className="overview-workflow-image agentic-visual-mobile">

![Мобільне подання агентного циклу розробки SiMa.ai Neat](@site/../docs/images/agentic-development-loop-mobile-animated.svg)

</div>

## Вимоги

Перш ніж розпочинати розробку програм, завершіть початкове налаштування:

- **Встановлення для моделі розгортання** — для Modalix DevKit, встановіть Neat.
  Бібліотеку можна встановити в Neat SDK або безпосередньо на пристрій. Для спільної обробки даних через PCIe встановіть `core/pciehost` на хост-машині та сумісну Neat Library на платі Modalix PCIe.
- **Артефакт моделі** — використовуйте попередньо скомпільовану модель із Model Zoo або скомпілюйте власну модель у архів, готовий для використання з Modalix.
- **Цільова платформа для середовища виконання** — запускайте нативні програми на DevKit, або створіть і запустіть
  запуск програми спільної обробки безпосередньо на хост-машині. З’єднайте та синхронізуйте DevKit під час перехресної компіляції нативних програм C++ у Neat SDK.

Сторінки «Hello Neat!» допоможуть вам запустити першу модель для виведення результатів, сторінки «Робочий процес розробки» детальніше пояснюють основні концепції, а навчальні матеріали показують, як застосувати їх до реальних прикладів програм.

Щоб ознайомитися з повними прикладами програм, які можна вивчити, адаптувати та запустити, перегляньте [приклади застосування](https://developer.sima.ai/examples).

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Почніть тут.</h2>
    <p>Почніть з налаштованого робочого середовища та розробіть основний робочий процес застосунку SiMa.ai Neat.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Привіт, Neat!</strong><span>Запустіть мінімальну програму Neat і перевірте цикл розробки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/development-workflow/overview/"><strong>Процес розробки</strong><span>Детальніше ознайомтеся з процесом роботи `Model`, `Graph` та `Run`.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>Навчальні матеріали</strong><span>Дотримуйтесь наведених прикладів, які демонструють реальні сценарії використання SiMa.ai та Neat.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-explore">
    <h2>Створюйте більше.</h2>
    <p>Використовуйте ці розділи, коли будете готові створювати складніші застосунки або вивчати можливості API.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/"><strong>Розширені концепції</strong><span>Розумійте структуру графів, формати даних, використання пам’яті, роботу з потоками та поведінку в середовищі виконання.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>Посилання</strong><span>Перегляньте матеріали з C++, Python, Model Compiler, інструкції з усунення несправностей та додаткові матеріали.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>Зробіть внесок.</strong><span>Зрозумійте принципи архітектури, процес створення вихідного коду, вимоги до тестування та правила, що діють у репозиторії.</span></a></li>
    </ul>
  </section>
</div>
