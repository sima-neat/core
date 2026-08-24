---
title: "Розширені концепції"
description: "Деталі проєктування для створення більш функціональних застосунків Neat."
sidebar_position: 1
slug: /develop-apps/advanced-concepts/
---

# Розширені концепції

Використовуйте ці сторінки, коли вашій програмі потрібні можливості, що перевищують базовий набір `Model`, `Graph` та `Run` для робочого процесу. Вони пояснюють структуру та поведінку в середовищі виконання, що лежать в основі більш складних програм Neat.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-app">
    <h2>Проєктування застосунку</h2>
    <p>Розробіть структуру вашої програми та визначте, як вона надаватиме результати.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/graphs/"><strong>Графи</strong><span>Створюйте моделі, вузли, іменовані вхідні та вихідні дані, гілки та сценарії виконання.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mipi-camera-input/"><strong>Використовуйте камеру MIPI.</strong><span>Створіть граф, що відображає зв’язок між даними з камери та моделлю, використовуючи вхідні дані від libcamera та попередню обробку CVU.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/video_sender/"><strong>Надіслати відео</strong><span>Транслюйте відеопотік з програми Neat за допомогою протоколів H.264 RTP/UDP.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/metadata_sender/"><strong>Надсилайте метадані у форматі JSON.</strong><span>Публікуйте структуровані метадані застосунку за допомогою UDP JSON.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Модель виконання</h2>
    <p>Зрозумійте, як організовано виконання завдань, як вони розподіляються між потоками та як узгоджуються в межах шару конвеєра.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/timing_model/"><strong>Модель часових параметрів</strong><span>Зрозумійте принципи синхронного та асинхронного виконання, механізми надсилання/отримання даних, а також визначте, коли виконуються певні завдання.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/threading/"><strong>Модель багатопоточності</strong><span>З’ясуйте, які потоки існують і де може виконуватися код програми.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/gstreamer_layer/"><strong>GStreamer шар.</strong><span>Дізнайтеся, які аспекти абстрагує Neat і коли важливі деталі «сирого» GStreamer.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Договори щодо даних та моделей</h2>
    <p>Зрозумійте структуру тензорів, пам’яті та моделей, на яких базуються конвеєри.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/data_formats/"><strong>Формати даних</strong><span>Відображайте маркери формату карти на структуру, розмір і семантику площини тензора.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/dtype_contract/"><strong>Тип даних, що використовується в контракті.</strong><span>Відстежуйте, як змінюється точність тензора на різних етапах: попередньої обробки, обробки за допомогою MLA та подальшої обробки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/memory_model/"><strong>Модель пам’яті</strong><span>Зрозумійте, що таке буфери без копіювання, фізичні адреси та принципи роботи кешу.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-start">
    <h2>Середовище виконання моделі.</h2>
    <p>Дослідіть детальніше артефакти скомпільованої моделі та апаратні платформи, на яких вони виконуються.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mpk_contract/"><strong>Контракт MPK</strong><span>Перегляньте, як скомпільовані архіви моделей визначають етапи та умови виконання.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/processor_backends/"><strong>Процесорні модулі</strong><span>Зрозумійте, які функції виконують A65, CVU, MLA, MLASHM, APU, TVM та M4.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/cvu_kernels/"><strong>Ядра CVU</strong><span>Перегляньте попередню та кінцеву обробку складових графа.</span></a></li>
    </ul>
  </section>
</div>
