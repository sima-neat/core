---
title: "Розширені концепції"
description: "Подробиці проєктування для створення складніших застосунків Neat"
sidebar_position: 1
slug: /develop-apps/advanced-concepts/
---

# Розширені концепції

Використовуйте ці сторінки, коли вашому застосунку потрібно більше, ніж базовий робочий процес із `Model`, `Graph` і `Run`. Вони пояснюють контракти й поведінку середовища виконання, на яких ґрунтуються складніші застосунки Neat.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-app">
    <h2>Проєктування застосунків</h2>
    <p>Визначте, як компонується ваш застосунок і як він надає вихідні дані.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/graphs/"><strong>Графи</strong><span>Компонуйте моделі, вузли, іменовані входи й виходи, гілки та запуски.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mipi-camera-input/"><strong>Використання камери MIPI</strong><span>Побудуйте граф від камери до моделі з власним джерелом, входом libcamera та попередньою обробкою на CVU.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/usb-camera-input/"><strong>Використання USB-камери</strong><span>Зчитуйте вебкамеру UVC MJPEG за її стабільним ідентифікатором, декодуйте її кадри на місці або пересилайте її JPEG.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/video_sender/"><strong>Передавання відео</strong><span>Передавайте вихідне відео застосунку Neat потоком H.264 через RTP/UDP.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/metadata_sender/"><strong>Надсилання метаданих JSON</strong><span>Публікуйте структуровані метадані застосунку у форматі JSON через UDP.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Модель виконання</h2>
    <p>Зрозумійте, як робота планується, розподіляється між потоками та відображається на рівень конвеєра.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/timing_model/"><strong>Часові характеристики</strong><span>Зрозумійте синхронне й асинхронне виконання, push/pull і те, коли виконується робота.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/threading/"><strong>Модель багатопоточності</strong><span>Дізнайтеся, які потоки існують і де може виконуватися код застосунку.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/gstreamer_layer/"><strong>Рівень GStreamer</strong><span>Дізнайтеся, що абстрагує Neat і коли важливі деталі самого GStreamer.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Контракти даних і моделей</h2>
    <p>Зрозумійте контракти тензорів, пам’яті та моделей, на які спираються конвеєри.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/data_formats/"><strong>Формати даних</strong><span>Зіставляйте токени форматів із розкладкою, формою та семантикою площин тензора.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/dtype_contract/"><strong>Контракт dtype</strong><span>Простежте, як змінюється точність тензора під час попередньої обробки, на MLA і під час постобробки.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/memory_model/"><strong>Модель пам’яті</strong><span>Зрозумійте буфери без копіювання, фізичні адреси та поведінку кешу.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-start">
    <h2>Середовище виконання моделі</h2>
    <p>Заглибтеся в артефакти скомпільованих моделей і апаратні бекенди, які їх виконують.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mpk_contract/"><strong>Контракт MPK</strong><span>Дізнайтеся, як скомпільовані архіви моделей визначають етапи й контракти інференсу.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/processor_backends/"><strong>Процесорні бекенди</strong><span>Зрозумійте ролі A65, CVU, MLA, MLASHM, APU, TVM і M4.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/cvu_kernels/"><strong>Ядра CVU</strong><span>Перегляньте складові графів попередньої обробки та постобробки.</span></a></li>
    </ul>
  </section>
</div>
