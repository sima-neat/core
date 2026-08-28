---
title: "Python-зв’язки (pyneat)"
description: "Створіть, протестуйте та запакуйте бібліотеки PyNeat як учасник розробки."
sidebar_position: 3
slug: /develop-apps/contribute/python_bindings
---

# Зв’язки для Python (`pyneat`)

Ця сторінка призначена для розробників і тих, хто підтримує проєкт `pyneat`.

`pyneat` — це шар зв’язування для Python, який використовується для SiMa.ai Neat. Він створений за допомогою `nanobind` і запакований за допомогою `scikit-build-core`.

Перегляньте [Довідник з API Python](/reference/pythonapi/modules/pyneat), щоб отримати доступ до згенерованої документації API.

## Необхідні умови

`pyneat` використовує ті самі нативні залежності, що й бібліотека C++, зокрема:

- Пакети для розробки та середовища виконання GStreamer.
- Пакети для розробки та середовища виконання OpenCV.
- Набір інструментів C++ (`cmake`, компілятор, `pkg-config`)

Див. [Створити](/develop-apps/contribute/build) для отримання інструкцій щодо налаштування хоста.

## Встановіть з вихідного коду.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Можливість редагування під час встановлення для розробки:

```bash
python -m pip install -e .[dev]
```

## Запустіть тести.

```bash
pytest -q python/tests
```

## Упаковка

Файл `pyproject.toml`, що розташований у кореневій теці репозиторію, визначає конфігурацію збірки для створення пакетів wheel/sdist для `pyneat`.
