---
title: "Прив’язки для Python (pyneat)"
description: "Збирайте, тестуйте та пакуйте прив’язки pyneat як учасник розробки"
sidebar_position: 3
slug: /develop-apps/contribute/python_bindings
---

# Прив’язки для Python (`pyneat`)

Ця сторінка призначена для учасників розробки та супровідників `pyneat`.

`pyneat` — це шар прив’язок Python для SiMa.ai Neat, зібраний за допомогою `nanobind` і запакований за допомогою `scikit-build-core`.

Згенеровану документацію API див. у [Довіднику з Python API](/reference/pythonapi/modules/pyneat).

## Необхідні умови

`pyneat` компонується з тими самими нативними залежностями, що й бібліотека C++, зокрема:

- Пакети GStreamer для розробки та середовища виконання
- Пакети OpenCV для розробки та середовища виконання
- Набір інструментів C++ (`cmake`, компілятор, `pkg-config`)

Інструкції з налаштування хоста див. на сторінці [Збірка](/develop-apps/contribute/build).

## Встановлення з початкового коду

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Встановлення в режимі редагування (editable) для розробки:

```bash
python -m pip install -e .[dev]
```

## Запуск тестів

```bash
pytest -q python/tests
```

## Пакування

Файл `pyproject.toml` у кореневій теці репозиторію визначає конфігурацію збірки wheel/sdist для `pyneat`.
