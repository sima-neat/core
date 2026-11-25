---
title: "Перелік інструментів і скриптів"
description: "Що міститься в `core/scripts/` і `core/tools/` і коли варто звертатися до кожного з них."
sidebar_position: 90
---

# Перелік інструментів і скриптів

Фреймворк постачається з двома теками допоміжних скриптів. Ця сторінка допоможе в них зорієнтуватися.

## `core/tools/` — допоміжні скрипти для документації та збирання

| Скрипт | Призначення |
|---|---|
| `generate_api_docs.sh` | Запускає doxygen2docusaurus для XML-файлів Doxygen і генерує Markdown для сайту довідки з C++ API. Запускайте після редагування публічних заголовних файлів. |
| `generate_python_api_docs.py` | Генерує Markdown довідки з Python API на основі рядків документації модуля `pyneat`. |
| `generate_tutorial_docs.py` | (Навчальні матеріали оголошуються застарілими — цей скрипт поступово виводиться з ужитку.) |
| `postprocess_d2d_links.py` | Виправляє слаги посилань doxygen2docusaurus після генерації. Автоматично викликається скриптом `generate_api_docs.sh`. |
| `strip_empty_programlisting.py` | Обхідне рішення для порожніх елементів `<programlisting>`, які збивають з пантелику doxygen2docusaurus. |
| `compute_version.sh` | Обчислює рядок версії пакета фреймворку з поля `package-version` у `deps/manifest.json`, а для збірок гілок додає метадані git. Використовується в CI і під час пакування. |
| `expand_code_tabs.py` | Розгортає вкладки з кодом кількома мовами у файлах початкового коду навчальних матеріалів. |
| `run_clean_env.sh` | Запускає команду в чистому середовищі оболонки (без дивної поведінки через успадковані `LD_*` / `PATH`). |
| `tutorial_quality_lint.py` / `tutorial_scorecard.py` | Перевіряє Markdown навчальних матеріалів лінтером / оцінює його. (Виводяться з ужитку разом із навчальними матеріалами.) |

Типовий порядок дій під час редагування публічних заголовних файлів:

```bash
cd core
doxygen docs/doxygen/Doxyfile      # regenerate XML
bash tools/generate_api_docs.sh    # regenerate Markdown
cd website && yarn start           # preview the site
```

## `core/scripts/` — перевірки на рівні репозиторію та допоміжні скрипти для розробки

| Скрипт | Призначення |
|---|---|
| `check_format.sh` | Запускає clang-format для дерева коду C++; завершується з помилкою, якщо є розбіжності. |
| `check_cmake_format.sh` / `check_cmake_style.py` | Запускає cmake-format / лінтер для файлів `CMakeLists.txt`. |
| `check_duplicate_includes.{sh,py}` | Виявляє повторювані рядки `#include` у заголовних файлах. |
| `check_internal_headers.sh` | Перевіряє, що рівень наскрізного доступу `core/src/pipeline/internal/sima/` дотримується межі між публічною та внутрішньою частинами. |
| `run_cpp_tidy.sh` | Запускає clang-tidy для всього дерева коду. |
| `route_refactor_validation.sh` | Цільова регресійна перевірка планувальника маршрутів (викликається з CI). |
| `install_neat_plugins.sh` | Встановлює плагіни GStreamer фреймворку в системну теку плагінів. |
| `install_codex_skill.sh` | Встановлює навичку NEAT для Codex CLI (для зручності розробників). |
| `sync_neatdecoder.sh` / `use_neatdecoder.sh` | Перемикають між вбудованою та зовнішньою збірками декодера. |

### `core/scripts/ci/`, `core/scripts/dev/`, `core/scripts/release/`

У цих підтеках містяться скрипти, які належать до відповідних робочих процесів: набір `ci/` запускає CI, `dev/` розробники запускають за потреби, а `release/` — інженери з випуску релізів. Не робіть код застосунку залежним від них.

`scripts/ci/prepare_scheduler_sdk.sh` експортує `NEAT_SDK_VERIFICATION_JSON` — шлях до свого файлу з перевіреними даними спостереження за SDK у `deps/`. `build.sh` оновлює цей файл після компіляції та передає його генераторові метаданих пакета. Ця змінна передає внутрішній стан збирання. Щоб вибрати пакети залежностей, використовуйте `--scheduler`, а щоб запросити оновлення SDK — `--update-sysroot`.

## Запуск генератора документації з чистої копії репозиторію

```bash
sudo apt-get install -y doxygen   # if not installed
cd core
doxygen docs/doxygen/Doxyfile      # generates docs/doxygen/out/xml/
bash tools/generate_api_docs.sh    # populates docs/reference/cppapi/
python3 tools/generate_python_api_docs.py   # populates docs/reference/pythonapi/
cd website && yarn install && yarn start    # serve at http://localhost:3000/
```

## Додаткові матеріали

- «Інструменти та скрипти» — §55 поглибленого огляду архітектури.
- Файл `core/AGENTS.md` у репозиторії містить домовленість для учасників про те, які інструменти потрібно запускати перед фіксацією змін.
