---
title: "Перелік інструментів і скриптів."
description: "Що міститься в `core/scripts/` і `core/tools/`, і коли слід використовувати кожен із них."
sidebar_position: 90
---

# Перелік інструментів і скриптів.

У складі фреймворку передбачено дві директорії з набором допоміжних функцій. На цій сторінці наведено їхній перелік.

## `core/tools/` — інструменти для роботи з документацією та збірки.


| Сценарій | Призначення |
|---|---|
| `generate_api_docs.sh` | Запустіть doxygen2docusaurus для обробки XML-файлів, згенерованих Doxygen, і створіть Markdown-документацію для сайту з довідковою інформацією щодо API C++. Запускайте після редагування публічних заголовних файлів. |
| `generate_python_api_docs.py` | Створює Markdown-документацію з описом API Python на основі рядків документації модуля `pyneat`. |
| `generate_tutorial_docs.py` | (Навчальні матеріали поступово виводяться з використання — цей скрипт більше не підтримується). |
| `postprocess_d2d_links.py` | Виправлення посилань Doxygen2Docusaurus після їх створення. Автоматично викликається скриптом `generate_api_docs.sh`. |
| `strip_empty_programlisting.py` | Обхідне рішення для випадків, коли порожні елементи `<programlisting>` викликають проблеми в doxygen2docusaurus. |
| `compute_version.sh` | Обчислює рядок версії пакета фреймворку на основі `deps/manifest.json` та `package-version`, а також метаданих Git для збірок гілок. Використовується в системах безперервної інтеграції та для пакування. |
| `expand_code_tabs.py` | Розгорніть багатомовні вкладки в джерелах навчальних матеріалів. |
| `run_clean_env.sh` | Запустіть команду в чистому середовищі оболонки (щоб уникнути непередбачуваних наслідків, пов’язаних із успадкованими значеннями `LD_*` / `PATH`). |
| `tutorial_quality_lint.py` / `tutorial_scorecard.py` | Перевірка якості Markdown-файлів навчальних матеріалів / оцінювання. (Застаріває разом із навчальними матеріалами). |

Типовий процес редагування загальнодоступних заголовків:

```bash
cd core
doxygen docs/doxygen/Doxyfile      # regenerate XML
bash tools/generate_api_docs.sh    # regenerate Markdown
cd website && yarn start           # preview the site
```

## `core/scripts/` — перевірки на рівні репозиторію та допоміжні інструменти для розробників.


| Сценарій | Призначення |
|---|---|
| `check_format.sh` | Запустіть clang-format для коду C++; завершуйте роботу з помилкою, якщо виявлено відмінності. |
| `check_cmake_format.sh` / `check_cmake_style.py` | Запустіть cmake-format / lint для `CMakeLists.txt` файли. |
| `check_duplicate_includes.{sh,py}` | Виявляйте повторювані рядки `#include` у заголовках. |
| `check_internal_headers.sh` | Перевірте, чи шар `core/src/pipeline/internal/sima/` конвеєра дотримується меж між публічною та внутрішньою зонами. |
| `run_cpp_tidy.sh` | Запустіть clang-tidy для всього проєкту. |
| `route_refactor_validation.sh` | Спеціалізована перевірка регресії планувальника маршрутів (виконується системою безперервної інтеграції). |
| `install_neat_plugins.sh` | Встановіть плагіни GStreamer для фреймворку в системну директорію плагінів. |
| `install_codex_skill.sh` | Встановіть NEAT-навичку для командного рядка Codex (для зручності розробників). |
| `fix_devkit_runtime.sh` | Виправляє шляхи та бібліотеки середовища виконання свіжовстановленого набору інструментів розробника та перезапускає сопроцесори. Запускає лише M4, коли працює `simaai-appcomplex.service`. |
| `sync_neatdecoder.sh` / `use_neatdecoder.sh` | Перемикання між вбудованою та зовнішньою версіями декодера. |

### `core/scripts/ci/`, `core/scripts/dev/`, `core/scripts/release/`

Ці підкаталоги містять скрипти, які належать відповідним робочим процесам: `ci/` запускається системою безперервної інтеграції, `dev/` запускається розробниками за потреби, а `release/` запускається командою, що відповідає за підготовку релізів. Не слід використовувати їх у коді застосунку.

## Запуск генератора документації з чистої копії репозиторію.

```bash
sudo apt-get install -y doxygen   # if not installed
cd core
doxygen docs/doxygen/Doxyfile      # generates docs/doxygen/out/xml/
bash tools/generate_api_docs.sh    # populates docs/reference/cppapi/
python3 tools/generate_python_api_docs.py   # populates docs/reference/pythonapi/
cd website && yarn install && yarn start    # serve at http://localhost:3000/
```

## Для подальшого ознайомлення

- «Інструменти та скрипти» — розділ 55 у детальному аналізі проєкту.
- У репозиторії `core/AGENTS.md` міститься угода для учасників, в якій зазначено, які інструменти мають запускатися перед фіксацією змін.
