---
title: "Мінімальний"
description: "Переконайтеся, що ваша інсталяція Neat працює належним чином і що всі необхідні модулі імпортуються."
sidebar_position: 1
mdx:
  format: mdx
---

# Мінімальний

![Найпростіший приклад: напишіть скрипт, запустіть його на DevKit, і переконайтеся, що середовище виконання реагує.](@site/../docs/images/minimal-example-flow.svg)

Після встановлення почніть звідси, щоб переконатися, що налаштування Neat виконано правильно.

На цій сторінці передбачено мінімальний приклад для першого запуску: створіть невелику програму, переконайтеся, що заголовки/імпорти правильно розпізнаються та програма успішно компілюється, а потім запустіть її на DevKit, щоб перевірити, чи реагує середовище виконання. Після успішного виконання перейдіть до [Запустіть програму.](./run_an_app.md), щоб запустити реальну модель у невеликій графічній програмі.

:::note Neat SDK Необхідні умови.
Щоб запускати команди безпосередньо в DevKit зсередини SDK (наприклад, `dk build/sima_neat_hello` або `dk hello_neat.py`), спочатку налаштуйте пару DevKit.

**Необхідне налаштування:** [DevKit Sync](/getting-started/dev-environment/devkit-sync/)

У вас ще немає DevKit? Почніть з [Складіть модель.](/compile-a-model/), яка повністю виконується в SDK.
:::

## Мінімальна кількість заявок.

Створіть робочу директорію для прикладу, а потім використовуйте вкладки **Python / C++**, щоб обрати потрібну мову. Ваш вибір відповідатиме загальносайтовому селектору мови, тому він буде однаковим у всій документації.

<CodeTabs>
<CodeTab label="Python" lang="python">

**Створіть сценарій:**

1. `hello_neat.py` імпортує `pyneat` і підтверджує, що середовище виконання Python DevKit готове до роботи.

   ```python
   from pyneat import DeviceType

   def main():
       print("Hello from sima-neat")
       print("DeviceType.CPU =", DeviceType.CPU)

   if __name__ == "__main__":
       main()
   ```

2. Ваша робоча директорія має виглядати так:

   ```text
   sima-neat-hello/
   └── hello_neat.py
   ```

**Запустити:**

* **Щодо DevKit**
  ```bash
  source ~/pyneat/bin/activate
  python3 hello_neat.py
  ```
* **На хості Neat SDK**
  ```bash
  dk hello_neat.py
  ```

:::note Розташування середовища виконання Python.
`pyneat` встановлюється в середовищі виконання DevKit, навіть якщо ви запускаєте інсталятор Neat з контейнера Neat SDK.

Коли ви запускаєте `dk hello_neat.py`, `dk` виконує скрипт на з’єднаному DevKit, використовуючи середовище DevKit `pyneat`.
:::

</CodeTab>
<CodeTab label="C++" lang="cpp">

**Створіть два файли:**

1. `CMakeLists.txt` містить інструкції для CMake щодо збірки та компонування застосунку.

   ```cmake title="CMakeLists.txt" {19,24-28}
   cmake_minimum_required(VERSION 3.16)
   project(sima_neat_hello LANGUAGES CXX)

   set(CMAKE_CXX_STANDARD 20)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)
   set(CMAKE_CXX_EXTENSIONS OFF)

   # Supports both:
   # - DevKit/native installs (system paths)
   # - Cross builds with SYSROOT exported (SDK sysroot paths)
   if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
     list(APPEND CMAKE_PREFIX_PATH
       "$ENV{SYSROOT}/usr"
       "$ENV{SYSROOT}/usr/lib"
       "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu"
     )
   endif()

   find_package(SimaNeat REQUIRED CONFIG)
   find_package(PkgConfig REQUIRED)
   pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

   add_executable(sima_neat_hello main.cpp)
   target_link_libraries(sima_neat_hello
     PRIVATE
       SimaNeat::sima_neat
       PkgConfig::OPENCV
   )
   ```

   Дві виділені рядки визначають, як Neat інтегрується у ваш проєкт: `find_package(SimaNeat REQUIRED CONFIG)` знаходить встановлений пакет Neat, а `target_link_libraries(sima_neat_hello PRIVATE SimaNeat::sima_neat ...)` пов’язує ваш виконуваний файл із ним — імпортована ціль `SimaNeat::sima_neat` також включає заголовні файли та транзитивні залежності Neat, тому вам не потрібно вручну вказувати шляхи до заголовних файлів або бібліотек. (Рядки, що стосуються OpenCV, тут лише тому, що цей приклад декодує зображення).

2. `main.cpp` містить невелику програму Neat.

   ```cpp title="main.cpp"
   #include <iostream>
   #include <pipeline/TensorCore.h>

   int main() {
     auto storage = simaai::neat::make_cpu_owned_storage(64);
     if (!storage) {
       std::cerr << "Failed to allocate CPU tensor storage\n";
       return 1;
     }
     std::cout << "Hello from sima-neat\n";
     return 0;
   }
   ```

3. Ваша робоча директорія має виглядати так:

   ```text
   sima-neat-hello/
   ├── CMakeLists.txt
   └── main.cpp
   ```

**Зберіть приклад:**

<ShellCommand prompt="sdk|devkit">
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
</ShellCommand>

**Запустити:**

<ShellCommand prompt="devkit">
./build/sima_neat_hello
</ShellCommand>

<ShellCommand prompt="sdk">
dk build/sima_neat_hello
</ShellCommand>

</CodeTab>
</CodeTabs>

Ви повинні побачити:

```text
Hello from sima-neat
```

Якщо програма успішно компілюється, усі необхідні модулі імпортуються, і на екрані з’являється привітання, це означає, що ваша інсталяція Neat завершена та готова до використання.

:::tip Про `dk` / `devkit-run`
`dk` (псевдонім для `devkit-run`) — це функція оболонки в контейнері SDK, визначена у `~/devkit-sync.rc` і завантажена за допомогою `~/.bashrc`.

Оскільки це функція оболонки, команди, такі як `which devkit-run`, можуть не повертати жодного результату в оболонці SDK. Використовуйте `dk <file>`, щоб запустити скомпільований бінарний файл або файл із точкою входу Python на відповідному DevKit.
:::

## Далі

Після того, як мінімальна версія застосунку запрацює, продовжуйте роботу, використовуючи [Запустіть програму.](./run_an_app.md), щоб запустити реальну модель розпізнавання об’єктів у невеликому графічному застосунку.
