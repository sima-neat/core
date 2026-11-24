---
title: "Мінімальний приклад"
description: "Переконайтеся, що інсталяція Neat працює, а імпорти розпізнаються"
sidebar_position: 1
mdx:
  format: mdx
---

# Мінімальний приклад

![Мінімальний приклад: напишіть скрипт, запустіть його на DevKit і переконайтеся, що середовище виконання відповідає](@site/../docs/images/minimal-example-flow.svg)

Після встановлення почніть звідси, щоб переконатися, що налаштування Neat виконано правильно.

Перший запуск на цій сторінці навмисно мінімальний: створіть крихітний застосунок, переконайтеся, що заголовки/імпорти розпізнаються і застосунок збирається, а потім запустіть його на DevKit, щоб перевірити, чи відповідає середовище виконання. Коли це запрацює, перейдіть до сторінки [Запуск застосунку](./run_an_app.md), щоб запустити справжню модель у невеликому графовому застосунку.

:::note Передумова для Palette SDK
Щоб запускати команди на DevKit безпосередньо зсередини SDK (наприклад, `dk build/sima_neat_hello` або `dk hello_neat.py`), спочатку налаштуйте з’єднання з DevKit.

**Необхідне налаштування:** [DevKit Sync](/getting-started/dev-environment/devkit-sync/)

У вас ще немає DevKit? Почніть з розділу [Компіляція моделі](/compile-a-model/), де все виконується в SDK.
:::

## Мінімальний застосунок

Створіть робочий каталог для прикладу, а потім скористайтеся вкладками **Python / C++**, щоб обрати свою мову. Ваш вибір відповідає перемикачу мови для всього сайту, тому він однаковий у всій документації.

<CodeTabs>
<CodeTab label="Python" lang="python">

**Створіть скрипт:**

1. `hello_neat.py` імпортує `pyneat` і підтверджує, що середовище виконання Python DevKit готове до роботи.

   ```python
   from pyneat import DeviceType

   def main():
       print("Hello from sima-neat")
       print("DeviceType.CPU =", DeviceType.CPU)

   if __name__ == "__main__":
       main()
   ```

2. Ваш робочий каталог має виглядати так:

   ```text
   sima-neat-hello/
   └── hello_neat.py
   ```

**Запуск:**

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
python3 hello_neat.py
</ShellCommand>

<ShellCommand prompt="sdk">
dk hello_neat.py
</ShellCommand>

:::note Розташування середовища виконання Python
`pyneat` встановлюється в середовищі виконання DevKit, навіть якщо ви запускаєте інсталятор Neat з контейнера Palette SDK.

Коли ви запускаєте `dk hello_neat.py`, `dk` виконує скрипт на з’єднаному DevKit, використовуючи середовище DevKit `pyneat`.
:::

</CodeTab>
<CodeTab label="C++" lang="cpp">

**Створіть два файли:**

1. `CMakeLists.txt` містить інструкції для CMake щодо збирання застосунку та підключення до нього бібліотек.

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

   Саме два виділені рядки підключають Neat до вашої збірки: `find_package(SimaNeat REQUIRED CONFIG)` знаходить встановлений пакет Neat, а `target_link_libraries(sima_neat_hello PRIVATE SimaNeat::sima_neat ...)` підключає його до вашого виконуваного файлу — імпортована ціль `SimaNeat::sima_neat` також підтягує заголовні файли та транзитивні залежності Neat, тому вам не потрібно вручну вказувати шляхи до заголовних файлів або бібліотек. (Рядки, що стосуються OpenCV, тут лише тому, що цей приклад декодує зображення.)

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

3. Ваш робочий каталог має виглядати так:

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

**Запуск:**

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

Якщо програма збирається, імпорти розпізнаються і вона виводить привітання, ваша інсталяція Neat готова до роботи.

:::tip Про `dk` / `devkit-run`
`dk` (псевдонім для `devkit-run`) — це функція оболонки в контейнері SDK, визначена у `~/devkit-sync.rc`, яку завантажує `~/.bashrc`.

Оскільки це функція оболонки, команди на кшталт `which devkit-run` можуть нічого не повертати в оболонці SDK. Використовуйте `dk <file>`, щоб запустити зібраний бінарний файл або Python-файл точки входу на з’єднаному DevKit.
:::

## Далі

Коли мінімальний застосунок запрацює, перейдіть до сторінки [Запуск застосунку](./run_an_app.md), щоб запустити справжню модель виявлення об’єктів у невеликому графовому застосунку.
