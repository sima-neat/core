---
title: "最小範例"
description: "確認 Neat 已正確安裝且能解析匯入項目"
sidebar_position: 1
mdx:
  format: mdx
---

# 最小範例

![最小範例：撰寫指令碼、在 DevKit 上執行，並確認執行階段有所回應](@site/../docs/images/minimal-example-flow.svg)

安裝完成後，請先從這裡確認 Neat 設定已正確連接。

第一次執行刻意保持精簡：建立一個小型應用程式，確認能解析標頭或匯入項目並成功建置，再於 DevKit 上執行，以確認執行階段有所回應。完成後，請繼續前往[執行應用程式](./run_an_app)，在小型 Graph 應用程式中執行實際模型。

:::note Neat SDK 必要條件
若要直接從 SDK 內部在 DevKit 上執行命令（例如 `dk build/sima_neat_hello` 或 `dk hello_neat.py`），必須先設定 DevKit 配對。

**必要設定：** [DevKit Sync](/getting-started/dev-environment/devkit-sync/)

還沒有 DevKit？請先從完全在 SDK 中執行的[編譯模型](/compile-a-model/)開始。
:::

## 最小應用程式

為範例建立工作目錄，然後使用 **Python / C++** 分頁依照所選語言進行。您的選擇會套用至整個網站的程式語言選擇器，因此在其他檔案中也會保持一致。

<CodeTabs>
<CodeTab label="Python" lang="python">

**建立指令碼：**

1. `hello_neat.py` 會匯入 `pyneat`，並確認 DevKit Python 執行階段已準備就緒。

   ```python
   from pyneat import DeviceType

   def main():
       print("Hello from sima-neat")
       print("DeviceType.CPU =", DeviceType.CPU)

   if __name__ == "__main__":
       main()
   ```

2. 工作目錄應如下所示：

   ```bash
   sima-neat-hello/
   └── hello_neat.py
   ```

**執行：**

* **在 DevKit 上**
  ```bash
  source ~/pyneat/bin/activate
  python3 hello_neat.py
  ```
* **在 Neat SDK 主機上**
  ```bash
  dk hello_neat.py
  ```

:::note Python 執行階段位置
即使您從 Neat SDK 容器內執行 Neat 安裝程式，`pyneat` 仍會安裝在 DevKit 執行階段一側。

執行 `dk hello_neat.py` 時，`dk` 會使用已配對 DevKit 上的 `pyneat` 環境執行指令碼。
:::

</CodeTab>
<CodeTab label="C++" lang="cpp">

**建立兩個檔案：**

1. `CMakeLists.txt` 告訴 CMake 如何建置及連結應用程式。

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

   兩個醒目提示的位置會將 Neat 納入建置：`find_package(SimaNeat REQUIRED CONFIG)` 尋找已安裝的 Neat 套件，而 `target_link_libraries(sima_neat_hello PRIVATE SimaNeat::sima_neat ...)` 將執行檔連結至該套件。匯入的 `SimaNeat::sima_neat` 目標也會帶入 Neat 標頭和可傳遞相依性，因此不必手動設定 include 或程式庫路徑。OpenCV 相關行只因本範例需要解碼影像而存在。

2. `main.cpp` 包含這個小型 Neat 程式。

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

3. 工作目錄應如下所示：

   ```bash
   sima-neat-hello/
   ├── CMakeLists.txt
   └── main.cpp
   ```

**建置範例：**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**執行：**

* **在 DevKit 上**
  ```bash
  ./build/sima_neat_hello
  ```
* **在 Neat SDK 主機上**
  ```bash
  dk build/sima_neat_hello
  ```

</CodeTab>
</CodeTabs>

您應該會看到：

```text
Hello from sima-neat
```

若程式能成功建置、解析匯入項目並顯示問候訊息，表示 Neat 安裝已準備就緒。

:::tip 關於 `dk` / `devkit-run`
`dk` 是 `devkit-run` 的別名，也是 SDK 容器內的 shell 函式。它定義於 `~/devkit-sync.rc`，並由 `~/.bashrc` 載入。

由於它是 shell 函式，在 SDK shell 中執行 `which devkit-run` 等命令可能不會傳回任何結果。請使用 `dk <file>`，在已配對的 DevKit 上執行建置完成的二進位檔或 Python 進入點檔案。
:::

## 後續步驟

最小應用程式正常執行後，請繼續前往[執行應用程式](./run_an_app)，在小型 Graph 應用程式中執行實際的物件偵測模型。
