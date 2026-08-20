---
title: "最小構成"
description: "Neatが正しくインストールされ、インポートを解決できることを確認する"
sidebar_position: 1
mdx:
  format: mdx
---

# 最小構成

![最小例：スクリプトを作成してDevKitで実行し、ランタイムの応答を確認する](@site/../docs/images/minimal-example-flow.svg)

インストール後、まずここでNeatのセットアップが正しく接続されていることを確認します。

最初の実行は意図的に小さくしています。簡単なアプリを作成し、ヘッダーまたはインポートが解決されてビルドできることを確認してから、DevKitで実行してランタイムの応答を確認します。完了したら、[アプリの実行](./run_an_app)に進み、小さなGraphアプリケーション内で実際のモデルを実行してください。

:::note Neat SDKの前提条件
SDK内からDevKit上のコマンド（`dk build/sima_neat_hello`や`dk hello_neat.py`など）を直接実行するには、先にDevKitのペアリングを設定します。

**必須設定：** [DevKit Sync](/getting-started/dev-environment/devkit-sync/)

まだDevKitがない場合は、SDK内だけで実行できる[モデルのコンパイル](/compile-a-model/)から始めてください。
:::

## 最小アプリケーション

例用の作業ディレクトリを作成し、**Python / C++**タブから使用する言語の手順を進めます。選択内容はサイト全体の言語セレクターに反映され、ほかのドキュメントでも維持されます。

<CodeTabs>
<CodeTab label="Python" lang="python">

**スクリプトを作成します：**

1. `hello_neat.py`で`pyneat`をインポートし、DevKitのPythonランタイムが利用できることを確認します。

   ```python
   from pyneat import DeviceType

   def main():
       print("Hello from sima-neat")
       print("DeviceType.CPU =", DeviceType.CPU)

   if __name__ == "__main__":
       main()
   ```

2. 作業ディレクトリは次のようになります。

   ```bash
   sima-neat-hello/
   └── hello_neat.py
   ```

**実行：**

* **DevKit上**
  ```bash
  source ~/pyneat/bin/activate
  python3 hello_neat.py
  ```
* **Neat SDKホスト上**
  ```bash
  dk hello_neat.py
  ```

:::note Pythonランタイムの場所
Neat SDKコンテナ内からNeatインストーラーを実行した場合でも、`pyneat`はDevKitのランタイム側にインストールされます。

`dk hello_neat.py`を実行すると、`dk`はペアリング済みDevKitの`pyneat`環境を使ってスクリプトを実行します。
:::

</CodeTab>
<CodeTab label="C++" lang="cpp">

**2つのファイルを作成します：**

1. `CMakeLists.txt`で、アプリのビルド方法とリンク方法をCMakeに指定します。

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

   強調表示された2か所がNeatをビルドに取り込みます。`find_package(SimaNeat REQUIRED CONFIG)`はインストール済みのNeatパッケージを検出し、`target_link_libraries(sima_neat_hello PRIVATE SimaNeat::sima_neat ...)`は実行ファイルをパッケージへリンクします。インポートされた`SimaNeat::sima_neat`ターゲットがNeatのヘッダーと推移的依存関係も提供するため、インクルードパスやライブラリパスを手動で設定する必要はありません。OpenCVの行は、この例で画像をデコードするためにだけ必要です。

2. `main.cpp`には小さなNeatプログラムを記述します。

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

3. 作業ディレクトリは次のようになります。

   ```bash
   sima-neat-hello/
   ├── CMakeLists.txt
   └── main.cpp
   ```

**例をビルドします：**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**実行：**

* **DevKit上**
  ```bash
  ./build/sima_neat_hello
  ```
* **Neat SDKホスト上**
  ```bash
  dk build/sima_neat_hello
  ```

</CodeTab>
</CodeTabs>

次の出力が表示されます。

```text
Hello from sima-neat
```

プログラムがビルドされ、インポートを解決でき、メッセージが表示されれば、Neatのインストールは完了です。

:::tip `dk` / `devkit-run`について
`dk`は`devkit-run`のエイリアスで、SDKコンテナ内のシェル関数です。`~/devkit-sync.rc`で定義され、`~/.bashrc`から読み込まれます。

シェル関数のため、SDKシェルで`which devkit-run`などを実行しても何も返されない場合があります。`dk <file>`を使用し、ビルド済みバイナリまたはPythonエントリポイントをペアリング済みDevKitで実行してください。
:::

## 次のステップ

最小アプリが動作したら、[アプリの実行](./run_an_app)に進み、小さなGraphアプリケーション内で実際の物体検出モデルを実行します。
