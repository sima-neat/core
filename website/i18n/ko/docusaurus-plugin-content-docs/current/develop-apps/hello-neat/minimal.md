---
title: "최소 예제"
description: "Neat가 올바르게 설치되고 import가 정상 동작하는지 확인"
sidebar_position: 1
mdx:
  format: mdx
---

# 최소 예제

![최소 예제: 스크립트를 작성하고 DevKit에서 실행하여 런타임 응답 확인](@site/../docs/images/minimal-example-flow.svg)

설치가 끝나면 여기서 Neat 설정이 올바르게 연결되었는지 확인하십시오.

첫 실행은 의도적으로 간단하게 구성했습니다. 작은 앱을 만들고 헤더 또는 import가 정상적으로 해석되어 빌드되는지 확인한 다음, DevKit에서 실행해 런타임이 응답하는지 확인합니다. 완료되면 [앱 실행](./run_an_app)으로 이동하여 작은 Graph 애플리케이션에서 실제 모델을 실행하십시오.

:::note Neat SDK 사전 요구 사항
SDK 내부에서 DevKit 명령(예: `dk build/sima_neat_hello` 또는 `dk hello_neat.py`)을 직접 실행하려면 먼저 DevKit 페어링을 설정해야 합니다.

**필수 설정:** [DevKit Sync](/getting-started/dev-environment/devkit-sync/)

아직 DevKit이 없다면 SDK에서 완전히 실행되는 [모델 컴파일](/compile-a-model/)부터 시작하십시오.
:::

## 최소 애플리케이션

예제용 작업 디렉터리를 만든 다음 **Python / C++** 탭에서 원하는 언어의 단계를 따르십시오. 선택한 언어는 사이트 전체 언어 선택기에 반영되어 다른 문서에서도 유지됩니다.

<CodeTabs>
<CodeTab label="Python" lang="python">

**스크립트 만들기:**

1. `hello_neat.py`는 `pyneat`를 import하고 DevKit의 Python 런타임이 준비되었는지 확인합니다.

   ```python
   from pyneat import DeviceType

   def main():
       print("Hello from sima-neat")
       print("DeviceType.CPU =", DeviceType.CPU)

   if __name__ == "__main__":
       main()
   ```

2. 작업 디렉터리는 다음과 같아야 합니다.

   ```text
   sima-neat-hello/
   └── hello_neat.py
   ```

**실행:**

* **DevKit에서**
  ```bash
  source ~/pyneat/bin/activate
  python3 hello_neat.py
  ```
* **Neat SDK 호스트에서**
  ```bash
  dk hello_neat.py
  ```

:::note Python 런타임 위치
Neat SDK 컨테이너 안에서 Neat 설치 프로그램을 실행하더라도 `pyneat`는 DevKit 런타임 쪽에 설치됩니다.

`dk hello_neat.py`를 실행하면 `dk`가 페어링된 DevKit의 `pyneat` 환경에서 스크립트를 실행합니다.
:::

</CodeTab>
<CodeTab label="C++" lang="cpp">

**두 파일 만들기:**

1. `CMakeLists.txt`는 앱을 빌드하고 링크하는 방법을 CMake에 지정합니다.

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

   강조 표시된 두 부분이 Neat를 빌드에 포함합니다. `find_package(SimaNeat REQUIRED CONFIG)`는 설치된 Neat 패키지를 찾고, `target_link_libraries(sima_neat_hello PRIVATE SimaNeat::sima_neat ...)`는 실행 파일을 해당 패키지에 링크합니다. 가져온 `SimaNeat::sima_neat` 대상이 Neat 헤더와 전이 종속성도 함께 제공하므로 include 또는 라이브러리 경로를 직접 설정할 필요가 없습니다. OpenCV 관련 줄은 이 예제에서 이미지를 디코딩하기 때문에 필요합니다.

2. `main.cpp`에는 작은 Neat 프로그램이 들어갑니다.

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

3. 작업 디렉터리는 다음과 같아야 합니다.

   ```text
   sima-neat-hello/
   ├── CMakeLists.txt
   └── main.cpp
   ```

**예제 빌드:**

<ShellCommand prompt="sdk|devkit">
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
</ShellCommand>

**실행:**

<ShellCommand prompt="devkit">
./build/sima_neat_hello
</ShellCommand>

<ShellCommand prompt="sdk">
dk build/sima_neat_hello
</ShellCommand>

</CodeTab>
</CodeTabs>

다음 출력이 표시되어야 합니다.

```text
Hello from sima-neat
```

프로그램이 빌드되고 import가 정상적으로 해석되며 인사말이 출력되면 Neat 설치가 준비된 것입니다.

:::tip `dk` / `devkit-run` 정보
`dk`는 `devkit-run`의 별칭인 SDK 컨테이너의 셸 함수입니다. `~/devkit-sync.rc`에 정의되어 있으며 `~/.bashrc`에서 로드됩니다.

셸 함수이므로 SDK 셸에서 `which devkit-run`과 같은 명령은 아무 결과도 반환하지 않을 수 있습니다. `dk <file>`을 사용하여 빌드된 바이너리 또는 Python 엔트리 포인트 파일을 페어링된 DevKit에서 실행하십시오.
:::

## 다음 단계

최소 앱이 정상적으로 동작하면 [앱 실행](./run_an_app)으로 이동하여 작은 Graph 애플리케이션에서 실제 객체 감지 모델을 실행하십시오.
