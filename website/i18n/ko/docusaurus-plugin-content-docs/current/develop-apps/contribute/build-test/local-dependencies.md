# Scheduler 및 LLiMa 의존성으로 Core 빌드

Core는 Scheduler와 LLiMa라는 두 가지 의존성을 기반으로 빌드합니다. Neat SDK의 Core
체크아웃에서 `build.sh`를 실행합니다.

```bash
./build.sh --all --no-doc
```

기본적으로 `build.sh`는 다음을 수행합니다.

1. Vulcan에서 공개된 Scheduler 집합과 LLiMa 패키지를 다운로드합니다.
2. 둘 다 SDK sysroot에 설치합니다.
3. Core를 빌드합니다.

## 로컬 Scheduler 또는 LLiMa 패키지 사용

직접 빌드한 Scheduler 또는 LLiMa로 Core를 빌드하려면 해당 DEB 폴더를 전달합니다.

```bash
./build.sh --scheduler ../scheduler-debs --llima ../llima/dist/debs
```

각 옵션은 단독으로 사용할 수 있으며, 다른 의존성은 `build.sh`가 계속 다운로드합니다.
지정한 패키지는 sysroot의 공개된 패키지를 대체합니다.

Scheduler 폴더에는 Scheduler가 빌드될 때 사용한 전체 집합이 있어야 합니다.

- `simaai-scheduler`
- Scheduler가 빌드될 때 사용한 `simaai-runner` 및 `simaai-kernels-dev` 패키지
- EV가 활성화된 빌드를 위한 일치하는 `simaai-kernels-firmware` 패키지
- Scheduler가 Codecs를 사용하는 경우 `neat-codecs` 및 `neat-codecs-dev` 쌍
- 선택적으로 `simaai-runtime`, 그리고 테스트 빌드에는 `simaai-scheduler-client-test-dev`

Core는 Kernels, Runner 또는 Codecs를 선택하지 않습니다. 이 중 하나의 변경 사항을
테스트하려면 Scheduler의 빌드 가이드에 따라 해당 구성 요소로 Scheduler를 빌드하고,
생성된 패키지를 한 폴더에 모아 `--scheduler`로 전달합니다. Scheduler의 코디네이터는
패키지가 서로 일치하는지 확인하며, 일치하지 않으면 빌드를 중단합니다.

`NEAT_KERNELS_*`, `NEAT_RUNNER_*`, `NEAT_SCHEDULER_*` 또는 `NEAT_CODECS_*` 소스 또는
패키지 변수가 설정되어 있으면 `build.sh`는 오류와 함께 중단됩니다. LLiMa 없이 빌드하려면
`--no-llima`를 사용합니다.

## 플랫폼 의존성

Kernels, Runner, Scheduler 및 Core를 동일한 Platform 3.0 SDK/툴체인으로 빌드합니다.
기본값이 아닌 툴체인을 선택할 때는 `CMAKE_TOOLCHAIN_FILE`을 설정합니다. Kernels와 Runner 영수증의
타깃/컴파일러, 커널 계약 및 소비한 빌드 구성이 일치해야 합니다. 정식 패키지
워크플로는 소비한 Kernels 페이로드의 정확한 식별 정보도 요구합니다. Kernels
패키지가 바뀌면 Runner 소스가 바뀌지 않았더라도 선택한 SDK로 Runner 패키지를
다시 빌드합니다. 그러면 Core는 일치하는 Scheduler 클라이언트 구현에 다시 링크하며,
Scheduler는 하위 수치 연산 패키지를 소비합니다.
Core의 소스 우선 DEB는 선택된 Scheduler 및 수치 연산 SDK 버전을 고정합니다.
Runner는 Kernels를 고정하고, Scheduler는 Runner, Kernels 및 활성화된 코덱 백엔드를
고정합니다. 테스트가 활성화된 Core 빌드는 동일한 생성기의 별도
`simaai-scheduler-client-test-dev` 패키지도 같은 Scheduler 폴더에 필요합니다.
코디네이터는 테스트 전용 CMake 구성 요소를 노출하기 전에 해당
프로덕션 영수증, 소스, 페이로드 및 아카이브를 확인합니다. 런타임 전용 빌드에는
이 패키지가 필요하지 않습니다.
빌드는 정확한 Runner 및 Scheduler 의존성을 가진 메타데이터 전용 `simaai-runtime`
호환성 패키지를 구성합니다. 정식 패키지는 각 파일을 한 번만 소유하며 기존 로더
구성은 Runner만 소유합니다. 패키지 하나를 교체하지 말고 항상 일치하는 배포본
전체를 설치합니다.

### GStreamer 헤더 소유권 이전

아직 릴리스되지 않은 Core `0.6.0`에서는
`/usr/include/gst/SimaaiGstErrors.h`의 소유권이 Scheduler로 이전됩니다. 포함 경로는
변하지 않습니다. Core는 자체 구현을 포함하거나 이 파일을 설치하지 않습니다.
소비자는 설치된 Scheduler 헤더와 `SimaScheduler::Headers` 타깃을 사용합니다.

완전하고 일치하는 배포본으로 Core와 Scheduler를 함께 업그레이드하세요.
Scheduler는 수정된 Core 0.6.0 패키지를 비호환으로 처리하지 않고, 아직 릴리스되지 않은 0.6.0 계열 패키지의 헤더 소유권 이전을 선언합니다.
APT가 파일을 변경하기 전에 설치 프로그램은 인증된 Scheduler 소유권 검사를 사용합니다.
이전 소유 패키지는 같은 패키지 집합에서 해당 파일을 포함하지 않는 패키지로 교체되거나, 명시적으로 선언되고 승인된 폐기 대상이어야 합니다.
새 패키지 사이의 중복 소유권과 관련 없는 소유권 충돌은 거부됩니다.

Platform 3.0에서는 완전한 분할 패키지 집합으로 업그레이드할 때 모놀리식
`neat-runtime`, `neat-gst-plugins`, `neat-internals-dev`의 0.4, 0.5, 0.6.0 계열도 개발 빌드를 포함해 폐기합니다.
폐기하려면 새 소유 패키지에 해당 버전 범위를 제한하는 일치하는 `Breaks`와 `Replaces` 선언이 있어야 합니다.
`0.6.1~` 이상 버전은 대상이 아닙니다. 관련 없는 패키지 제거를 허용하거나 폐기된 API와의 호환성을 보장하지 않습니다.
런타임과 플러그인 파일 이전은 Scheduler가 선언하며, 완전한 런타임에는 실행 의존성으로 일치하는 Runner도 필요합니다.
Runner만 설치하면 모놀리식 0.6.0 패키지는 폐기되지 않습니다.

EV가 활성화된 Modalix 배포본에는 Scheduler가 선택한 일치하는 `simaai-kernels-firmware` 패키지가 필요합니다.
펌웨어가 없거나 일치하지 않으면 승인 검증에 실패합니다.
기존 `neat-ev74-firmware` 패키지를 교체하려면 선택된 바로 그 펌웨어 패키지가 해당 버전과 일치하는 `Replaces` 및 `Breaks` 또는 `Conflicts`를 선언해야 합니다. 일반 트랜잭션 및 제거 검사도 계속 적용됩니다.
펌웨어 활성화에는 활성화 도우미를 제공하는 유일한 새 패키지만 사용합니다.
설치된 패키지의 식별 정보, 도우미의 소유권, 실행 모드 및 바이트가 선택한 패키지와 일치해야 합니다.
선택한 도우미가 일치하지 않으면 설치가 실패합니다.
EV74 펌웨어가 필요하지 않은 배포본에서는 도우미가 없어도 기존 펌웨어를 그대로 둡니다.
설치 프로그램은 이전 설치에서 남은 도우미를 사용하지 않습니다.

DevKit에서 설치가 중단되었다면 검증된 전체 배포본을 다시 실행하세요.
설치 프로그램은 미완료 패키지가 새 패키지 집합에 속하고 아키텍처가 일치할 때만 복구할 수 있습니다.
복구는 관련 없는 패키지를 변경하지 않고 선택된 정확한 버전을 설치해야 합니다.
새 코호트가 폐기를 명시적으로 선언하고 기존 제거 검사가 승인한, 설정이 완료되고 보류되지 않은 이전 패키지만 제거할 수 있습니다.
복구는 먼저 승인된 전환을 완료하고 패키지 상태를 확인한 다음, 제거를 금지한 상태에서 선택된 모든 패키지를 다시 설치합니다. 이 최종 재설치를 완료해야만 설치가 성공합니다.
관련 없는 패키지 데이터베이스 오류는 별도로 복구해야 합니다. 설치 프로그램은 일반적인 APT 복구를 수행하거나 롤백을 보장하지 않습니다.

기존 Core 개발 패키지가 있는 환경에 Scheduler DEB만 설치하거나, `--force-overwrite`를 사용하거나, dpkg 소유 파일을 삭제하지 마세요.
독립형 Scheduler 테스트 이미지에는 Core 설치가 필요하지 않지만, 기존 Core 설치가 있다면 함께 업그레이드해야 합니다.

## 증분 빌드 및 설치

빌드할 때마다 선택된 Scheduler 집합을 `deps/scheduler/`에서 새로 가져오고 해당
패키지를 `deps/backend/`에 스테이징합니다. CMake는 Core를 증분 빌드합니다.
`--clean`은 Core 빌드도 제거합니다.

링크 전에 Scheduler 의존성 코디네이터는 설치된 커널 헤더를 영수증과 대조하고,
Runner가 소비한 계약과 Scheduler의 정확한 Runner 입력을 확인합니다. 배포 패키지를
스테이징하기 전에는 각 Debian 패키지의 `/usr` 페이로드가 선택된 설치 트리와
일치하는지 확인합니다. 이전 Internals 패키지는 선택하거나 다운로드하지 않습니다.

Core는 형제 소스의 포함 경로나 하위 개발 인터페이스가 아니라 설치된 `SimaScheduler`
클라이언트 구성 요소를 소비합니다. 하위 런타임 식별 정보는 검증된 의존성 집합에
유지됩니다. 생성된 의존성 선택은 `build/resolved_manifest.json`과
`build/buildinfo.json`에 기록됩니다. 빌드가 완료되어야 최종 설치에 도달합니다.
이것이 SDK 또는 장치 설치의 트랜잭션 롤백을 보장하는 것은 아닙니다.

패키지 빌드 정보는 CMake 구성 전에는 임시 정보입니다. 컴파일 후 `build.sh`는
설치 확인이나 패키징에 앞서 해당 빌드의 CMake 캐시, 구성된 컴파일러, 타깃 Python
헤더 및 빌드된 확장 모듈로 정보를 갱신합니다. 타깃 Python ABI 플래그는 확장 모듈
접미사와 일치해야 하며 호스트 Python 버전은 사용하지 않습니다. Python이 비활성화된
빌드는 Python ABI 없이 해당 프로파일을 기록합니다.

`compatibility.host_sdk`는 빌드 호스트에서 관측한 SDK 릴리스를 나타내고,
`compatibility.target_sdk`는 선택한 sysroot와 컴파일러를 나타냅니다. 호스트에 SDK
릴리스 정보가 있어도 타깃 SDK 릴리스가 없으면 사용할 수 없는 상태로 남습니다.
매니페스트의 플랫폼 버전은 별도의 호환성 요구사항이며 관측하지 못한 SDK 버전을
대체하지 않습니다. `ELXR_SDK_RELEASE_FILE`은 호스트 릴리스 파일을 선택합니다.
`NEAT_TARGET_SDK_RELEASE_FILE`로 타깃 릴리스 파일을 명시적으로 제공할 수 있지만,
선택한 sysroot의 릴리스 파일과 일치해야 합니다. `NEAT_BUILDINFO_TOOLCHAIN`은
구성된 컴파일러 식별 정보를 대체하지 않고 별도의 표시용 레이블을 보존합니다.

메타데이터 검사는 충돌하는 컴파일 sysroot, 헤더 및 확장 모듈 ABI 태그를 거부합니다.
이 메타데이터 경로는 별도의 컴파일/링크 sysroot나 추가 CMake 컴파일러
인수/타깃/툴체인 컨텍스트를 지원하지 않습니다. 검증하지 않은 타깃 식별 정보를
보고하는 대신 실패합니다. 마커로 한정된 NumPy 요구사항은 하나의 무조건 최소
버전으로 축소하지 않고 `pyproject.toml`에서 그대로 유지합니다.

소스 기록에는 실제 체크아웃 커밋과 변경 여부를 사용합니다. 추적 중인 변경과
무시되지 않은 미추적 파일/심볼릭 링크 식별 정보를 별도로 기록합니다. 무시된
빌드 결과물은 소스 스냅샷이 아닙니다. 자기 참조를 피하기 위해 메타데이터 출력
자체는 미추적 콘텐츠 다이제스트에서 제외합니다. 변경된 로컬 후보를 깨끗한 공개
아티팩트로 식별하지 않습니다.

헤더 전용 의존성 준비의 경우 `--install-deps-only`는 `--scheduler` 폴더를 사용하거나
공개된 집합을 다운로드합니다. A65/EV 코드를 빌드하거나 실행하지 않고 헤더만 복사합니다.
네이티브 clang-tidy 분석에는 선택한 Scheduler와 일치하는 테스트 지원 패키지의
접두사로 `scripts/run_cpp_tidy.sh`를 실행합니다. `TIDY_INSTALL_DEPS=1`은 네이티브
호스트 분석 전제 조건을 설치합니다. 스크립트는 각 패키지의 공개 CMake 내보내기를
직접 사용하며 SDK sysroot를 활성화하거나 수정하지 않습니다. LLiMa 기반 GenAI 구현 파일은
이 호스트 전용 분석 프로파일에서 제외됩니다. 스크립트는 선택한 빌드의 의존성
대신 주변 SDK의 LLiMa 패키지를 대입하지 않습니다. 문서 전용 빌드는 네이티브
의존성을 준비하지 않습니다.

EV 장치 빌드는 별도의 라이선스가 필요한 워크플로입니다. 이 명령들은 MetaWare를
암묵적으로 호출하거나 펌웨어를 교체하거나 펌웨어 변경을 검증하지 않습니다.

## 공개된 CI 의존성 {#published-ci-dependencies}

일반 Vulcan 빌드는 `deps/manifest.json`의 Scheduler 정책을 `sima-cli`로
해석한 뒤 선택한 불변 공개본을 다운로드합니다. LLiMa와 마찬가지로 브랜치 `snap`은
현재 Core 브랜치를 먼저 시도합니다. 해석에 실패하고 Scheduler의 공개 브랜치 목록에서
해당 브랜치가 없음을 확인한 경우에만 폴백을 알리고 `develop:latest`를 다시 시도합니다.
목록을 가져올 수 없거나 목록이 유효하지 않은 경우를 포함한 다른 조회 실패는 오류로
처리합니다. 태그 snap과 명시적인 브랜치/커밋 선택자는 폴백하지 않습니다.
장치와 PCIe 호스트 획득은 이 정책을 공유하며, PCIe `--scheduler-spec <commit>`도
엄격하게 적용됩니다. 해석 후 다운로드, 검증, 부트스트랩 또는 스테이징이 실패하면
오류로 처리합니다. 기록에는 실제 선택한 브랜치와 불변 커밋이 보존됩니다.
`tools/ci/acquire_scheduler.py`는 Scheduler 부트스트랩을 실행하기 전에 게시자
해시를 검증합니다. 설치된 Scheduler 코디네이터가 번들된 Runner, Kernels 및 Codecs
아카이브를 분리하고 검증합니다. Core는 해당 저장소들을 독립적으로 해석하지 않습니다.

선택한 Scheduler 패키지는 정확한 SDK 정책을 설치합니다. SDK에서 `build.sh`는
`scripts/ci/prepare_scheduler_sdk.sh`를 실행하며, 이 스크립트는 해당 리비전의 공식
sysroot 설정 명령을 사용하고 다운로드한 플랫폼 아카이브를 보존합니다. 코디네이터는 실제 GCC14 버전/타깃과
타깃 SDK 식별 정보를 확인한 다음 `NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`를 통해
Runner가 소비한 정확한 런타임 제공자 아카이브를 선택합니다. 네이티브 ARM64와
크로스 컴파일러의 실행 파일 바이트가 같을 필요는 없습니다. 타깃 ABI와 선택한
플랫폼은 일치해야 합니다. 이 단계에서는 장치 설치나 펌웨어 검증을 수행하지 않습니다.

`--scheduler` 폴더는 다운로드한 집합과 섞이거나 그것으로 대체되지 않습니다. 기존
`tools/ci/materialize_dependencies.py --manifest ... --output-root ...` 명령은 별도로
검토된 불변 매니페스트를 위한 선택적 전송 수단으로 남습니다. 기본 워크플로가
아니며 암묵적인 `ci/dependencies.json`도 없습니다.

코드 위생 검사는 `tools/ci/acquire_scheduler.py`로 동일한 의존성 정책에 따라 공개된
Scheduler를 독립적으로 가져오므로 Core 빌드가 끝날 때까지 기다리지 않습니다.
Scheduler 부트스트랩이 패키지 집합을 검증합니다. 호스트 분석은 SDK 활성화나 타깃
실행 없이 프런트엔드 및 일치하는 테스트 지원 CMake 내보내기를 사용합니다.

보호된 성능 레인에 필요한 타깃 측 감독자는 [하드웨어 테스트 소유권](./hardware-test-safety.md)을 참조하세요.

## 완성된 배포본 검증

`build.sh --all`은 패키지 메타데이터 생성 후 완성된 배포본을 검사합니다.
`scripts/ci/verify_core_dist.sh`는 CI에서 선택한 `sima-cli` Python 환경으로 동일한
읽기 전용 검사를 실행합니다. 인터프리터가 선택한 CLI 실행 파일 옆에 없으면
`SIMA_CLI_PYTHON`을 명시적으로 설정합니다.

검사는 의존성 패키지 바이트, 호환성 필드, 설치 프로그램 허용 목록, 모든 리소스
체크섬, 공개 wheel 격리 및 설치된 CTest 트리를 검증합니다. 기존 생성기 영수증을
소비하며 의존성을 해석하거나 복구하지 않습니다. 다음 문서는 구분해서 유지합니다.

- `dist/manifest.json`: 플랫폼, Model Zoo 및 ABI 호환성.
- `dist/resolved-deps-manifest.json`: 정확히 복사된 선택 백엔드 영수증.
- `build/resolved_manifest.json`: 더 완전한 Core 빌드 해석 정보.

이 검사 후에는 파일을 교체하거나 선택된 체크섬을 다시 계산하지 마세요.
아티팩트가 바뀌면 메타데이터를 다시 빌드하고 전체 배포본을 다시 검증합니다.


## 명시적 플랫폼 제공자 패키지

Core의 프로덕션 박스/마스크 게시는 int32 좌표를 사용합니다. 정식 Kernels 렌더링과
일치하는 Runner 실행 API로 빌드된 Scheduler를 선택합니다.
`SIMANEAT_REQUIRE_INT32_RENDERER`는 기본적으로 런타임 아티팩트 요구사항을 따르며,
설치된 Scheduler에 이 기능이 없으면 구성에 실패합니다. Core 박스/마스크 렌더링을
사용하지 않는 Core 프로파일에서만 `OFF`로 설정합니다.

렌더링은 별도의 플랫폼 렌더러 아카이브가 아니라 선택한 Kernels/Runner 패키지를
사용합니다. 요청 API 리비전 2에 맞춰 구성 요소 집합을 함께 다시 빌드합니다.

SDK에서는 `build.sh`가 `scripts/ci/prepare_scheduler_sdk.sh`를 통해 다음을
설정합니다. SDK 밖에서는 직접 제공합니다.

- `NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`: Runner가 선택한 정확한 런타임 제공자 DEB의
  절대 경로를 담은 JSON 배열입니다. 필수 SimaHeap 런타임 아카이브를 포함하고,
  해당 백엔드가 활성화되면 MLA 및 TVM 런타임 아카이브도 포함합니다. 이 배열에는
  개발 아카이브를 포함하지 마세요. Runner는 실제 로더 이름과 라이브러리 바이트를
  검증합니다. Scheduler는 정확한 아카이브 식별 정보와 페이로드 목록을 빌드 증거로
  유지합니다. Runner가 Platform 소유로 선언한 제공자는 Core의 설치 가능한 DEB
  집합에서 제외되며, 선언되지 않은 제공자는 정확한 아카이브 구성을 유지합니다.

`--scheduler` 폴더를 사용해도 이 선택은 명시적으로 유지됩니다. 불변 네 구성 요소 구체화 도구는 플랫폼 제공자 아카이브를 가져오지
않습니다. Core를 호출하기 전에 지원되는 생성기에서 획득하세요. 제공자가 없거나
소비되지 않으면 sysroot로 폴백하지 않고 실패합니다. 컴파일에는 일치하는 개발
패키지를 사용합니다. DevKit에서는 Scheduler의 인증된 허용 검사 도구가 패키지
변경 전에 설치된 Platform 제공자를 들어오는 소비자와 대조합니다. 설치 프로그램은
이 제공자를 교체하거나 제거하는 APT 계획을 거부하고, 이후 식별 정보와 설치된
소비자의 의존성 집합을 검증합니다. `FORCE_INSTALL`로도 이 검사를 면제할 수 없습니다.
비공개 DSO를 릴리스된 제공자로 표시하기 위한 목적으로 DEB를 다시 빌드하지 마세요.

패키지 구성 검사는 겹치지 않는 페이로드 소유권, 정확한 의존성 버전 및 변하지 않은
아카이브를 확인합니다. 이러한 빌드 시점 검사는 수치 실행, 동적 마스크 또는 하드웨어
작업 종료를 검증하지 않습니다. 해당 검증에는 선택한 타깃 구성 요소 집합에서의
통합 테스트가 필요합니다.
