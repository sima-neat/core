# 로컬 Kernels, Runner 및 Scheduler로 Core 빌드

Neat 개발 환경의 Core 체크아웃에서 평소 사용하는 명령을 실행합니다.

```bash
./build.sh --all --no-doc --install
```

빌드는 **Kernels → Runner → Scheduler → Core** 순서로 준비한 다음 기존 최종
설치/배포를 수행합니다. 하위 빌드는 SDK에 설치하거나 DevKit에 배포하지 않으며,
타깃 테스트 실행, EV 펌웨어 컴파일, 펌웨어 활성화도 수행하지 않습니다.

소스를 자동으로 찾을 수 있도록 체크아웃을 나란히 배치합니다.

```text
development/
├── core/
├── kernels/
├── runner/
└── scheduler/
```

Runner는 프레임워크에 독립적인 실행, 준비, 메모리 및 네이티브 전송을 담당합니다.
Scheduler는 GStreamer 통합/플러그인과 호환성 어댑터를 담당합니다. 각각 별도의
소스 루트와 설치 패키지를 갖습니다. 체크아웃에 `.git` 디렉터리가 없어도 됩니다.
워크트리와 소스 스냅샷도 지원합니다.

## 소스 또는 기존 로컬 패키지 선택

체크아웃이 다른 위치에 있으면 `NEAT_KERNELS_SOURCE`, `NEAT_RUNNER_SOURCE` 또는 `NEAT_SCHEDULER_SOURCE`를 설정합니다.

```bash
NEAT_RUNNER_SOURCE=/path/to/simaai-runtime ./build.sh --all --no-doc
```

이미 로컬에서 빌드한 패키지를 재사용하려면 `NEAT_KERNELS_PACKAGE_DIR`,
`NEAT_RUNNER_PACKAGE_DIR` 또는 `NEAT_SCHEDULER_PACKAGE_DIR`를 설정합니다. 각 패키지
디렉터리에는 설치된 `usr/` 트리, 빌드 영수증, 해당 구성 요소의 정식 패키지
워크플로가 생성한 일치하는 Debian 패키지가 있어야 합니다.
Kernels는 표준 CMake 구성/빌드/설치와 자체 패키지 생성기를 사용합니다.
Runner와 Scheduler는 각각 루트의 `build.sh` 진입점을 사용합니다.

```bash
NEAT_RUNNER_PACKAGE_DIR=/path/to/runner/dist/package \
  ./build.sh --all --no-doc
```

선택 우선순위는 명시적 재정의, 유효한 형제 소스, 형제 `dist/package` 순입니다.
명시적으로 패키지를 선택하면 형제 소스보다 우선합니다. 같은 구성 요소에 대해
소스와 패키지 재정의를 모두 설정하면 오류입니다. 로컬 빌드가 실패하면 전체
빌드가 중단되며, 이전 패키지로 폴백하지 않습니다.

**TODO: 네이티브 의존성 아티팩트 다운로드.** 구현되기 전까지 세 의존성 모두의
로컬 소스 또는 패키지를 제공해야 합니다. 입력이 누락되면 의존성 설치 전에
실패합니다. 기존 LLiMa 의존성 처리는 그대로 유지됩니다. 선택적 백엔드가 필요하지
않으면 `--no-llima`를 사용합니다.

## 플랫폼 의존성

네 저장소 모두 동일한 Platform 3.0 SDK/툴체인을 사용합니다. 기본값이 아닌 툴체인을
선택할 때는 `CMAKE_TOOLCHAIN_FILE`을 설정합니다. Kernels와 Runner 영수증의
타깃/컴파일러, 커널 계약 및 소비한 빌드 구성이 일치해야 합니다. 정식 패키지
워크플로는 소비한 Kernels 페이로드의 정확한 식별 정보도 요구합니다. Kernels
패키지가 바뀌면 Runner 소스가 바뀌지 않았더라도 선택한 SDK로 Runner 패키지를
다시 빌드합니다. 그러면 Core는 일치하는 Scheduler 클라이언트 구현에 다시 링크하며,
Scheduler는 하위 수치 연산 패키지를 소비합니다.
Core의 소스 우선 DEB는 선택된 Scheduler 및 수치 연산 SDK 버전을 고정합니다.
Runner는 Kernels를 고정하고, Scheduler는 Runner, Kernels 및 활성화된 코덱 백엔드를
고정합니다. 테스트가 활성화된 Core 빌드는 동일한 생성기의 별도
`simaai-scheduler-client-test-dev` 패키지도 필요합니다. 소스 빌드는
`--client-test-support`로 이를 요청하고, 패키지 빌드는 인접한
`client-test-support/` 디렉터리 또는 `NEAT_SCHEDULER_CLIENT_TEST_PACKAGE_DIR`를
선택합니다. 코디네이터는 테스트 전용 CMake 구성 요소를 노출하기 전에 해당
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

펌웨어 활성화에는 활성화 도우미를 제공하는 유일한 새 패키지만 사용합니다.
설치된 패키지의 식별 정보, 도우미의 소유권, 실행 모드 및 바이트가 선택한 패키지와 일치해야 합니다.
패키지 집합에 도우미가 없으면 이전 설치에서 남은 도우미를 호출하지 않고 기존 펌웨어를 그대로 둡니다.
선택한 도우미가 일치하지 않으면 설치가 실패합니다.

DevKit에서 설치가 중단되었다면 검증된 전체 배포본을 다시 실행하세요.
설치 프로그램은 미완료 패키지가 새 패키지 집합에 속하고 아키텍처가 일치할 때만 복구할 수 있습니다.
복구는 패키지를 제거하거나 관련 없는 패키지를 변경하지 않고 선택된 정확한 버전을 설치해야 합니다.
관련 없는 패키지 데이터베이스 오류는 별도로 복구해야 합니다. 설치 프로그램은 일반적인 APT 복구를 수행하거나 롤백을 보장하지 않습니다.

기존 Core 개발 패키지가 있는 환경에 Scheduler DEB만 설치하거나, `--force-overwrite`를 사용하거나, dpkg 소유 파일을 삭제하지 마세요.
독립형 Scheduler 테스트 이미지에는 Core 설치가 필요하지 않지만, 기존 Core 설치가 있다면 함께 업그레이드해야 합니다.

### 백엔드 제공자

Scheduler에는 별도로 설치된 플랫폼 백엔드가 필요합니다. 이 워크플로는 이를
Scheduler 안에 복사하거나 가져오지 않습니다.

- 별도로 빌드한 Codecs 패키지 디렉터리로 `NEAT_CODECS_PACKAGE_DIR`를 설정합니다.
  Scheduler가 코덱을 활성화하면 이 디렉터리에는 소비한 정확한 설치 `usr/` 접두사와
  일치하는 `neat-codecs` 및 `neat-codecs-dev` Debian 패키지가 있어야 합니다.
  Scheduler 의존성 코디네이터가 페이로드, 소유권 및 의존성 제어 정보를 검증합니다.
  Core는 검증된 의존성 집합을 유지하고 두 아카이브를 최종 배포본에 포함합니다.
  Codecs를 빌드하거나 다른 버전으로 대체하지는 않습니다.
- SDK의 일치하는 MLA Runtime, 렌더링 및 추적/로깅 라이브러리를 사용합니다.
- SDK 경로를 명시적으로 선택해야 하면 `NEAT_SCHEDULER_CMAKE_ARGS`로 CMake 정의를
  전달합니다. 예를 들어 `-DSIMA_RUNTIME_MLART_LIBRARY=/path/to/libmlart.so`와 일치하는
  포함 디렉터리를 전달합니다. Scheduler 빌드 가이드에 문서화된 옵션 이름을 사용합니다.

`NEAT_RUNNER_CMAKE_ARGS`와 `NEAT_SCHEDULER_CMAKE_ARGS`의 인수는 셸 방식의 단어로
구문 분석되며 셸 코드로 실행되지 않습니다. 선택한 의존성 패키지나 설치 레이아웃을
재정의할 수 없습니다.

## 증분 빌드 및 설치

Core가 소유한 의존성 빌드와 패키지는 각 저장소의 독립형 빌드 디렉터리와 분리된
`deps/backend/` 아래에 있습니다. CMake는 증분 빌드를 수행합니다. `--clean`은
선택된 로컬 의존성 빌드와 Core 빌드를 정리합니다.

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

헤더 전용 의존성 준비의 경우 `--install-deps-only`에는 로컬 패키지 재정의가
필요합니다. A65/EV 코드를 빌드하거나 실행하지 않고 헤더만 복사합니다.
네이티브 clang-tidy 분석에는 선택한 패키지 재정의 및 Scheduler 접두사와 함께
`scripts/run_cpp_tidy.sh`를 사용합니다. `TIDY_INSTALL_DEPS=1`은 네이티브 호스트
분석 전제 조건을 설치하고 설치된 Scheduler 코디네이터를 헤더 전용 모드로 호출합니다.
SDK sysroot를 활성화하거나 수정하지는 않습니다. LLiMa 기반 GenAI 구현 파일은
이 호스트 전용 분석 프로파일에서 제외됩니다. 스크립트는 선택한 빌드의 의존성
대신 주변 SDK의 LLiMa 패키지를 대입하지 않습니다. 문서 전용 빌드는 네이티브
의존성을 준비하지 않습니다.

EV 장치 빌드는 별도의 라이선스가 필요한 워크플로입니다. 이 명령들은 MetaWare를
암묵적으로 호출하거나 펌웨어를 교체하거나 펌웨어 변경을 검증하지 않습니다.

## 공개된 CI 의존성

일반 Vulcan 빌드는 `deps/manifest.json`의 Scheduler 정책을 `sima-cli`로 한 번
해석한 뒤 해당 불변 공개본을 다운로드합니다. snap 정책은 현재 브랜치를 선택하며,
공개되지 않은 브랜치는 다른 브랜치로 폴백하는 대신 실패합니다.
`tools/ci/acquire_scheduler.py`는 Scheduler 부트스트랩을 실행하기 전에 게시자
해시를 검증합니다. 설치된 Scheduler 코디네이터가 번들된 Runner, Kernels 및 Codecs
아카이브를 분리하고 검증합니다. Core는 해당 저장소들을 독립적으로 해석하지 않습니다.

선택한 Scheduler 패키지는 정확한 SDK 정책을 설치합니다. 일회용 CI SDK 컨테이너에서
`scripts/ci/prepare_scheduler_sdk.sh`는 해당 리비전의 공식 sysroot 설정 명령을 사용하고
다운로드한 플랫폼 아카이브를 보존합니다. 코디네이터는 실제 GCC14 버전/타깃과
타깃 SDK 식별 정보를 확인한 다음 `NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`를 통해
Runner가 소비한 정확한 런타임 제공자 아카이브를 선택합니다. 네이티브 ARM64와
크로스 컴파일러의 실행 파일 바이트가 같을 필요는 없습니다. 타깃 ABI와 선택한
플랫폼은 일치해야 합니다. 이 단계에서는 장치 설치나 펌웨어 검증을 수행하지 않습니다.

로컬 소스/패키지 재정의는 계속 우선합니다. CI 획득은 명시적으로 수행되며,
실패한 로컬 소스 빌드가 대체물을 다운로드하지 않습니다. 기존
`tools/ci/materialize_dependencies.py --manifest ... --output-root ...` 명령은 별도로
검토된 불변 매니페스트를 위한 선택적 전송 수단으로 남습니다. 기본 워크플로가
아니며 암묵적인 `ci/dependencies.json`도 없습니다.

코드 위생 검사는 이 실행의 `neat-core-dist`를 다운로드하고 검증한 뒤 실제 Core
위임 도구와 설치된 Scheduler 코디네이터를 헤더 전용 모드로 호출합니다. 코디네이터는
빌드에 사용한 동일한 하위 패키지와 인증된 플랫폼 빌드 증거를 확인합니다. Core
분석에는 Scheduler 프런트엔드 접두사만 노출됩니다. LLiMa는 Core의 직접 의존성으로
남습니다. 분석은 별도의 가변 최신 버전이 아니라 Core 빌드 영수증에 지정된 정확한
개발 아카이브를 다운로드합니다. 이는 호스트 도구 검증이며 타깃 실행이 아닙니다.

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

Core의 소스 우선 `build.sh`를 실행하기 전에 다음을 제공합니다.

- `NEAT_RUNNER_PLATFORM_RUNTIME_DEBS`: Runner가 선택한 정확한 런타임 제공자 DEB의
  절대 경로를 담은 JSON 배열입니다. 필수 SimaHeap 런타임 아카이브를 포함하고,
  해당 백엔드가 활성화되면 MLA 및 TVM 런타임 아카이브도 포함합니다. 이 배열에는
  개발 아카이브를 포함하지 마세요. Runner는 실제 로더 이름과 라이브러리 바이트를
  검증합니다. Scheduler는 정확한 아카이브 식별 정보와 페이로드 목록을 빌드 증거로
  유지합니다. Runner가 Platform 소유로 선언한 제공자는 Core의 설치 가능한 DEB
  집합에서 제외되며, 선언되지 않은 제공자는 정확한 아카이브 구성을 유지합니다.

미리 빌드된 Runner/Scheduler 패키지 재정의를 사용해도 이 선택은 명시적으로
유지됩니다. 불변 네 구성 요소 구체화 도구는 플랫폼 제공자 아카이브를 가져오지
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
