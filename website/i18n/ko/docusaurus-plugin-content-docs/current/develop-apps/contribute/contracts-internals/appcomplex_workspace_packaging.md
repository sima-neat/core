---
title: "AppComplex 작업 공간 패키징"
description: "게이트 방식으로 보호된 앱 복합 작업 공간 서비스 패키지를 구축하고 설치합니다."
sidebar_position: 3
slug: /develop-apps/contribute/appcomplex_workspace_packaging
---

# AppComplex 작업 공간 패키징

이 가이드에서는 `tmp/core/sima-ai-appcomplex`를 격리된 시스템 패키지로 패키징합니다. 사용자가 명시적으로 요청하지 않는 한, 이 패키지는 현재 실행 중인 `simaai-appcomplex.service`를 대체하지 않습니다.

## 설치되는 항목

- `/opt/simaai/appcomplex-workspace/` 아래의 바이너리 및 라이브러리
- Systemd 유닛: `simaai-appcomplex-workspace.service`
- 구성 파일: `/etc/default/simaai-appcomplex-workspace`

작업 공간 유닛은 기본적으로 격리된 엔드포인트를 사용합니다.

- 제어 소켓: `/tmp/mlactrl_workspace`
- SHM 객체: `/mlashmdata_workspace`
- MLA 초기화 게이트: `APP_COMPLEX_RUN_INIT=0` (병렬 실행의 경우 초기화 건너뛰기)

## 패키지 빌드

```bash
./scripts/release/build_appcomplex_workspace_deb.sh
```

스크립트는 생성된 `.deb` 경로를 `build/packages/`에 출력합니다.

## 설치 (기본 설정, 제한됨)

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb>
```

기본 설치 동작:

- `simaai-appcomplex.service`를 중지하거나 비활성화하지 않습니다.
- `simaai-appcomplex-workspace.service`를 자동으로 활성화하거나 시작하지 않습니다.

## 작업 공간 서비스를 수동으로 활성화합니다.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now simaai-appcomplex-workspace.service
```

작업 공간 시작 전에 MLA를 재초기화해야 하는 경우(마이그레이션 모드), 다음을 설정하십시오.

```bash
sudo sed -i 's/^APP_COMPLEX_RUN_INIT=.*/APP_COMPLEX_RUN_INIT=1/' /etc/default/simaai-appcomplex-workspace
```

## 선택적 전환(명시적 전환만 해당)

기존 서비스를 중지하고 워크스페이스 서비스를 활성화하도록 요청하려면:

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb> --activate --switch-system
```

또는 `/etc/default/simaai-appcomplex-workspace`를 업데이트합니다.

- `APP_COMPLEX_ACTIVATE_ON_INSTALL=1`
- `APP_COMPLEX_SWITCH_SYSTEM_SERVICE=1`

그런 다음 다음 명령을 실행합니다.

```bash
sudo dpkg-reconfigure simaai-appcomplex-workspace
```
