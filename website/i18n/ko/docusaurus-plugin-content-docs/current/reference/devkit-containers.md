---
title: "DevKit 컨테이너"
description: "Modalix DevKit에 Docker 설치, dk container 명령, Ubuntu 컨테이너 및 문제 해결"
sidebar_position: 11
---

# DevKit 컨테이너

이 페이지는 플랫폼 버전 3.0의 Modalix DevKit에서 컨테이너를 실행하기 위한 참조 문서입니다. 개요와 전체 워크플로는
[컨테이너 애플리케이션 개발](/getting-started/dev-environment/containerized-apps/)부터 시작하십시오.

## 가속기 액세스 {#accelerators}

Neat 애플리케이션은 컨테이너 내에서 MLA를 사용할 수 있습니다. 컨테이너에는 다음이 필요합니다.

- Docker의 `--device` 옵션으로 전달하는 `/dev/mla`, `/dev/cvu`, `/dev/dma_heap/linux,cma` 등의 필수 장치에 대한 액세스.
- 이미지에 포함하거나 DevKit에서 마운트한 호환 Neat 라이브러리, GStreamer 플러그인 및 플랫폼 사용자 공간 라이브러리.
- 호환되는 DevKit 플랫폼과 컨테이너 플랫폼 감지를 지원하는 Neat Core 버전. 애플리케이션은 런타임의 ABI와 일치해야 합니다.

최신 DevKit에서는 `--user UID:GID`를 사용하여 DevKit 사용자의 UID 및 GID로 워크로드를 실행하고 매핑된 장치에 대한 액세스 권한을 확보하십시오. Docker 액세스가 설정되면 MLA 장치 액세스를 위해 애플리케이션에 `sudo` 또는 `--privileged`가 필요하지 않습니다. 현재 공식 실행기는 보안 제한을 해제하는 옵션도 사용하지 않습니다. Docker 설치 및 호스트 설정을 위한 관리자 액세스는 애플리케이션의 장치 권한과 별개입니다.

커널 드라이버와 펌웨어는 여전히 DevKit에서 제공합니다. 장치를 전달하는 것만으로 임의의 Ubuntu 이미지에서 가속기를 사용할 수 있는 것은 아닙니다. DSP 워크로드에는 별도의 호환 런타임과 장치 설정이 필요하며, 이 가이드는 임의의 컨테이너 이미지에서 DSP 지원을 보장하지 않습니다.

전체 장치 및 라이브러리 설정은 공식 [Docker 병렬 앱 예제](https://github.com/sima-neat/apps/tree/develop/examples/docker/parallel-apps)를 참조하십시오. 선택한 예제 및 플랫폼 버전의 사전 요구 사항을 따르십시오.

## DevKit에 Docker 설치 {#install}

직접 관리하는 DevKit에서는 Palette SDK 또는 수동 방식으로 Docker를 설치합니다. 관리자가 관리하는 DevKit에서는 관리자에게 Docker 활성화를 요청하고, 직접 설치하거나 교체하지 말고 제공된 서비스와 함께 `dk container`를 사용하십시오. 아래 두 방법 중 하나로 새로 설치하면 다음 작업을 수행합니다.

- Docker의 Debian `trixie` 저장소에서 ARM64용 Docker CE(`docker-ce`, `docker-ce-cli`, `containerd.io`)를 설치합니다.
- 이미지가 루트 파일 시스템을 채우지 않도록 Docker 데이터는 `/data/docker`에, containerd 데이터는
  `/data/containerd`에 저장합니다.
- 컨테이너 네트워킹에 필요한 IP 포워딩을 켭니다.
- `docker` 및 `containerd` 서비스를 시작하고, 부팅할 때마다 시작되도록 설정합니다.
- 사용자가 `sudo` 없이 `docker`를 실행할 수 있도록 DevKit 사용자를 `docker` 그룹에 추가합니다.

두 방법 모두 DevKit의 인터넷 연결이 필요합니다.

### Palette SDK에서 설치 {#install-sdk}

먼저 `sima-cli sdk setup --devkit {devkit-ip}`로 DevKit를 페어링하십시오. 직접 관리하는 DevKit에서는 설치를 위한 비밀번호 없는 `sudo`가 설정됩니다. 관리자가 관리하는 DevKit에서는 기존 액세스 정책과 제공된 Docker 서비스를 사용합니다. 그런 다음 SDK 셸에서 다음을 실행합니다.

<ShellCommand prompt="sdk">
dk container setup
</ShellCommand>

DevKit에 Docker를 설정할지 물으면 `y`로 응답하십시오. 질문을 건너뛰려면 `dk container setup --yes`를 실행합니다. Docker가 이미 작동하면 재사용합니다. 관리자가 관리하는 DevKit에서는 무제한 `sudo` 권한 없이 제공된 서비스를 통해 레지스트리 액세스를 설정합니다. SDK가 관리하는 설치가 중단되면 명령을 다시 실행하여 재개하십시오.

직접 관리하는 DevKit에서는 다른 `dk container` 명령도 Docker가 없을 때 설치를 제안합니다.

### DevKit에서 수동 설치 {#install-manual}

Palette SDK 없이 직접 관리하는 DevKit에 Docker를 새로 설치할 때 이 방법을 사용하십시오. 관리자 액세스가 필요합니다. 기존 설치가 있다면 설정을 보존하고 SDK 설정 경로를 사용하십시오. 아래 모든 명령은 SSH 또는 시리얼 콘솔을 통해 DevKit에서 실행합니다. `sudo`가 비밀번호를 묻습니다. 단계가 실패하면 중지하고 문제 해결 전에 9단계에 따라 패키지 미러를 복원하십시오. 다음 단계로 진행하지 마십시오.

#### 1. eLxr 패키지 미러 목록 옮겨 두기

9단계에서 원래 위치로 되돌립니다.

<ShellCommand prompt="devkit">
sudo mkdir -p /root/apt-disabled
sudo mv /etc/apt/sources.list.d/0000mirror.list /root/apt-disabled/
</ShellCommand>

#### 2. Docker 데이터를 /data에 저장

<ShellCommand prompt="devkit">
sudo mkdir -p /etc/docker /data/docker
echo '{"data-root": "/data/docker", "firewall-backend": "nftables"}' | sudo tee /etc/docker/daemon.json
</ShellCommand>

#### 3. IP 포워딩 켜기

<ShellCommand prompt="devkit">
printf 'net.ipv4.ip_forward=1\nnet.ipv6.conf.all.forwarding=1\n' | sudo tee /etc/sysctl.d/99-docker-forward.conf
sudo sysctl --system
</ShellCommand>

#### 4. Docker 패키지 저장소 추가

<ShellCommand prompt="devkit">
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo 'deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian trixie stable' | sudo tee /etc/apt/sources.list.d/docker.list
</ShellCommand>

#### 5. Docker 설치

<ShellCommand prompt="devkit">
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
</ShellCommand>

#### 6. containerd 데이터를 /data로 이동

<ShellCommand prompt="devkit">
sudo systemctl stop docker.socket docker containerd
sudo sh -eu <<'SH'
if mountpoint -q /var/lib/containerd; then
  echo '/var/lib/containerd is already mounted; stop and inspect its storage.' >&2
  exit 1
fi
mkdir -p /var/lib/containerd /data/containerd
if [ -n "$(find /data/containerd -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo '/data/containerd is not empty; stop and inspect it before migrating.' >&2
  exit 1
fi
cp -a /var/lib/containerd/. /data/containerd/
diff -qr /var/lib/containerd /data/containerd
SH
</ShellCommand>

복사 및 비교가 성공한 경우에만 계속하십시오. 원본 데이터는 바인드 마운트로 가려지기 전까지 `/var/lib/containerd`에 남습니다. 설정 중에는 삭제하지 마십시오. 복사가 실패하면 서비스를 중지한 상태로 유지하고 재시도 전에 두 디렉터리를 확인하십시오.

#### 7. 부팅할 때마다 /data/containerd 마운트

<ShellCommand prompt="devkit">
sudo grep -Fqx '/data/containerd /var/lib/containerd none bind 0 0' /etc/fstab ||
  echo '/data/containerd /var/lib/containerd none bind 0 0' | sudo tee -a /etc/fstab
sudo mount /var/lib/containerd
</ShellCommand>

#### 8. 서비스 시작

<ShellCommand prompt="devkit">
sudo systemctl enable containerd docker
sudo systemctl restart containerd docker
</ShellCommand>

#### 9. eLxr 패키지 미러 목록 되돌리기

<ShellCommand prompt="devkit">
sudo mv /root/apt-disabled/0000mirror.list /etc/apt/sources.list.d/
</ShellCommand>

#### 10. 사용자를 docker 그룹에 추가

<ShellCommand prompt="devkit">
sudo usermod -aG docker "$USER"
</ShellCommand>

그룹 변경 사항을 적용하려면 로그아웃했다가 다시 로그인하십시오.

## dk container 명령 {#dk-container}

이 명령은 Palette SDK 셸에서 실행합니다. 명령은 SSH를 통해 페어링된 DevKit에 연결하고 그곳에서 Docker를 실행합니다.

| 명령 | 기능 | 로컬 레지스트리 필요 여부 |
| --- | --- | --- |
| `dk container setup [--yes]` | DevKit에 Docker를 설치하거나, 작동 중인 기존 설치를 유지합니다. | 아니요 |
| `dk container deploy <image> [options] [-- command]` | 로컬 레지스트리에서 DevKit로 이미지를 다운로드한 다음 시작합니다. | 예 |
| `dk container run <image> [options] [-- command]` | 이미 DevKit에 있는 이미지를 시작합니다. | 예 |
| `dk container pull <image>` | 로컬 레지스트리에서 DevKit로 이미지를 다운로드합니다. | 예 |
| `dk container images` | 로컬 레지스트리에서 가져온 DevKit의 이미지를 나열합니다. | 예 |
| `dk container list` | 실행 중이거나 중지된 DevKit의 모든 컨테이너를 나열합니다. | 아니요 |
| `dk container logs <container> [options]` | 컨테이너의 출력을 표시합니다. 계속 확인하려면 `--follow`를 추가하십시오. | 아니요 |
| `dk container stop <container>` | 컨테이너를 중지합니다. | 아니요 |
| `dk container remove <container> [options]` | 컨테이너를 제거합니다. `rm`도 사용할 수 있습니다. | 아니요 |
| `dk container help` | 명령 요약을 표시합니다. | 아니요 |

명령이 인수를 처리하는 방식은 다음과 같습니다.

- **이미지 이름.** 레지스트리 주소 없이 이미지 이름을 지정합니다. 예를 들면
  `my-app:latest`입니다. `dk`가 로컬 레지스트리의 DevKit 측 주소를 추가합니다.
  따라서 `deploy`, `run`, `pull`은 로컬 레지스트리의 이미지에서만 작동합니다.
- **옵션.** 이미지 이름 뒤와 `--` 앞의 인수는 `--detach`, `--name`, `--rm`, `--network host`와 같은 `docker run`
  옵션입니다. `--` 뒤의 인수는 컨테이너에서 실행할 명령입니다.
- **아키텍처.** `deploy` 및 `run`은 ARM64가 아닌 이미지를 거부합니다.

예를 들어 이미지를 시작하면서 명령을 재정의하려면 다음을 실행합니다.

<ShellCommand prompt="sdk">
dk container run my-app:latest --rm -- sh -c 'uname -m'
</ShellCommand>

## Ubuntu 컨테이너 실행 {#ubuntu}

Docker Hub의 Ubuntu 이미지는 DevKit의 Arm CPU에서 실행됩니다. 시험해 보려면 DevKit에서 다음 명령을 실행하십시오.

<ShellCommand prompt="devkit">
docker run --rm ubuntu:24.04 cat /etc/os-release
</ShellCommand>

`PRETTY_NAME="Ubuntu 24.04.5 LTS"` 또는 이후 포인트 릴리스가 출력됩니다. 대화형 셸을 사용하려면 `docker run --rm -it ubuntu:24.04 bash`를 실행하십시오. 컨테이너 안에서는
`apt-get`이 ARM64용 Ubuntu 패키지를 설치합니다.

컨테이너는 DevKit의 Linux 커널을 공유하며, 별도의 운영 체제 설치가 아닙니다. 컨테이너에서 변경한 파일은 컨테이너를 제거하면 사라집니다. 보관하려는 파일은 `-v`를 사용하여 DevKit 디렉터리를 컨테이너에 마운트하십시오.

Ubuntu 기반 애플리케이션을 배포하려면 `Dockerfile`을
`FROM ubuntu:<release>`로 시작하고, [컨테이너 애플리케이션 개발](/getting-started/dev-environment/containerized-apps/)에 설명된 대로 SDK에서 빌드하고 배포하십시오.
`dk container run ubuntu:24.04`는 작동하지 않습니다. `dk`가 로컬 레지스트리에서 이미지를 찾기 때문입니다.

위 Ubuntu 명령은 CPU 실행만 보여 줍니다. Neat 가속기 워크로드는 [가속기 액세스](#accelerators)를 따르십시오. 기본 이미지를 변경하는 것만으로 필요한 장치와 런타임 라이브러리가 제공되지는 않습니다.

## 설치 확인 {#check}

DevKit에서 다음을 실행합니다.

<ShellCommand prompt="devkit">
docker info --format '{{.DockerRootDir}}'
docker run --rm hello-world
</ShellCommand>

첫 번째 명령은 `/data/docker`를 출력합니다. 두 번째 명령은 `Hello from Docker!`를 출력합니다.

## /data의 여유 공간 {#storage}

Docker는 이미지와 컨테이너를 약 3 GB인 `/data`에 저장합니다. DevKit에서 여유 공간을 확인하십시오.

<ShellCommand prompt="devkit">
df -h /data
</ShellCommand>

공간을 확보하려면 더 이상 필요하지 않은 컨테이너와 이미지를 제거하십시오.

<ShellCommand prompt="devkit">
docker container prune
docker image prune -a
</ShellCommand>

`docker image prune -a`는 컨테이너에서 사용하지 않는 모든 이미지를 제거합니다.

## 문제 해결 {#troubleshooting}

| 메시지 | 해결 방법 |
| --- | --- |
| `The local container registry is not configured for this SDK shell.` | 호스트에서 `sima-cli sdk setup --devkit {devkit-ip}`를 실행하고 컨테이너 레지스트리 질문에 `y`로 응답한 다음 새 SDK 셸을 여십시오. |
| `Docker installation requires passwordless sudo` | 직접 관리하는 DevKit에서는 `sima-cli sdk setup --devkit {devkit-ip}`로 다시 페어링하거나 Docker를 [수동](#install-manual)으로 설치하십시오. 관리자가 관리하는 DevKit에서는 관리자에게 Docker 활성화를 요청하십시오. |
| `Image ... is for amd64, not ARM64.` | `docker buildx build --platform linux/arm64 ... --push`로 이미지를 다시 빌드하십시오. |
| `Image ... is not available on the DevKit.` | 먼저 `dk container deploy` 또는 `dk container pull`을 사용하여 이미지를 다운로드하십시오. |
| DevKit에서 `permission denied while trying to connect to the Docker daemon socket` | 사용자가 `docker` 그룹에 포함되도록 Docker 설정 후 로그아웃했다가 다시 로그인하십시오. |
| 빌드 중 `exec format error` | `Dockerfile`의 `RUN` 단계에 `x86_64` 호스트에서의 ARM64 에뮬레이션이 필요합니다. 호스트에서 `docker run --privileged --rm tonistiigi/binfmt --install arm64`로 설치한 다음 다시 빌드하십시오. |
