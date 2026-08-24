---
title: "macOS 호스트 관련 참고 사항"
description: "Neat SDK 및 DevKit-Sync를 위해 macOS 호스트를 준비합니다."
sidebar_position: 3
---

호스트 머신이 macOS이고 Neat 개발 환경(Neat SDK로 칭함)을 DevKit-Sync와 함께 실행하려는 경우 이 가이드를 참조하십시오.

## 필수 조건

- macOS 호스트.
- 호스트에 `sima-cli`가 설치되었습니다.
- Modalix 및 DevKit는 동일한 네트워크에서 연결 가능합니다.

:::info 네트워크 토폴로지
macOS에서는 DevKit을 USB/이더넷을 통해 호스트에 직접 연결하거나, 호스트와 DevKit을 기존 네트워크에 별도로 배치할 수 있습니다. 기존 네트워크에 있는 경우, 호스트와 DevKit이 SSH 및 NFS 트래픽을 통해 서로 통신할 수 있다면 특별한 공유 설정이 필요하지 않습니다.
:::

## Colima를 설치하고 실행합니다.

Colima를 설치하고 실행하여 macOS에서 Docker 워크로드가 실행될 수 있도록 합니다.

```bash
brew install colima docker
colima start
docker ps
```

Colima가 이미 설치되어 있다면, `sima-cli sdk setup`을 사용하기 전에 Colima가 실행 중인지 확인하세요.

## macOS에서 NFS 권한 설정

DevKit-Sync는 SDK 설정 중에 호스트 NFS 내보내기를 사용합니다. macOS에서는 `nfsd`에 전체 디스크 접근 권한이 있는지 확인하십시오. 그렇지 않으면 호스트 작업 공간 내보내기/마운트가 실패할 수 있습니다.

단계:

1. 시스템 설정을 엽니다.
2. 개인 정보 보호 및 보안 > 전체 디스크 접근 권한으로 이동합니다.
3. `+`을 클릭한 다음 `Cmd + Shift + G`를 누르고 `/sbin/`을 입력합니다.
4. `nfsd`를 선택하고 활성화되어 있는지 확인하세요.
5. 권한이 부여된 후 SDK 설정을 다시 실행합니다.

## Mac과 DevKit 간의 직접 연결을 위한 인터넷 공유

DevKit이 일반 네트워크에 직접 연결할 수 없고 Mac과의 직접 USB/이더넷 링크를 통해 인터넷에 액세스해야 하는 경우 이 방법을 사용하십시오.

1. DevKit에서 연결된 네트워크 인터페이스를 `DHCP`로 설정합니다(일반적으로 기본 설정입니다).
2. macOS에서 `System Settings > General > Sharing > Internet Sharing`을 엽니다.
3. **연결을 공유할 네트워크**를 `Wi-Fi`로 설정하세요.
4. DevKit에 연결된 USB/이더넷 동글 인터페이스를 통해 공유 기능을 활성화합니다.
5. Mac에서는 USB/이더넷 어댑터 인터페이스가 `DHCP`로 설정되어 있는지 확인하십시오.

인터넷 공유 기능이 활성화되면 Mac의 활성 USB/이더넷 인터페이스는 주소(예: `en0` 또는 `en1`, 어댑터 및 호스트 설정에 따라 다름)를 수신해야 합니다.

### DevKit에서 DNS 문제를 해결하는 방법

이 직접 연결 시나리오에서 DHCP가 성공적으로 완료된 후에도 DevKit의 DNS 설정이 잘못 구성된 상태로 유지될 수 있습니다. 이름 확인에 실패하면 DevKit의 `/etc/resolv.conf` 파일을 업데이트하십시오.

```bash
sudo nano /etc/resolv.conf
```

세트:

```text
nameserver 8.8.8.8
nameserver 127.0.0.1
```

## Colima UDP 포워딩을 사용하여 Insight 비디오 문제 해결

Insight가 브라우저에서 열리지만 실시간 비디오가 표시되지 않으면 UDP 패킷이 SDK 컨테이너에 제대로 전달되지 않을 수 있습니다. macOS에서 Colima를 사용하는 경우, Colima가 SSH 포트 포워딩을 사용할 때 이러한 문제가 발생할 수 있습니다. Docker는 예상되는 UDP 포트 매핑을 표시할 수 있지만, Colima의 호스트-VM 포워딩 경로가 컨테이너로 UDP 트래픽을 전달하지 못할 수 있습니다.

SDK는 일반적으로 다음 UDP 범위를 사용합니다.

- 비디오용 `9000-9079/udp`
- 메타데이터용 `9100-9179/udp`
- WebRTC용 `40000-40199/udp`

SDK 컨테이너가 실행 중이고 해당 UDP 매핑이 존재한다면, Colima 포트 포워더를 확인하세요. SSH 포워딩은 TCP만 지원하는 반면, gRPC 포워딩은 TCP와 UDP를 모두 지원합니다.

gRPC 포워딩을 사용하도록 Colima를 재구성하세요:

```bash
colima stop
colima start --edit
```

에디터에서 다음을 변경합니다.

```yaml
portForwarder: ssh
```

받는 사람:

```yaml
portForwarder: grpc
```

그런 다음 SDK를 다시 시작합니다.

```bash
sima-cli sdk stop
sima-cli sdk start
sima-cli sdk neat
```

Docker가 여전히 UDP 포트를 게시하는지 확인합니다.

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

SDK 컨테이너 목록에 `9000-9079/udp` 및 `9100-9179/udp`, 그리고 WebRTC UDP 범위가 포함되어 있는지 확인합니다.

비디오를 전송하는 동안 Insight에서 SDK 내부에서 들어오는 패킷을 감지하는지 확인합니다.

```bash
curl -k 'https://127.0.0.1:9900/api/ingest/stats?all=1&verbose=1'
```

예상 채널에서 `packets_received` 값이 증가하는지 확인합니다. 송신자가 SDK 컨테이너 IP가 아닌 Mac 호스트 IP와 올바른 UDP 포트를 대상으로 하는지 확인합니다. 채널 0은 UDP `9000`을 사용하고, 채널 1은 UDP `9001`을 사용하며, 이와 같은 방식으로 진행됩니다.

Colima를 gRPC 전달 방식으로 전환한 후에도 UDP가 여전히 Insight에 도달하지 않으면 Docker Desktop 또는 Linux 호스트로 테스트합니다. 이 경우 SDK 및 Docker 포트 매핑이 올바를 가능성이 높으며, 나머지 의심스러운 부분은 Colima의 macOS UDP 전달 경로입니다.

## 다음 단계

[Neat SDK](/getting-started/dev-environment/)로 돌아가서 설치/설정 작업을 계속하십시오.
