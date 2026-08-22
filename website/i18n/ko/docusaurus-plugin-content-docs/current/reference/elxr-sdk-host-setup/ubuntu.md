---
title: "Ubuntu 호스트 관련 참고 사항"
description: "Neat SDK 및 DevKit-Sync를 위해 Ubuntu 호스트를 준비합니다."
sidebar_position: 1
---

호스트 머신이 Ubuntu이고 Neat를 실행하려는 경우 이 가이드를 참조하십시오. 개발 환경(Neat SDK라고 함)을 DevKit-Sync와 함께 사용합니다.

## 필수 조건

- Ubuntu 22.04 또는 24.04 호스트.
- Docker 엔진이 설치되었고 정상적으로 작동합니다.
- 호스트에 `sima-cli`가 설치되었습니다.
- Modalix 및 DevKit는 동일한 네트워크에서 연결 가능합니다.

:::info 네트워크 토폴로지
Ubuntu에서는 DevKit을 USB/이더넷을 통해 호스트에 직접 연결하거나, 호스트와 DevKit을 기존 네트워크에 별도로 배치할 수 있습니다. 기존 네트워크에 있는 경우, 호스트와 DevKit이 SSH 및 NFS 트래픽을 통해 서로 통신할 수 있다면 특별한 공유 설정이 필요하지 않습니다.
:::

## Ubuntu에서 DevKit으로 직접 연결

DevKit이 USB/이더넷을 통해 Ubuntu 장치에 직접 연결되어 있고 Ubuntu 장치의 네트워크 연결을 공유해야 하는 경우 다음 단계를 따르십시오.

공유되는 DevKit에 연결된 네트워크 인터페이스에서 IPv6를 비활성화합니다. DevKit-Sync는 SSH 및 NFS를 위해 예측 가능한 IPv4 주소 지정을 사용하며, 공유 링크에서 IPv6를 활성화하면 장치 검색 및 경로 선택이 불안정해질 수 있습니다.

### NetworkManager GUI

1. Ubuntu 머신을 Wi-Fi 또는 다른 상위 인터페이스를 통해 인터넷에 연결합니다.
2. USB/이더넷 어댑터를 통해 DevKit을 Ubuntu 장치에 연결합니다.
3. DevKit에서 연결된 네트워크 인터페이스 설정을 `DHCP`로 유지합니다.
4. Ubuntu에서 `Settings > Network`를 엽니다.
5. DevKit에 연결된 유선 인터페이스의 설정을 엽니다.
6. `IPv4` 탭에서 `IPv4 Method`를 `Shared to other computers`로 설정합니다.
7. `IPv6` 탭에서 `IPv6 Method`를 `Disabled`로 설정합니다.
8. 변경 사항을 적용한 후 유선 인터페이스 연결을 해제했다가 다시 연결합니다.

링크가 활성화되면 Ubuntu에서 DevKit의 IPv4 주소를 찾으세요.

```bash
ip neigh
```

SDK 설정을 시작하기 전에 SSH 접속을 확인하세요.

```bash
ssh sima@<devkit-ip>
```

그런 다음 DevKit 페어링을 진행합니다.

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

### NetworkManager 명령줄 인터페이스

명령줄 설정을 선호하는 경우, DevKit에 연결되는 인터페이스를 확인하세요.

```bash
nmcli device status
```

IPv6를 비활성화하고 공유 IPv4 연결을 생성합니다.

```bash
sudo nmcli connection add type ethernet ifname <devkit-interface> con-name devkit-shared ipv4.method shared ipv6.method disabled
sudo nmcli connection up devkit-shared
```

해당 인터페이스에 대한 연결 프로필이 이미 존재하는 경우, 해당 프로필을 수정하십시오.

```bash
sudo nmcli connection modify "<connection-name>" ipv4.method shared ipv6.method disabled
sudo nmcli connection down "<connection-name>"
sudo nmcli connection up "<connection-name>"
```

## 방화벽 관련 참고 사항

Ubuntu 방화벽 규칙이 활성화된 경우, DevKit에 연결된 인터페이스 또는 서브넷에서 SSH 및 NFS 트래픽을 허용한 후 DevKit-Sync 설정을 실행합니다. 최소한 DevKit는 SSH에 접속할 수 있어야 하며, `sima-cli sdk setup --devkit`에서 생성된 호스트 NFS 내보내기에 접근할 수 있어야 합니다.

## 다음 단계

[Neat SDK](/getting-started/dev-environment/)로 돌아가서 설치/설정 작업을 계속하십시오.
