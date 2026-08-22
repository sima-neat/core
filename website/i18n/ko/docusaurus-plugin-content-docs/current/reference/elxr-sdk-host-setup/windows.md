---
title: "Windows 호스트 관련 정보"
description: "Windows 11 호스트를 Neat SDK 및 DevKit-Sync에 맞게 준비합니다."
sidebar_position: 2
---

호스트 머신이 Windows 11이고 DevKit-Sync를 사용하여 Neat 개발 환경(이하 Neat SDK)을 실행하려는 경우 이 가이드를 참조하십시오.

## 필수 조건

- Windows 11 호스트.
- [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)이 설치되었고 정상적으로 작동합니다.
- WSL 내부에 설치된 Docker 엔진입니다.
- `sima-cli`가 WSL 내부에 설치되었습니다.

## WSL에서 시작하세요.

WSL Linux 배포판 내에서 Neat SDK 명령을 실행하세요. PowerShell 또는 명령 프롬프트에서 실행하지 마세요. 여기에는 `sima-cli neat install sdk@release-2.1`이 포함됩니다.

## WSL 네트워크 모드

`%UserProfile%\\.wslconfig`를 구성합니다.

```ini
[wsl2]
networkingMode=mirrored
```

그런 다음 WSL을 다시 시작합니다.

```powershell
wsl --shutdown
```

이렇게 하면 WSL이 호스트 네트워크 구성을 공유할 수 있으며, 이는 DevKit-Sync 및 NFS 통신에 도움이 됩니다.

## 권장되는 연결 방식: Windows에서 DevKit으로 직접 연결

Windows 호스트의 경우, Windows 장치와 DevKit 간의 직접 USB/이더넷 연결이 권장되는 구성입니다. 일반적으로 DevKit을 더 넓은 공유 네트워크에 배치하는 것보다 설정하기가 더 간단하며, Windows 방화벽 변경 사항을 전체 네트워크가 아닌 로컬 DevKit에 연결된 인터페이스로 제한할 수 있습니다. Ubuntu 및 macOS와 달리, Windows에서는 공유 네트워크에 대해 이미 유효성이 검사된 방화벽 및 WSL 네트워킹 규칙이 있는 경우가 아니면 이 직접 연결 구성을 사용하는 것이 좋습니다.

DevKit이 직접 연결을 통해 Windows 장치의 네트워크 연결을 공유해야 하는 경우 인터넷 연결 공유(ICS)를 사용합니다.

1. Windows 장치를 Wi-Fi 또는 다른 네트워크 인터페이스를 통해 인터넷에 연결합니다.
2. USB/이더넷 어댑터를 통해 DevKit을 Windows 컴퓨터에 연결합니다.
3. DevKit에서 연결된 네트워크 인터페이스 설정을 `DHCP`로 유지합니다.
4. Windows에서 `Control Panel > Network and Internet > Network Connections`를 엽니다.
   또한 `Win + R` 키를 누르고, `ncpa.cpl`을 실행한 다음 Enter 키를 누를 수도 있습니다.
5. 인터넷에 연결된 어댑터를 마우스 오른쪽 버튼으로 클릭한 다음 `Properties`를 선택합니다.
6. `Sharing` 탭을 엽니다.
7. `Allow other network users to connect through this computer's Internet connection`를 활성화합니다.
8. 안에 `Home networking connection`USB/이더넷 어댑터를 선택하여 연결합니다. DevKit.
9. 변경 사항을 적용한 후, DevKit가 신호를 받지 못하는 경우 DevKit에 연결된 어댑터를 다시 연결하십시오.
   IPv4 주소입니다.

ICS가 활성화되면 Windows는 일반적으로 공유 어댑터에 `192.168.137.0/24`의 주소를 할당합니다.
WSL 또는 DevKit 콘솔에서 DevKit의 IPv4 주소를 찾은 다음 WSL에서 SSH 액세스가 가능한지 확인합니다.

```bash
ssh sima@<devkit-ip>
```

그런 다음 WSL에서 DevKit 페어링을 진행합니다.

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

:::note Windows에서 직접 링크를 통해 Insight에 접근
Windows의 직접 네트워크 공유 기능을 사용하면 Windows 방화벽 및 WSL 포트 전달 동작으로 인해 네트워크의 다른 장치에서 Neat Insight 웹 인터페이스에 접근하지 못할 수 있습니다. 이 경우 Windows Neat SDK 호스트에서 Insight를 직접 열어 사용합니다. 예를 들어 `https://localhost:9900`에서 열 수 있습니다.
:::

## NFS 방화벽 규칙 (PowerShell)

Windows 방화벽에서 NFS 관련 트래픽을 허용합니다. 관리자 권한으로 PowerShell을 실행하고 `New-NetFirewallRule`을 사용하여 필요한 NFS 포트/프로토콜에 대한 규칙을 추가합니다.

예시:

```powershell
New-NetFirewallRule -DisplayName "Allow NFS TCP 2049" -Direction Inbound -Protocol TCP -LocalPort 2049 -Action Allow
New-NetFirewallRule -DisplayName "Allow NFS UDP 2049" -Direction Inbound -Protocol UDP -LocalPort 2049 -Action Allow
```

NFS 서버/클라이언트 설정에 필요한 추가 포트를 추가하세요.

## 다음 단계

[Neat SDK](/getting-started/dev-environment/)로 돌아가서 설치/설정 작업을 계속하십시오.
