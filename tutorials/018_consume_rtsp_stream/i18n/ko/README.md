# 018 실시간 RTSP 스트림 사용

## Metadata
| Field | Value |
| --- | --- |
| Category | Cameras & Streaming |
| Difficulty | Intermediate |
| Estimated Read Time | 5-10 minutes |
| Model | None |
| Labels | rtsp, h264, h265, streaming, input-group, live-input |

## Concept

실시간 H.264 또는 H.265 RTSP 스트림을 `Graph`에 연결하고,
`RtspDecodedInput` 프래그먼트를 사용합니다. 이 프래그먼트는 RTSP에 연결되고, 일치하는 RTP 디패킷화 장치 및 파서를 선택한 다음, 스트림을 원시 프레임으로 디코딩합니다.

## Walkthrough

이 챕터는 입력이 *프로그램 외부*에서 시작되는 첫 번째 챕터입니다. 이전 챕터에서는 테스트 이미지를 생성하거나 디스크에서 파일을 읽었습니다. 여기서는 네트워크 스트림에서 프레임이 지속적으로 도착하며, 사용자는 가능한 한 빠르게 해당 프레임을 처리합니다. 이 메커니즘은 재사용 가능한 `Graph` 조각인 `RtspDecodedInput`이며, 이는 전체 RTSP-to-raw-frames 프런트 엔드를 하나의 인터페이스 뒤에 묶습니다.

이 챕터는 의도적으로 "디코딩된 프레임을 가져오는" 단계에서 중단됩니다. 해당 프레임을 `Model`에 공급하는 것은 다른 곳에서 다룹니다(001은 단일 모델 실행, 007은 모델을 파이프라인에 연결, 015는 모델을 그래프 내에 임베딩). 마지막에는 RTSP URL에 연결하고 각 디코딩된 프레임의 텐서 형태를 출력하여 스트림이 제대로 작동하는지 확인할 수 있습니다.

이것은 *소비자* 역할만 합니다. 스트림을 게시하려면 별도의 RTSP 서버(예: `mediamtx`)를 실행하고 `--url`을 해당 서버로 지정합니다.

### RTSP 클라이언트 구성 {#step-configure-rtsp}

`RtspDecodedInputOptions`는 소스와 디코더를 구성합니다. `url`은
`rtsp://...` 소스를 선택합니다. `codec`는 인코딩된 형식을 선택합니다. 기본값은 H.264이며, 이 튜토리얼에서는 `avc`, `h265` 및 `hevc`도 허용합니다. 여기서 AVC는 H.264와 같고 HEVC는 H.265와 같습니다.

소스의 프레임 속도를 이미 알고 있는 경우 `source_fps`를 설정합니다. 생략하면 이 튜토리얼은 RTSP 소스를 OpenCV로 열고, 보고된 FPS를 읽고, 감지된 값을 `RtspDecodedInput`에 공급합니다. 그룹 자체는 URL을 조사하지 않습니다. 조사 경로에만 OpenCV가 필요하며, Python 버전은 필요에 따라 가져오므로 `--source-fps`를 공급하면 OpenCV 없이도 실행됩니다. 조사를 수행하려면 `pip install opencv-python`을 사용하여 설치합니다. H.265의 경우 Neat는 이 값을 파싱된 스트림 캡과 디코더 구성에 전달합니다. 프레임 속도는 변경되지 않습니다. H.265 스트림은 HEVC Main 프로필, 8비트, 4:2:0 입력을 사용해야 합니다.

`tcp = true`를 설정하면 RTP를 TCP를 통해 전송합니다. TCP는 순서를 유지하고 손실된 세그먼트를 재전송하므로 UDP에 비해 눈에 띄는 손실을 줄일 수 있지만, 손실된 데이터를 복구하는 동안 지연 시간이 증가할 수 있습니다.

### 그래프 구성 {#step-compose-graph}

다음과 같이 번역합니다.

다음과 같이 번역합니다. `Graph` 단 두 단계로 구성됩니다. `RtspDecodedInput` 단편(원본)과 기본적인 `Output` 노드(데이터를 가져오는 엔드포인트). 이 부분을 추가하면 하나의 `add(...)` — 내부적으로 연결/패킷 분할/디코딩 요소로 확장되므로 사용자의 구성은 의도 수준으로 유지됩니다. 입력이 파이프라인 *내부*에서 시작되기 때문에, `build(RunOptions{})` 초기 샘플이 필요 없는 과부하: 프레임을 제공할 수 없습니다. `build()` 처음부터, 스트림에서 해당 데이터를 생성하므로.

### 디코딩된 프레임을 가져옵니다. {#step-pull-frames}

실행 중인 상태에서 루프를 반복하고 `pull(...)` 타임아웃과 함께. 성공적인 각 풀 요청은 다음을 반환합니다. `Sample` 텐서가 하나의 디코딩된 프레임인 경우입니다. 이 튜토리얼에서는 기본 NV12 출력을 사용하며, 이는 논리적으로 표현됩니다. `[H, W]` Y 및 UV 평면 메타데이터가 포함된 텐서입니다. 아무것도 반환하지 않거나 빈 텐서를 반환하는 요청은 다음을 출력합니다. `frame=N rtsp_timeout` 그리고 루프를 중단합니다. 이는 일반적으로 URL이 잘못되었거나 스트림이 제대로 전송되지 않음을 의미합니다. 타임아웃은 중단된 스트림으로 인해 프로그램이 멈추는 것을 방지합니다.

**C++:** 프레임이 추출됩니다. `tensors_from_sample(*sample, true)`; 루프는 데이터를 읽기 전에 빈 목록인지 확인합니다. `shape`.

**Python:** 첫 번째 항목에서 프레임을 읽어옵니다. `sample.tensors` 모양을 인쇄하기 전에.

## Run

이 장에서는 실시간 RTSP 스트림을 사용하므로 연결 가능한
`--url`카메라가 없는 경우, RTSP 서버를 통해 호환되는 비디오를 게시하고 해당 비디오를 가리킵니다. `--url` 해당 작업을 수행합니다. **Python** 및 **C++(미리 빌드된 버전)** 명령을 실행합니다.Neat root 디렉터리(해당 디렉터리에 포함된)를 설치합니다. `share/` 그리고
`lib/`); 소스 코드를 기반으로 빌드하는 명령어를 실행하고, **저장소의 루트 디렉터리**에서 실행합니다.

자동화된 튜토리얼 회귀 테스트는 두 가지 코덱을 모두 실행합니다. 사용 가능한 첫 번째 URL을 읽습니다. `SIMANEAT_TEST_RTSP_H264_URL` 또는 `SIMANEAT_TEST_RTSP_H264_URLS`그리고
부터 `SIMANEAT_TEST_RTSP_H265_URL` 또는 `SIMANEAT_TEST_RTSP_H265_URLS`. 테스트는 각 소스를 검사하고, 감지된 FPS 값을 RTSP 그룹에 전달합니다.

**Python:**
```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --source-fps 30 --frames 5
```

H.265의 경우:

```bash
python3 share/sima-neat/tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py \
  --url rtsp://host:port/stream --codec hevc --source-fps 30 --frames 5
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_018_consume_rtsp_stream \
  --url rtsp://host:port/stream --codec h265 --source-fps 30 --frames 5
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_018_consume_rtsp_stream
./build/tutorials-standalone/tutorial_018_consume_rtsp_stream \
  --url rtsp://host:port/stream --codec h265 --source-fps 30 --frames 5
```

예상 출력(형식은 스트림의 해상도와 디코더 형식에 따라 달라짐):

```text
frame=0 shape=[720, 1280]
frame=1 shape=[720, 1280]
frame=2 shape=[720, 1280]
frame=3 shape=[720, 1280]
frame=4 shape=[720, 1280]
```

스트림에 연결할 수 없는 경우 대신 `frame=0 rtsp_timeout`가 표시됩니다. 이 장의 C++ 소스 코드를 사용자 지정 `CMakeLists.txt`와 함께 자신의 프로젝트에 통합하려면 (추가 폴더는 필요하지 않음), 랜딩 페이지의 [튜토리얼 실행 방법](/tutorials#compile-a-copy-yourself)을 참조하십시오.

## 소스 파일
- C++: `tutorials/018_consume_rtsp_stream/consume_rtsp_stream.cpp`
- Python: `tutorials/018_consume_rtsp_stream/consume_rtsp_stream.py`
- Python OpenCV FPS 프로브: `tutorials/018_consume_rtsp_stream/probe_rtsp_fps.py`
