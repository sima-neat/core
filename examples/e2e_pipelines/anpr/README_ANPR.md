# ANPR Quick Start

## 1) Start RTSP server

```bash
./utils/run_rtsp_server.sh --host-port 8555
```

## 2) Stream a video to RTSP

```bash
./utils/stream_cam.sh \
  --video-path /path/to/video.mp4 \
  --rtsp-host <RTSP_SERVER_IP> \
  --rtsp-port 8555 \
  --rtsp-path cam1 \
  --fps 25
```

## 3) Compile and run ANPR C++ (`main_anpr.cpp`)

From repo root (for example, `/root/workspace/PipelineSession`):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target anpr_demo -j 8
```

Run:

```bash
./build/anpr_demo \
  --rtsp-url rtsp://<RTSP_SERVER_IP>:8555/cam1 \
  --vehicle-model examples/e2e_pipelines/anpr/models/YOLO_TRAFFIC_384H_640W_accurate_mod_mpk.tar.gz \
  --ocr-model examples/e2e_pipelines/anpr/models/YOLO_OCR_256_mod_2_0_INT8_accurate_mpk.tar.gz \
  --udp-host <RECEIVER_IP> \
  --udp-port 5000 \
  --rtsp-fps 25
```

### C++ flags (`main_anpr.cpp`)

- `--vehicle-model <path>`
- Optional path to vehicle detector model pack (`.tar.gz`).
- If not provided, defaults to `<REPO_ROOT_SIMA>/models/YOLO_TRAFFIC_384H_640W_accurate_mod_mpk.tar.gz`.
- `--ocr-model <path>`
- Optional path to OCR model pack (`.tar.gz`).
- If not provided, defaults to `<REPO_ROOT_SIMA>/models/YOLO_OCR_256_mod_2_0_INT8_accurate_mpk.tar.gz`.
- `--rtsp-url <url>`
- Full RTSP URL. If set, this takes priority over `--rtsp-host/--rtsp-port`.
- `--rtsp-list <file>`
- Text file with RTSP URLs; first line is used.
- `--rtsp-host <host>`
- RTSP host used when `--rtsp-url` is not set.
- `--rtsp-port <port>`
- RTSP port used when `--rtsp-url` is not set.
- `--udp-host <host>`
- Destination host for encoded output (`udpsink`).
- `--udp-port <port>`
- Destination UDP port for encoded output.
- `--rtsp-fps <n>`
- Fallback FPS used when RTSP caps do not provide framerate.

### C++ env fallbacks

- `RTSP_HOST`, `RTSP_PORT`
- Used when `--rtsp-url` and corresponding host/port flags are not provided.
- `UDP_HOST` or `OUTPUT_HOST`, `UDP_PORT` or `VIDEO_PORT`
- Used when UDP flags are not provided.
- `RTSP_FPS`
- Used when `--rtsp-fps` is not provided.
- `OCR_THROTTLE`
- `1/true` enables OCR throttling (default), `0/false` disables it.
- `REPO_ROOT_SIMA`
- Base path for model and label files used by C++ pipeline.
- Default is `examples/e2e_pipelines/anpr`.

## 4) Visualize result stream

```bash
./utils/gst_receiver.sh --port 5000
```

## Notes

- Start step 1 before step 2, otherwise FFmpeg shows `Connection refused`.
