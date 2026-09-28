# Encoder performance

The existing performance runner has a separate encoder suite so new references do not replace historical Core/decoder baselines:

```sh
python3 tests/perf/tools/run_perf_matrix.py --suite encoder --prebuilt-tests-dir build/tests --results-dir build/encoder-performance
```

Run it exclusively on Modalix with the intended Core library and matching native encoder plugin/runtime installed. The regular performance gate and Vulcan workflow run this suite sequentially after the existing matrix. Functional encoder and sender cases remain in ordinary CTest.

Each case warms the same session with 200 frames and measures at least 500 frames and two seconds three times. A separate run submits at least 500 frames, scheduled over at least two seconds at 95% of the median completed-frame rate. All measurements use four native outputs and prepared inputs. The median must reach 90% of its saved matching reference; attempted, accepted, encoded and delivered frames must agree. Sender cases also require every emitted RTP packet to arrive in order. Producer shortfall fails independently of accepted-frame loss. The producer submits the complete scheduled count instead of discarding the last frame at a timer boundary. Producer and delivery duration must stay within 5% of the scheduled duration plus the larger of 20 milliseconds or one frame period; lateness remains reported separately. Existing process and drain timeouts remain unchanged.

For longer qualification measurements, set `SIMA_PERF_ENCODER_QUALIFICATION=1` on the same runner command. This retains the previous 1,000-frame/ten-second throughput minimums and sixty-second paced measurement. The setting changes duration only; configuration coverage, warmup, repetitions, reference identity and loss checks stay the same.

The 13 cases cover standalone H.264/H.265/MJPEG with CPU and DMA input, legacy H.264 with DMA input, raw senders with DMA input and encoded passthrough for all three codecs. The legacy baseline and new H.264 DMA path use identical inputs and settings. Input hashes, raw plane layout and native encoder settings must match the saved reference. The hash covers every input frame and its caps; references store one shared raw layout and only settings relevant to the selected codec. Passthrough has no encoder settings. Encoded passthrough measures transport using prepared fixtures, not encoding throughput.

Each run retains JSON, diagnostics and a warmup bitstream capture. Completed-frame rate, encoder completion timing, output latency and paced counts are separate fields. Independent software decode/pixel checks belong outside the measured window. The emitter fails on incomplete drain or hung teardown.

References live in `baselines/v2/modalix_encoder`. They are first measured references for these Core paths, not evidence of improvement over an earlier implementation. Compare native Labs results only with matching standalone DMA settings. Keep actual environment and library identities with the qualification record when refreshing references; do not copy thresholds from another SDK or alter the input hash to make a mismatched run pass.

## Measured reference

These first references were measured on 2026-09-26 with Platform 3.0.0 build 1493, GCC 14 and GStreamer 1.26.2. Each value is the median of three warmed runs. Every corresponding 60-second run at 95% of its measured rate completed without frame loss, packet loss or producer shortfall.

| Path | Codec | Input | FPS |
|---|---|---|---:|
| legacy | H264 | DMA | 191.760651 |
| standalone | H264 | CPU | 189.256333 |
| standalone | H264 | DMA | 191.874578 |
| standalone | H265 | CPU | 239.566898 |
| standalone | H265 | DMA | 241.984181 |
| standalone | MJPEG | CPU | 390.583008 |
| standalone | MJPEG | DMA | 395.151141 |
| raw-sender | H264 | DMA | 191.371369 |
| raw-sender | H265 | DMA | 240.996281 |
| raw-sender | MJPEG | DMA | 24.989478 |
| encoded-sender | H264 | CPU | 3696.254126 |
| encoded-sender | H265 | CPU | 2132.899615 |
| encoded-sender | MJPEG | CPU | 572.802254 |

Encoded sender rows measure forwarding of prepared compressed inputs; their FPS is not encoder throughput. These references establish the 90% regression floor for matching workloads.

Raw MJPEG sending is limited to about 25 FPS in this measurement, while standalone MJPEG encoding reaches about 395 FPS and encoded passthrough reaches about 573 FPS. The raw-sender bottleneck remains unresolved; the saved reference records that limitation and does not establish its cause.
